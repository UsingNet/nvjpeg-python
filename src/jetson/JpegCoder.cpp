#include <JpegCoder.hpp>
#include <NvJpegDecoder.h>
#include <NvJpegEncoder.h>
#include "Utils/ColorSpace.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <CUDAHelper.h>
#include <map>

#ifndef NVJPEG_MAX_COMPONENT
#define NVJPEG_MAX_COMPONENT 4
#endif

void* JpegCoder::_global_context = nullptr;

typedef struct
{
    NvJPEGEncoder* nv_encoder;
    NvJPEGDecoder* nv_decoder;
    std::map<long, CUcontext*>* cudaContextMap;
}NvJpegGlobalContext;

#define JPEGCODER_GLOBAL_CONTEXT ((NvJpegGlobalContext*)(JpegCoder::_global_context))

// typedef struct
// {
// }NvJpegLocalContext;

// #define JPEGCODER_LOCAL_CONTEXT ((NvJpegLocalContext*)(this->_local_context))

JpegCoderImage::JpegCoderImage(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling){
    this->img = malloc(width * height * nChannel);
    this->height = height;
    this->width = width;
    this->nChannel = nChannel;
    this->subsampling = subsampling;
}

void JpegCoderImage::fill(const unsigned char* data){
    memcpy(img, data, width*height*nChannel);
    this->subsampling = JPEGCODER_CSS_444;
}

unsigned char* JpegCoderImage::buffer(){
    void* rtn = malloc(width * height * nChannel);
    memcpy(rtn, img, width * height * nChannel);
    return (unsigned char*)rtn;
}

JpegCoderImage::~JpegCoderImage(){
    free(this->img);
    this->img = nullptr;
}


JpegCoder::JpegCoder(){
    if(JpegCoder::_global_context == nullptr){
        JpegCoder::_global_context = malloc(sizeof(NvJpegGlobalContext));
        JPEGCODER_GLOBAL_CONTEXT->cudaContextMap = new std::map<long, CUcontext*>();
        JPEGCODER_GLOBAL_CONTEXT->nv_decoder = NvJPEGDecoder::createJPEGDecoder("nvjpeg-python:decoder");
        JPEGCODER_GLOBAL_CONTEXT->nv_encoder = NvJPEGEncoder::createJPEGEncoder("nvjpeg-python:encoder");
    }
}

JpegCoder::~JpegCoder(){
    // ArgusSamples::cleanupCUDA((CUcontext*)_local_context);
    // free(_local_context);
    _local_context = nullptr;
}

void JpegCoder::cleanUpEnv(){
    if(JpegCoder::_global_context != nullptr) {
      delete(JPEGCODER_GLOBAL_CONTEXT->nv_decoder);
      delete(JPEGCODER_GLOBAL_CONTEXT->nv_encoder);
      for(auto cudaContext: *(JPEGCODER_GLOBAL_CONTEXT->cudaContextMap)){
         ArgusSamples::cleanupCUDA(cudaContext.second);
      }
      delete(JPEGCODER_GLOBAL_CONTEXT->cudaContextMap);
    //   ArgusSamples::cleanupCUDA(&(JPEGCODER_GLOBAL_CONTEXT->g_cudaContext));
      free(JpegCoder::_global_context);
      JpegCoder::_global_context = nullptr;
    }
}

void JpegCoder::ensureThread(long threadIdent){
    // printf("threadIdent Id: %ld\n", threadIdent);
    if(JPEGCODER_GLOBAL_CONTEXT->cudaContextMap->count(threadIdent) == 0){
        CUcontext* context = (CUcontext*)malloc(sizeof(CUcontext));
        ArgusSamples::initCUDA(context);
        (*JPEGCODER_GLOBAL_CONTEXT->cudaContextMap)[threadIdent] = context;
    }
}

JpegCoderImage* JpegCoder::decode(const unsigned char* jpegData, size_t length){
    NvJPEGDecoder* nv_decoder = JPEGCODER_GLOBAL_CONTEXT->nv_decoder;

    uint32_t pixfmt, width, height;
    NvBuffer* buffer;
    int nReturnCode = nv_decoder->decodeToBuffer(&buffer, (unsigned char*)jpegData, length, &pixfmt, &width, &height);
    if (nReturnCode != 0){
        throw JpegCoderError(nReturnCode, "NvJpeg Decoder Error");
    }
    JpegCoderChromaSubsampling subsampling;
    switch (pixfmt)
    {
    case V4L2_PIX_FMT_YUV420M:
        subsampling = JPEGCODER_CSS_420;
        break;
    case V4L2_PIX_FMT_YUV444M:    
        subsampling = JPEGCODER_CSS_444;
        break;
    default:
        throw JpegCoderError(pixfmt, "Unknown pixfmt");
    }
    JpegCoderImage* imgdesc = new JpegCoderImage(width, height, NVJPEG_MAX_COMPONENT, subsampling);
    char* img_data = (char*)imgdesc->img;
    int frameSize = 0;
    for (u_int32_t i = 0; i < buffer->n_planes; i++)
    {
        NvBuffer::NvBufferPlane &plane = buffer->planes[i];
        size_t bytes_to_write =
            plane.fmt.bytesperpixel * plane.fmt.width * plane.fmt.height;
        memcpy(img_data, plane.data, bytes_to_write);
        img_data += bytes_to_write;
        frameSize += bytes_to_write;
    }
    CUdeviceptr dpFrame = 0, nv12Frame = 0;
    CUresult error_code = cuMemAlloc(&dpFrame, width * height * 4);
    if(error_code != CUDA_SUCCESS){
        throw JpegCoderError(error_code, "cuMemAlloc Error");
    }
    error_code = cuMemAlloc(&nv12Frame, frameSize);
    if(error_code != CUDA_SUCCESS){
        throw JpegCoderError(error_code, "cuMemAlloc Error");
    }
    cudaError_t eCopy = cudaMemcpy((void*)nv12Frame, imgdesc->img, frameSize, cudaMemcpyHostToDevice);
    if(eCopy != cudaSuccess){
        throw JpegCoderError(error_code, cudaGetErrorString(eCopy));
    }
    switch(subsampling){
        case JPEGCODER_CSS_420:
            YUV420ToColor32<BGRA32>((uint8_t*)nv12Frame, width, (uint8_t *)dpFrame, 4 * width, width, height);
        break;
        case JPEGCODER_CSS_444:
            YUV444ToColor32<BGRA32>((uint8_t*)nv12Frame, width, (uint8_t *)dpFrame, 4 * width, width, height);
        break;
        default:
            throw JpegCoderError(pixfmt, "Unknown pixfmt");
    }
    int output_size = width * height * 4;
    eCopy = cudaMemcpy(imgdesc->img, (uint8_t*)dpFrame, output_size, cudaMemcpyDeviceToHost);
    if(eCopy != cudaSuccess){
        throw JpegCoderError(error_code, cudaGetErrorString(eCopy));
    }
    cuMemFree(dpFrame);
    cuMemFree(nv12Frame);
    delete(buffer);
    return imgdesc;
}

JpegCoderBytes* JpegCoder::encode(JpegCoderImage* img, int quality){
    NvJPEGEncoder *nv_encodere = JPEGCODER_GLOBAL_CONTEXT->nv_encoder;

    NvBuffer buffer(V4L2_PIX_FMT_YUV420M, img->width, img->height, 0);
    buffer.allocateMemory();

    
    CUdeviceptr bgrFrame = 0, yuvFrame = 0;
    CUresult error_code = cuMemAlloc(&bgrFrame, img->width * img->height * 3);
    if(error_code != CUDA_SUCCESS){
        throw JpegCoderError(error_code, "cuMemAlloc Error");
    }
    size_t yuvframeSize = img->width*img->height + ((int)(img->width/2) * (int)(img->height/2))*2;
    char* yuv_data = (char*)malloc(yuvframeSize);
    error_code = cuMemAlloc(&yuvFrame, yuvframeSize);
    if(error_code != CUDA_SUCCESS){
        throw JpegCoderError(error_code, "cuMemAlloc Error");
    }
    cudaError_t eCopy = cudaMemcpy((void*)bgrFrame, img->img, img->width * img->height * 3, cudaMemcpyHostToDevice);
    if(eCopy != cudaSuccess){
        throw JpegCoderError(error_code, cudaGetErrorString(eCopy));
    }
    BGRToYUV420((uint8_t*)bgrFrame, (uint8_t*)yuvFrame, img->width, img->height);
    eCopy = cudaMemcpy(yuv_data, (uint8_t*)yuvFrame, yuvframeSize, cudaMemcpyDeviceToHost);
    if(eCopy != cudaSuccess){
        throw JpegCoderError(error_code, cudaGetErrorString(eCopy));
    }

    char* img_data = yuv_data;
    for (uint32_t i = 0; i < buffer.n_planes; i++)
    {
        NvBuffer::NvBufferPlane &plane = buffer.planes[i];
        char* data = (char *) plane.data;
        plane.bytesused = plane.fmt.stride * plane.fmt.height;
        memcpy(data, img_data, plane.bytesused);
        img_data+=plane.bytesused;
    }
    free(yuv_data);
    cuMemFree(bgrFrame);
    cuMemFree(yuvFrame);


    unsigned long out_buf_size = img->width * img->height * 3 / 2;
    unsigned char *out_buf = new unsigned char[out_buf_size];
    int nReturnCode = nv_encodere->encodeFromBuffer(buffer, JCS_YCbCr, &out_buf, out_buf_size, quality);
    if (0 != nReturnCode){
        throw JpegCoderError(nReturnCode, "NvJpeg Encoder Error");
    }
    
    JpegCoderBytes* jpegData = new JpegCoderBytes(out_buf, out_buf_size);
    return jpegData;
}

