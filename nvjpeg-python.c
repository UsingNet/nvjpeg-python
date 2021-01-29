#include <stdio.h>
#include <nvjpeg.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


typedef struct
{
    nvjpegImage_t img;
    int nComponent;
    nvjpegChromaSubsampling_t subsampling;
    size_t height;
    size_t width;
}NvJpegPythonImage;

typedef struct{
    size_t size;
    unsigned char* data;
}NvJpegJpegData;

typedef struct
{
    nvjpegHandle_t nv_handle;
    nvjpegJpegState_t nv_statue;
    nvjpegEncoderState_t nv_enc_state;
}NvJpegPythonHandle;

NvJpegPythonHandle* NvJpegPython_createHandle(){
    NvJpegPythonHandle* handle = (NvJpegPythonHandle*)malloc(sizeof(NvJpegPythonHandle));
    
    nvjpegCreateSimple(&(handle->nv_handle));
    nvjpegJpegStateCreate(handle->nv_handle, &(handle->nv_statue));
    nvjpegEncoderStateCreate(handle->nv_handle, &(handle->nv_enc_state), NULL);

    return handle;
}

void NvJpegPython_destoryHandle(NvJpegPythonHandle** handle){
    nvjpegJpegStateDestroy((*handle)->nv_statue);
    nvjpegEncoderStateDestroy((*handle)->nv_enc_state);
    nvjpegDestroy((*handle)->nv_handle);
    free(*handle);
    *handle = NULL;
}

NvJpegPythonImage* NvJpegPython_createImage(int width, int height, int nComponent, nvjpegChromaSubsampling_t subsampling){
    unsigned char * pBuffer = NULL; 
    cudaError_t eCopy = cudaMalloc((void **)&pBuffer, width * height * NVJPEG_MAX_COMPONENT);
    if (cudaSuccess != eCopy)
    {
        printf("cudaMalloc failed : %s\n", cudaGetErrorString(eCopy));
        return NULL;
    }
    NvJpegPythonImage *imgdesc = (NvJpegPythonImage*)malloc(sizeof(NvJpegPythonImage));
    for(int i = 0;i<NVJPEG_MAX_COMPONENT;i++){
        imgdesc->img.channel[i] = pBuffer + (width*height*i);
        imgdesc->img.pitch[i] = width;
    }

    imgdesc->height = height;
    imgdesc->width = width;
    imgdesc->nComponent = nComponent;
    imgdesc->subsampling = subsampling;
    return imgdesc;
}

NvJpegPythonImage* NvJpegPython_createImageFromHost(int width, int height, const unsigned char* data, int nComponent){
    unsigned char * pBuffer = NULL; 
    cudaError_t eCopy = cudaMalloc((void **)&pBuffer, width * height * NVJPEG_MAX_COMPONENT);
    if (cudaSuccess != eCopy)
    {
        printf("cudaMalloc failed : %s\n", cudaGetErrorString(eCopy));
        return NULL;
    }
    NvJpegPythonImage *imgdesc = (NvJpegPythonImage*)malloc(sizeof(NvJpegPythonImage));
    imgdesc->img.channel[0] = pBuffer;
    
    for(int i = 0;i<NVJPEG_MAX_COMPONENT;i++){
        imgdesc->img.channel[i] = pBuffer + (width*height*i);
        imgdesc->img.pitch[i] = width;
    }
    cudaMemcpy(imgdesc->img.channel[0], data, width*height*3, cudaMemcpyHostToDevice);

    imgdesc->height = height;
    imgdesc->width = width;
    imgdesc->nComponent = nComponent;
    imgdesc->subsampling = NVJPEG_CSS_444;
    return imgdesc;
}

void NvJpegPython_destoryImage(NvJpegPythonImage** img){
    cudaFree((*img)->img.channel[0]);
    free(*img);
    *img = NULL;
}

NvJpegPythonImage* NvJpegPython_decode(NvJpegPythonHandle* handle, const unsigned char* jpegData, size_t length){
    nvjpegHandle_t nv_handle = handle->nv_handle;
    nvjpegJpegState_t nv_statue = handle->nv_statue;
    // nvjpegCreateSimple(&nv_handle);
    // nvjpegJpegStateCreate(nv_handle, &nv_statue);

    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    int nComponent = 0;
    nvjpegChromaSubsampling_t subsampling;
    nvjpegGetImageInfo(nv_handle, jpegData, length, &nComponent, &subsampling, widths, heights);

    NvJpegPythonImage* imgdesc = NvJpegPython_createImage(widths[0], heights[0], nComponent, subsampling);
    int nReturnCode = nvjpegDecode(nv_handle, nv_statue, jpegData, length, NVJPEG_OUTPUT_BGR, &(imgdesc->img), NULL);

    if(nReturnCode != 0)
    {
        printf("Error in nvjpegDecode. %d\n", nReturnCode);
        return NULL;
    }

    // nvjpegJpegStateDestroy(nv_statue);
    // nvjpegDestroy(nv_handle);

    return imgdesc;
}

unsigned char* NvJpegPythonImage2HostMemory(NvJpegPythonImage* img){
    size_t size = img->height*img->width*3;
    unsigned char* buffer = (unsigned char*)malloc(size);
    cudaMemcpy(buffer, img->img.channel[0], size, cudaMemcpyDeviceToHost);
    return buffer;
}

NvJpegJpegData* NvJpegPython_createJpegData(size_t size){
    NvJpegJpegData* jpegData = (NvJpegJpegData*)malloc(sizeof(NvJpegJpegData));
    jpegData->data = (unsigned char*)malloc(size);
    jpegData->size = size;
    return jpegData;
}

void NvJpegPython_destoryJpegData(NvJpegJpegData** jpegData){
    free((*jpegData)->data);
    free(*jpegData);
    *jpegData = NULL;
}

NvJpegJpegData* NvJpegPython_encode(NvJpegPythonHandle* handle, NvJpegPythonImage* img, int quality){
    nvjpegHandle_t nv_handle = handle->nv_handle;
    // nvjpegCreateSimple(&nv_handle);

    nvjpegEncoderState_t nv_enc_state = handle->nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;

    // nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, NULL);
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, NULL);

    nvjpegEncoderParamsSetQuality(nv_enc_params, quality, NULL);
    nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 1, NULL);
    nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, img->subsampling, NULL);

    nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &(img->img), NVJPEG_INPUT_BGR, img->width, img->height, NULL);

    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, NULL);
    
    NvJpegJpegData* jpegData = NvJpegPython_createJpegData(length);
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, jpegData->data, &(jpegData->size), NULL);

    nvjpegEncoderParamsDestroy(nv_enc_params);
    // nvjpegEncoderStateDestroy(nv_enc_state);
    // nvjpegDestroy(nv_handle);
    return jpegData;
}


#ifdef BUILD_TEST

int NvJpegPython_test(char* inputJpegFilePath, char* outputRawPath, char* outputJpegFilePath){
    struct stat fileInfo;
    if(stat(inputJpegFilePath, &fileInfo) == -1){
        printf("File %s Not Exist\n", inputJpegFilePath);
        return -1;
    }

    NvJpegPythonHandle* handle = NvJpegPython_createHandle();
    FILE* fp = fopen(inputJpegFilePath, "rb");
    unsigned char* jpegData = (unsigned char*)malloc(fileInfo.st_size);
    size_t size = fread(jpegData, 1, fileInfo.st_size, fp);
    printf("Read file with %ld size\n", size);
    fclose(fp);

    NvJpegPythonImage* img = NvJpegPython_decode(handle, jpegData, size);
    free(jpegData);
    int height = img->height;
    int width = img->width;
    int nComponent = img->nComponent;

    printf("Image Size: %d x %d\n", width, height);
    for(int i = 0;i < nComponent;i++){
        printf("Channel %d size: %ld \n", i, (long int)img->img.pitch[i]);
    }

    unsigned char* buffer = NvJpegPythonImage2HostMemory(img);
    NvJpegPython_destoryImage(&img);

    fp = fopen(outputRawPath, "wb");
    fwrite(buffer, 1, width*height*nComponent, fp);
    fclose(fp);

    const unsigned char* channels[NVJPEG_MAX_COMPONENT];
    for(int i = 0;i<nComponent;i++){
        channels[i] = buffer + (width*height*i);
    }

    img = NvJpegPython_createImageFromHost(width, height, channels[0], nComponent);
    free(buffer);

    NvJpegJpegData* jd = NvJpegPython_encode(handle, img, 70);
    printf("Jpeg Data Size: %ld\n", jd->size);

    fp = fopen(outputJpegFilePath, "wb");
    fwrite(jd->data, 1, jd->size, fp);
    fclose(fp);

    NvJpegPython_destoryImage(&img);
    NvJpegPython_destoryJpegData(&jd);

    NvJpegPython_destoryHandle(&handle);
    return 0;
}

int main(int args, char** argv){
    NvJpegPython_test(
        "./tests/test.jpg", 
        "./tests/out/c-test.bgr",
        "./tests/out/c-test.jpg"
    );
    return 0;
}

#endif

typedef struct
{
    PyObject_HEAD
    long long m_handle;
}NvJpeg;


static PyMemberDef NvJpeg_DataMembers[] =
{
        {(char*)"m_handle",   T_LONGLONG, offsetof(NvJpeg, m_handle),   0, (char*)"NvJpeg handle ptr"},
        {NULL, 0, 0, 0, NULL}
};


static void NvJpeg_init(NvJpeg* Self, PyObject* pArgs)
{
    Self->m_handle = (long long)(NvJpegPython_createHandle());
}

static void NvJpeg_Destruct(NvJpeg* Self)
{
    NvJpegPythonHandle* m_handle = (NvJpegPythonHandle*)Self->m_handle;
    NvJpegPython_destoryHandle(&m_handle);
    Py_TYPE(Self)->tp_free((PyObject*)Self);
}

static PyObject* NvJpeg_Str(NvJpeg* Self)
{
    return Py_BuildValue("s", "<nvjpeg-python.nvjpeg>");
}

static PyObject* NvJpeg_Repr(NvJpeg* Self)
{
    return NvJpeg_Str(Self);
}

static PyObject* NvJpeg_decode(NvJpeg* Self, PyObject* Argvs)
{
    NvJpegPythonHandle* m_handle = (NvJpegPythonHandle*)Self->m_handle;
    
    unsigned char* jpegData;
    int len;
    if(!PyArg_ParseTuple(Argvs, "y#", &jpegData, &len)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should jpegData byte string!");
        return NULL;
    }
    NvJpegPythonImage* img = NvJpegPython_decode(m_handle, (const unsigned char*)jpegData, len);

    unsigned char* data = NvJpegPythonImage2HostMemory(img);
    npy_intp dims[1] = {(npy_intp)((img->width)*(img->height))*3};
    PyObject* temp = PyArray_SimpleNewFromData(1, dims, NPY_UINT8, data);
    free(data);
    
    PyObject* shape = Py_BuildValue("(i,i,i)", 3, img->height, img->width);
    NvJpegPython_destoryImage(&img);
    PyObject* rtn = PyArray_Reshape((PyArrayObject*)temp, shape);
    Py_DECREF(shape);
    Py_DECREF(temp);
    temp = rtn;

    rtn = PyArray_SwapAxes((PyArrayObject*)temp, 0, 2);
    Py_DECREF(temp);
    temp = rtn;

    rtn = PyArray_SwapAxes((PyArrayObject*)temp, 0, 1);
    Py_DECREF(temp);

    return rtn;
}

static PyObject* NvJpeg_encode(NvJpeg* Self, PyObject* Argvs)
{
    PyArrayObject *vecin;
    unsigned int quality = 70;
    if (!PyArg_ParseTuple(Argvs, "O!|I", &PyArray_Type, &vecin, &quality)){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass BGR image numpy array!");
        return NULL;
    }

    if (NULL == vecin){
        Py_INCREF(Py_None);
        return Py_None;
    }

    if (PyArray_NDIM(vecin) != 3){
        PyErr_SetString(PyExc_ValueError, "Parse the argument FAILED! You should pass BGR image numpy array by height*width*channel!");
        return NULL;
    }

    if(quality>100){
        quality = 100;
    }

    NvJpegPythonHandle* m_handle = (NvJpegPythonHandle*)Self->m_handle;
    NvJpegPythonImage* img = NvJpegPython_createImageFromHost(PyArray_DIM(vecin, 0), PyArray_DIM(vecin, 1), (const unsigned char*)PyArray_BYTES(vecin), 3);
    NvJpegJpegData* data = NvJpegPython_encode(m_handle, img, quality);

    PyObject* rtn = PyByteArray_FromStringAndSize((const char*)data->data, data->size);
    NvJpegPython_destoryImage(&img);
    return rtn;
}


static PyMethodDef NvJpeg_MethodMembers[] =
{
        {"encode",  (PyCFunction)NvJpeg_encode,  METH_VARARGS,  "encode jpge"},
        {"decode", (PyCFunction)NvJpeg_decode, METH_VARARGS,  "decode jpeg"},
        {NULL, NULL, 0, NULL}
};


static PyTypeObject NvJpeg_ClassInfo =
{
        PyVarObject_HEAD_INIT(NULL, 0)"nvjpeg.NvJpeg",
        sizeof(NvJpeg),
        0,
        (destructor)NvJpeg_Destruct,
        NULL,NULL,NULL,NULL,
        (reprfunc)NvJpeg_Repr,
        NULL,NULL,NULL,NULL,NULL,
        (reprfunc)NvJpeg_Str,
        NULL,NULL,NULL,
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
        "NvJpeg Python Objects---Extensioned by nvjpeg",
        NULL,NULL,NULL,0,NULL,NULL,
        NvJpeg_MethodMembers,
        NvJpeg_DataMembers,
        NULL,NULL,NULL,NULL,NULL,0,
        (initproc)NvJpeg_init,
        NULL,
};


static PyModuleDef ModuleInfo =
{
        PyModuleDef_HEAD_INIT,
        "NvJpeg Module",
        "NvJpeg by Nvjpeg",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC
PyInit_nvjpeg(void) {
    PyObject * pReturn = NULL;
    NvJpeg_ClassInfo.tp_new = PyType_GenericNew;


    if(PyType_Ready(&NvJpeg_ClassInfo) < 0)
        return NULL;

    pReturn = PyModule_Create(&ModuleInfo);
    if(pReturn == NULL)
        return NULL;

    Py_INCREF(&ModuleInfo);
    PyModule_AddObject(pReturn, "NvJpeg", (PyObject*)&NvJpeg_ClassInfo);
    import_array();
    return pReturn;
}

