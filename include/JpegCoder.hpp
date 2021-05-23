#pragma once

#include <sys/types.h>
#include <malloc.h>
#include <memory.h>
#include <iostream>
#include <exception>

class JpegCoderError: public std::runtime_error{
protected:
    int _code;
public:
    JpegCoderError(int code, const std::string& str):std::runtime_error(str){
        this->_code = code;
    }
    JpegCoderError(int code, const char* str):std::runtime_error(str){
        this->_code = code;
    }
    int code(){
        return this->_code;
    }
};

typedef enum
{
    JPEGCODER_CSS_444 = 0,
    JPEGCODER_CSS_422 = 1,
    JPEGCODER_CSS_420 = 2,
    JPEGCODER_CSS_440 = 3,
    JPEGCODER_CSS_411 = 4,
    JPEGCODER_CSS_410 = 5,
    JPEGCODER_CSS_GRAY = 6,
    JPEGCODER_CSS_UNKNOWN = -1
} JpegCoderChromaSubsampling;

typedef enum{
    JPEGCODER_PIXFMT_RGB         = 3,
    JPEGCODER_PIXFMT_BGR         = 4, 
    JPEGCODER_PIXFMT_RGBI        = 5, 
    JPEGCODER_PIXFMT_BGRI        = 6,
}JpegCoderColorFormat;

class JpegCoderImage{
public:
    void* img;
    JpegCoderChromaSubsampling subsampling;
    size_t height;
    size_t width;
    short nChannel;
    
    JpegCoderImage(size_t width, size_t height, short nChannel, JpegCoderChromaSubsampling subsampling);
    ~JpegCoderImage();
    void fill(const unsigned char* data);
    unsigned char* buffer();
};

class JpegCoderBytes{
public:
    size_t size;
    unsigned char* data;
    JpegCoderBytes(size_t size){
        this->data = (unsigned char*)malloc(size);
        this->size = size;
    }

    JpegCoderBytes(unsigned char* data, size_t size){
        this->data = data;
        this->size = size;
    }

    ~JpegCoderBytes(){
        if(this->data!=nullptr){
            free(this->data);
        }
    }
};

class JpegCoder{
protected:
    static void* _global_context;
    void* _local_context;
public:
    JpegCoder();
    ~JpegCoder();
    void ensureThread(long threadIdent);
    JpegCoderImage* decode(const unsigned char* jpegData, size_t length);
    JpegCoderBytes* encode(JpegCoderImage* img, int quality);
    static void cleanUpEnv();
};
