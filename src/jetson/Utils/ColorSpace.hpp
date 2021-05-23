#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <vector_types.h>

typedef enum ColorSpaceStandard {
    ColorSpaceStandard_BT709 = 1,
    ColorSpaceStandard_Unspecified = 2,
    ColorSpaceStandard_Reserved = 3,
    ColorSpaceStandard_FCC = 4,
    ColorSpaceStandard_BT470 = 5,
    ColorSpaceStandard_BT601 = 6,
    ColorSpaceStandard_SMPTE240M = 7,
    ColorSpaceStandard_YCgCo = 8,
    ColorSpaceStandard_BT2020 = 9,
    ColorSpaceStandard_BT2020C = 10
} ColorSpaceStandard;

union BGRA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t b, g, r, a;
    } c;
};

union RGBA32 {
    uint32_t d;
    uchar4 v;
    struct {
        uint8_t r, g, b, a;
    } c;
};


#ifdef __cplusplus
}
#endif


template <class COLOR32>
void YUV420ToColor32(uint8_t *dpYUV420, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);

template <class COLOR32>
void YUV444ToColor32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix = 0);

void BGRToYUV420(uint8_t *dpBgra, uint8_t *dpYUV420, int nWidth, int nHeight, int iMatrix = 0);