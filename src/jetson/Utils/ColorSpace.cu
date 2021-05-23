/*
* Copyright 2017-2020 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <cuda_runtime.h>
#include "ColorSpace.hpp"

__constant__ float matYuv2Rgb[3][3];
__constant__ float matRgb2Yuv[3][3];


void inline GetConstants(int iMatrix, float &wr, float &wb, int &black, int &white, int &max) {
    black = 16; white = 235;
    max = 255;

    switch (iMatrix)
    {
    case ColorSpaceStandard_BT709:
    default:
        wr = 0.2126f; wb = 0.0722f;
        break;

    case ColorSpaceStandard_FCC:
        wr = 0.30f; wb = 0.11f;
        break;

    case ColorSpaceStandard_BT470:
    case ColorSpaceStandard_BT601:
        wr = 0.2990f; wb = 0.1140f;
        break;

    case ColorSpaceStandard_SMPTE240M:
        wr = 0.212f; wb = 0.087f;
        break;

    case ColorSpaceStandard_BT2020:
    case ColorSpaceStandard_BT2020C:
        wr = 0.2627f; wb = 0.0593f;
        // 10-bit only
        black = 64 << 6; white = 940 << 6;
        max = (1 << 16) - 1;
        break;
    }
}

void SetMatYuv2Rgb(int iMatrix) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        1.0f, 0.0f, (1.0f - wr) / 0.5f,
        1.0f, -wb * (1.0f - wb) / 0.5f / (1 - wb - wr), -wr * (1 - wr) / 0.5f / (1 - wb - wr),
        1.0f, (1.0f - wb) / 0.5f, 0.0f,
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * max / (white - black) * mat[i][j]);
        }
    }
    cudaMemcpyToSymbol(matYuv2Rgb, mat, sizeof(mat));
}


void SetMatRgb2Yuv(int iMatrix) {
    float wr, wb;
    int black, white, max;
    GetConstants(iMatrix, wr, wb, black, white, max);
    float mat[3][3] = {
        wr, 1.0f - wb - wr, wb,
        -0.5f * wr / (1.0f - wb), -0.5f * (1 - wb - wr) / (1.0f - wb), 0.5f,
        0.5f, -0.5f * (1.0f - wb - wr) / (1.0f - wr), -0.5f * wb / (1.0f - wr),
    };
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat[i][j] = (float)(1.0 * (white - black) / max * mat[i][j]);
        }
    }
    cudaMemcpyToSymbol(matRgb2Yuv, mat, sizeof(mat));
}

template<class T>
__device__ static T Clamp(T x, T lower, T upper) {
    return x < lower ? lower : (x > upper ? upper : x);
}

template<class Rgb, class YuvUnit>
__device__ inline Rgb YuvToRgbForPixel(YuvUnit y, YuvUnit u, YuvUnit v) {
    const int 
        low = 1 << (sizeof(YuvUnit) * 8 - 4),
        mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    float fy = (int)y - low, fu = (int)u - mid, fv = (int)v - mid;
    const float maxf = (1 << sizeof(YuvUnit) * 8) - 1.0f;
    YuvUnit 
        r = (YuvUnit)Clamp(matYuv2Rgb[0][0] * fy + matYuv2Rgb[0][1] * fu + matYuv2Rgb[0][2] * fv, 0.0f, maxf),
        g = (YuvUnit)Clamp(matYuv2Rgb[1][0] * fy + matYuv2Rgb[1][1] * fu + matYuv2Rgb[1][2] * fv, 0.0f, maxf),
        b = (YuvUnit)Clamp(matYuv2Rgb[2][0] * fy + matYuv2Rgb[2][1] * fu + matYuv2Rgb[2][2] * fv, 0.0f, maxf);
    
    Rgb rgb{};
    const int nShift = abs((int)sizeof(YuvUnit) - (int)sizeof(rgb.c.r)) * 8;
    if (sizeof(YuvUnit) >= sizeof(rgb.c.r)) {
        rgb.c.r = r >> nShift;
        rgb.c.g = g >> nShift;
        rgb.c.b = b >> nShift;
    } else {
        rgb.c.r = r << nShift;
        rgb.c.g = g << nShift;
        rgb.c.b = b << nShift;
    }
    return rgb;
}

template<class Rgb, class RgbIntx2>
__global__ static void Yuv420ToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t u = *(pYuv + nWidth * nHeight + y*((uint32_t)(nWidth/2))/2 + x/2);
    uint8_t v = *(pYuv + nWidth * nHeight + ((uint32_t)(nWidth/2))*((uint32_t)(nHeight/2)) + y*((uint32_t)(nWidth/2))/2 + x/2);
    
    
    for(int j=0;j<2;j++){
        for(int k=0;k<2;k++){
            uint8_t* pY = pYuv + nWidth*(y+k) + (x+j);
            Rgb c = YuvToRgbForPixel<Rgb>(pY[0], u, v);
            uint8_t* pC = (uint8_t*)&c;
            uint8_t *pOutput = (pRgb + (x+j) * 3 + (y+k) * nWidth * 3);
            pOutput[0] = pC[0];
            pOutput[1] = pC[1];
            pOutput[2] = pC[2];
        }
    }
}


template<class YuvUnitx2, class Rgb, class RgbIntx2>
__global__ static void Yuv444ToRgbKernel(uint8_t *pYuv, int nYuvPitch, uint8_t *pRgb, int nRgbPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    for(int j=0;j<2;j++){
        for(int k=0;k<2;k++){
            uint8_t Y = *(pYuv + (y+k)*nWidth + (x+j));
            uint8_t U = *(pYuv + nWidth * nHeight + (y+k)*nWidth + (x+j));
            uint8_t V = *(pYuv + nWidth * nHeight * 2 + (y+k)*nWidth + (x+j));

            Rgb c = YuvToRgbForPixel<Rgb>(Y, U, V);
            uint8_t* pC = (uint8_t*)&c;
            uint8_t *pOutput = (pRgb + (x+j) * 3 + (y+k) * nWidth * 3);
            pOutput[0] = pC[0];
            pOutput[1] = pC[1];
            pOutput[2] = pC[2];
        }
    }
}


template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToY(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit low = 1 << (sizeof(YuvUnit) * 8 - 4);
    return matRgb2Yuv[0][0] * r + matRgb2Yuv[0][1] * g + matRgb2Yuv[0][2] * b + low;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToU(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[1][0] * r + matRgb2Yuv[1][1] * g + matRgb2Yuv[1][2] * b + mid;
}

template<class YuvUnit, class RgbUnit>
__device__ inline YuvUnit RgbToV(RgbUnit r, RgbUnit g, RgbUnit b) {
    const YuvUnit mid = 1 << (sizeof(YuvUnit) * 8 - 1);
    return matRgb2Yuv[2][0] * r + matRgb2Yuv[2][1] * g + matRgb2Yuv[2][2] * b + mid;
}


template<class YuvUnitx2>
__global__ static void BgrToYuvKernel(uint8_t *pRgb, uint8_t *pYuv, int nYuvPitch, int nWidth, int nHeight) {
    int x = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    int y = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
    if (x + 1 >= nWidth || y + 1 >= nHeight) {
        return;
    }

    uint8_t *pSrc = pRgb + x * 3 + y * nWidth * 3;
    
    uint8_t *int2a = pSrc;
    uint8_t *int2b = pSrc + nWidth * 3;

    uint8_t b = (int2a[0] + int2a[3] + int2b[0] + int2b[3]) / 4,
        g = (int2a[1] + int2a[4] + int2b[1] + int2b[4]) / 4,
        r = (int2a[2] + int2a[5] + int2b[2] + int2b[5]) / 4;

    uint8_t *pDst = pYuv + x + y * nWidth;
    
    pDst[0] = RgbToY<uint8_t, uint8_t>(int2a[0+2], int2a[0+1], int2a[0+0]);
    pDst[1] = RgbToY<uint8_t, uint8_t>(int2a[1*3+2], int2a[1*3+1], int2a[1*3+0]);
    pDst[nWidth] = RgbToY<uint8_t, uint8_t>(int2b[0+2], int2b[0+1], int2b[0+0]);
    pDst[nWidth + 1] = RgbToY<uint8_t, uint8_t>(int2b[1*3+2], int2b[1*3+1], int2b[1*3+0]);
    *(pYuv + nWidth * nHeight + (size_t)(nWidth/2)*((size_t)(y/2)) + x/2) = RgbToU<uint8_t, uint8_t>(r, g, b);
    *(pYuv + nWidth * nHeight + (size_t)(nWidth/2) * (size_t)(nHeight/2) + (size_t)(nWidth/2)*((size_t)(y/2)) + x/2) = RgbToV<uint8_t, uint8_t>(r, g, b);
}


template <class COLOR32>
void YUV420ToColor32(uint8_t *dpYuv420, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv420ToRgbKernel<COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpYuv420, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

template <class COLOR32>
void YUV444ToColor32(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix) {
    SetMatYuv2Rgb(iMatrix);
    Yuv444ToRgbKernel<uchar2, COLOR32, uint2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2), dim3(32, 2) >>>
        (dpYUV444, nPitch, dpBgra, nBgraPitch, nWidth, nHeight);
}

void BGRToYUV420(uint8_t *dpBgra, uint8_t *dpYUV420, int nWidth, int nHeight, int iMatrix) {
    SetMatRgb2Yuv(iMatrix);
    BgrToYuvKernel<ushort2>
        <<<dim3((nWidth + 63) / 32 / 2, (nHeight + 3) / 2 / 2), dim3(32, 2)>>>
        (dpBgra, dpYUV420, 4*nWidth, nWidth, nHeight);
}


// Explicit Instantiation
template void YUV420ToColor32<BGRA32>(uint8_t *dpYuv420, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
template void YUV444ToColor32<BGRA32>(uint8_t *dpYUV444, int nPitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);

void BGRToYUV420(uint8_t *dpBgra, uint8_t *dpYUV420, int nWidth, int nHeight, int iMatrix);
