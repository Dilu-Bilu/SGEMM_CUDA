#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint x = blockDim.x * blockIdx.x + threadIdx.x;
    const uint y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < M && y < N) {
        float temp = 0.0;
        for (int i=0; i < K; i++) {
            
        }
    }
} 