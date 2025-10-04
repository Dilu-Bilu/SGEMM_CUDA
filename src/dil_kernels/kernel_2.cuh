#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/
template <const uint BLOCKSIZE>
__global__ void kernel_2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint x = blockIdx.x*BLOCKSIZE + (threadIdx.x/BLOCKSIZE);
    const uint y = blockIdx.y*BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
        tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
    