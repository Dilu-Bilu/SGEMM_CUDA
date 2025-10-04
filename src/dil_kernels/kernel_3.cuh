/*

Matrix sizes:
MxK * KxN = MxN

*/
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void kernel_3(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    // Let's first calculate exactly which block of C we need to output 
    const uint Crow = blockIdx.x;
    const uint Ccol = blockIdx.y;

    // Now let's load our blocks of shared memory, remember that this needs to be as big as one of our blocks  
    __shared__ float As[BLOCKSIZE*BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE*BLOCKSIZE];

    // Now we can load the thread's information to calculate which warp our thread is in with threadrow 
    const uint threadrow = threadIdx.x / BLOCKSIZE; 
    // threadcol is used to see what our thread's position is inside of our warp 
    const uint threadcol = threadIdx.x % BLOCKSIZE; 

    // Let's now move our pointers to where they should be starting in A and B and C
    A += K*Crow*BLOCKSIZE; // [Crow, 0] is where want to start
    B += Ccol*BLOCKSIZE; // [0, Ccol] is where we start for this 
    C +=  N*Crow*BLOCKSIZE + Ccol*BLOCKSIZE; // [Crow, Ccol], since this is MxN

    // Now let's start the main loop REMEMBER that K is just a number like 4092 and we increment by 32 each time 
    float temp = 0.0;
    for (int blkidx=0; blkidx < K; blkidx += BLOCKSIZE) {
        // We don't need to worry about blockidx because we just increment the pointers 
        As[threadrow*BLOCKSIZE + threadcol] = A[threadrow*K + threadcol];
        Bs[threadrow*BLOCKSIZE + threadcol] = B[threadrow*N + threadcol];

        __syncthreads();
        // Next, we increment the pointers 
        A += BLOCKSIZE;
        B +=  BLOCKSIZE*N; 
        // Now we can execute the inside of the loaded block 
        for (int i=0; i < BLOCKSIZE; i++) {
            // remember that the row we are loading is the thread row inside of the block and we need to increment by blocksize whenever we want to access a row 
            temp += As[threadrow*BLOCKSIZE + i] * Bs[threadcol + BLOCKSIZE*i];
        
        }
        // Final synch thrads before we load the next block in shared memory 
        __syncthreads();

    }
    C[threadrow * N + threadcol] = alpha * temp + beta * C[threadrow * N + threadcol];
}
    