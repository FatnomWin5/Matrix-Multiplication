#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <cstdlib>

using namespace std;

__global__ void matrixMult(int N, const int* matrix_one, const int* matrix_two, int* matrix_res) {
    int i = N * (blockDim.y * blockIdx.y + threadIdx.y);
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int sum = 0;

    for (int k = 0; k < N; k++){
        sum += matrix_one[i + k] * matrix_two[k * N + j];
    }

    int ind = N * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    matrix_res[ind] = sum;
}

void randomiseMatrix(int* matrix, int N) {

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = rand() % 100;
        }
    }
    return;
}

int main(int argc, char** argv) {

    int N;
    int threads;

    printf("Enter matrix size (integer value): ");
    cin >> N;

    printf("Enter max number of threads for grid block (integer value): ");
    cin >> threads;

    int* matrix_one;
    int* matrix_two;
    int* matrix_res;

    size_t size = N * N * sizeof(int);

    matrix_one = (int*)malloc(size);
    matrix_two = (int*)malloc(size);
    matrix_res = (int*)malloc(size);

    randomiseMatrix(matrix_one, N);
    randomiseMatrix(matrix_two, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* calcMatrix_one = NULL;
    int* calcMatrix_two = NULL;
    int* calcMatrix_res = NULL;

    cudaMalloc((void**)&calcMatrix_one, size);
    cudaMalloc((void**)&calcMatrix_two, size);
    cudaMalloc((void**)&calcMatrix_res, size);

    cudaMemcpy(calcMatrix_one, matrix_one, size, cudaMemcpyHostToDevice);
    cudaMemcpy(calcMatrix_two, matrix_two, size, cudaMemcpyHostToDevice);

    float time;

    for (int thr = 1; thr <= threads; thr++) {
        dim3 threadsPerBlock = dim3(thr, thr);
        dim3 blocksPerGrid = dim3(N / thr, N / thr);

        cudaEventRecord(start, 0);
        matrixMult <<<blocksPerGrid, threadsPerBlock>>> (N, calcMatrix_one, calcMatrix_two, calcMatrix_res);
        cudaEventRecord(stop, 0);

        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&time, start, stop);

        printf("Number of threads for grid block: %d; Blocks per grid: %d; Number of seconds: %f", thr, (N / thr), (time / 1000));
        printf("\n");

        cudaMemcpy(matrix_res, calcMatrix_res, size, cudaMemcpyDeviceToHost);
    }

    cudaFree(calcMatrix_one);
    cudaFree(calcMatrix_two);
    cudaFree(calcMatrix_res);
    free(matrix_one);
    free(matrix_two);
    free(matrix_res);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
