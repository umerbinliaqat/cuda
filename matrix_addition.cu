/*

//1 DIMENTIONAL MATRIX ADDITION

#include <stdio.h>
#define SIZE	1024

__global__
void VectorAdd(int* a, int* b, int* c, int n)
{
	int i = threadIdx.x;

    //for (i = 0; i < n; ++i)
    //enusres only the required number of threads are used in the execution
	if (i < n) 
		c[i] = a[i] + b[i];
}

int main()
{
	int *a, *b,  *c;

    //allow shared GPU and CPU access to the variables
	cudaMallocManaged(&a, SIZE * sizeof(int));
	cudaMallocManaged(&b, SIZE * sizeof(int));
	cudaMallocManaged(&c, SIZE * sizeof(int));

	for (int i = 0; i < SIZE; ++i)
	{
		a[i] = i;
		b[i] = i;
		c[i] = 0;
	}

    //kernel invocation
	VectorAdd <<< 1, SIZE >>> (a, b, c, SIZE);

	cudaDeviceSynchronize();

	for (int i = 0; i < 10; ++i)
		printf("c[%d] = %d\n", i, c[i]);
    
    //free up the used space after execution
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	return 0;
}
*/


//2 DIMENTIONAL MATRIX ADDITION

#include<stdio.h>
#include<cuda.h>
__global__ void matadd(int* l, int* m, int* n)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int id = gridDim.x * y + x;

    n[id] = l[id] + m[id];
}
int main()
{
    int a[2][3];
    int b[2][3];
    int c[2][3];
    int* d, * e, * f;
    int i, j;
    printf("\n Enter elements of first matrix of size 2 * 3\n");
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            scanf("%d", &a[i][j]);
        }
    }
    printf("\n Enter elements of second matrix of size 2 * 3\n");
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            scanf("%d", &b[i][j]);
        }
    }

    //allocate space for the arrays in the GPU memory
    cudaMalloc((void**)&d, 2 * 3 * sizeof(int));
    cudaMalloc((void**)&e, 2 * 3 * sizeof(int));
    cudaMalloc((void**)&f, 2 * 3 * sizeof(int));

    //transfer the data from the host to the device
    cudaMemcpy(d, a, 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e, b, 2 * 3 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(3, 2);
    //defining two dimensional Grid(collection of blocks) structure. Syntax is dim3 grid(no. of columns,no. of rows) */

    //kernel invocation
    matadd << <grid, 1 >> > (d, e, f);

    cudaDeviceSynchronize();

    cudaMemcpy(c, f, 2 * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSum of two matrices:\n ");
    for (i = 0; i < 2; i++)
    {
        for (j = 0; j < 3; j++)
        {
            printf("%d\t", c[i][j]);
        }
        printf("\n");
    }

    //Relenquish the memory space after execution 
    cudaFree(d);
    cudaFree(e);
    cudaFree(f);

    return 0;
}

// This program computes the sum of two vectors of length N
// By: Nick from CoffeeBeforeArch

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

// CUDA kernel for vector addition
// __global__ means this is called from the CPU, and runs on the GPU
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
    // Calculate global thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Boundary check
    if (tid < N) c[tid] = a[tid] + b[tid];
}

// Check vector add result
void verify_result(std::vector<int>& a, std::vector<int>& b,
    std::vector<int>& c) {
    for (int i = 0; i < a.size(); i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

void matrix_init(int* a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
    }
}

void error_check(int* a, int* b, int* c, int N) {
    for (int i = 0; i < N; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    // Array size of 2^16 (65536 elements)
    int N = 1 << 16;
    int *h_a, *h_b, *h_c;
    int* d_a, * d_b, * d_c;
    size_t bytes = sizeof(int) * N;

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrix_init(h_a, N);
    matrix_init(h_b, N);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;

    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, N);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    error_check(h_a, h_b, h_c, N);

    printf("COMPLETED SUCCESSFULLY\n");

    return 0;
}


