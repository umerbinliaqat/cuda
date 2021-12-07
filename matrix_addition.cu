/*

//Umer Bin Liaqat - ubl203
//Presentation # 2 - GPUs in Robotics
//NVIDIA CUDA Matrix addition demostration

//1 DIMENTIONAL MATRIX ADDITION

#include <stdio.h>
#define SIZE	1024
__global__
void VectorAdd(int* a, int* b, int* c, int n)
{
    //Assigning a unique address to each thread
    int i = threadIdx.x;

    //for (i = 0; i < n; ++i) c[i] = a[i] + b[i]; 
    //enusres only the required number of threads are used in the execution
    if (i < n)
        c[i] = a[i] + b[i];
}
int main()
{
    int *a, *b,  *c;

    //allows shared GPU and CPU access to the variables
    cudaMallocManaged(&a, SIZE * sizeof(int));
    cudaMallocManaged(&b, SIZE * sizeof(int));
    cudaMallocManaged(&c, SIZE * sizeof(int));

    //initializing the arrays
    for (int i = 0; i < SIZE; ++i)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }

    //kernel invocation, code executed in one go as a single block with 1024 threads
    VectorAdd <<< 1, SIZE >>> (a, b, c, SIZE);

    cudaDeviceSynchronize();
    
    //display output
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
    //assigning unique id to each thread
    int id = gridDim.x * y + x;

    //martix addition
    n[id] = l[id] + m[id];
}
int main()
{
    int a[2][3];
    int b[2][3];
    int c[2][3];
    int* d, * e, * f;
    int i, j;

    printf("Enter elements of first matrix of size 2 * 3\n");

    //input taken from the user
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
