#include "stdio.h"

__global__ void zzpmemset(int *a,int *b,int size,int value)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < size)
    {
        a[i] = b[i] = value;
    }
    printf("i am in: %d\n",i);
}