#include <stdio.h>

__global__ void helloFromGPU(void)
{
	printf("Hello World from GPU thread: %d\n", threadIdx.x);
}

int main(void)
{
	printf("Hello World from the CPU!\n");
	
	helloFromGPU<<<1, 10>>>();
	cudaDeviceReset();
	return 0;
}