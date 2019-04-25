#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define A values[0]
#define B values[1]
#define C values[2]
#define D values[3]
#define E values[4]
#define F values[5]
#define G values[6]
#define H values[7]

#define SHR(x,n) (x>>n)

#define ROTR(x,n) (SHR(x,n) | (x<<(32-n)))

#define s0(x) (ROTR(x,7)^ROTR(x,18)^SHR(x,3))
#define s1(x) (ROTR(x,17)^ROTR(x,19)^SHR(x,10))

#define S1(x) (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
#define S0(x) (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))

#define ch(x, y, z) ((x & (y ^ z)) ^ z)
#define maj(x, y, z) ((x & (y | z)) | (y & z))

__device__
static const uint32_t h[8] =
{
	0x6a09e667,
	0xbb67ae85,
	0x3c6ef372,
	0xa54ff53a,
	0x510e527f,
	0x9b05688c,
	0x1f83d9ab,
	0x5be0cd19
};

__device__
static const uint32_t k[64] =
{
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
   	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
   	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
   	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
   	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
   	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
   	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__
void sha256Check(uint32_t *out, int dif, int out_stride, uint8_t *state, int idx)
{
	if ((out[out_stride] >> (32-dif)) == 0)
	{
		state[idx] = 1;
	}
	else
		state[idx] = 0;
}

__device__
void sha256Compute(uint32_t *input, uint32_t *out, int idx, int input_stride, int out_stride)
{
	uint32_t pad[16];

	#pragma unroll 8
	for (int i=0;i< 8;i++)
		pad[i]  =  input[i+input_stride];

	pad[8] =  0x80000000;

	#pragma unroll 6
	for (int i=9;i<15;i++)
		pad[i] =  0x00000000;
	
	pad[15] =  0x00000100;

	uint32_t s[64];
	uint32_t values[8];

	#pragma unroll 16
	for(int i=0;i<16;i++)
		s[i]  = pad[i];
	
	#pragma unroll 48
	for(int i=16;i<64;i++)
		s[i] = s[i-16]+(s0(s[i-15]))+s[i-7]+(s1(s[i-2]));  
	
	A = h[0];
	B = h[1];
	C = h[2];
	D = h[3];
	E = h[4];
	F = h[5];
	G = h[6];
	H = h[7];

	uint32_t temp1;
	uint32_t temp2;

	#pragma unroll 64
	for (int i=0;i<64;i++)
	{
		temp1 = H+(S1(E))+(ch(E,F,G))+k[i]+s[i];
		temp2 = (S0(A))+(maj(A,B,C));

		H = G;
		G = F;
		F = E;
		E = D+temp1;
		D = C;
		C = B;
		B = A;
		A = temp1+temp2;
	}

	out[0+out_stride] = h[0]+A;
	out[1+out_stride] = h[1]+B;
	out[2+out_stride] = h[2]+C;
	out[3+out_stride] = h[3]+D;
	out[4+out_stride] = h[4]+E;
	out[5+out_stride] = h[5]+F;
	out[6+out_stride] = h[6]+G;
	out[7+out_stride] = h[7]+H;

}


__device__
void generateStrings(uint32_t *input, int inlen, uint32_t nonce, int input_stride, int idx)
{

	// Appending nonce to the end of the random arrays. 
	inlen = inlen-1;

	curandState state;
	curand_init(clock64(), idx, 0, &state);

	#pragma unroll 7
	for (int i = 0; i < inlen; i++)
	{
		uint32_t rand = curand_uniform(&state)*100000;
		input[i+input_stride] = rand;	
	}

	input[inlen+input_stride] = nonce;
}

__global__
void sha256Init(uint32_t *input, int inlen, uint32_t nonce, uint32_t *out, uint8_t *state, int dif)
{

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int input_stride = idx*inlen;
	int out_stride = idx*8;

	generateStrings(input,inlen,nonce,input_stride,idx);
	sha256Compute(input,out,idx,input_stride,out_stride);
	sha256Check(out,dif,out_stride,state,idx);
}

int main(int argc, char *argv[])
{
	int inlen = 8;
	int threads_per_block = 512;
	int blocks = 64;
	int threads = threads_per_block * blocks;
	uint32_t nonce;
	int dif = atoi(argv[1]);

	clock_t start, end;
	int counter = 0;
	int end_counter = atoi(argv[2]);

	start = clock();
	while(1)
	{
		uint32_t *d_input;	
		uint32_t *d_output;
		uint8_t *d_state;

		cudaMallocManaged(&d_input, threads*256);
		cudaMallocManaged(&d_output, threads*32);
		cudaMallocManaged(&d_state, threads);

		nonce = 0x41414141;

		sha256Init<<<blocks, threads_per_block>>>(d_input,inlen,nonce,d_output,d_state,dif);
		cudaDeviceSynchronize();

		for (int i=0; i<threads;i++)
		{
			if (counter >= end_counter)
				break;
			if(d_state[i] == 1)
			{
				counter++;
				//for(int j = 0; j < inlen; j++)
				//	printf("%02x", c_input[j+(i*inlen)]);
				//printf("\n");
			}
		}

		cudaFree(d_input);
		cudaFree(d_output);
		cudaFree(d_state);

		if (counter >= end_counter)
			break;

	}
	end = clock();
	double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("%d solutions in %f seconds at %d difficulty.\n", counter,cpu_time_used,dif);
	printf("%f MH/s\n", counter/cpu_time_used/1000000);

	cudaDeviceReset();

	return 0;
}