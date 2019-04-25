/*
Author: Eleftherios Amperiadis
*/

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define UL64(x) x##ULL

#define SHR(x, n) (x >> n)
#define ROTR(x, n) (SHR(x, n) | (x << (64 - n)))

#define S0(x) (ROTR(x, 1) ^ ROTR(x, 8) ^ SHR(x, 7))
#define S1(x) (ROTR(x, 19) ^ ROTR(x, 61) ^ SHR(x, 6))

#define S2(x) (ROTR(x, 28) ^ ROTR(x, 34) ^ ROTR(x, 39))
#define S3(x) (ROTR(x, 14) ^ ROTR(x, 18) ^ ROTR(x, 41))

#define F0(x, y, z) ((x & y) | (z & (x | y)))
#define F1(x, y, z) (z ^ (x & (y ^ z)))

#define P(a, b, c, d, e, f, g, h, x, K)      \
  {                                          \
    temp1 = h + S3(e) + F1(e, f, g) + K + x; \
    temp2 = S2(a) + F0(a, b, c);             \
    d += temp1;                              \
    h = temp1 + temp2;                       \
  }


__device__ static const uint64_t K[80] = 
{
    UL64(0x428A2F98D728AE22), UL64(0x7137449123EF65CD),
    UL64(0xB5C0FBCFEC4D3B2F), UL64(0xE9B5DBA58189DBBC),
    UL64(0x3956C25BF348B538), UL64(0x59F111F1B605D019),
    UL64(0x923F82A4AF194F9B), UL64(0xAB1C5ED5DA6D8118),
    UL64(0xD807AA98A3030242), UL64(0x12835B0145706FBE),
    UL64(0x243185BE4EE4B28C), UL64(0x550C7DC3D5FFB4E2),
    UL64(0x72BE5D74F27B896F), UL64(0x80DEB1FE3B1696B1),
    UL64(0x9BDC06A725C71235), UL64(0xC19BF174CF692694),
    UL64(0xE49B69C19EF14AD2), UL64(0xEFBE4786384F25E3),
    UL64(0x0FC19DC68B8CD5B5), UL64(0x240CA1CC77AC9C65),
    UL64(0x2DE92C6F592B0275), UL64(0x4A7484AA6EA6E483),
    UL64(0x5CB0A9DCBD41FBD4), UL64(0x76F988DA831153B5),
    UL64(0x983E5152EE66DFAB), UL64(0xA831C66D2DB43210),
    UL64(0xB00327C898FB213F), UL64(0xBF597FC7BEEF0EE4),
    UL64(0xC6E00BF33DA88FC2), UL64(0xD5A79147930AA725),
    UL64(0x06CA6351E003826F), UL64(0x142929670A0E6E70),
    UL64(0x27B70A8546D22FFC), UL64(0x2E1B21385C26C926),
    UL64(0x4D2C6DFC5AC42AED), UL64(0x53380D139D95B3DF),
    UL64(0x650A73548BAF63DE), UL64(0x766A0ABB3C77B2A8),
    UL64(0x81C2C92E47EDAEE6), UL64(0x92722C851482353B),
    UL64(0xA2BFE8A14CF10364), UL64(0xA81A664BBC423001),
    UL64(0xC24B8B70D0F89791), UL64(0xC76C51A30654BE30),
    UL64(0xD192E819D6EF5218), UL64(0xD69906245565A910),
    UL64(0xF40E35855771202A), UL64(0x106AA07032BBD1B8),
    UL64(0x19A4C116B8D2D0C8), UL64(0x1E376C085141AB53),
    UL64(0x2748774CDF8EEB99), UL64(0x34B0BCB5E19B48A8),
    UL64(0x391C0CB3C5C95A63), UL64(0x4ED8AA4AE3418ACB),
    UL64(0x5B9CCA4F7763E373), UL64(0x682E6FF3D6B2B8A3),
    UL64(0x748F82EE5DEFB2FC), UL64(0x78A5636F43172F60),
    UL64(0x84C87814A1F0AB72), UL64(0x8CC702081A6439EC),
    UL64(0x90BEFFFA23631E28), UL64(0xA4506CEBDE82BDE9),
    UL64(0xBEF9A3F7B2C67915), UL64(0xC67178F2E372532B),
    UL64(0xCA273ECEEA26619C), UL64(0xD186B8C721C0C207),
    UL64(0xEADA7DD6CDE0EB1E), UL64(0xF57D4F7FEE6ED178),
    UL64(0x06F067AA72176FBA), UL64(0x0A637DC5A2C898A6),
    UL64(0x113F9804BEF90DAE), UL64(0x1B710B35131C471B),
    UL64(0x28DB77F523047D84), UL64(0x32CAAB7B40C72493),
    UL64(0x3C9EBE0A15C9BEBC), UL64(0x431D67C49C100D4C),
    UL64(0x4CC5D4BECB3E42B6), UL64(0x597F299CFC657E2A),
    UL64(0x5FCB6FAB3AD6FAEC), UL64(0x6C44198C4A475817)
};

__device__ static const uint64_t h[8] = 
{
   	UL64(0x6A09E667F3BCC908),
	UL64(0xBB67AE8584CAA73B),
	UL64(0x3C6EF372FE94F82B),
	UL64(0xA54FF53A5F1D36F1),
	UL64(0x510E527FADE682D1),
	UL64(0x9B05688C2B3E6C1F),
	UL64(0x1F83D9ABFB41BD6B),
	UL64(0x5BE0CD19137E2179)
};

__device__
void sha512Compute(uint64_t *in, uint64_t *out, int stride)
{
	uint64_t pad[16];
	
	#pragma unroll 8
	for(int i=0;i<8;i++)
		pad[i] = in[i+stride];

	pad[8] = 0x8000000000000000;

	#pragma unroll 6
	for (int i=9;i<16;i++)
		pad[i] = 0x0000000000000000;

	pad[16] = 0x0000000000000200;

	uint64_t w[80];
	uint64_t A, B, C, D, E, F, G, H, temp1, temp2;
	
	#pragma unroll 16
	for(int i=0;i<16;i++)
		w[i] = pad[i];

	#pragma unroll 64
	for(int i=16;i<80;i++)
		w[i] = w[i-16] + (S0(w[i-15])) + w[i-7] + (S1(w[i-2]));		
	
	A = h[0];
  	B = h[1];
  	C = h[2];
  	D = h[3];
  	E = h[4];
  	F = h[5];
  	G = h[6];
  	H = h[7];
  
  	#pragma unroll 10
  	for (int i=0;i<80;)
  	{
	    P(A, B, C, D, E, F, G, H, w[i], K[i]);
	    i++;
	    P(H, A, B, C, D, E, F, G, w[i], K[i]);
	    i++;
	    P(G, H, A, B, C, D, E, F, w[i], K[i]);
	    i++;
	    P(F, G, H, A, B, C, D, E, w[i], K[i]);
	    i++;
	    P(E, F, G, H, A, B, C, D, w[i], K[i]);
	    i++;
	    P(D, E, F, G, H, A, B, C, w[i], K[i]);
	    i++;
	    P(C, D, E, F, G, H, A, B, w[i], K[i]);
	    i++;
	    P(B, C, D, E, F, G, H, A, w[i], K[i]);
    	i++;
  	} 

  	out[0+stride] += h[0]+A;
  	out[1+stride] += h[1]+B;
  	out[2+stride] += h[2]+C;
  	out[3+stride] += h[3]+D;
  	out[4+stride] += h[4]+E;
  	out[5+stride] += h[5]+F;
  	out[6+stride] += h[6]+G;
  	out[7+stride] += h[7]+H;	
}

__device__
void generateInput(uint64_t *in, uint64_t nonce, int stride, int idx)
{
	curandState state;
	curand_init(clock64(), idx, 0, &state);

	#pragma unroll 7
	for (int i = 0; i < 7; i++)
	{
		uint64_t rand = curand_uniform(&state)*100000;
		in[i+stride] = rand;	
	}
	in[7+stride] = nonce;
}

__device__
void sha512Validate(uint64_t *out, uint8_t *state, int dif, int idx, int stride)
{
	if ((out[stride] >> (64-dif)) == 0)
		state[idx] = 1;
	else
		state[idx] = 0;
}

__global__
void sha512Init(uint64_t *in, uint64_t *out, uint64_t nonce, int dif, uint8_t *state)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int stride = idx*8;

	generateInput(in,nonce,stride,idx);
	sha512Compute(in,out,stride);
	sha512Validate(out,state,dif,idx,stride);
}

int main(int argc, char *argv[])
{
	int thread_count = 256;
	int block_count = 128;
	int threads = thread_count*block_count; 
	uint64_t nonce = 0xFFFFFFFFFFFFFFFF;
	
	int dif = atoi(argv[1]);
	int end_counter = atoi(argv[2]); 

	clock_t start, end;
	int counter = 0;

	start = clock();
	while(1)
	{
		uint64_t *in;
		uint64_t *out;
		uint8_t *state;

		cudaMallocManaged(&in, threads*512);
		cudaMallocManaged(&out, threads*512);
		cudaMallocManaged(&state, threads);

		sha512Init<<<block_count,thread_count>>>(in,out,nonce,dif,state);
		cudaDeviceSynchronize();

		for(int i=0;i<threads;i++)
		{
			if (state[i] == true)
			{	
				if (counter >= end_counter)
					break;
				counter++;

				//for(int j=0;j<64;j++)
				//	printf("%.2x", hashed_array[i*64+j]);
				//printf("\n");
			}
		}
		
		cudaFree(in);
		cudaFree(out);
		cudaFree(state);

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
