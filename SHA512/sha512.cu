/*
Author: Eleftherios Amperiadis
Date: 02.17.2019
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "sha512_init.c"

void verifyLeadingZeroes(unsigned char *hash, int leading_zero, int hash_offset, bool* hashed_winner,
	int idx);

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

__device__ static const uint64_t H_array[8] = 
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
void computeHash(unsigned char *padded_array, int size, unsigned char *hashed_array,
	int idx, bool* hashed_winner)
{
	uint64_t s0, s1;
	uint64_t w[80];
	uint64_t A, B, C, D, E, F, G, H, temp1, temp2;
	uint64_t state[8];

	int thread_offset_pad = idx*128;
	int thread_offset_sha512 = idx*64;

	for(int i=0;i<16;i++)
	{	
		for(int j=0;j<8;j++)
		{
			w[i] <<= 8;
			w[i] |= (uint64_t)padded_array[i*8+j+thread_offset_pad];
		}
	}

	for(int i=16;i<80;i++)
	{	
			s0 = S0(w[i-15]);
			s1 = S1(w[i-2]);
			w[i] = w[i-16] + s0 + w[i-7] + s1;		
	}

	A = H_array[0];
  	B = H_array[1];
  	C = H_array[2];
  	D = H_array[3];
  	E = H_array[4];
  	F = H_array[5];
  	G = H_array[6];
  	H = H_array[7];
  	int i = 0;

  	do {
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
  	} while (i < 80);

  	state[0] = H_array[0];
  	state[1] = H_array[1];
  	state[2] = H_array[2];
  	state[3] = H_array[3];
  	state[4] = H_array[4];
  	state[5] = H_array[5];
  	state[6] = H_array[6];
  	state[7] = H_array[7];

  	state[0] += A;
  	state[1] += B;
  	state[2] += C;
  	state[3] += D;
  	state[4] += E;
  	state[5] += F;
  	state[6] += G;
  	state[7] += H;
	
  	for(int i=0;i<8;i++)
  	{
	  	hashed_array[thread_offset_sha512+(i*8)]    = state[i] >> 56;
		hashed_array[thread_offset_sha512+(i*8)+1]  = state[i] >> 48;
		hashed_array[thread_offset_sha512+(i*8)+2]  = state[i] >> 40;
		hashed_array[thread_offset_sha512+(i*8)+3]  = state[i] >> 32;
		hashed_array[thread_offset_sha512+(i*8)+4]  = state[i] >> 24;
	    hashed_array[thread_offset_sha512+(i*8)+5]  = state[i] >> 16;
	    hashed_array[thread_offset_sha512+(i*8)+6]  = state[i] >>  8;
	    hashed_array[thread_offset_sha512+(i*8)+7]  = state[i];
	}

	int difficulty = 10;
	
	verifyLeadingZeroes(hashed_array, difficulty, thread_offset_sha512, hashed_winner, idx);
}

__device__
void verifyLeadingZeroes(unsigned char *hash, int leading_zero, int hash_offset, bool* hashed_winner, int idx)
{
	for(int i=0;i<64;i++)
	{
		for (int j=0;j<8;j++)
		{
			if(leading_zero == 0)
			{
				hashed_winner[idx] = true;
				i = 64;
				break;
			}
			if(((hash[i+hash_offset] >> j) & 0x01) != 0)
			{
				i = 64;
				break;
			}
			else
				leading_zero--;
		}
	}
}

__global__
void padding(unsigned char *message, int size, unsigned char *hashed_array, 
	unsigned char *padded_array, bool* hashed_winner)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("X: %d\n", idx);

	int thread_offset = idx*128;
	int message_offset = idx*size;
	int pad_offset = 112;
	


	for(int i=0;i<size;i++)
		padded_array[i+thread_offset]=message[i+message_offset];
	
	padded_array[size+thread_offset] = 0x80;

	for(int i=size+1+thread_offset;i<pad_offset+thread_offset;i++)
		padded_array[i] = 0x00;

	uint64_t val = size*8;

	for(int i=pad_offset+thread_offset;i<pad_offset+thread_offset+8;i++)
		padded_array[i] = 0x00;

	padded_array[pad_offset+thread_offset+8]  =  val >> 56;
	padded_array[pad_offset+thread_offset+9]  =  val >> 48;
	padded_array[pad_offset+thread_offset+10] =  val >> 40;
	padded_array[pad_offset+thread_offset+11] =  val >> 32;
	padded_array[pad_offset+thread_offset+12] =  val >> 24;
    padded_array[pad_offset+thread_offset+13] =  val >> 16;
    padded_array[pad_offset+thread_offset+14] =  val >>  8;
    padded_array[pad_offset+thread_offset+15] =  val >>  0;

	computeHash((unsigned char*)padded_array, 128, (unsigned char*)hashed_array, idx, hashed_winner);
}

int main(void)
{
	int array_len = 1024;
	int cuda_blocks = 1;
	int string_len = 10;
	int counter = 0;

	clock_t t;
	t = clock();
	while(1)
	{
		unsigned char* temp_array = (unsigned char*)malloc(array_len*string_len*sizeof(unsigned char));

		unsigned char* message_array;
		unsigned char* hashed_array;
		unsigned char* padded_array;
		bool* hashed_winner;

		cudaMallocManaged(&message_array, (cuda_blocks*array_len*string_len*sizeof(unsigned char)));
		cudaMallocManaged(&hashed_array, (array_len*cuda_blocks*64*sizeof(unsigned char)));
		cudaMallocManaged(&padded_array, (array_len*cuda_blocks*128*sizeof(unsigned char)));
		cudaMallocManaged(&hashed_winner, (array_len*cuda_blocks*sizeof(bool)));
	
		temp_array = generateArray(array_len*cuda_blocks, string_len);
		for(int i=0; i<array_len*string_len;i++)
		{
			message_array[i] = temp_array[i];
		}

		free(temp_array);
		
		padding<<<cuda_blocks,array_len>>>(message_array, string_len, hashed_array, padded_array, hashed_winner);
		cudaDeviceSynchronize();

	
		for(int i=0;i<array_len*cuda_blocks;i++)
		{
			if (hashed_winner[i] == true)
			{
			//	counter++;
				//printf("%d\n", counter);
				for(int j=0;j<64;j++)
					printf("%.2x", hashed_array[i*64+j]);
				printf("\n");
			}
		}
		

		cudaFree(message_array);
		cudaFree(hashed_array);
		cudaFree(padded_array);
		cudaFree(hashed_winner);
		//break;
	}
	t = clock() - t;
	double time_taken = t/CLOCKS_PER_SEC;
	//printf("%d hashes in %f seconds.\n", counter, time_taken);
	cudaDeviceReset();
	return 0;
}