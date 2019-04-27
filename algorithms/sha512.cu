/*
Author: Eleftherios Amperiadis
*/

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <curand_kernel.h>

#define DEVICE 0

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

__global__
void sha512Compute(uint64_t *valid_in, int dif,unsigned long long int *counter)
{
	uint64_t w[80];
	uint64_t A, B, C, D, E, F, G, H, temp1, temp2;
	bool passed = false;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curandState state;
	curand_init(clock64(), idx, 0, &state);

	while(1)
	{
		w[0] = curand(&state);
		w[1] = curand(&state);
		w[2] = curand(&state);
		w[3] = curand(&state);
		w[4] = curand(&state);
		w[5] = curand(&state);
		w[6] = curand(&state);
		w[7] = curand(&state);

		if (passed == false)
		{
			w[8]  = 0x8000000000000000;

			w[9]  = 0x0000000000000000;
			w[10] = 0x0000000000000000;
			w[11] = 0x0000000000000000;
			w[12] = 0x0000000000000000;
			w[13] = 0x0000000000000000;
			w[14] = 0x0000000000000000;

			w[15] = 0x0000000000000200;
			passed = true;
		}

		
		w[16] = w[0] + (S0(w[1])) + w[9] + (S1(w[14]));
		w[17] = w[1] + (S0(w[2])) + w[10] + (S1(w[15]));
		w[18] = w[2] + (S0(w[3])) + w[11] + (S1(w[16]));
		w[19] = w[3] + (S0(w[4])) + w[12] + (S1(w[17]));
		w[20] = w[4] + (S0(w[5])) + w[13] + (S1(w[18]));
		w[21] = w[5] + (S0(w[6])) + w[14] + (S1(w[19]));
		w[22] = w[6] + (S0(w[7])) + w[15] + (S1(w[20]));
		w[23] = w[7] + (S0(w[8])) + w[16] + (S1(w[21]));


		w[24] = w[8] + (S0(w[9])) + w[17] + (S1(w[22]));
		w[25] = w[9] + (S0(w[10])) + w[18] + (S1(w[23]));
		w[26] = w[10] + (S0(w[11])) + w[19] + (S1(w[24]));
		w[27] = w[11] + (S0(w[12])) + w[20] + (S1(w[25]));
		w[28] = w[12] + (S0(w[13])) + w[21] + (S1(w[26]));
		w[29] = w[13] + (S0(w[14])) + w[22] + (S1(w[27]));
		w[30] = w[14] + (S0(w[15])) + w[23] + (S1(w[28]));


		w[31] = w[15] + (S0(w[16])) + w[24] + (S1(w[29]));
		w[32] = w[16] + (S0(w[17])) + w[25] + (S1(w[30]));
		w[33] = w[17] + (S0(w[18])) + w[26] + (S1(w[31]));
		w[34] = w[18] + (S0(w[19])) + w[27] + (S1(w[32]));
		w[35] = w[19] + (S0(w[20])) + w[28] + (S1(w[33]));
		w[36] = w[20] + (S0(w[21])) + w[29] + (S1(w[34]));
		w[37] = w[21] + (S0(w[22])) + w[30] + (S1(w[35]));
		w[38] = w[22] + (S0(w[23])) + w[31] + (S1(w[36]));
		w[39] = w[23] + (S0(w[24])) + w[32] + (S1(w[37]));

	
		w[40] = w[24] + (S0(w[25])) + w[33] + (S1(w[38]));
		w[41] = w[25] + (S0(w[26])) + w[34] + (S1(w[39]));
		w[42] = w[26] + (S0(w[27])) + w[35] + (S1(w[40]));
		w[43] = w[27] + (S0(w[28])) + w[36] + (S1(w[41]));
		w[44] = w[28] + (S0(w[29])) + w[37] + (S1(w[42]));
		w[45] = w[29] + (S0(w[30])) + w[38] + (S1(w[43]));



		w[46] = w[30] + (S0(w[31])) + w[39] + (S1(w[44]));
		w[47] = w[31] + (S0(w[32])) + w[40] + (S1(w[45]));
		w[48] = w[32] + (S0(w[33])) + w[41] + (S1(w[46]));
		w[49] = w[33] + (S0(w[34])) + w[42] + (S1(w[47]));
		w[50] = w[34] + (S0(w[35])) + w[43] + (S1(w[48]));
		w[51] = w[35] + (S0(w[36])) + w[44] + (S1(w[49]));
		w[52] = w[36] + (S0(w[37])) + w[45] + (S1(w[50]));
		w[53] = w[37] + (S0(w[38])) + w[46] + (S1(w[51]));
		w[54] = w[38] + (S0(w[39])) + w[47] + (S1(w[52]));
		w[55] = w[39] + (S0(w[40])) + w[48] + (S1(w[53]));
		w[56] = w[40] + (S0(w[41])) + w[49] + (S1(w[54]));
		w[57] = w[41] + (S0(w[42])) + w[50] + (S1(w[55]));
		w[58] = w[42] + (S0(w[43])) + w[51] + (S1(w[56]));
		w[59] = w[43] + (S0(w[44])) + w[52] + (S1(w[57]));
		w[60] = w[44] + (S0(w[45])) + w[53] + (S1(w[58]));
		w[61] = w[45] + (S0(w[46])) + w[54] + (S1(w[59]));
		w[62] = w[46] + (S0(w[47])) + w[55] + (S1(w[60]));
		w[63] = w[47] + (S0(w[48])) + w[56] + (S1(w[61]));
		w[64] = w[48] + (S0(w[49])) + w[57] + (S1(w[62]));
		w[65] = w[49] + (S0(w[50])) + w[58] + (S1(w[63]));
		w[66] = w[50] + (S0(w[51])) + w[59] + (S1(w[64]));
		w[67] = w[51] + (S0(w[52])) + w[60] + (S1(w[65]));
		w[68] = w[52] + (S0(w[53])) + w[61] + (S1(w[66]));
		w[69] = w[53] + (S0(w[54])) + w[62] + (S1(w[67]));
		w[70] = w[54] + (S0(w[55])) + w[63] + (S1(w[68]));
		w[71] = w[55] + (S0(w[56])) + w[64] + (S1(w[69]));
		w[72] = w[56] + (S0(w[57])) + w[65] + (S1(w[70]));
		w[73] = w[57] + (S0(w[58])) + w[66] + (S1(w[71]));
		w[74] = w[58] + (S0(w[59])) + w[67] + (S1(w[72]));
		w[75] = w[59] + (S0(w[60])) + w[68] + (S1(w[73]));
		w[76] = w[60] + (S0(w[61])) + w[69] + (S1(w[74]));
		w[77] = w[61] + (S0(w[62])) + w[70] + (S1(w[75]));
		w[78] = w[62] + (S0(w[63])) + w[71] + (S1(w[76]));
		w[79] = w[63] + (S0(w[64])) + w[72] + (S1(w[77]));	
				
		
		A = h[0];
	  	B = h[1];
	  	C = h[2];
	  	D = h[3];
	  	E = h[4];
	  	F = h[5];
	  	G = h[6];
	  	H = h[7];
	  
		P(A, B, C, D, E, F, G, H, w[0], K[0]);  
		P(H, A, B, C, D, E, F, G, w[1], K[1]); 
		P(G, H, A, B, C, D, E, F, w[2], K[2]);  
		P(F, G, H, A, B, C, D, E, w[3], K[3]);  
		P(E, F, G, H, A, B, C, D, w[4], K[4]);
		P(D, E, F, G, H, A, B, C, w[5], K[5]);
		P(C, D, E, F, G, H, A, B, w[6], K[6]);
		P(B, C, D, E, F, G, H, A, w[7], K[7]);

		P(A, B, C, D, E, F, G, H, w[8], K[8]);  
		P(H, A, B, C, D, E, F, G, w[9], K[9]); 
		P(G, H, A, B, C, D, E, F, w[10], K[10]);  
		P(F, G, H, A, B, C, D, E, w[11], K[11]);  
		P(E, F, G, H, A, B, C, D, w[12], K[12]);
		P(D, E, F, G, H, A, B, C, w[13], K[13]);
		P(C, D, E, F, G, H, A, B, w[14], K[14]);
		P(B, C, D, E, F, G, H, A, w[15], K[15]);

		P(A, B, C, D, E, F, G, H, w[16], K[16]);  
		P(H, A, B, C, D, E, F, G, w[17], K[17]); 
		P(G, H, A, B, C, D, E, F, w[18], K[18]);  
		P(F, G, H, A, B, C, D, E, w[19], K[19]);  
		P(E, F, G, H, A, B, C, D, w[20], K[20]);
		P(D, E, F, G, H, A, B, C, w[21], K[21]);
		P(C, D, E, F, G, H, A, B, w[22], K[22]);
		P(B, C, D, E, F, G, H, A, w[23], K[23]);

		P(A, B, C, D, E, F, G, H, w[24], K[24]);  
		P(H, A, B, C, D, E, F, G, w[25], K[25]); 
		P(G, H, A, B, C, D, E, F, w[26], K[26]);  
		P(F, G, H, A, B, C, D, E, w[27], K[27]);  
		P(E, F, G, H, A, B, C, D, w[28], K[28]);
		P(D, E, F, G, H, A, B, C, w[29], K[29]);
		P(C, D, E, F, G, H, A, B, w[30], K[30]);
		P(B, C, D, E, F, G, H, A, w[31], K[31]);

		P(A, B, C, D, E, F, G, H, w[32], K[32]);  
		P(H, A, B, C, D, E, F, G, w[33], K[33]); 
		P(G, H, A, B, C, D, E, F, w[34], K[34]);  
		P(F, G, H, A, B, C, D, E, w[35], K[35]);  
		P(E, F, G, H, A, B, C, D, w[36], K[36]);
		P(D, E, F, G, H, A, B, C, w[37], K[37]);
		P(C, D, E, F, G, H, A, B, w[38], K[38]);
		P(B, C, D, E, F, G, H, A, w[39], K[39]);

		P(A, B, C, D, E, F, G, H, w[40], K[40]);  
		P(H, A, B, C, D, E, F, G, w[41], K[41]); 
		P(G, H, A, B, C, D, E, F, w[42], K[42]);  
		P(F, G, H, A, B, C, D, E, w[43], K[43]);  
		P(E, F, G, H, A, B, C, D, w[44], K[44]);
		P(D, E, F, G, H, A, B, C, w[45], K[45]);
		P(C, D, E, F, G, H, A, B, w[46], K[46]);
		P(B, C, D, E, F, G, H, A, w[47], K[47]);

		P(A, B, C, D, E, F, G, H, w[48], K[48]);  
		P(H, A, B, C, D, E, F, G, w[49], K[49]); 
		P(G, H, A, B, C, D, E, F, w[50], K[50]);  
		P(F, G, H, A, B, C, D, E, w[51], K[51]);  
		P(E, F, G, H, A, B, C, D, w[52], K[52]);
		P(D, E, F, G, H, A, B, C, w[53], K[53]);
		P(C, D, E, F, G, H, A, B, w[54], K[54]);
		P(B, C, D, E, F, G, H, A, w[55], K[55]);

		P(A, B, C, D, E, F, G, H, w[56], K[56]);  
		P(H, A, B, C, D, E, F, G, w[57], K[57]); 
		P(G, H, A, B, C, D, E, F, w[58], K[58]);  
		P(F, G, H, A, B, C, D, E, w[59], K[59]);  
		P(E, F, G, H, A, B, C, D, w[60], K[60]);
		P(D, E, F, G, H, A, B, C, w[61], K[61]);
		P(C, D, E, F, G, H, A, B, w[62], K[62]);
		P(B, C, D, E, F, G, H, A, w[63], K[63]);

		P(A, B, C, D, E, F, G, H, w[64], K[64]);  
		P(H, A, B, C, D, E, F, G, w[65], K[65]); 
		P(G, H, A, B, C, D, E, F, w[66], K[66]);  
		P(F, G, H, A, B, C, D, E, w[67], K[67]);  
		P(E, F, G, H, A, B, C, D, w[68], K[68]);
		P(D, E, F, G, H, A, B, C, w[69], K[69]);
		P(C, D, E, F, G, H, A, B, w[70], K[70]);
		P(B, C, D, E, F, G, H, A, w[71], K[71]);

		P(A, B, C, D, E, F, G, H, w[72], K[72]);  
		P(H, A, B, C, D, E, F, G, w[73], K[73]); 
		P(G, H, A, B, C, D, E, F, w[74], K[74]);  
		P(F, G, H, A, B, C, D, E, w[75], K[75]);  
		P(E, F, G, H, A, B, C, D, w[76], K[76]);
		P(D, E, F, G, H, A, B, C, w[77], K[77]);
		P(C, D, E, F, G, H, A, B, w[78], K[78]);
		P(B, C, D, E, F, G, H, A, w[79], K[79]);
	 

	  	atomicAdd(counter, 1);
	  	if (h[0]+A>>64-dif == 0)
	  	{
			valid_in[0] = w[0];
			valid_in[1] = w[1];
			valid_in[2] = w[2];
			valid_in[3] = w[3];
			valid_in[4] = w[4];
			valid_in[5] = w[5];
			valid_in[6] = w[6];
			valid_in[7] = w[7];
			asm("trap;");
	  	}
	  	__syncthreads();
	}
}


int main(int argc, char *argv[])
{
	int thread_count = atoi(argv[2]);
	int block_count = atoi(argv[3]);
	int threads = thread_count*block_count; 
	
	int dif = atoi(argv[1]);

	clock_t start, end;
	

	int device;
	cudaGetDeviceCount(&device);

	printf("%d cuda device(s)\n", device);

	cudaDeviceProp prop;
	for (int dev = 0; dev < device; dev++) 
	{
		cudaGetDeviceProperties(&prop, dev);
		printf("%s compute capability ", prop.name);
		printf("%d.%d\n", prop.major, prop.minor);
	}
	
	device = DEVICE;
	cudaGetDeviceProperties(&prop, device);
	printf("Using %s\n\n", prop.name);
	cudaSetDevice(device);

	start = clock();

	unsigned long long int *counter;
	uint64_t *in;
	
	cudaMallocManaged(&in, 512);
	cudaMallocManaged(&counter, 64);

	sha512Compute<<<block_count,thread_count>>>(in,dif,counter);
	cudaDeviceSynchronize();
	
	for(int j=0;j<8;j++)
		printf("%016lx", in[j]);
	printf("\n");


	end = clock();
	double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("%lld hash attempts performed in %f seconds at %d difficulty.\n", *counter,cpu_time_used,dif);
	printf("%f MH/s\n", *counter/cpu_time_used/1000000);

	cudaFree(in);
	cudaFree(counter);
	cudaDeviceReset();
	return 0;
}
