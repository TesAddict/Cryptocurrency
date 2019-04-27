#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define SHR(x,n) (x>>n)

#define ROTR(x,n) (SHR(x,n) | (x<<(32-n)))

#define s0(x) (ROTR(x,7)^ROTR(x,18)^SHR(x,3))
#define s1(x) (ROTR(x,17)^ROTR(x,19)^SHR(x,10))

#define S1(x) (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
#define S2(x) (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))

#define F0(x, y, z) ((x & y) | (z & (x | y)))
#define F1(x, y, z) (z ^ (x & (y ^ z)))

#define P(a, b, c, d, e, f, g, h, x, K)      \
  {                                          \
    temp1 = h + S1(e) + F1(e, f, g) + K + x; \
    temp2 = S2(a) + F0(a, b, c);             \
    d += temp1;                              \
    h = temp1 + temp2;                       \
  }

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
static const uint32_t K[64] =
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


__global__
void sha256Compute(uint32_t *valid_in, uint32_t nonce, int dif, unsigned long long int *counter, int threads)
{
	uint32_t w[64];
	uint32_t A,B,C,D,E,F,G,H,temp1,temp2;
	bool passed = false; 
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idx_mod = idx; 
	curandState state;
	curand_init(clock64(), idx, 0, &state);

	while(1)
	{	
		w[1] = curand(&state);
		w[2] = curand(&state);
		w[3] = nonce+idx_mod;
		
		if(passed == false)
		{
			w[0]  = curand(&state);
			w[4]  = 0x80000000;
			w[5]  = 0x00000000;
			w[6]  = 0x00000000;
			w[7]  = 0x00000000;
			w[8]  = 0x00000000;

			w[9]  = 0x00000000;
			w[10] = 0x00000000;
			w[11] = 0x00000000;
			w[12] = 0x00000000;
			w[13] = 0x00000000;
			w[14] = 0x00000000;
			
			w[15] = 0x00000080;
			passed = true;
		}
		

		w[16] = w[0]+(s0(w[1]))+w[9]+(s1(w[14]));
		w[17] = w[1]+(s0(w[2]))+w[10]+(s1(w[15]));
		w[18] = w[2]+(s0(w[3]))+w[11]+(s1(w[16]));
		w[19] = w[3]+(s0(w[4]))+w[12]+(s1(w[17]));
		w[20] = w[4]+(s0(w[5]))+w[13]+(s1(w[18]));
		w[21] = w[5]+(s0(w[6]))+w[14]+(s1(w[19]));
		w[22] = w[6]+(s0(w[7]))+w[15]+(s1(w[20]));
		w[23] = w[7]+(s0(w[8]))+w[16]+(s1(w[21]));
		w[24] = w[8]+(s0(w[9]))+w[17]+(s1(w[22]));
		w[25] = w[9]+(s0(w[10]))+w[18]+(s1(w[23]));
		w[26] = w[10]+(s0(w[11]))+w[19]+(s1(w[24]));
		w[27] = w[11]+(s0(w[12]))+w[20]+(s1(w[25]));
		w[28] = w[12]+(s0(w[13]))+w[21]+(s1(w[26]));
		w[29] = w[13]+(s0(w[14]))+w[22]+(s1(w[27]));
		w[30] = w[14]+(s0(w[15]))+w[23]+(s1(w[28]));
		w[31] = w[15]+(s0(w[16]))+w[24]+(s1(w[29]));
		w[32] = w[16]+(s0(w[17]))+w[25]+(s1(w[30]));
		w[33] = w[17]+(s0(w[18]))+w[26]+(s1(w[31]));
		w[34] = w[18]+(s0(w[19]))+w[27]+(s1(w[32]));
		w[35] = w[19]+(s0(w[20]))+w[28]+(s1(w[33]));
		w[36] = w[20]+(s0(w[21]))+w[29]+(s1(w[34]));
		w[37] = w[21]+(s0(w[22]))+w[30]+(s1(w[35]));
		w[38] = w[22]+(s0(w[23]))+w[31]+(s1(w[36]));
		w[39] = w[23]+(s0(w[24]))+w[32]+(s1(w[37]));
		w[40] = w[24]+(s0(w[25]))+w[33]+(s1(w[38]));
		w[41] = w[25]+(s0(w[26]))+w[34]+(s1(w[39]));
		w[42] = w[26]+(s0(w[27]))+w[35]+(s1(w[40]));
		w[43] = w[27]+(s0(w[28]))+w[36]+(s1(w[41]));
		w[44] = w[28]+(s0(w[29]))+w[37]+(s1(w[42]));
		w[45] = w[29]+(s0(w[30]))+w[38]+(s1(w[43]));
		w[46] = w[30]+(s0(w[31]))+w[39]+(s1(w[44]));
		w[47] = w[31]+(s0(w[32]))+w[40]+(s1(w[45]));
		w[48] = w[32]+(s0(w[33]))+w[41]+(s1(w[46]));
		w[49] = w[33]+(s0(w[34]))+w[42]+(s1(w[47]));
		w[50] = w[34]+(s0(w[35]))+w[43]+(s1(w[48]));
		w[51] = w[35]+(s0(w[36]))+w[44]+(s1(w[49]));
		w[52] = w[36]+(s0(w[37]))+w[45]+(s1(w[50]));
		w[53] = w[37]+(s0(w[38]))+w[46]+(s1(w[51]));
		w[54] = w[38]+(s0(w[39]))+w[47]+(s1(w[52]));
		w[55] = w[39]+(s0(w[40]))+w[48]+(s1(w[53]));
		w[56] = w[40]+(s0(w[41]))+w[49]+(s1(w[54]));
		w[57] = w[41]+(s0(w[42]))+w[50]+(s1(w[55]));
		w[58] = w[42]+(s0(w[43]))+w[51]+(s1(w[56]));
		w[59] = w[43]+(s0(w[44]))+w[52]+(s1(w[57]));
		w[60] = w[44]+(s0(w[45]))+w[53]+(s1(w[58]));
		w[61] = w[45]+(s0(w[46]))+w[54]+(s1(w[59]));
		w[62] = w[46]+(s0(w[47]))+w[55]+(s1(w[60]));
		w[63] = w[47]+(s0(w[48]))+w[56]+(s1(w[61]));


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


		atomicAdd(counter,1);

		if (dif <= 32 && h[0]+A >> 32-dif == 0)
		{	
			valid_in[0] = w[0];
			valid_in[1] = w[1];
			valid_in[2] = w[2];
			valid_in[3] = w[3];
			asm("trap;");
		}
		else if ( (dif<=64) && (h[0]+A==0) && (h[1]+B>>(64-dif)==0) )
		{
			valid_in[0] = w[0];
			valid_in[1] = w[1];
			valid_in[2] = w[2];
			valid_in[3] = w[3];
			asm("trap;");
		}
		
		idx_mod = idx+threads;
		__syncthreads();
	}
}

int main(int argc, char *argv[])
{
	int threads_per_block;
	int blocks;
	
	uint32_t nonce = 0x00000000;
	int dif = atoi(argv[1]);

	clock_t start, end;

	threads_per_block = atoi(argv[2]);
	blocks = atoi(argv[3]);

	int threads = threads_per_block * blocks;
	uint32_t *input;	
	unsigned long long int *counter;

	cudaMallocManaged(&input, 128);
	cudaMallocManaged(&counter,64);

	start = clock();

	sha256Compute<<<blocks, threads_per_block>>>(input,nonce,dif,counter,threads);
	cudaDeviceSynchronize();
	
	for(int j = 0; j < 4; j++)
		printf("%08x", input[j]);
	printf("\n");
		
				
		
	end = clock();
	double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("%lld hashes attempted in %f seconds at %d difficulty.\n", counter[0],cpu_time_used,dif);
	printf("%f MH/s\n", counter[0]/cpu_time_used/1000000);
	cudaFree(input);
	cudaFree(counter);
	cudaDeviceReset();

	return 0;
}