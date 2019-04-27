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

__global__
void sha256Compute(uint32_t *valid_in, uint32_t nonce, int dif, uint8_t *byte)
{
	uint32_t w[64];
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	curandState state;
	curand_init(clock64(), idx, 0, &state);

	w[0] = curand(&state);
	w[1] = curand(&state);
	w[2] = curand(&state);
	w[3] = curand(&state);
	w[4] = curand(&state);
	w[5] = curand(&state);
	w[6] = curand(&state);
	w[7] = nonce;	

	w[8]  = 0x80000000;

	w[9]  = 0x00000000;
	w[10] = 0x00000000;
	w[11] = 0x00000000;
	w[12] = 0x00000000;
	w[13] = 0x00000000;
	w[14] = 0x00000000;
	
	w[15] = 0x00000100;


	uint32_t values[8];
	

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

	uint32_t temp1;
	uint32_t temp2;

	#pragma unroll 64
	for (int i=0;i<64;i++)
	{

		temp1 = H+(S1(E))+(ch(E,F,G))+k[i]+w[i];
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



	if (((h[0]+A) >> (32-dif)) == 0)
	{
		byte[0] = 1;
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
}

int main(int argc, char *argv[])
{
	int threads_per_block;
	int blocks;
	
	uint32_t nonce = 0xFFFFFFFF;;
	int dif = atoi(argv[1]);

	clock_t start, end;
	long int counter = 0;
	threads_per_block = atoi(argv[2]);
	blocks = atoi(argv[3]);

	int threads = threads_per_block * blocks;

	start = clock();
	while(1)
	{
		uint32_t *input;	
		uint8_t *byte;

		cudaMallocManaged(&input, 256);
		cudaMallocManaged(&byte, 1);

		sha256Compute<<<blocks, threads_per_block>>>(input,nonce,dif,byte);
		cudaDeviceSynchronize();


		counter += threads;

		if(byte[0]==1)
		{
			for(int j = 0; j < 8; j++)
				printf("%08x", input[j]);
			printf("\n");
			break;
		}	
				
		cudaFree(input);
		cudaFree(byte);
	}
	end = clock();
	double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("%ld hashes attempted in %f seconds at %d difficulty.\n", counter,cpu_time_used,dif);
	printf("%f MH/s\n", counter/cpu_time_used/1000000);

	cudaDeviceReset();

	return 0;
}