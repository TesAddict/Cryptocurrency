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
		state[idx] = 1;
	else
		state[idx] = 0;
}


__device__
void sha256Compute(uint8_t *pad, int idx, int pad_stride, uint32_t *out, uint8_t *state, int dif, int out_stride)
{
	uint32_t s[64];
	uint32_t values[8];

	s[0]  = pad[0+pad_stride]<<24  | pad[1+pad_stride]<<16  | pad[2+pad_stride]<<8  | pad[3+pad_stride];
	s[1]  = pad[4+pad_stride]<<24  | pad[5+pad_stride]<<16  | pad[6+pad_stride]<<8  | pad[7+pad_stride];
	s[2]  = pad[8+pad_stride]<<24  | pad[9+pad_stride]<<16  | pad[10+pad_stride]<<8 | pad[11+pad_stride];
	s[3]  = pad[12+pad_stride]<<24 | pad[13+pad_stride]<<16 | pad[14+pad_stride]<<8 | pad[15+pad_stride];
	s[4]  = pad[16+pad_stride]<<24 | pad[17+pad_stride]<<16 | pad[18+pad_stride]<<8 | pad[19+pad_stride];
	s[5]  = pad[20+pad_stride]<<24 | pad[21+pad_stride]<<16 | pad[22+pad_stride]<<8 | pad[23+pad_stride];
	s[6]  = pad[24+pad_stride]<<24 | pad[25+pad_stride]<<16 | pad[26+pad_stride]<<8 | pad[27+pad_stride];
	s[7]  = pad[28+pad_stride]<<24 | pad[29+pad_stride]<<16 | pad[30+pad_stride]<<8 | pad[31+pad_stride];
	s[8]  = pad[32+pad_stride]<<24 | pad[33+pad_stride]<<16 | pad[34+pad_stride]<<8 | pad[35+pad_stride];
	s[9]  = pad[36+pad_stride]<<24 | pad[37+pad_stride]<<16 | pad[38+pad_stride]<<8 | pad[39+pad_stride];
	s[10] = pad[40+pad_stride]<<24 | pad[41+pad_stride]<<16 | pad[42+pad_stride]<<8 | pad[43+pad_stride];
	s[11] = pad[44+pad_stride]<<24 | pad[45+pad_stride]<<16 | pad[46+pad_stride]<<8 | pad[47+pad_stride];
	s[12] = pad[48+pad_stride]<<24 | pad[49+pad_stride]<<16 | pad[50+pad_stride]<<8 | pad[51+pad_stride];
	s[13] = pad[52+pad_stride]<<24 | pad[53+pad_stride]<<16 | pad[54+pad_stride]<<8 | pad[55+pad_stride];
	s[14] = pad[56+pad_stride]<<24 | pad[57+pad_stride]<<16 | pad[58+pad_stride]<<8 | pad[59+pad_stride];
	s[15] = pad[60+pad_stride]<<24 | pad[61+pad_stride]<<16 | pad[62+pad_stride]<<8 | pad[63+pad_stride];
	
	#pragma unroll 48
	for(int i=16;i<64;i++)
		s[16] = s[i-16] + (s0(s[i-15]))  + s[i-7]   + (s1(s[i-2]));  
	

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

	sha256Check(out,dif,out_stride,state,idx);
}

__device__
void sha256Pad(uint8_t *input, int inlen, uint32_t *output, uint8_t *pad, int idx, int input_stride, 
			   uint8_t *state, int dif, int pad_stride)
{
	#pragma unroll 30
	for (int i=0;i< 30;i++)
		pad[i+pad_stride]  =  input[i+input_stride];
	
	pad[30+pad_stride] =  0x80;

	#pragma unroll 32
	for (int i=31;i<63;i++)
		pad[i+pad_stride] =  0x00;
	
	pad[63+pad_stride] =  0xf0;

}


__device__
void generateStrings(uint8_t *input, int inlen, uint8_t *nonce, uint32_t *output, 
					 uint8_t *pad, uint8_t *state_a, int dif, int idx, int input_stride)
{

	// Appending nonce to the end of the random arrays. 
	inlen = inlen-4;

	curandState state;
	curand_init(clock64(), idx, 0, &state);

	for (int i = 0; i < inlen; i++)
	{
		uint64_t rand = curand_uniform(&state)*100000;
		input[i+input_stride] = (uint8_t)rand;
		
	}

	input[inlen+input_stride] =   nonce[0];
	input[inlen+input_stride+1] = nonce[1];
	input[inlen+input_stride+2] = nonce[2];
	input[inlen+input_stride+3] = nonce[3];	
}

__global__
void sha256Init(uint8_t *input, int inlen, uint8_t *nonce, uint32_t *out, 
					 uint8_t *pad, uint8_t *state, int dif)
{

	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int input_stride = idx*inlen;
	int pad_stride = idx*64;
	int out_stride = idx*8;


	generateStrings(input,inlen,nonce,out,pad,state,dif,idx,input_stride);
	sha256Pad(input,inlen,out,pad,idx,input_stride,state,dif,pad_stride);
	sha256Compute(pad,idx,pad_stride,out,state,dif,out_stride);
	sha256Check(out,dif,out_stride,state,idx);
}

int main(int argc, char *argv[])
{
	int inlen = 30;
	int threads_per_block = 256;
	int blocks = 128;
	int threads = threads_per_block * blocks;
	int dif = atoi(argv[1]);

	clock_t start, end;
	int counter = 0;
	int end_counter = atoi(argv[2]);

	start = clock();
	while(1)
	{
		uint8_t *d_input;
		uint8_t *d_nonce;
		uint32_t *d_output;
		uint8_t *d_state;
		uint8_t *d_pad;

		cudaMallocManaged(&d_input, threads*inlen);
		cudaMallocManaged(&d_nonce, 4);
		cudaMallocManaged(&d_output, threads*32);
		cudaMallocManaged(&d_pad, threads*64);
		cudaMallocManaged(&d_state, threads);

		d_nonce[0] = 0x41;
		d_nonce[1] = 0x41;
		d_nonce[2] = 0x41;
		d_nonce[3] = 0x41;


		sha256Init<<<blocks, threads_per_block>>>(d_input,inlen,d_nonce,d_output,d_pad,d_state,dif);
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
		cudaFree(d_nonce);
		cudaFree(d_output);
		cudaFree(d_pad);
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