#include <curand_kernel.h>
#include <stdio.h>
#include <time.h>

#define A 0
#define B 1
#define C 2
#define D 3
#define E 4
#define F 5
#define G 6
#define H 7

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
void sha256Compute(uint8_t *pad, uint32_t *s, int idx, int pad_stride, uint32_t *out, uint8_t *state, int dif)
{
	int out_stride = idx*8;

	s[0+pad_stride]  = pad[0+pad_stride]<<24  | pad[1+pad_stride]<<16  | pad[2+pad_stride]<<8  | pad[3+pad_stride];
	s[1+pad_stride]  = pad[4+pad_stride]<<24  | pad[5+pad_stride]<<16  | pad[6+pad_stride]<<8  | pad[7+pad_stride];
	s[2+pad_stride]  = pad[8+pad_stride]<<24  | pad[9+pad_stride]<<16  | pad[10+pad_stride]<<8 | pad[11+pad_stride];
	s[3+pad_stride]  = pad[12+pad_stride]<<24 | pad[13+pad_stride]<<16 | pad[14+pad_stride]<<8 | pad[15+pad_stride];
	s[4+pad_stride]  = pad[16+pad_stride]<<24 | pad[17+pad_stride]<<16 | pad[18+pad_stride]<<8 | pad[19+pad_stride];
	s[5+pad_stride]  = pad[20+pad_stride]<<24 | pad[21+pad_stride]<<16 | pad[22+pad_stride]<<8 | pad[23+pad_stride];
	s[6+pad_stride]  = pad[24+pad_stride]<<24 | pad[25+pad_stride]<<16 | pad[26+pad_stride]<<8 | pad[27+pad_stride];
	s[7+pad_stride]  = pad[28+pad_stride]<<24 | pad[29+pad_stride]<<16 | pad[30+pad_stride]<<8 | pad[31+pad_stride];
	s[8+pad_stride]  = pad[32+pad_stride]<<24 | pad[33+pad_stride]<<16 | pad[34+pad_stride]<<8 | pad[35+pad_stride];
	s[9+pad_stride]  = pad[36+pad_stride]<<24 | pad[37+pad_stride]<<16 | pad[38+pad_stride]<<8 | pad[39+pad_stride];
	s[10+pad_stride] = pad[40+pad_stride]<<24 | pad[41+pad_stride]<<16 | pad[42+pad_stride]<<8 | pad[43+pad_stride];
	s[11+pad_stride] = pad[44+pad_stride]<<24 | pad[45+pad_stride]<<16 | pad[46+pad_stride]<<8 | pad[47+pad_stride];
	s[12+pad_stride] = pad[48+pad_stride]<<24 | pad[49+pad_stride]<<16 | pad[50+pad_stride]<<8 | pad[51+pad_stride];
	s[13+pad_stride] = pad[52+pad_stride]<<24 | pad[53+pad_stride]<<16 | pad[54+pad_stride]<<8 | pad[55+pad_stride];
	s[14+pad_stride] = pad[56+pad_stride]<<24 | pad[57+pad_stride]<<16 | pad[58+pad_stride]<<8 | pad[59+pad_stride];
	s[15+pad_stride] = pad[60+pad_stride]<<24 | pad[61+pad_stride]<<16 | pad[62+pad_stride]<<8 | pad[63+pad_stride];
	


	#pragma unroll 47
	for(int i=16;i<64;i++)
		s[i+pad_stride] = s[i-16+pad_stride] + (s0(s[i-15+pad_stride])) + s[i-7+pad_stride] + (s1(s[i-2 +pad_stride]));
	

	int A_ = A+out_stride;
	int B_ = B+out_stride;
	int C_ = C+out_stride;
	int D_ = D+out_stride;
	int E_ = E+out_stride;
	int F_ = F+out_stride;
	int G_ = G+out_stride;
	int H_ = H+out_stride;

	out[A_] = h[0];
	out[B_] = h[1];
	out[C_] = h[2];
	out[D_] = h[3];
	out[E_] = h[4];
	out[F_] = h[5];
	out[G_] = h[6];
	out[H_] = h[7];

	uint32_t temp1;
	uint32_t temp2;

	#pragma unroll 64
	for (int i=0;i<64;i++)
	{
		temp1 = out[H_]+(S1(out[E_]))+(ch(out[E_],out[F_],out[G_]))+k[i]+s[i+pad_stride];
		temp2 = (S0(out[A_]))+(maj(out[A_],out[B_],out[C_]));

		out[H_] = out[G_];
		out[G_] = out[F_];
		out[F_] = out[E_];
		out[E_] = out[D_]+temp1;
		out[D_] = out[C_];
		out[C_] = out[B_];
		out[B_] = out[A_];
		out[A_] = temp1+temp2;
	}

	out[A_] = h[0]+out[A_];
	out[B_] = h[1]+out[B_];
	out[C_] = h[2]+out[C_];
	out[D_] = h[3]+out[D_];
	out[E_] = h[4]+out[E_];
	out[F_] = h[5]+out[F_];
	out[G_] = h[6]+out[G_];
	out[H_] = h[7]+out[H_];

	sha256Check(out,dif,out_stride,state,idx);
}

__device__
void sha256Pad(uint8_t *input, int inlen, uint32_t *output, uint8_t *pad, int idx, int input_stride, 
			   uint32_t *schedule, uint8_t *state, int dif)
{
	int pad_stride = idx*64;

	pad[0+pad_stride]  =  input[0+input_stride];
	pad[1+pad_stride]  =  input[1+input_stride];
	pad[2+pad_stride]  =  input[2+input_stride];
	pad[3+pad_stride]  =  input[3+input_stride];
	pad[4+pad_stride]  =  input[4+input_stride];
	pad[5+pad_stride]  =  input[5+input_stride];
	pad[6+pad_stride]  =  input[6+input_stride];
	pad[7+pad_stride]  =  input[7+input_stride];
	pad[8+pad_stride]  =  input[8+input_stride];
	pad[9+pad_stride]  =  input[9+input_stride];
	pad[10+pad_stride] =  input[10+input_stride];
	pad[11+pad_stride] =  input[11+input_stride];
	pad[12+pad_stride] =  input[12+input_stride];
	pad[13+pad_stride] =  input[13+input_stride];
	pad[14+pad_stride] =  input[14+input_stride];
	pad[15+pad_stride] =  input[15+input_stride];
	pad[16+pad_stride] =  input[16+input_stride];
	pad[17+pad_stride] =  input[17+input_stride];
	pad[18+pad_stride] =  input[18+input_stride];
	pad[19+pad_stride] =  input[19+input_stride];
	pad[20+pad_stride] =  input[20+input_stride];
	pad[21+pad_stride] =  input[21+input_stride];
	pad[22+pad_stride] =  input[22+input_stride];
	pad[23+pad_stride] =  input[23+input_stride];
	pad[24+pad_stride] =  input[24+input_stride];
	pad[25+pad_stride] =  input[25+input_stride];
	pad[26+pad_stride] =  input[26+input_stride];
	pad[27+pad_stride] =  input[27+input_stride];
	pad[28+pad_stride] =  input[28+input_stride];
	pad[29+pad_stride] =  input[29+input_stride];
	
	pad[30+pad_stride] =  0x80;

	pad[31+pad_stride] =  0x00;
	pad[32+pad_stride] =  0x00; 
	pad[33+pad_stride] =  0x00;
	pad[34+pad_stride] =  0x00;
	pad[35+pad_stride] =  0x00;
	pad[36+pad_stride] =  0x00;
	pad[37+pad_stride] =  0x00;
	pad[38+pad_stride] =  0x00;
	pad[39+pad_stride] =  0x00;
	pad[40+pad_stride] =  0x00;
	pad[41+pad_stride] =  0x00;
	pad[42+pad_stride] =  0x00;
	pad[43+pad_stride] =  0x00;
	pad[44+pad_stride] =  0x00;
	pad[45+pad_stride] =  0x00;
	pad[46+pad_stride] =  0x00;
	pad[47+pad_stride] =  0x00;
	pad[48+pad_stride] =  0x00;
	pad[49+pad_stride] =  0x00;
	pad[50+pad_stride] =  0x00;
	pad[51+pad_stride] =  0x00;
	pad[52+pad_stride] =  0x00;
	pad[53+pad_stride] =  0x00;
	pad[54+pad_stride] =  0x00;
	pad[55+pad_stride] =  0x00;
	
	// Padding length of input (constant length of 240)
	pad[56+pad_stride] =  0x00;
	pad[57+pad_stride] =  0x00;
	pad[58+pad_stride] =  0x00;
	pad[59+pad_stride] =  0x00;
	pad[60+pad_stride] =  0x00;
	pad[61+pad_stride] =  0x00;
	pad[62+pad_stride] =  0x00;
	pad[63+pad_stride] =  0xf0;

	sha256Compute(pad,schedule,idx,pad_stride,output,state,dif);
}


__global__
void generateStrings(uint8_t *input, int inlen, uint8_t *nonce, uint32_t *output, 
					 uint8_t *pad, uint32_t *schedule, uint8_t *state_a, int dif)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int input_stride = idx*inlen;

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

	sha256Pad(input,inlen+4,output,pad,idx,input_stride,schedule,state_a,dif);
}

int main()
{
	uint8_t *c_input, *d_input;
	uint8_t *c_nonce, *d_nonce;
	uint32_t *d_output;
	uint8_t *c_state, *d_state;
	uint8_t *d_pad;
	uint32_t *d_schedule;

	int inlen = 30;
	int threads_per_block = 256;
	int blocks = 128;
	int threads = threads_per_block * blocks;
	int dif = 20;

	while(1)
	{

		c_input  =  (uint8_t*)malloc(threads*inlen);
		c_nonce  =  (uint8_t*)malloc(4);
		c_state  =  (uint8_t*)malloc(threads);

		c_nonce[0] = 0x41;
		c_nonce[1] = 0x41;
		c_nonce[2] = 0x41;
		c_nonce[3] = 0x41;

		cudaMalloc(&d_input, threads*inlen);
		cudaMalloc(&d_nonce, 4);
		cudaMalloc(&d_output, threads*32);
		cudaMalloc(&d_pad, threads*64);
		cudaMalloc(&d_schedule, threads*256);
		cudaMalloc(&d_state, threads);

		cudaMemcpy(d_nonce, c_nonce, 4, cudaMemcpyHostToDevice);

		generateStrings<<<blocks, threads_per_block>>>(d_input,inlen,d_nonce,d_output,d_pad,d_schedule,d_state,dif);
		cudaDeviceSynchronize();

		cudaMemcpy(c_input, d_input, threads*inlen, cudaMemcpyDeviceToHost);
		cudaMemcpy(c_state, d_state, threads, cudaMemcpyDeviceToHost);

		for (int i=0; i<threads;i++)
		{
			if(c_state[i] == 1)
			{
				for(int j = 0; j < inlen; j++)
					printf("%02x", c_input[j+(i*inlen)]);
				printf("\n");
			}
		}

		cudaFree(d_input);
		cudaFree(d_nonce);
		cudaFree(d_output);
		cudaFree(d_pad);
		cudaFree(d_schedule);
		cudaFree(d_state);
		free(c_input);
		free(c_nonce);
		free(c_state);

	}

	cudaDeviceReset();

	return 0;
}