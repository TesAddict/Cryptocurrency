#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <curand_kernel.h>
#include <time.h>

#define THREADS 1
#define BLOCKS 1
#define TOTAL_STATES (1*1)

#define U8TO32_BIG(p)               \
  (((uint32_t)((p)[0]) << 24) | ((uint32_t)((p)[1]) << 16) |  \
   ((uint32_t)((p)[2]) <<  8) | ((uint32_t)((p)[3])      ))

#define U32TO8_BIG(p, v)                \
  (p)[0] = (uint8_t)((v) >> 24); (p)[1] = (uint8_t)((v) >> 16); \
  (p)[2] = (uint8_t)((v) >>  8); (p)[3] = (uint8_t)((v)      );

#define U8TO64_BIG(p) \
  (((uint64_t)U8TO32_BIG(p) << 32) | (uint64_t)U8TO32_BIG((p) + 4))

#define U64TO8_BIG(p, v)          \
  U32TO8_BIG((p),     (uint32_t)((v) >> 32)); \
  U32TO8_BIG((p) + 4, (uint32_t)((v)      ));


__device__ const uint8_t sigma[][16] =
{
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
  {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
  {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
  { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
  {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13 , 0 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
  {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
  {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
  { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
  { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
  { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 }
};


__device__ const uint64_t u512[16] =
{
  0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL, 
  0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL,
  0x452821e638d01377ULL, 0xbe5466cf34e90c6cULL, 
  0xc0ac29b7c97c50ddULL, 0x3f84d5b5b5470917ULL,
  0x9216d5d98979fb1bULL, 0xd1310ba698dfb5acULL, 
  0x2ffd72dbd01adfb7ULL, 0xb8e1afed6a267e96ULL,
  0xba7c9045f12c7f99ULL, 0x24a19947b3916cf7ULL, 
  0x0801f2e2858efc16ULL, 0x636920d871574e69ULL
};


__device__ static const uint8_t padding[129] =
{
  0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

__device__ unsigned char charset[] = "0123456789"
                     "abcdefghijklmnopqrstuvwxyz"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

__device__
void blake512_compress_buf(uint64_t *h, uint64_t *t, uint64_t *s, int *nullt, int idx, uint8_t *buf, 
                           int buf_stride, int h_stride, int t_stride, int s_stride)
{
  uint64_t v[16], m[16], i;

  #define ROT(x,n) (((x)<<(64-n))|( (x)>>(n)))

  #define G(a,b,c,d,e)          \
  v[a] += (m[sigma[i][e]] ^ u512[sigma[i][e+1]]) + v[b];\
  v[d] = ROT( v[d] ^ v[a],32);        \
  v[c] += v[d];           \
  v[b] = ROT( v[b] ^ v[c],25);        \
  v[a] += (m[sigma[i][e+1]] ^ u512[sigma[i][e]])+v[b];  \
  v[d] = ROT( v[d] ^ v[a],16);        \
  v[c] += v[d];           \
  v[b] = ROT( v[b] ^ v[c],11);

  for( i = 0; i < 16; ++i )  m[i] = U8TO64_BIG( buf[buf_stride] + i * 8 );

  for( i = 0; i < 8; ++i )  v[i] = h[i+h_stride];

  v[ 8] = s[0+s_stride] ^ u512[0];
  v[ 9] = s[1+s_stride] ^ u512[1];
  v[10] = s[2+s_stride] ^ u512[2];
  v[11] = s[3+s_stride] ^ u512[3];
  v[12] =  u512[4];
  v[13] =  u512[5];
  v[14] =  u512[6];
  v[15] =  u512[7];

  /* don't xor t when the block is only padding */
  if ( !nullt[idx] )
  {
    v[12] ^= t[0+t_stride];
    v[13] ^= t[0+t_stride];
    v[14] ^= t[1+t_stride];
    v[15] ^= t[1+t_stride];
  }

  for( i = 0; i < 16; ++i )
  {
    /* column step */
    G( 0, 4, 8, 12, 0 );
    G( 1, 5, 9, 13, 2 );
    G( 2, 6, 10, 14, 4 );
    G( 3, 7, 11, 15, 6 );
    /* diagonal step */
    G( 0, 5, 10, 15, 8 );
    G( 1, 6, 11, 12, 10 );
    G( 2, 7, 8, 13, 12 );
    G( 3, 4, 9, 14, 14 );
  }

  for( i = 0; i < 16; ++i )  h[(i % 8)+h_stride] ^= v[i];

  for( i = 0; i < 8 ; ++i )  h[i+h_stride] ^= s[(i % 4)+s_stride];
}

__device__
void blake512_compress_in(uint64_t *h, uint64_t *t, uint64_t *s, int *nullt, int idx, uint8_t *in, 
                           int in_stride, int h_stride, int t_stride, int s_stride)
{
  uint64_t v[16], m[16], i;

  #define ROT(x,n) (((x)<<(64-n))|( (x)>>(n)))

  #define G(a,b,c,d,e)          \
  v[a] += (m[sigma[i][e]] ^ u512[sigma[i][e+1]]) + v[b];\
  v[d] = ROT( v[d] ^ v[a],32);        \
  v[c] += v[d];           \
  v[b] = ROT( v[b] ^ v[c],25);        \
  v[a] += (m[sigma[i][e+1]] ^ u512[sigma[i][e]])+v[b];  \
  v[d] = ROT( v[d] ^ v[a],16);        \
  v[c] += v[d];           \
  v[b] = ROT( v[b] ^ v[c],11);

  for( i = 0; i < 16; ++i )  m[i] = U8TO64_BIG( in[in_stride] + i * 8 );

  for( i = 0; i < 8; ++i )  v[i] = h[i+h_stride];

  v[ 8] = s[0+s_stride] ^ u512[0];
  v[ 9] = s[1+s_stride] ^ u512[1];
  v[10] = s[2+s_stride] ^ u512[2];
  v[11] = s[3+s_stride] ^ u512[3];
  v[12] =  u512[4];
  v[13] =  u512[5];
  v[14] =  u512[6];
  v[15] =  u512[7];

  /* don't xor t when the block is only padding */
  if ( !nullt[idx] )
  {
    v[12] ^= t[0+t_stride];
    v[13] ^= t[0+t_stride];
    v[14] ^= t[1+t_stride];
    v[15] ^= t[1+t_stride];
  }

  for( i = 0; i < 16; ++i )
  {
    /* column step */
    G( 0, 4, 8, 12, 0 );
    G( 1, 5, 9, 13, 2 );
    G( 2, 6, 10, 14, 4 );
    G( 3, 7, 11, 15, 6 );
    /* diagonal step */
    G( 0, 5, 10, 15, 8 );
    G( 1, 6, 11, 12, 10 );
    G( 2, 7, 8, 13, 12 );
    G( 3, 4, 9, 14, 14 );
  }

  for( i = 0; i < 16; ++i )  h[(i % 8)+h_stride] ^= v[i];

  for( i = 0; i < 8 ; ++i )  h[i+h_stride] ^= s[(i % 4)+s_stride];
}

__device__
void blake512_init(uint64_t *h, uint64_t *t, uint64_t *s, int *buflen, int *nullt, int idx, 
                   int h_stride, int t_stride, int s_stride)
{


  h[0+h_stride] = 0x6a09e667f3bcc908ULL;
  h[1+h_stride] = 0xbb67ae8584caa73bULL;
  h[2+h_stride] = 0x3c6ef372fe94f82bULL;
  h[3+h_stride] = 0xa54ff53a5f1d36f1ULL;
  h[4+h_stride] = 0x510e527fade682d1ULL;
  h[5+h_stride] = 0x9b05688c2b3e6c1fULL;
  h[6+h_stride] = 0x1f83d9abfb41bd6bULL;
  h[7+h_stride] = 0x5be0cd19137e2179ULL;
  t[0+t_stride] = t[1+t_stride] = buflen[idx] = nullt[idx] = 0;
  s[0+s_stride] = s[1+s_stride] = s[2+s_stride] = s[3+s_stride] = 0;
}

__device__
void blake512_update(uint64_t *h, uint64_t *t, uint64_t *s, int *buflen, int idx, 
                    int h_stride, int t_stride, int s_stride, int buf_stride, int in_stride, 
                    uint8_t *in, uint64_t inlen, uint64_t *buf, int *nullt)
{
  int left = buflen[idx];
  int fill = 128 - left;

  /* data left and data received fill a block  */
  if( left && ( inlen >= fill ) )
  {
    memcpy( ( void * ) ( buf[buf_stride] + left ), ( void * ) in[in_stride], fill );
    t[0+t_stride] += 1024;

    if ( t[0+t_stride] == 0 ) t[1+t_stride]++;

    blake512_compress_buf(h,t,s,nullt,idx,buf,buf_stride,h_stride,t_stride,s_stride);
  

    in[in_stride] += fill;
    inlen  -= fill;
    left = 0;
  }

  /* compress blocks of data received */
  while( inlen >= 128 )
  {
    t[0+t_stride] += 1024;

    if ( t[0+t_stride] == 0 ) t[1+t_stride]++;

    blake512_compress_in(h,t,s,nullt,idx,in,in_stride,h_stride,t_stride,s_stride);

    in[in_stride] += 128;
    inlen -= 128;
  }

  /* store any data left */
  if( inlen > 0 )
  {
    memcpy( ( void * ) ( buf[buf_stride] + left ),   \
            ( void * ) in[in_stride], ( size_t ) inlen );
    buflen[idx] = left + ( int )inlen;
  }
  else buflen[idx] = 0;
}


__device__
void blake512_final(uint64_t *h, uint64_t *t, uint64_t *s, int *buflen, int idx, 
                    int h_stride, int t_stride, int s_stride, int buf_stride, int in_stride,
                    uint8_t *in, uint64_t inlen, uint64_t *buf, int *nullt, uint8_t *out, int out_stride)
{
  uint8_t msglen[16], zo = 0x01, oo = 0x81;
  uint64_t lo = t[0+t_stride] + ( buflen[idx] << 3 ), hi = t[1+t_stride];

  /* support for hashing more than 2^32 bits */
  if ( lo < ( buflen[idx] << 3 ) ) hi++;

  U64TO8_BIG(  msglen + 0, hi );
  U64TO8_BIG(  msglen + 8, lo );

  if ( buflen[idx] == 111 )   /* one padding byte */
  {
    t[0+t_stride] -= 8;
    blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,idx,&oo,1,buf,nullt);
  }
  else
  {
    if ( buflen[idx] < 111 )  /* enough space to fill the block */
    {
      if ( buflen[idx] ) nullt[idx] = 1;

      t[0+t_stride] -= 888 - ( buflen[idx] << 3 );
      blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,idx*(111-buflen[idx]),padding,111-buflen[idx],buf,nullt);
    }
    else   /* need 2 compressions */
    {
      t[0+t_stride] -= 1024 - ( buflen[idx] << 3 );
      blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,idx*(128-buflen[idx]),padding,128-buflen[idx],buf,nullt);
      t[0+t_stride] -= 888;
      blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,idx*111,padding+1,111,buf,nullt);
      nullt[idx] = 1;
    }

    blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,idx,&zo,1,buf,nullt);
    t[0+t_stride] -= 8;
  }

  t[0+t_stride] -= 128;

  blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,idx*16,msglen,16,buf,nullt);
  U64TO8_BIG( out[out_stride] + 0, h[0+h_stride] );
  U64TO8_BIG( out[out_stride] + 8, h[1+h_stride] );
  U64TO8_BIG( out[out_stride] + 16, h[2+h_stride] );
  U64TO8_BIG( out[out_stride] + 24, h[3+h_stride] );
  U64TO8_BIG( out[out_stride] + 32, h[4+h_stride] );
  U64TO8_BIG( out[out_stride] + 40, h[5+h_stride] );
  U64TO8_BIG( out[out_stride] + 48, h[6+h_stride] );
  U64TO8_BIG( out[out_stride] + 56, h[7+h_stride] );
}

/* 
  out is thread count * 128 (uint8_t)
  in is thread count * inlen (uint8_t)
  h is thread count * 8 (uint64_t)
  s is thread count * 4 (uint64_t)
  t is thread count * 2 (uint64_t)
  buf is thread count * 128 (uint8_t)
  buflen is thread count (int)
  nullt is thread count (int)
*/

__global__
void blake512_hash(uint8_t *out, uint8_t *in, uint64_t inlen, uint64_t *h, uint64_t *s, 
                   uint64_t *t, uint8_t *buf, int *buflen, int *nullt)
{

  int idx = blockIdx.x*blockDim.x+threadIdx.x;
  int in_stride = (inlen*idx);
  int h_stride = (8*idx);
  int t_stride = (2*idx);
  int s_stride = (4*idx);
  int buf_stride = (128*idx);
  int out_stride = (128*idx);

  curandState state;
  curand_init(clock64(), idx, 0, &state);

  for (int i = 0; i < inlen; i++)
  {
    unsigned int rand = curand_uniform(&state)*100000;
    in[in_stride] = (uint8_t)charset[(rand%63)];
  }

  blake512_init(h,t,s,buflen,nullt,idx,h_stride,t_stride,s_stride);

  blake512_update(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,in_stride,in,inlen,buf,nullt);

  blake512_final(h,t,s,buflen,idx,h_stride,t_stride,s_stride,buf_stride,in_stride,in,inlen,buf,nullt,out,out_stride);
}

int main(int argc, char **argv)
{
  uint8_t *in;
  uint8_t *out;
  uint64_t *h;
  uint64_t *s;
  uint64_t *t;
  uint8_t *buf;
  int *buflen;
  int *nullt;

  int inlen = 30;

  int threads = THREADS * BLOCKS;

  cudaMallocManaged(&in, threads*inlen);
  cudaMallocManaged(&out, threads*128); // 128 from output size.
  cudaMallocManaged(&h, threads*64); // 64 from uint64_t size * 8.
  cudaMallocManaged(&s, threads*32); // 32 from uint64_t size * 4.
  cudaMallocManaged(&t, threads*16); // 16 from uint64_t size * 2.
  cudaMallocManaged(&buf, threads*128); // 128 from output size.
  cudaMallocManaged(&buflen, threads);
  cudaMallocManaged(&nullt, threads);

  blake512_hash<<<BLOCKS, THREADS>>>(out, in, inlen, h, s, t, buf, buflen, nullt);
  cudaDeviceSynchronize();

  cudaFree(in);
  cudaFree(out);
  cudaFree(h);
  cudaFree(s);
  cudaFree(t);
  cudaFree(buf);
  cudaFree(buflen);
  cudaFree(nullt);


  return 0;
}