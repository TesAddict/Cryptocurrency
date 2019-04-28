#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

uint32_t target[8];

void generateTargetThreshold(uint32_t nbits)
{
	uint32_t mantissa = nbits & 0x00FFFFFF;
	uint32_t exponent = (nbits & 0xFF000000) >> 24;
	uint32_t offset = 0x20 - (exponent - 0x03);
	uint8_t inter[32]={0};
	
	inter[offset-1]   = (mantissa & 0x000000FF);

	inter[offset-2] = (mantissa & 0x0000FF00) >> 8;

	inter[offset-3] = (mantissa & 0x00FF0000) >> 16;

	target[0] = inter[0]<<24  | inter[1]<<16  | inter[2]<<8  | inter[3];
	target[1] = inter[4]<<24  | inter[5]<<16  | inter[6]<<8  | inter[7];
	target[2] = inter[8]<<24  | inter[9]<<16  | inter[10]<<8 | inter[11];
	target[3] = inter[12]<<24 | inter[13]<<16 | inter[14]<<8 | inter[15];
	target[4] = inter[16]<<24 | inter[17]<<16 | inter[18]<<8 | inter[19];
	target[5] = inter[20]<<24 | inter[21]<<16 | inter[22]<<8 | inter[23];
	target[6] = inter[24]<<24 | inter[25]<<16 | inter[26]<<8 | inter[27];
	target[7] = inter[28]<<24 | inter[29]<<16 | inter[30]<<8 | inter[31];
}

int main()
{
	uint32_t nbits = 0x1d1bc330;

	generateTargetThreshold(nbits);
	
	printf("NBits: %08x\n", nbits);


	for (int i=0;i < 8;i++)
		printf("%08x", target[i]);
	printf("\n");

	return 0;

}