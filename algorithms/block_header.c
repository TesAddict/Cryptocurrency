#include "block_header.h"

uint32_t textToHex(uint8_t *input)
{
	uint32_t output = 0x00000000;

	for(int i=7;i>=0;i--)
	{
		switch(input[i])
		{
			case 0x30:
				output = output | (0x0<<(4*(7-i)));
				break;
			case 0x31:
				output = output |(0x1<<(4*(7-i)));
				break;
			case 0x32:
				output = output |(0x2<<(4*(7-i)));
				break;
			case 0x33:
				output = output |(0x3<<(4*(7-i)));
				break;
			case 0x34:
				output = output |(0x4<<(4*(7-i)));
				break;
			case 0x35:
				output = output |(0x5<<(4*(7-i)));
				break;
			case 0x36:
				output = output |(0x6<<(4*(7-i)));
				break;
			case 0x37:
				output = output |(0x7<<(4*(7-i)));
				break;
			case 0x38:
				output = output |(0x8<<(4*(7-i)));
				break;
			case 0x39:
				output = output |(0x9<<(4*(7-i)));
				break;
			case 0x61:
				output = output |(0xa<<(4*(7-i)));
				break;
			case 0x62:
				output = output |(0xb<<(4*(7-i)));
				break;
			case 0x63:
				output = output |(0xc<<(4*(7-i)));
				break;
			case 0x64:
				output = output |(0xd<<(4*(7-i)));
				break;
			case 0x65:
				output = output |(0xe<<(4*(7-i)));
				break;
			case 0x66:
				output = output |(0xf<<(4*(7-i)));
		}
	}
	return output;
}

void processLine(char *buf, uint32_t *output)
{
	uint8_t *input;
	input = (uint8_t*)malloc(8);
	char *Version = "Version";
	char *PreviousHash = "PreviousHash";
	char *MerkleRoot = "MerkleRoot";
	char *UnixTime = "UnixTime";
	char *Target = "Target";
	char *Nonce = "Nonce";

	if(strstr(buf,Version) != NULL)
	{
		memcpy(input,&buf[8],8);
		output[0] = textToHex(input);
	}
	if(strstr(buf,PreviousHash) != NULL)
	{
		int j = 5;
		for(int i=0;i<8;i++)
		{
			memcpy(input,&buf[j+=8],8);
			output[i+1] = textToHex(input);
		}
	}
	if(strstr(buf,MerkleRoot) != NULL)
	{
		int j = 3;
		for(int i=0;i<8;i++)
		{
			memcpy(input,&buf[j+=8],8);
			output[i+9] = textToHex(input);
		}
	}
	if(strstr(buf,UnixTime) != NULL)
	{
		memcpy(input,&buf[9],8);
		output[17] = textToHex(input);
	}
	if(strstr(buf,Target) != NULL)
	{
		memcpy(input,&buf[7],8);
		output[18] = textToHex(input);
	}
	if(strstr(buf,Nonce) != NULL)
	{
		memcpy(input,&buf[6],8);
		output[19] = textToHex(input);
	}
}

void getBlockHeader(uint32_t *block_header,char *filename)
{
	char buf[1024];
	
	FILE *fp;
	
	fp = fopen(filename, "r");

	if (fp == NULL)
	{
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}
		
	while(fgets(buf,sizeof(buf),fp) != NULL)
	{
		buf[strlen(buf) - 1] = '\0';
		processLine(buf,block_header);
	}

	fclose(fp);
}