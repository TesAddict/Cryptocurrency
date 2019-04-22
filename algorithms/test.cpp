#include <iostream>
#include <stdint.h>

int main()
{
	uint32_t nonce = 0xDEADBEEF;

	unsigned char message[4];

	message[0] = nonce >> 24;
	message[1] = nonce >> 16;
	message[2] = nonce >> 8;
	message[3] = nonce;

	for (int i = 0; i < 4; i++)
	{
		printf("%x\n", message[i]);
	}

	printf("%d", sizeof(uint32_t));
	return 0;
}