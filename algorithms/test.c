#include <stdio.h>
#include <stdint.h>

int main()
{
	uint32_t a;
	uint8_t b[4] = {0x02, 0x01, 0x02, 0x03};

	a = b[0]<<24| b[1]<<16 | b[2]<<8 | b[3];
	printf("%.2x", a);
	return 0;
}