#include <stdio.h>
#include <stdlib.h>

int main()
{
	unsigned char *test = malloc(10*sizeof(unsigned char));
	test[0] = 0x00;
	test[1] = 0x00;
	test[2] = 0x11;

	for(int i=0; i<3; i++)
		printf("%x", test[i]);

	return 0;
}