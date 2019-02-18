#include <stdio.h>
#include <stdlib.h>

unsigned char* generateString(unsigned char *dest, int length);

unsigned char* generateArray(int array_length, int string_length)
{
	unsigned char *value_ptr = malloc(string_length*sizeof(unsigned char));
	unsigned char *array_ptr = malloc(string_length*array_length*sizeof(unsigned char));
	
	for (int i=0;i<array_length;i++)
	{
		generateString(value_ptr, string_length);
		for(int j=0;j<string_length;j++)
			array_ptr[i*string_length+j] = value_ptr[j];
	}
	return array_ptr;
}


unsigned char* generateString(unsigned char *dest, int length)
{
	unsigned char charset[] = "0123456789"
                     "abcdefghijklmnopqrstuvwxyz"
                     "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    for (int i=0;i<length;i++) {
        size_t index = (double) rand() / RAND_MAX * (sizeof charset - 1);
        dest[i] = (unsigned char)charset[index];
    }
    return dest;
}
