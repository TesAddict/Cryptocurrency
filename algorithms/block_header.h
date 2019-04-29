#ifndef BLOCK_HEADER_H_
#define BLOCK_HEADER_H_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

void getBlockHeader(uint32_t *block_header,char *filename);
void processLine(char *buf, uint32_t *output);
uint32_t textToHex(uint8_t *input);

#endif


