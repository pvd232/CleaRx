#ifndef KWIC_M1_INPUT_H
#define KWIC_M1_INPUT_H

#include <stdio.h>

#include "data.h"

int input_read(FILE *stream, LineTable *lines);
void input_free(LineTable *lines);

#endif
