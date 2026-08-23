#ifndef KWIC_M1_OUTPUT_H
#define KWIC_M1_OUTPUT_H

#include <stdio.h>

#include "data.h"

void output_write(FILE *stream,
                  const LineTable *lines,
                  const ShiftTable *sorted);

#endif
