#ifndef KWIC_M1_CIRCULAR_SHIFT_H
#define KWIC_M1_CIRCULAR_SHIFT_H

#include "data.h"

int circular_shift_build(const LineTable *lines, ShiftTable *shifts);
void shift_table_free(ShiftTable *shifts);

#endif
