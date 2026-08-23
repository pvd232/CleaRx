#ifndef KWIC_M1_ALPHABETIZER_H
#define KWIC_M1_ALPHABETIZER_H

#include "data.h"

int alphabetize(const LineTable *lines,
                const ShiftTable *unsorted,
                ShiftTable *sorted);

#endif
