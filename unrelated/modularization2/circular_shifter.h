#ifndef KWIC_M2_CIRCULAR_SHIFTER_H
#define KWIC_M2_CIRCULAR_SHIFTER_H

#include <stddef.h>

/* Clients see a virtual sequence of shifts, not the shift table. */
int cs_setup(void);
size_t cs_count(void);
size_t cs_words(size_t shift);
size_t cs_chars(size_t shift, size_t word);
int cs_char(size_t shift, size_t word, size_t character);
size_t cs_original_line(size_t shift);
void cs_dispose(void);

#endif
