#ifndef KWIC_M2_LINE_STORAGE_H
#define KWIC_M2_LINE_STORAGE_H

#include <stddef.h>

/* The representation of lines and words is absent from this interface. */
void ls_reset(void);
int ls_append_line(size_t word_count,
                   const size_t word_lengths[],
                   size_t *line_index_out);
int ls_set_char(size_t line, size_t word, size_t character, int value);
int ls_char(size_t line, size_t word, size_t character);
size_t ls_line_count(void);
size_t ls_words(size_t line);
size_t ls_chars(size_t line, size_t word);
void ls_dispose(void);

#endif
