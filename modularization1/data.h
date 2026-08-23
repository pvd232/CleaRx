#ifndef KWIC_M1_DATA_H
#define KWIC_M1_DATA_H

#include <stddef.h>

/* Every processing module can see and depend on these representations. */
typedef struct {
    char **words;
    size_t word_count;
} Line;

typedef struct {
    Line *items;
    size_t count;
    size_t capacity;
} LineTable;

typedef struct {
    size_t line_index;
    size_t first_word;
} ShiftRef;

typedef struct {
    ShiftRef *items;
    size_t count;
} ShiftTable;

#endif
