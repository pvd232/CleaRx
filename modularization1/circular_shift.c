#include "circular_shift.h"

#include <stdlib.h>

int circular_shift_build(const LineTable *lines, ShiftTable *shifts)
{
    size_t total = 0;
    size_t r;
    size_t i = 0;

    for (r = 0; r < lines->count; ++r) {
        total += lines->items[r].word_count;
    }

    shifts->items = calloc(total, sizeof(*shifts->items));
    shifts->count = total;
    if (total != 0 && shifts->items == NULL) {
        shifts->count = 0;
        return -1;
    }

    for (r = 0; r < lines->count; ++r) {
        size_t first;
        for (first = 0; first < lines->items[r].word_count; ++first) {
            shifts->items[i].line_index = r;
            shifts->items[i].first_word = first;
            ++i;
        }
    }
    return 0;
}

void shift_table_free(ShiftTable *shifts)
{
    free(shifts->items);
    shifts->items = NULL;
    shifts->count = 0;
}
