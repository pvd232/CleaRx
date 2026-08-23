#include "circular_shifter.h"

#include <stdlib.h>

#include "line_storage.h"

typedef struct {
    size_t line;
    size_t first_word;
} HiddenShift;

static HiddenShift *shifts;
static size_t shift_count;

int cs_setup(void)
{
    size_t r;
    size_t i = 0;

    shift_count = 0;
    for (r = 0; r < ls_line_count(); ++r) {
        shift_count += ls_words(r);
    }
    shifts = calloc(shift_count, sizeof(*shifts));
    if (shift_count != 0 && shifts == NULL) {
        shift_count = 0;
        return -1;
    }
    for (r = 0; r < ls_line_count(); ++r) {
        size_t first;
        for (first = 0; first < ls_words(r); ++first) {
            shifts[i].line = r;
            shifts[i].first_word = first;
            ++i;
        }
    }
    return 0;
}

size_t cs_count(void)
{
    return shift_count;
}

size_t cs_words(size_t shift)
{
    return shift < shift_count ? ls_words(shifts[shift].line) : 0;
}

static size_t original_word(size_t shift, size_t word)
{
    size_t count = cs_words(shift);
    return (shifts[shift].first_word + word) % count;
}

size_t cs_chars(size_t shift, size_t word)
{
    if (shift >= shift_count || word >= cs_words(shift)) {
        return 0;
    }
    return ls_chars(shifts[shift].line, original_word(shift, word));
}

int cs_char(size_t shift, size_t word, size_t character)
{
    if (shift >= shift_count || word >= cs_words(shift)) {
        return -1;
    }
    return ls_char(shifts[shift].line,
                   original_word(shift, word),
                   character);
}

size_t cs_original_line(size_t shift)
{
    return shift < shift_count ? shifts[shift].line : (size_t)-1;
}

void cs_dispose(void)
{
    free(shifts);
    shifts = NULL;
    shift_count = 0;
}
