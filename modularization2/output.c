#include "output.h"

#include <stddef.h>

#include "alphabetizer.h"
#include "circular_shifter.h"

int output_write(FILE *stream)
{
    size_t rank;

    for (rank = 0; rank < alph_count(); ++rank) {
        size_t shift = ith(rank);
        size_t w;
        for (w = 0; w < cs_words(shift); ++w) {
            size_t c;
            if (w != 0 && fputc(' ', stream) == EOF) {
                return -1;
            }
            for (c = 0; c < cs_chars(shift, w); ++c) {
                if (fputc(cs_char(shift, w, c), stream) == EOF) {
                    return -1;
                }
            }
        }
        if (fputc('\n', stream) == EOF) {
            return -1;
        }
    }
    return 0;
}
