#include "output.h"

void output_write(FILE *stream,
                  const LineTable *lines,
                  const ShiftTable *sorted)
{
    size_t i;

    for (i = 0; i < sorted->count; ++i) {
        ShiftRef shift = sorted->items[i];
        const Line *line = &lines->items[shift.line_index];
        size_t offset;

        for (offset = 0; offset < line->word_count; ++offset) {
            size_t word = (shift.first_word + offset) % line->word_count;
            if (offset != 0) {
                fputc(' ', stream);
            }
            fputs(line->words[word], stream);
        }
        fputc('\n', stream);
    }
}
