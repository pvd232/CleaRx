#include "line_storage.h"

#include <stdlib.h>

typedef struct {
    char **words;
    size_t *lengths;
    size_t word_count;
} StoredLine;

static StoredLine *lines;
static size_t line_count;
static size_t line_capacity;

void ls_reset(void)
{
    lines = NULL;
    line_count = 0;
    line_capacity = 0;
}

int ls_append_line(size_t word_count,
                   const size_t word_lengths[],
                   size_t *line_index_out)
{
    StoredLine line = {0};
    size_t w;

    line.words = calloc(word_count, sizeof(*line.words));
    line.lengths = calloc(word_count, sizeof(*line.lengths));
    if ((word_count != 0 && line.words == NULL) ||
        (word_count != 0 && line.lengths == NULL)) {
        free(line.words);
        free(line.lengths);
        return -1;
    }

    line.word_count = word_count;
    for (w = 0; w < word_count; ++w) {
        line.words[w] = calloc(word_lengths[w] + 1, 1);
        if (line.words[w] == NULL) {
            while (w > 0) {
                free(line.words[--w]);
            }
            free(line.words);
            free(line.lengths);
            return -1;
        }
        line.lengths[w] = word_lengths[w];
    }

    if (line_count == line_capacity) {
        size_t capacity = line_capacity == 0 ? 8 : line_capacity * 2;
        StoredLine *larger = realloc(lines, capacity * sizeof(*lines));
        if (larger == NULL) {
            for (w = 0; w < word_count; ++w) {
                free(line.words[w]);
            }
            free(line.words);
            free(line.lengths);
            return -1;
        }
        lines = larger;
        line_capacity = capacity;
    }

    lines[line_count] = line;
    *line_index_out = line_count++;
    return 0;
}

int ls_set_char(size_t line, size_t word, size_t character, int value)
{
    if (line >= line_count || word >= lines[line].word_count ||
        character >= lines[line].lengths[word]) {
        return -1;
    }
    lines[line].words[word][character] = (char)value;
    return 0;
}

int ls_char(size_t line, size_t word, size_t character)
{
    if (line >= line_count || word >= lines[line].word_count ||
        character >= lines[line].lengths[word]) {
        return -1;
    }
    return (unsigned char)lines[line].words[word][character];
}

size_t ls_line_count(void)
{
    return line_count;
}

size_t ls_words(size_t line)
{
    return line < line_count ? lines[line].word_count : 0;
}

size_t ls_chars(size_t line, size_t word)
{
    if (line >= line_count || word >= lines[line].word_count) {
        return 0;
    }
    return lines[line].lengths[word];
}

void ls_dispose(void)
{
    size_t r;

    for (r = 0; r < line_count; ++r) {
        size_t w;
        for (w = 0; w < lines[r].word_count; ++w) {
            free(lines[r].words[w]);
        }
        free(lines[r].words);
        free(lines[r].lengths);
    }
    free(lines);
    ls_reset();
}
