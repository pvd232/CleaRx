#include "input.h"

#include <ctype.h>
#include <stdlib.h>

#include "line_storage.h"

static int read_line(FILE *stream, char **text_out)
{
    size_t length = 0;
    size_t capacity = 64;
    char *text = malloc(capacity);
    int ch;

    if (text == NULL) {
        return -1;
    }
    while ((ch = fgetc(stream)) != EOF && ch != '\n') {
        if (length + 1 >= capacity) {
            char *larger;
            capacity *= 2;
            larger = realloc(text, capacity);
            if (larger == NULL) {
                free(text);
                return -1;
            }
            text = larger;
        }
        text[length++] = (char)ch;
    }
    if (ch == EOF && length == 0) {
        free(text);
        return 0;
    }
    if (length > 0 && text[length - 1] == '\r') {
        --length;
    }
    text[length] = '\0';
    *text_out = text;
    return 1;
}

static int store_line(char *text)
{
    size_t word_count = 0;
    size_t capacity = 8;
    char **starts = malloc(capacity * sizeof(*starts));
    size_t *lengths = malloc(capacity * sizeof(*lengths));
    char *cursor = text;
    size_t line;
    size_t w;

    if (starts == NULL || lengths == NULL) {
        free(starts);
        free(lengths);
        return -1;
    }

    while (*cursor != '\0') {
        char *start;
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        start = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (cursor == start) {
            continue;
        }
        if (word_count == capacity) {
            size_t new_capacity = capacity * 2;
            char **larger_starts = realloc(starts, new_capacity * sizeof(*starts));
            size_t *larger_lengths;
            if (larger_starts == NULL) {
                free(starts);
                free(lengths);
                return -1;
            }
            starts = larger_starts;
            larger_lengths = realloc(lengths, new_capacity * sizeof(*lengths));
            if (larger_lengths == NULL) {
                free(starts);
                free(lengths);
                return -1;
            }
            lengths = larger_lengths;
            capacity = new_capacity;
        }
        starts[word_count] = start;
        lengths[word_count] = (size_t)(cursor - start);
        ++word_count;
    }

    if (word_count == 0) {
        free(starts);
        free(lengths);
        return 0;
    }
    if (ls_append_line(word_count, lengths, &line) != 0) {
        free(starts);
        free(lengths);
        return -1;
    }
    for (w = 0; w < word_count; ++w) {
        size_t c;
        for (c = 0; c < lengths[w]; ++c) {
            if (ls_set_char(line, w, c, (unsigned char)starts[w][c]) != 0) {
                free(starts);
                free(lengths);
                return -1;
            }
        }
    }
    free(starts);
    free(lengths);
    return 0;
}

int input_read(FILE *stream)
{
    int status;

    ls_reset();
    for (;;) {
        char *text = NULL;
        status = read_line(stream, &text);
        if (status <= 0) {
            if (status < 0) {
                ls_dispose();
            }
            return status < 0 ? -1 : 0;
        }
        if (store_line(text) != 0) {
            free(text);
            ls_dispose();
            return -1;
        }
        free(text);
    }
}
