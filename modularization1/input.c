#include "input.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

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

static int append_word(Line *line, const char *start, size_t length)
{
    char **larger = realloc(line->words,
                            (line->word_count + 1) * sizeof(*line->words));
    char *word;

    if (larger == NULL) {
        return -1;
    }
    line->words = larger;

    word = malloc(length + 1);
    if (word == NULL) {
        return -1;
    }
    memcpy(word, start, length);
    word[length] = '\0';
    line->words[line->word_count++] = word;
    return 0;
}

static int parse_line(char *text, Line *line)
{
    char *cursor = text;

    line->words = NULL;
    line->word_count = 0;

    while (*cursor != '\0') {
        char *start;

        while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        start = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (cursor != start && append_word(line, start, (size_t)(cursor - start)) != 0) {
            return -1;
        }
    }
    return 0;
}

static int append_line(LineTable *lines, Line line)
{
    if (lines->count == lines->capacity) {
        size_t capacity = lines->capacity == 0 ? 8 : lines->capacity * 2;
        Line *larger = realloc(lines->items, capacity * sizeof(*lines->items));
        if (larger == NULL) {
            return -1;
        }
        lines->items = larger;
        lines->capacity = capacity;
    }
    lines->items[lines->count++] = line;
    return 0;
}

static void free_line(Line *line)
{
    size_t w;

    for (w = 0; w < line->word_count; ++w) {
        free(line->words[w]);
    }
    free(line->words);
    line->words = NULL;
    line->word_count = 0;
}

void input_free(LineTable *lines)
{
    size_t r;

    for (r = 0; r < lines->count; ++r) {
        free_line(&lines->items[r]);
    }
    free(lines->items);
    lines->items = NULL;
    lines->count = 0;
    lines->capacity = 0;
}

int input_read(FILE *stream, LineTable *lines)
{
    int status;

    lines->items = NULL;
    lines->count = 0;
    lines->capacity = 0;

    for (;;) {
        char *text = NULL;
        Line line;

        status = read_line(stream, &text);
        if (status <= 0) {
            if (status < 0) {
                input_free(lines);
            }
            return status < 0 ? -1 : 0;
        }

        if (parse_line(text, &line) != 0) {
            free(text);
            free_line(&line);
            input_free(lines);
            return -1;
        }
        free(text);

        if (line.word_count == 0) {
            free(line.words);
            continue;
        }
        if (append_line(lines, line) != 0) {
            free_line(&line);
            input_free(lines);
            return -1;
        }
    }
}
