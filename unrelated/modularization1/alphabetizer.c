#include "alphabetizer.h"

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

/* qsort has no portable context argument, so this module holds one temporarily. */
static const LineTable *sort_lines;

static int word_compare(const char *left, const char *right)
{
    while (*left != '\0' && *right != '\0') {
        int a = tolower((unsigned char)*left);
        int b = tolower((unsigned char)*right);
        if (a != b) {
            return a < b ? -1 : 1;
        }
        ++left;
        ++right;
    }
    if (*left == *right) {
        return 0;
    }
    return *left == '\0' ? -1 : 1;
}

static int shift_compare(const void *left_ptr, const void *right_ptr)
{
    const ShiftRef *left = left_ptr;
    const ShiftRef *right = right_ptr;
    const Line *left_line = &sort_lines->items[left->line_index];
    const Line *right_line = &sort_lines->items[right->line_index];
    size_t i;
    size_t limit = left_line->word_count < right_line->word_count
                       ? left_line->word_count
                       : right_line->word_count;

    for (i = 0; i < limit; ++i) {
        const char *left_word = left_line->words[(left->first_word + i) % left_line->word_count];
        const char *right_word = right_line->words[(right->first_word + i) % right_line->word_count];
        int result = word_compare(left_word, right_word);
        if (result != 0) {
            return result;
        }
    }
    if (left_line->word_count != right_line->word_count) {
        return left_line->word_count < right_line->word_count ? -1 : 1;
    }
    if (left->line_index != right->line_index) {
        return left->line_index < right->line_index ? -1 : 1;
    }
    if (left->first_word != right->first_word) {
        return left->first_word < right->first_word ? -1 : 1;
    }
    return 0;
}

int alphabetize(const LineTable *lines,
                const ShiftTable *unsorted,
                ShiftTable *sorted)
{
    sorted->count = unsorted->count;
    sorted->items = malloc(sorted->count * sizeof(*sorted->items));
    if (sorted->count != 0 && sorted->items == NULL) {
        sorted->count = 0;
        return -1;
    }
    if (sorted->count != 0) {
        memcpy(sorted->items, unsorted->items,
               sorted->count * sizeof(*sorted->items));
    }

    sort_lines = lines;
    if (sorted->count > 1) {
        qsort(sorted->items, sorted->count,
              sizeof(*sorted->items), shift_compare);
    }
    sort_lines = NULL;
    return 0;
}
