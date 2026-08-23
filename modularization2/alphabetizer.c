#include "alphabetizer.h"

#include <ctype.h>
#include <stdlib.h>

#include "circular_shifter.h"

static size_t *order;
static size_t order_count;

static int word_compare(size_t left_shift,
                        size_t left_word,
                        size_t right_shift,
                        size_t right_word)
{
    size_t left_length = cs_chars(left_shift, left_word);
    size_t right_length = cs_chars(right_shift, right_word);
    size_t limit = left_length < right_length ? left_length : right_length;
    size_t c;

    for (c = 0; c < limit; ++c) {
        int a = tolower((unsigned char)cs_char(left_shift, left_word, c));
        int b = tolower((unsigned char)cs_char(right_shift, right_word, c));
        if (a != b) {
            return a < b ? -1 : 1;
        }
    }
    if (left_length == right_length) {
        return 0;
    }
    return left_length < right_length ? -1 : 1;
}

static int shift_compare(const void *left_ptr, const void *right_ptr)
{
    size_t left = *(const size_t *)left_ptr;
    size_t right = *(const size_t *)right_ptr;
    size_t left_words = cs_words(left);
    size_t right_words = cs_words(right);
    size_t limit = left_words < right_words ? left_words : right_words;
    size_t w;

    for (w = 0; w < limit; ++w) {
        int result = word_compare(left, w, right, w);
        if (result != 0) {
            return result;
        }
    }
    if (left_words != right_words) {
        return left_words < right_words ? -1 : 1;
    }
    return left < right ? -1 : left > right;
}

int alph(void)
{
    size_t i;

    order_count = cs_count();
    order = malloc(order_count * sizeof(*order));
    if (order_count != 0 && order == NULL) {
        order_count = 0;
        return -1;
    }
    for (i = 0; i < order_count; ++i) {
        order[i] = i;
    }
    if (order_count > 1) {
        qsort(order, order_count, sizeof(*order), shift_compare);
    }
    return 0;
}

size_t alph_count(void)
{
    return order_count;
}

size_t ith(size_t rank)
{
    return rank < order_count ? order[rank] : (size_t)-1;
}

void alph_dispose(void)
{
    free(order);
    order = NULL;
    order_count = 0;
}
