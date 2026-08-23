#include <stdio.h>

#include "alphabetizer.h"
#include "circular_shifter.h"
#include "input.h"
#include "line_storage.h"
#include "output.h"

int main(void)
{
    int status = 1;

    if (input_read(stdin) != 0) {
        fputs("input failed\n", stderr);
        return 1;
    }
    if (cs_setup() != 0) {
        fputs("circular shift failed\n", stderr);
        goto cleanup_lines;
    }
    if (alph() != 0) {
        fputs("alphabetization failed\n", stderr);
        goto cleanup_shifts;
    }
    if (output_write(stdout) != 0) {
        fputs("output failed\n", stderr);
        goto cleanup_alphabetizer;
    }

    status = 0;

cleanup_alphabetizer:
    alph_dispose();
cleanup_shifts:
    cs_dispose();
cleanup_lines:
    ls_dispose();
    return status;
}
