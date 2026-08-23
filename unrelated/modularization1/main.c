#include <stdio.h>

#include "alphabetizer.h"
#include "circular_shift.h"
#include "input.h"
#include "output.h"

int main(void)
{
    LineTable lines;
    ShiftTable shifts;
    ShiftTable sorted;

    /* Master Control explicitly sequences the four processing stages. */
    if (input_read(stdin, &lines) != 0) {
        fputs("input failed\n", stderr);
        return 1;
    }
    if (circular_shift_build(&lines, &shifts) != 0) {
        fputs("circular shift failed\n", stderr);
        input_free(&lines);
        return 1;
    }
    if (alphabetize(&lines, &shifts, &sorted) != 0) {
        fputs("alphabetization failed\n", stderr);
        shift_table_free(&shifts);
        input_free(&lines);
        return 1;
    }

    output_write(stdout, &lines, &sorted);

    shift_table_free(&sorted);
    shift_table_free(&shifts);
    input_free(&lines);
    return 0;
}
