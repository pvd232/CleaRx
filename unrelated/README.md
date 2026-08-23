# Parnas's two KWIC modularizations in C

These two programs implement the same KWIC behavior with different module
boundaries. Each nonempty input line is a title. The program generates one
circular shift per word and prints every shift in case-insensitive alphabetical
order.

This code interprets the two designs in D. L. Parnas, "On the Criteria To Be
Used in Decomposing Systems into Modules," *Communications of the ACM* 15(12),
1972, pp. 1053-1058, DOI 10.1145/361598.361623. The paper specifies module
responsibilities and interfaces and leaves the source code unspecified. These
files supply illustrative C11 implementations of those responsibilities.

## The controlled comparison

Both executables use the same input, sorting rule, and output. They differ in
the information that crosses module boundaries.

| Design | Boundary | Consequence visible in the code |
|---|---|---|
| Modularization 1 | Processing stages exchange `LineTable` and `ShiftTable` structures from `data.h`. | The circular shifter, alphabetizer, and output module all depend on the storage representation. |
| Modularization 2 | Modules exchange queries such as `ls_char`, `cs_char`, and `ith`. | Only `line_storage.c` knows how lines are stored; only `circular_shifter.c` knows whether shifts are stored or computed. |

The programs make three local choices that the paper leaves open: words are
separated by spaces or tabs, punctuation remains part of a word, and sorting
folds ASCII letter case. Every word remains eligible because the paper's KWIC
definition includes every circular shift.

## Build and run

The project requires a C11 compiler and `make`.

```sh
make
printf 'Gone with the Wind\nThe Old Man and the Sea\n' | build/kwic_m1
printf 'Gone with the Wind\nThe Old Man and the Sea\n' | build/kwic_m2
make test
```

`make test` runs both programs on `sample.txt` and requires byte-for-byte equal
output.

## Reading order

Start with `modularization1/data.h`. Every processing module includes that
header, which makes the shared representation easy to see. Then open
`modularization2/line_storage.h`, `circular_shifter.h`, and `alphabetizer.h`.
Those headers publish operations. The arrays remain private to the `.c` files.
