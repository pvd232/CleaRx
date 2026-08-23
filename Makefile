CC ?= cc
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -O2

M1 = modularization1/main.c \
	modularization1/input.c \
	modularization1/circular_shift.c \
	modularization1/alphabetizer.c \
	modularization1/output.c

M2 = modularization2/main.c \
	modularization2/line_storage.c \
	modularization2/input.c \
	modularization2/circular_shifter.c \
	modularization2/alphabetizer.c \
	modularization2/output.c

.PHONY: all test clean

all: build/kwic_m1 build/kwic_m2

build:
	mkdir -p build

build/kwic_m1: $(M1) | build
	$(CC) $(CFLAGS) $(M1) -o $@

build/kwic_m2: $(M2) | build
	$(CC) $(CFLAGS) $(M2) -o $@

test: all
	@build/kwic_m1 < sample.txt > build/m1.out
	@build/kwic_m2 < sample.txt > build/m2.out
	@cmp build/m1.out build/m2.out
	@cmp build/m1.out expected.txt
	@echo "Both modularizations produced the expected output."

clean:
	rm -rf build
