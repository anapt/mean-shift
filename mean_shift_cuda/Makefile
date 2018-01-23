SHELL := /bin/bash

# ============================================
# COMMANDS

CC = nvcc
RM = rm -f
CFLAGS=-lm -O3 -Wall -I.
OBJ=meanshift.o meanshift_declarations.o
DEPS=meanshift_declarations.h

# ==========================================
# TARGETS

EXECUTABLES = meanshift

.PHONY: all clean

all: $(EXECUTABLES)

# ==========================================
# DEPENDENCIES (HEADERS)

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PRECIOUS: $(EXECUTABLES) $(OBJ)

# ==========================================
# EXECUTABLE (MAIN)

$(EXECUTABLES): $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	$(RM) *.o *~ $(EXECUTABLES)
