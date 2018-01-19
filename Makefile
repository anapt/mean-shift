SHELL := /bin/bash

# ============================================
# COMMANDS

CC = gcc
RM = rm -f
CFLAGS=-lm -O3 -Wall -I.
OBJ=serial.o serialDeclarations.o
DEPS=serialDeclarations.h

# ==========================================
# TARGETS

EXECUTABLES = serial

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
