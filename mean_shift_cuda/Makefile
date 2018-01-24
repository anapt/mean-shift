SHELL := /bin/bash

# ============================================
# COMMANDS

CC = /usr/local/cuda/bin/nvcc
RM = rm -f
CFLAGS= -arch=sm_21 -lm -O0 -I.
OBJ=meanshift.o meanshift_utils.o meanshift_kernels.o
DEPS=meanshift_utils.h meanshift_kernels.h

# ==========================================
# TARGETS

EXECUTABLES = meanshift

.PHONY: all clean

all: $(EXECUTABLES)

# ==========================================
# DEPENDENCIES (HEADERS)

%.o: %.cu $(DEPS)
	$(CC) -x cu $(CFLAGS) -dc $< -o $@

.PRECIOUS: $(EXECUTABLES) $(OBJ)

# ==========================================
# EXECUTABLE (MAIN)

$(EXECUTABLES): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

clean:
	$(RM) *.o *~ $(EXECUTABLES)