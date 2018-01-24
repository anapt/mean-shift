SHELL := /bin/bash

# ============================================
# COMMANDS

CC = /usr/local/cuda/bin/nvcc
RM = rm -f
HOST_COMPILER = -ccbin gcc
CFLAGS= -arch=sm_21 -lm -O0 -I. -Wno-deprecated-gpu-targets
OBJ = meanshift.o meanshift_utils.o meanshift_kernels.o
DEPS = meanshift_utils.h meanshift_kernels.h

# ==========================================
# TARGETS

EXECUTABLES = meanshift

.PHONY: all clean

all: $(EXECUTABLES)

# ==========================================
# DEPENDENCIES (HEADERS)

%.o: %.cu $(DEPS)
	$(CC) $(HOST_COMPILER) -x cu $(CFLAGS) -dc $< -o $@

.PRECIOUS: $(EXECUTABLES) $(OBJ)

# ==========================================
# EXECUTABLE (MAIN)

$(EXECUTABLES): $(OBJ)
	$(CC) $(HOST_COMPILER) $(CFLAGS) $(OBJ) -o $@

clean:
	$(RM) *.o *~ $(EXECUTABLES)