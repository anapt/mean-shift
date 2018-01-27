SHELL := /bin/bash

# ============================================
# COMMANDS

CC = nvcc
HOST_COMPILER = -ccbin gcc
CUDA_FLAGS = -arch=sm_21 -Wno-deprecated-gpu-targets -lcublas
C_FLAGS = -lm -O3 -I.

COMPILE_FLAGS = $(HOST_COMPILER) -x cu $(CUDA_FLAGS) -dc $(C_FLAGS)
LINK_FLAGS = $(HOST_COMPILER) $(CUDA_FLAGS) $(C_FLAGS)

OBJ = meanshift.o meanshift_utils.o meanshift_gpu_utils.o meanshift_kernels.o
DEPS = meanshift_utils.h meanshift_kernels.h

RM = rm -f

# ==========================================
# TARGETS

EXECUTABLES = meanshift

.PHONY: all clean

all: $(EXECUTABLES)

# ==========================================
# DEPENDENCIES (HEADERS)

%.o: %.cu $(DEPS)
	$(CC) $(COMPILE_FLAGS) $< -o $@

.PRECIOUS: $(EXECUTABLES) $(OBJ)

# ==========================================
# EXECUTABLE (MAIN)

$(EXECUTABLES): $(OBJ)
	$(CC) $(LINK_FLAGS) $(OBJ) -o $@

clean:
	$(RM) *.o *~ $(EXECUTABLES)