#include "meanshift_kernels.h"
#include <stdio.h>

__global__ void multiply_kernel(Matrix matrix1, Matrix matrix2, Matrix output){
    // Each thread computes one element of output
    // by accumulating results into cell_value
    double cell_value = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output.height && col < output.width){
        for (int element_index = 0; element_index < matrix1.width; ++element_index){
            cell_value += matrix1.elements[row * matrix1.width + element_index]
                * matrix2.elements[element_index * matrix2.width + col];
        }
        printf("%f\n", cell_value);
        output.elements[row * output.width + col] = cell_value;
    }
}