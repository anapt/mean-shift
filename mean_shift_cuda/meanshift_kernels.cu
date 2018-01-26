#include "meanshift_kernels.h"
#include <stdio.h>

__global__ void multiply_kernel(Matrix matrix1, Matrix matrix2, Matrix output){
    // Each thread computes one element of output
    // by accumulating results into cell_value
    double cell_value = 0;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row + col < output.height * output.width){
        for (int element_index = 0; element_index < matrix1.width; ++element_index){
            cell_value += matrix1.elements[row * matrix1.width + element_index]
                * matrix2.elements[element_index * matrix2.width + col];
        }
        output.elements[row * output.width + col] = cell_value;
    }
}

__global__ void calculate_kernel_matrix_kernel(Matrix shifted_points, Matrix original_points
    , double deviation, Matrix kernel_matrix){
    // Each thread calculates one element of kernel_matrix
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row * kernel_matrix.width + col > kernel_matrix.width * kernel_matrix.height){
        return;
    }

    int dimensions = shifted_points.width;
    // calculate distance
    double sum = 0, dif;
    for (int i=0; i<dimensions; i++){
        dif = shifted_points.elements[row * dimensions + i] - original_points.elements[col * dimensions + i];
        sum += dif * dif;
    }
    double distance = sqrt(sum);

    double deviation_square = deviation*deviation;
    if (distance < deviation_square){
        // computes kernel matrix
        double pow = ((-1)*(distance * distance))/(2*(deviation_square));
        kernel_matrix.elements[row * kernel_matrix.width + col] = exp(pow);
    } else {
        kernel_matrix.elements[row * kernel_matrix.width + col] = 0;
    }
    if (row == col){
        kernel_matrix.elements[row * kernel_matrix.width + col] += 1;
    }
}

__global__ void denominator_kernel<<<dimGrid, dimBlock>>>(Matrix denominator, Matrix kernel_matrix, int total){

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;


    if (row * denominator.width + col > denominator.width * denominator.height){
        return;
    }

    denominator[col]=0;
    denominator[row] += kernel_matrix[row*denominator.width + col];

}

// serial

// calculate denominator
for (int i=0; i<NUMBER_OF_POINTS; i++){
    double sum = 0;
    for (int j=0; j<NUMBER_OF_POINTS; j++){
        sum = sum + kernel_matrix[i][j];
    }
    denominator[i] = sum;
}