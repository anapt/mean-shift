#include "meanshift_kernels.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void calculate_kernel_matrix_kernel(Matrix shifted_points, Matrix original_points,
    double deviation, Matrix kernel_matrix){
    // each thread calculates one element of kernel_matrix
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // performs calculations only if thread's indexes are within matrix bounds
    if (row * kernel_matrix.width + col >= kernel_matrix.width * kernel_matrix.height){
        return;
    }

    int dimensions = shifted_points.width;
    // calculate distance
    double sum = 0, dif;
    for (int i=0; i<dimensions; i++){
        dif = shifted_points.elements[row * dimensions + i]
            - original_points.elements[col * dimensions + i];
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

__global__ void denominator_kernel(Matrix denominator, Matrix kernel_matrix){
    // each thread computes one element of denominator_kernel
    // by accumulating results into cell_value
    double cell_value = 0;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // performs calculations only if thread's indexes are within matrix bounds
    if (row >= denominator.height){
        return;
    }

    for (int column = 0; column < kernel_matrix.width; ++column){
         cell_value += kernel_matrix.elements[row * kernel_matrix.width + column];
    }
    denominator.elements[row] = cell_value;
}

__global__ void shift_points_kernel(Matrix original_points, Matrix kernel_matrix,
    Matrix shifted_points, Matrix new_shift, Matrix denominator, Matrix mean_shift_vector){
    int BLOCK_SIZE = blockDim.y;
    int block_row = blockIdx.x;
    int block_col = blockIdx.y;
    
    // each thread computes one element of new_shift by accumulating results into cell_value
    double cell_value = 0;

    // Thread row and column within sub_new_shift
    int row = threadIdx.x;
    int col = threadIdx.y;

        // performs calculations only if thread's indexes are within matrix bounds
    //if (row * new_shift.width + col >= new_shift.width * new_shift.height){
    /*if (new_shift.stride * BLOCK_SIZE * block_row + BLOCK_SIZE * block_col >=
        new_shift.width * new_shift.height){*/
    if (BLOCK_SIZE * block_row >= new_shift.height || BLOCK_SIZE * block_col >= new_shift.width){
        return;
    }

    // Each thread block computes one sub-matrix sub_new_shift of C
    Matrix sub_new_shift = GetSubMatrix(new_shift, block_row, block_col, BLOCK_SIZE);

    // shared memory used to store sub_kernel_matrix and sub_original_points respectively
    __shared__ double *s_sub_kernel_matrix;
    s_sub_kernel_matrix = (double*)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    __shared__ double *s_sub_original_points;
    s_sub_original_points = (double*)malloc(BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    // loops over all the sub-matrices of kernel_matrix and original_points that are required to
    //compute sub_new_shift, multiplies each pair of sub-matrices and accumulates the results
    for (int sub_matrix_index = 0; sub_matrix_index < (kernel_matrix.width / BLOCK_SIZE); ++sub_matrix_index) {

        // gets sub-matrix sub_kernel_matrix of kernel_matrix
        Matrix sub_kernel_matrix = GetSubMatrix(kernel_matrix, block_row, sub_matrix_index, BLOCK_SIZE);
        // gets sub-matrix sub_original_points of original_points
        Matrix sub_original_points = GetSubMatrix(original_points, sub_matrix_index, block_col, BLOCK_SIZE);

        // loads s_sub_kernel_matrix and s_sub_original_points from device global memory to shared
        //memory, each thread loads one element of each sub-matrix
        s_sub_kernel_matrix[row * BLOCK_SIZE + col] =
            sub_kernel_matrix.elements[row * sub_kernel_matrix.stride + col];
        s_sub_original_points[row * BLOCK_SIZE + col] =
            sub_original_points.elements[row * sub_original_points.stride + col];

        // synchronizes to make sure the sub-matrices are loaded before starting the computation
        __syncthreads();

        // multiplies sub_kernel_matrix and sub_original_points
        for (int element_index = 0; element_index < BLOCK_SIZE; ++element_index){
            cell_value += s_sub_kernel_matrix[row * sub_kernel_matrix.stride + element_index] *
                s_sub_original_points[element_index * sub_original_points.stride + col];
        }

        // synchronizes to make sure that the preceding computation is done before loading two new
        // sub-matrices of kernel_matrix and original_points in the next iteration
        __syncthreads();
    }

    // new_shift elements are calculated by dividing with the denominator
    int cell_row = (block_row * BLOCK_SIZE + row) * new_shift.stride;
    int cell_col = block_col * BLOCK_SIZE + col;
    //sub_new_shift.elements[cell_row + cell_col] = cell_value / denominator.elements[cell_row];
    sub_new_shift.elements[row * sub_new_shift.stride + col] =
        cell_value / denominator.elements[block_row * BLOCK_SIZE + row];

    // calculates mean-shift vector
    /*mean_shift_vector.elements[(block_row * BLOCK_SIZE + row) * mean_shift_vector.stride
        + (block_col * BLOCK_SIZE + col)] =
        sub_new_shift.elements[row * sub_new_shift.stride + col] -
        shifted_points.elements[(block_row * BLOCK_SIZE + row) * shifted_points.stride
        + (block_col * BLOCK_SIZE + col)];*/

    /*free(s_sub_kernel_matrix);
    free(s_sub_original_points);*/
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix GetSubMatrix(Matrix A, int row, int col, int BLOCK_SIZE){
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = BLOCK_SIZE;
    Asub.elements = &(A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col]);
    return Asub;
}