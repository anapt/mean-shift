#include "meanshift_kernels.h"
#include <stdio.h>

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
    // each thread computes one element of new_shift
    // by accumulating results into cell_value
    double cell_value = 0;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // performs calculations only if thread's indexes are within matrix bounds
    if (row * new_shift.width + col >= new_shift.width * new_shift.height){
        return;
    }

    // calculates new_shift
    // builds nominator by multiplying kernel_matrix and original_points
    for (int element_index = 0; element_index < kernel_matrix.width; ++element_index){
        cell_value += kernel_matrix.elements[row * kernel_matrix.width + element_index]
            * original_points.elements[element_index * original_points.width + col];
    }
    // new_shift elements are calculated by dividing with the denominator
    new_shift.elements[row * new_shift.width + col] =
        cell_value / denominator.elements[row];

    // calculates mean-shift vector
    mean_shift_vector.elements[row * new_shift.width + col] =
        new_shift.elements[row * new_shift.width + col] -
        shifted_points.elements[row * new_shift.width + col];
}

__global__ void norm(Matrix mean_shift_vector, double *current_norm) {
    // each thread computes one element of new_shift
    // by accumulating results into cell_value
    double cell_value = 0;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // performs calculations only if thread's indexes are within matrix bounds
    if (row * mean_shift_vector.width + col >= mean_shift_vector.width * mean_shift_vector.height){
        return;
    }

    for (int element_index = 0; element_index < mean_shift_vector.width; ++element_index){
        cell_value += mean_shift_vector.elements[row * mean_shift_vector.width + element_index]
                      * mean_shift_vector.elements[row * mean_shift_vector.width + element_index];
    }

    &current_norm = sqrt(cell_value);


//    // new_shift elements are calculated by dividing with the denominator
//    new_shift.elements[row * new_shift.width + col] =
//            cell_value / denominator.elements[row];
//
//    // calculates mean-shift vector
//    mean_shift_vector.elements[row * new_shift.width + col] =
//            new_shift.elements[row * new_shift.width + col] -
//            shifted_points.elements[row * new_shift.width + col];

//    int n_tid = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
//    int i = 1;
//    int initial_tid = n_tid / 2;
//    int limit = gridDim.x * blockDim.x;

//    int block_end = 2 * (blockIdx.x * blockDim.x + blockDim.x) - 1;
//
//    if (n_tid < (2 * limit)){
//
//        while ( (i < (2 * blockDim.x)) && n_tid < block_end &&
//                (n_tid + i) <= block_end){
//
//            norms[n_tid] += norms[n_tid + i];
//            n_tid = n_tid + i * (initial_tid * 2 - 2 * (blockIdx.x * blockDim.x));
//            i *= 2;
//            __syncthreads();
//        }
//
//
//        if (!((initial_tid) % blockDim.x))
//            norm_per_block[blockIdx.x] = norms[n_tid];
//
//    }
}