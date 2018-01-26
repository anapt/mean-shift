#ifndef SERIAL_KERNELS_H    /*    Include guard    */
#define SERIAL_KERNELS_H

typedef struct {
    int width;
    int height;
    double *elements;
} Matrix;

__global__ void calculate_kernel_matrix_kernel(Matrix shifted_points, Matrix original_points,
    double deviation, Matrix kernel_matrix);

//Function multiply_kernel calculates the product of matrices 1 and 2 into output.
__global__ void shift_points_kernel(Matrix original_points, Matrix kernel_matrix, Matrix shifted_points,
    Matrix new_shift, Matrix denominator, Matrix mean_shift_vector);

__global__ void denominator_kernel(Matrix denominator, Matrix kernel_matrix);

#endif //SERIAL_KERNELS_H