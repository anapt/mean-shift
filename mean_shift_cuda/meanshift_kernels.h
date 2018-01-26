#ifndef SERIAL_KERNELS_H    /*    Include guard    */
#define SERIAL_KERNELS_H

typedef struct{
    int width;
    int height;
    double *elements;
} Matrix;

//Function multiply_kernel calculates the product of matrices 1 and 2 into output.
__global__ void multiply_kernel(Matrix matrix1, Matrix matrix2, Matrix output);

__global__ void calculate_kernel_matrix_kernel(Matrix shifted_points, Matrix original_points
    , double deviation, Matrix kernel_matrix);

__global__ void denominator_kernel(Matrix denominator, Matrix kernel_matrix, int T);

#endif //SERIAL_KERNELS_H