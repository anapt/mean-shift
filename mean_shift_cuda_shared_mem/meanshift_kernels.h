#ifndef SERIAL_KERNELS_H    /*    Include guard    */
#define SERIAL_KERNELS_H

/*      Structures     */

//Matrix is used to describe matrices
typedef struct {
    int width;
    int height;
    int stride;
    double *elements;
} Matrix;

//Kernel calculate_kernel_matrix_kernel calculates the current kernel matrix
__global__ void calculate_kernel_matrix_kernel(Matrix shifted_points, Matrix original_points,
    double deviation, Matrix kernel_matrix);

//Kernel denominator_kernel calculates the sum in the denominator of the fraction used to find new
//(shifted) positions of the points
__global__ void denominator_kernel(Matrix denominator, Matrix kernel_matrix);

//Kernel shift_points_kernel shifts the positions of all points and calculates the new mean shift
//vector according to the new point array
__global__ void shift_points_kernel(Matrix original_points, Matrix kernel_matrix,
    Matrix shifted_points, Matrix new_shift, Matrix denominator, Matrix mean_shift_vector);

__device__ Matrix get_sub_matrix(Matrix A, int row, int col, int ROW_BLOCK_SIZE,
    int COLUMN_BLOCK_SIZE);

#endif //SERIAL_KERNELS_H