#ifndef SERIAL_GPU_UTILS_H    /*    Include guard    */
#define SERIAL_GPU_UTILS_H

#include "meanshift_kernels.h"

//GPU error check snippet taken from:
//              https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
   if (code != cudaSuccess){
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/*        Global variables        */
extern int DEVIATION;
extern int NUMBER_OF_POINTS;
extern int DIMENSIONS;
extern const char* POINTS_FILENAME;
extern const char* LABELS_FILENAME;
extern Parameters params;
extern cudaDeviceProp device_properties;

//Function set_GPU parses available GPU devices, selects the one with the most multi-processors for
//usage and stores its properties in global struct device_properties
void set_GPU();

//Function meanshift recursively shifts original points according to the mean-shift algorithm saving
//the result to shiftedPoints, h is the desired deviation
int meanshift(double **original_points, double ***shifted_points, int h);

//Function init_device_memory allocates memory for necessary arrays in the device
void init_device_memory(double **original_points, double **shifted_points,
    Matrix *d_original_points, Matrix *d_shifted_points, Matrix *d_kernel_matrix,
    Matrix *d_denominator, Matrix *d_new_shift);

//Function calculate_kernel_matrix is a wrapper for the kernel call of the corresponding kernel
//"calculate_kernel_matrix_kernel" that calculates the kernel matrix
void calculate_kernel_matrix(Matrix d_shifted_points, Matrix d_original_points,
    Matrix d_kernel_matrix, double deviation, double ***kernel_matrix, double *w_memcpy_time);

//Function calculate_denominator is a wrapper for the kernel call of the corresponding kernel
//"calculate_denominator_kernel" that calculates the denominator of shifted points fraction
void calculate_denominator(Matrix d_kernel_matrix, Matrix d_denominator);

//Function shift_points is a wrapper for the kernel call of the corresponding kernel
//"shift_points_kernel" that shifts the positions of all points
void shift_points(Matrix d_kernel_matrix, Matrix d_original_points, Matrix d_shifted_points,
    Matrix d_new_shift, Matrix d_denominator, Matrix d_mean_shift_vector, double **kernel_matrix,
    double **original_points, double ***new_shift, double ***mean_shift_vector,
    double *w_memcpy_time);

//Function calculate_norm is a wrapper for the kernel call of the corresponing kernel
//"norm" that calculate the norm of the mean_shift_vector matrix
void calculate_norm(Matrix d_mean_shift_vector, double *current_norm);

//Function free_device_memory frees device's previously allocated memory
void free_device_memory(Matrix d_original_points, Matrix d_kernel_matrix, Matrix d_denominator,
    Matrix d_shifted_points);

#endif //SERIAL_GPU_UTILS_H