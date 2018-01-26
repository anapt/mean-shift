#ifndef SERIAL_GPU_UTILS_H    /*    Include guard    */
#define SERIAL_GPU_UTILS_H

#include "meanshift_kernels.h"

//GPU error check snippet taken from:
//https://stackoverflow.com/a/14038590
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
extern char* POINTS_FILENAME;
extern char* LABELS_FILENAME;
extern parameters params;
extern cudaDeviceProp device_properties;

void set_GPU();

//Function meanshift recursively shifts original points according to th
//mean-shift algorithm saving the result to shiftedPoints. Struct opt has user
//options, h is the desirable deviation.
int meanshift(double **original_points, double ***shifted_points, int h
    , parameters *opt);

void init_device_memory(double **original_points, double **shifted_points,
    Matrix *d_original_points, Matrix *d_shifted_points,
    Matrix *d_kernel_matrix, Matrix *d_denominator, Matrix *d_new_shift);

void calculate_kernel_matrix(Matrix d_shifted_points, Matrix d_original_points,
    Matrix d_kernel_matrix, double deviation, double ***kernel_matrix);

//Function multiply allocates memory in GPU, sends the data and calls the 
//multiply kernel function.
void shift_points(Matrix d_kernel_matrix, Matrix d_original_points, Matrix d_shifted_points,
    Matrix d_new_shift, Matrix d_denominator, Matrix d_mean_shift_vector, double **kernel_matrix,
    double **original_points, double ***new_shift, double ***mean_shift_vector);

void free_device_memory(Matrix d_original_points, Matrix d_kernel_matrix, Matrix d_denominator,
    Matrix d_new_shift);

//Function calculate_denominator allocates memory in GPU, sends the data and calls the
//denominator kernel function.
void calculate_denominator(Matrix d_kernel_matrix, Matrix d_denominator);

#endif //SERIAL_GPU_UTILS_H