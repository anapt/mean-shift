#ifndef SERIAL_UTILS_H    /*    Include guard    */
#define SERIAL_UTILS_H

#include <stdbool.h>
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

/*        Structs       */
typedef struct parameters {
    double epsilon;
    bool verbose;
    bool display;
} parameters;

/*        Global variables        */
extern int DEVIATION;
extern int NUMBER_OF_POINTS;
extern int DIMENSIONS;
extern char* POINTS_FILENAME;
extern char* LABELS_FILENAME;
extern parameters params;
extern cudaDeviceProp device_properties;

//Function get_args parses command line arguments.
void get_args(int argc, char **argv, parameters *params);

//Function init reads the dataset and label arrays from the corresponding files.
void init(double ***vectors, char **labels);

void set_GPU();

//Function meanshift recursively shifts original points according to th
//mean-shift algorithm saving the result to shiftedPoints. Struct opt has user
//options, h is the desirable deviation.
int meanshift(double **original_points, double ***shifted_points, int h
    , parameters *opt);

void init_device_memory(double **original_points, double **shifted_points,
    Matrix *d_original_points, Matrix *d_shifted_points,
    Matrix *d_kernel_matrix, Matrix *d_denominator, Matrix *d_new_shift);

//Function norm returns the second norm of matrix of dimensions rowsXcols.
double norm(double **matrix, int rows, int cols);

void calculate_kernel_matrix(Matrix d_shifted_points, Matrix d_original_points,
    Matrix d_kernel_matrix, double deviation, double ***kernel_matrix);

//Function multiply allocates memory in GPU, sends the data and calls the 
//multiply kernel function.
void shift_points(Matrix d_kernel_matrix, Matrix d_original_points, Matrix d_shifted_points,
    Matrix d_new_shift, Matrix d_denominator, Matrix d_mean_shift_vector, double **kernel_matrix,
    double **original_points, double ***new_shift, double ***mean_shift_vector);

void free_device_memory(Matrix d_original_points, Matrix d_kernel_matrix, Matrix d_denominator,
    Matrix d_new_shift);

//Function calculateDistance returns the distance between x and y vectors.
double calculateDistance(double *y, double *x);

//Function alloc_2d_double allocates rows*cols bytes of continuous memory.
double **alloc_2d_double(int rows, int cols);

//Function duplicate copies the values of source array to dest array.
void duplicate(double **source, int rows, int cols, double ***dest);

//Function print_matrix prints array of dimensions rowsXcols to the console.
void print_matrix(double **array, int rows, int cols);

//Function save_matrix prints matrix in a csv file with path/filename
//"output/output_iteration". If a file already exists new lines are concatenated.
void save_matrix(double **matrix
    , int iteration);

//Function calculate_denominator allocates memory in GPU, sends the data and calls the
//denominator kernel function.
double * calculate_denominator(double **kernel_matrix);

#endif //SERIAL_UTILS_H