#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <sys/time.h>

#include <cublas_v2.h>

#include "meanshift_utils.h"
#include "meanshift_gpu_utils.h"

cudaDeviceProp device_properties;

struct timeval start, end;
double seq;

//Based on:
//          https://www.cs.virginia.edu/~csadmin/wiki/index.php/CUDA_Support/Choosing_a_GPU
void set_GPU(){
    int devices_count = 0, max_multiprocessors = 0, max_device = 0;

    // gets devices count checking for errors like no devices or no drivers to check for
    // devices available
    gpuErrchk( cudaGetDeviceCount(&devices_count) );
    for(int device_index = 0; device_index < devices_count; ++device_index){
        // gets current index device's properties
        cudaDeviceProp this_device_properties;
        gpuErrchk( cudaGetDeviceProperties(&this_device_properties, device_index) );

        // stores best available device's index
        // only devices with compute capability >= 2.0 are able to run the code
        if (max_multiprocessors < this_device_properties.multiProcessorCount
            && this_device_properties.major >= 2 && this_device_properties.minor >= 0){
            // stores devices properties for later use
            device_properties = this_device_properties;
            max_multiprocessors = this_device_properties.multiProcessorCount;
            max_device = device_index;
        }
    }
    // sets the device
    gpuErrchk( cudaSetDevice(max_device) );
    if (params.verbose){
        printf("Device chosen is \"%s\"\n"
            "Device has %d multi processors and compute capability %d.%d\n"
            "Max threads per block supported are %d\n\n"
            , device_properties.name
            , device_properties.multiProcessorCount, device_properties.major, device_properties.minor
            , device_properties.maxThreadsPerBlock);
    }
}

int meanshift(double **original_points, double ***shifted_points, int deviation
    , parameters *opt){
    // host variables
    int size = 0;
    static int iteration = 0;
    static double **kernel_matrix, **mean_shift_vector;
    double **new_shift, current_norm = 0;

    // device variables
    static Matrix d_original_points, d_shifted_points, d_kernel_matrix, d_denominator,
        d_mean_shift_vector;
    Matrix d_new_shift;

    // allocates memory and copies original points on first iteration
    if (iteration == 0 || (*shifted_points) == NULL){
        // allocates memory for shifted points array and copies original points into it
        (*shifted_points) = alloc_double(NUMBER_OF_POINTS, DIMENSIONS);
        duplicate(original_points, NUMBER_OF_POINTS, DIMENSIONS, shifted_points);

        // allocates memory for mean shift vector
        mean_shift_vector = alloc_double(NUMBER_OF_POINTS, DIMENSIONS);
        // initializes elements of mean_shift_vector to inf
        for (int i=0;i<NUMBER_OF_POINTS;i++){
            for (int j=0;j<DIMENSIONS;j++){
                mean_shift_vector[i][j] = DBL_MAX;
            }
        }

        // allocates memory for kernel_matrix
        kernel_matrix = alloc_double(NUMBER_OF_POINTS, NUMBER_OF_POINTS);

        // tic
        gettimeofday (&start, NULL);

        // allocates corresponding memory in device
        init_device_memory(original_points, *shifted_points, &d_original_points, &d_shifted_points,
            &d_kernel_matrix, &d_denominator, &d_mean_shift_vector);
        // toc
        gettimeofday (&end, NULL);
        seq = (double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec);

//        printf("%s wall clock time = %f\n","Device memory allocation", seq);
        // to create output data file
        printf("%f ", seq);
    }

    // finds pairwise distance matrix (inside radius)
    // [I, D] = rangesearch(x,y,h);
    calculate_kernel_matrix(d_shifted_points, d_original_points, d_kernel_matrix, deviation,
        &kernel_matrix);

    // calculates denominator
    calculate_denominator(d_kernel_matrix, d_denominator);

    // creates new y vector
    // allocates memory in every recursion
    new_shift = alloc_double(NUMBER_OF_POINTS, DIMENSIONS);
    // allocates corresponding memory in device
    d_new_shift.width = DIMENSIONS;
    d_new_shift.height = NUMBER_OF_POINTS;
    size = NUMBER_OF_POINTS * DIMENSIONS * sizeof(double);
    gpuErrchk( cudaMalloc(&(d_new_shift.elements), size) );

    shift_points(d_kernel_matrix, d_original_points, d_shifted_points, d_new_shift, d_denominator,
        d_mean_shift_vector, kernel_matrix, original_points, &new_shift, &mean_shift_vector);

    // frees previously shifted points, they're now garbage
    free((*shifted_points)[0]);
    // updates shifted points pointer to the new array address
    shifted_points = &new_shift;
    d_shifted_points.elements = d_new_shift.elements;

    if (params.display){
        save_matrix((*shifted_points), iteration);
    }

    // calculates norm of the new mean shift vector in GPU using "cuBlas" library function
    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS){
        exit(cublas_status);
    }
    cublas_status = cublasDnrm2(handle, NUMBER_OF_POINTS * DIMENSIONS, d_mean_shift_vector.elements,
        1, &current_norm);
    if (cublas_status != CUBLAS_STATUS_SUCCESS){
        exit(cublas_status);
    }
    cublas_status = cublasDestroy(handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS){
        exit(cublas_status);
    }

    if (params.verbose){
        printf("Iteration n. %d, error\t%f \n", iteration, current_norm);
    }

    // iterates until convergence
    if (current_norm > opt->epsilon) {
        ++iteration;
        meanshift(original_points, shifted_points, deviation, opt);
    }

    if (iteration == 0){
        // cleans up allocations
        free(mean_shift_vector[0]);
        free(mean_shift_vector);
        free(kernel_matrix[0]);
        free(kernel_matrix);

        free_device_memory(d_original_points, d_kernel_matrix, d_denominator, d_new_shift);
    }

    return iteration;
}

void init_device_memory(double **original_points, double **shifted_points,
    Matrix *d_original_points, Matrix *d_shifted_points, Matrix *d_kernel_matrix,
    Matrix *d_denominator, Matrix *d_mean_shift_vector){
    int size;

    // allocates memory for original_points in GPU and copies the array
    d_original_points->width = DIMENSIONS;
    d_original_points->height = NUMBER_OF_POINTS;
    size = NUMBER_OF_POINTS * DIMENSIONS * sizeof(double);
    gpuErrchk( cudaMalloc(&(d_original_points->elements), size) );
    gpuErrchk( cudaMemcpy(d_original_points->elements, &(original_points[0][0])
        , size, cudaMemcpyHostToDevice) );

    // allocates memory for shifted_points in GPU and copies the array
    d_shifted_points->width = DIMENSIONS;
    d_shifted_points->height = NUMBER_OF_POINTS;
    size = DIMENSIONS * NUMBER_OF_POINTS * sizeof(double);
    gpuErrchk( cudaMalloc(&(d_shifted_points->elements), size) );
    gpuErrchk( cudaMemcpy(d_shifted_points->elements, &(shifted_points[0][0])
        , size, cudaMemcpyHostToDevice) );

    // allocates memory for kernel_matrix in GPU
    d_kernel_matrix->width = NUMBER_OF_POINTS;
    d_kernel_matrix->height = NUMBER_OF_POINTS;
    size = NUMBER_OF_POINTS * NUMBER_OF_POINTS * sizeof(double);
    gpuErrchk( cudaMalloc(&(d_kernel_matrix->elements), size) );

    // allocates memory for denominator in GPU
    d_denominator->width = 1;
    d_denominator->height = NUMBER_OF_POINTS;
    size = NUMBER_OF_POINTS * sizeof(double);
    gpuErrchk( cudaMalloc(&(d_denominator->elements), size) );

    // allocates memory for mean_shift_vector in GPU
    d_mean_shift_vector->width = DIMENSIONS;
    d_mean_shift_vector->height = NUMBER_OF_POINTS;
    size = NUMBER_OF_POINTS * DIMENSIONS * sizeof(double);
    gpuErrchk( cudaMalloc(&(d_mean_shift_vector->elements), size) );
}

void calculate_kernel_matrix(Matrix d_shifted_points, Matrix d_original_points,
    Matrix d_kernel_matrix, double deviation, double ***kernel_matrix){
    int size;
    static bool first_iter = true;
    // gets max block size supported from the device
    static int max_block_size = device_properties.maxThreadsPerBlock;
    static int requested_block_size = (int)sqrt(max_block_size);
    bool block_size_too_big = true;

    dim3 dimBlock;
    dim3 dimGrid;
    do {
        dimBlock.x = requested_block_size;
        dimBlock.y = requested_block_size;
        dimGrid.x = (d_kernel_matrix.height + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = (d_kernel_matrix.width + dimBlock.y - 1) / dimBlock.y;

        calculate_kernel_matrix_kernel<<<dimGrid, dimBlock>>>(d_shifted_points, d_original_points
            , deviation, d_kernel_matrix);
        if (cudaGetLastError() != cudaSuccess){
            --requested_block_size;
        } else {
            block_size_too_big = false;
            gpuErrchk( cudaDeviceSynchronize() );
        }
    } while(block_size_too_big);
    
    if (first_iter && params.verbose){
        printf("calculate_kernel_matrix_kernel called with:\n");
        printf("dimBlock.x = %d, dimBlock.y = %d\n", dimBlock.x, dimBlock.y);
        printf("dimGrid.x = %d, dimGrid.y = %d\n\n", dimGrid.x, dimGrid.y);
        first_iter = false;
    }

    size = NUMBER_OF_POINTS * NUMBER_OF_POINTS * sizeof(double);

    // tic
    gettimeofday (&start, NULL);
    gpuErrchk( cudaMemcpy(&((*kernel_matrix)[0][0]), d_kernel_matrix.elements
        , size, cudaMemcpyDeviceToHost) );

    // toc
    gettimeofday (&end, NULL);
    seq = (double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec);

//    printf("%s wall clock time = %f\n","Copying from device to host", seq);
    // to create output data file
        printf("%f ", seq);
}

void calculate_denominator(Matrix d_kernel_matrix, Matrix d_denominator){
    static bool first_iter = true;
    // gets max block size supported from the device
    static int requested_block_size = device_properties.maxThreadsPerBlock;
    bool block_size_too_big = true;

    dim3 dimBlock;
    dim3 dimGrid;
    do {
        dimBlock.x = requested_block_size;
        dimBlock.y = 1;
        dimGrid.x = (d_kernel_matrix.height + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = 1;

        denominator_kernel<<<dimGrid, dimBlock>>>(d_denominator, d_kernel_matrix);
        if (cudaGetLastError() != cudaSuccess){
            --requested_block_size;
        } else {
            block_size_too_big = false;
            gpuErrchk( cudaDeviceSynchronize() );
        }
    } while(block_size_too_big);
    
    if (first_iter && params.verbose){
        printf("calculate_denominator called with:\n");
        printf("dimBlock.x = %d, dimBlock.y = %d\n", dimBlock.x, dimBlock.y);
        printf("dimGrid.x = %d, dimGrid.y = %d\n\n", dimGrid.x, dimGrid.y);
        first_iter = false;
    }

}

void shift_points(Matrix d_kernel_matrix, Matrix d_original_points, Matrix d_shifted_points,
    Matrix d_new_shift, Matrix d_denominator, Matrix d_mean_shift_vector, double **kernel_matrix,
    double **original_points, double ***new_shift, double ***mean_shift_vector){
    int size;
    static bool first_iter = true;
    // gets max block size supported from the device
    static int max_block_size = device_properties.maxThreadsPerBlock;
    static int requested_block_size = (int)(max_block_size / d_new_shift.width);
    bool block_size_too_big = true;

    dim3 dimBlock;
    dim3 dimGrid;
    do {
        dimBlock.x = requested_block_size;
        dimBlock.y = d_new_shift.width;
        dimGrid.x = (d_denominator.height + dimBlock.x - 1) / dimBlock.x;
        dimGrid.y = 1;

        shift_points_kernel<<<dimGrid, dimBlock>>>(d_original_points, d_kernel_matrix, d_shifted_points,
            d_new_shift, d_denominator, d_mean_shift_vector);
        if (cudaGetLastError() != cudaSuccess){
            --requested_block_size;
        } else {
            block_size_too_big = false;
            gpuErrchk( cudaDeviceSynchronize() );
        }
    } while(block_size_too_big);

    if (first_iter && params.verbose){
        printf("shift_points_kernel called with:\n");
        printf("dimBlock.x = %d, dimBlock.y = %d\n", dimBlock.x, dimBlock.y);
        printf("dimGrid.x = %d, dimGrid.y = %d\n\n", dimGrid.x, dimGrid.y);
        first_iter = false;
    }

    size = NUMBER_OF_POINTS * DIMENSIONS * sizeof(double);

    // tic
    gettimeofday (&start, NULL);

    gpuErrchk( cudaMemcpy(&((*new_shift)[0][0]), d_new_shift.elements
        , size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&((*mean_shift_vector)[0][0]), d_mean_shift_vector.elements
        , size, cudaMemcpyDeviceToHost) );

    // toc
    gettimeofday (&end, NULL);
    seq = (double)((end.tv_usec - start.tv_usec)/1.0e6 + end.tv_sec - start.tv_sec);

//    printf("%s wall clock time = %f\n","Copying from device to host", seq);
    // to create output data file
        printf("%f ", seq);

}

void free_device_memory(Matrix d_original_points, Matrix d_kernel_matrix, Matrix d_denominator,
    Matrix d_new_shift){
    // frees all memory previously allocated in device
    gpuErrchk( cudaFree(d_original_points.elements) );
    gpuErrchk( cudaFree(d_kernel_matrix.elements) );
    gpuErrchk( cudaFree(d_denominator.elements) );
    gpuErrchk( cudaFree(d_new_shift.elements) );
}