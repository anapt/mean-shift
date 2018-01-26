#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "meanshift_utils.h"

#define OUTPUT_PREFIX "../output/output_"

cudaDeviceProp device_properties;

void get_args(int argc, char **argv, parameters *params){
    if (argc < 7) {
        printf("Usage: %s h e N D Pd Pl\nwhere:\n"
        "\th is the variance\n"
        "\te is the min distance, between two points, that is taken into account in computations\n"
        "\tN is the the number of points\n"
        "\tD is the number of dimensions of each point\n"
        "\tPd is the path of the dataset file\n"
        "\tPl is the path of the labels file\n"
        "\n\t--verbose | -v is an optional flag to enable execution information output"
        "\n\t--output | -o is an optional flag to enable points output in each iteration", argv[0]);
        exit(1);
    }

    DEVIATION = atoi(argv[1]);
    params->epsilon = atof(argv[2]);
    NUMBER_OF_POINTS = atoi(argv[3]);
    DIMENSIONS = atoi(argv[4]);
    POINTS_FILENAME = argv[5];
    LABELS_FILENAME = argv[6];
    params->verbose = false;
    params->display = false;

    if (argc > 7){
        for (int index=7; index<argc; ++index){
            if (!strcmp(argv[index], "--verbose") || !strcmp(argv[index], "-v")){
                params->verbose = true;
            } else if (!strcmp(argv[index], "--output") || !strcmp(argv[index], "-o")){
                params->display = true;
            } else {
                printf("Couldn't parse argument %d: %s\n", index, argv[index]);
                exit(EXIT_FAILURE);
            }
        }
    }

    /*printf("DEVIATION = %d\n"
        "epsilon = %f\n"
        "NUMBER_OF_POINTS = %d\n"
        "DIMENSIONS = %d\n"
        "POINTS_FILENAME = %s\n"
        "LABELS_FILENAME = %s\n"
        "verbose = %d\n"
        "display = %d\n", DEVIATION, params->epsilon, NUMBER_OF_POINTS, DIMENSIONS, POINTS_FILENAME
            , LABELS_FILENAME, params->verbose, params->display);*/
}

void init(double ***vectors, char **labels){
    int bytes_read = 0;

    set_GPU();

    if (params.verbose){
        printf("Reading dataset and labels...\n");
    }

    // initializes vectors
    FILE *points_file;
    points_file = fopen(POINTS_FILENAME, "rb");
    if (points_file != NULL){
        // allocates memory for the array
        (*vectors) = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
        // reads vectors dataset from file
        for (int i=0; i<NUMBER_OF_POINTS; i++){
            bytes_read = fread((*vectors)[i], sizeof(double), DIMENSIONS, points_file);
            if ( bytes_read != DIMENSIONS ){
                if(feof(points_file)){
                    printf("Premature end of file reached.\n");
                } else{
                    printf("Error reading points file.");
                }
                fclose(points_file);
                exit(EXIT_FAILURE);
            }
        }
    } else {
        printf("Error reading dataset file.\n");
        exit(EXIT_FAILURE);
    }
    fclose(points_file);

    // initializes file that will contain the labels (train)
    FILE *labels_file;
    labels_file = fopen(LABELS_FILENAME, "rb");
    if (labels_file != NULL){
        // NOTE : Labels were classified as <class 'numpy.uint8'>
        // variables of type uint8 are stored as 1-byte (8-bit) unsigned integers
        // gets number of labels
        fseek(labels_file, 0L, SEEK_END);
        long int pos = ftell(labels_file);
        rewind(labels_file);
        int label_elements = pos/ sizeof(char);

        // allocates memory for the array
        *labels = (char*)malloc(label_elements* sizeof(char));
        fseek(labels_file, 0L, SEEK_SET);
        bytes_read = fread((*labels), sizeof(char), label_elements, labels_file);
        if ( bytes_read != label_elements ){
            if(feof(points_file)){
                printf("Premature end of file reached.\n");
            } else{
                printf("Error reading points file.");
            }
            fclose(labels_file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(labels_file);

    if (params.verbose){
        printf("Done.\n\n");
    }
}

//Based on https://stackoverflow.com/a/28113186
//Poio psagmeno link https://www.cs.virginia.edu/~csadmin/wiki/index.php/CUDA_Support/Choosing_a_GPU
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
    static double **kernel_matrix, *denominator, **mean_shift_vector;
    double **new_shift;

    // device variables
    static Matrix d_original_points, d_shifted_points, d_kernel_matrix, d_denominator,
        d_mean_shift_vector;
    Matrix d_new_shift;

    // allocates memory and copies original points on first iteration
    if (iteration == 0 || (*shifted_points) == NULL){
        // allocates memory for shifted points array and copies original points into it
        (*shifted_points) = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
        duplicate(original_points, NUMBER_OF_POINTS, DIMENSIONS, shifted_points);

        // allocates memory for mean shift vector
        mean_shift_vector = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
        // initializes elements of mean_shift_vector to inf
        for (int i=0;i<NUMBER_OF_POINTS;i++){
            for (int j=0;j<DIMENSIONS;j++){
                mean_shift_vector[i][j] = DBL_MAX;
            }
        }

        // allocates memory for other arrays needed
        kernel_matrix = alloc_2d_double(NUMBER_OF_POINTS, NUMBER_OF_POINTS);
        denominator = (double *)malloc(NUMBER_OF_POINTS * sizeof(double));

        // allocates corresponding memory in device
        init_device_memory(original_points, *shifted_points, &d_original_points, &d_shifted_points,
            &d_kernel_matrix, &d_denominator, &d_mean_shift_vector);
    }

    // finds pairwise distance matrix (inside radius)
    // [I, D] = rangesearch(x,y,h);
    calculate_kernel_matrix(d_shifted_points, d_original_points, d_kernel_matrix, deviation,
        &kernel_matrix);

    // calculates denominator
    calculate_denominator(d_kernel_matrix, d_denominator, &denominator);

    size = NUMBER_OF_POINTS * sizeof(double);
    gpuErrchk( cudaMemcpy(d_denominator.elements, &(denominator[0])
        , size, cudaMemcpyHostToDevice) );

    // creates new y vector
    // allocates memory in every recursion
    new_shift = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
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

    // calculates norm of the new mean shift vector
    double current_norm = norm(mean_shift_vector, NUMBER_OF_POINTS, DIMENSIONS);
    if (params.verbose){
        printf("Iteration n. %d, error %f \n", iteration, current_norm);
    }

    /** iterates until convergence **/
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
        free(denominator);

        free_device_memory(d_original_points, d_kernel_matrix, d_denominator, d_new_shift);
    }

    return iteration;
}

// TODO check why there's is a difference in the norm calculate in matlab
double norm(double **matrix, int rows, int cols){
    double sum=0, temp_mul=0;
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            temp_mul = matrix[i][j] * matrix[i][j];
            sum = sum + temp_mul;
        }
    }
    double norm = sqrt(sum);
    return norm;
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
    gpuErrchk( cudaMemcpy(&((*kernel_matrix)[0][0]), d_kernel_matrix.elements
        , size, cudaMemcpyDeviceToHost) );
}

void calculate_denominator(Matrix d_kernel_matrix, Matrix d_denominator, double **denominator){
    int size;
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

    size = NUMBER_OF_POINTS * sizeof(double);
    gpuErrchk( cudaMemcpy(&((*denominator)[0]), d_denominator.elements
    	, size, cudaMemcpyDeviceToHost) );
}

void shift_points(Matrix d_kernel_matrix, Matrix d_original_points, Matrix d_shifted_points,
    Matrix d_new_shift, Matrix d_denominator, Matrix d_mean_shift_vector, double **kernel_matrix,
    double **original_points, double ***new_shift, double ***mean_shift_vector){
    int size;
    static bool first_iter = true;
    // gets max block size supported from the device
    static int max_block_size = device_properties.maxThreadsPerBlock;
    static int requested_block_size = (int)(max_block_size / 2);
    bool block_size_too_big = true;

    dim3 dimBlock;
    dim3 dimGrid;
    do {
        dimBlock.x = requested_block_size;
        dimBlock.y = 2;
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
    gpuErrchk( cudaMemcpy(&((*new_shift)[0][0]), d_new_shift.elements
        , size, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy(&((*mean_shift_vector)[0][0]), d_mean_shift_vector.elements
        , size, cudaMemcpyDeviceToHost) );
}

void free_device_memory(Matrix d_original_points, Matrix d_kernel_matrix, Matrix d_denominator,
    Matrix d_new_shift){
    // frees all memory previously allocated in device
    gpuErrchk( cudaFree(d_original_points.elements) );
    gpuErrchk( cudaFree(d_kernel_matrix.elements) );
    //gpuErrchk( cudaFree(d_shifted_points.elements) );
    gpuErrchk( cudaFree(d_denominator.elements) );
    gpuErrchk( cudaFree(d_new_shift.elements) );
}

double calculateDistance(double *y, double *x){
    double sum = 0, dif;
    for (int i=0; i<DIMENSIONS; i++){
        dif = y[i]-x[i];
        sum += dif * dif;
    }
    double distance = sqrt(sum);
    return distance;
}

double **alloc_2d_double(int rows, int cols) {
    double *data = (double *) malloc(rows*cols*sizeof(double));
    double **array = (double **) malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);
    return array;
}

void duplicate(double **source, int rows, int cols, double ***dest){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            (*dest)[i][j] = source[i][j];
        }
    }
}

void print_matrix(double **array, int rows, int cols){
    for (int i=0; i<cols; i++){
        for (int j=0; j<rows; j++){
            printf("%f ", array[j][i]);
        }
        printf("\n");
    }
}

void save_matrix(double **matrix, int iteration){
    char filename[50];
    snprintf(filename, sizeof(filename), "%s%d", "../output/output_", iteration);
    FILE *file;
    file = fopen(filename, "w");
    for (int rows=0; rows<NUMBER_OF_POINTS; ++rows){
        for (int cols=0; cols<DIMENSIONS; ++cols){
            fprintf(file, "%f", matrix[rows][cols]);
            if (cols != DIMENSIONS - 1){
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
}