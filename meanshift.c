#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "serial_declarations.h"

int NUMBER_OF_POINTS = 600;
int DIMENSIONS = 2;
char* POINTS_FILENAME = "data/X.bin";
char* LABELS_FILENAME = "data/L.bin";

struct timeval startwtime, endwtime;
double seq_time;

int meanshift(double **original_points, double ***shifted_points, int h
        , parameters *opt, int iteration);

int main(int argc, char **argv){
    int h = 1;

    //get_args(argc, argv, &h); commented out while in development

    FILE *f;
//    f = fopen(X, "rb");
//    fseek(f, 0L, SEEK_END);
//    long int pos = ftell(f);
//    fclose(f);
//    int elements = pos / sizeof(double);  // number of total elements (points*dimension)
//    int points = elements/DIMENSIONS;
//    //printf("points : %d \n", points);
    f = fopen(POINTS_FILENAME, "rb");
    double **vectors;
    vectors = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
    for (int i=0; i<NUMBER_OF_POINTS; i++){
        int out = fread(vectors[i], sizeof(double), DIMENSIONS, f);
    }

    save_matrix(vectors, 0);

    // initializing file that will contain the labels (train)
    f = fopen(LABELS_FILENAME, "rb");
    // NOTE : Labels were classified as <class 'numpy.uint8'>
    // variables of type uint8 are stored as 1-byte (8-bit) unsigned integers
    fseek(f, 0L, SEEK_END);
    long int pos = ftell(f);
    rewind(f);
    //printf("position : %ld \n", pos);
    int label_elements = pos/ sizeof(char);
    char *labels = (char*)malloc(label_elements* sizeof(char));
    fseek(f, 0L, SEEK_SET);
    int out = fread(labels, sizeof(char), label_elements, f);
    fclose(f);

    // MEAN SHIFT OPTIONS
    parameters params;
    params.epsilon = 0.0001;
    params.verbose = false;
    params.display = false;
    parameters *opt;
    opt = &params;

    double **shifted_points;
    // tic
    gettimeofday (&startwtime, NULL);

    int iterations = meanshift(vectors, &shifted_points, h, opt, 1);

    // toc
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("%s wall clock time = %f\n","Mean Shift", seq_time);

    //TODO write output points to file -> plot later
    //save_matrix(shifted_points, iterations);
}

int meanshift(double **original_points, double ***shifted_points, int h
        , parameters *opt, int iteration){

    // allocates space and copies original points on first iteration
    if (iteration == 1){
        (*shifted_points) = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
        duplicate(original_points, NUMBER_OF_POINTS, DIMENSIONS, shifted_points);
    }

    // mean shift vector
    double **mean_shift_vector;
    mean_shift_vector = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
    // initialize elements of mean_shift_vector to inf
    for (int i=0;i<NUMBER_OF_POINTS;i++){
        for (int j=0;j<DIMENSIONS;j++){
            mean_shift_vector[i][j] = DBL_MAX;
        }
    }


    /** allocate memory **/
    double **kernel_matrix = alloc_2d_double(NUMBER_OF_POINTS, NUMBER_OF_POINTS);
    double *denominator = malloc(NUMBER_OF_POINTS * sizeof(double));
    // create new y vector
    double **new_shift = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);


    double * d_kernel_matrix;
    cudaMalloc(&d_kernel_matrix, NUMBER_OF_POINTS * NUMBER_OF_POINTS * sizeof(double));
    double * d_denominator;
    cudaMalloc(&d_denominator, NUMBER_OF_POINTS * sizeof(double));
    double * d_new_shift;
    cudaMalloc(&d_new_shift, NUMBER_OF_POINTS * DIMENSIONS * sizeof(double));

//    (*shifted_points) = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
//    duplicate(original_points, NUMBER_OF_POINTS, DIMENSIONS, shifted_points);
    double * d_shifted_points;
    cudaMalloc(&d_shifted_points, NUMBER_OF_POINTS * DIMENSIONS * sizeof(double));

//    double **mean_shift_vector;
//    mean_shift_vector = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
    double * d_mean_shift_vector;
    cudaMalloc(&d_mean_shift_vector, NUMBER_OF_POINTS * DIMENSIONS * sizeof(double));

    //Copy vectors from host memory to device memory
    cudaMemcpy(d_shifted_points, *shifted_points, NUMBER_OF_POINTS * DIMENSIONS * sizeof(double),
               cudaMemcpyHostToDevice);
    // y[i][j] == d_y[COLUMNS*i + j]
    cudaMemcpy(d_mean_shift_vector, mean_shift_vector, NUMBER_OF_POINTS * DIMENSIONS * sizeof(double),
               cudaMemcpyHostToDevice);


    // TODO REFACTOR AS A KERNEL
    // find pairwise distance matrix (inside radius)
    // [I, D] = rangesearch(x,y,h);
    for (int i=0; i<NUMBER_OF_POINTS; i++){
        double sum = 0;
        for (int j=0; j<NUMBER_OF_POINTS; j++){
            double dist_sum = 0;
            for (int p=0; p<DIMENSIONS; p++){
                double dif = ((*shifted_points)[i])[p]-(original_points[j])[p];
                dist_sum += dif * dif;
            }
            double dist = sqrt(dist_sum);

            if (dist < h*h){
                kernel_matrix[i][j] = dist * dist;
                // compute kernel matrix
                double pow = ((-1)*(kernel_matrix[i][j]))/(2*(h*h));
                kernel_matrix[i][j] = exp(pow);
            } else {
                kernel_matrix[i][j] = 0;
            }
            if (i==j){
                kernel_matrix[i][j] += 1;
            }
            sum = sum + kernel_matrix[i][j];
        }
        denominator[i] = sum;

        // build nominator
        for (int j=0; j<DIMENSIONS; j++){
            new_shift[i][j] = 0;
            for (int k=0; k<NUMBER_OF_POINTS; k++){
                new_shift[i][j] += kernel_matrix[i][k] * original_points[k][j];
            }
            // divide element-wise
            new_shift[i][j] = new_shift[i][j] / denominator[i];
            // calculate mean-shift vector at the same time
            mean_shift_vector[i][j] = new_shift[i][j] - (*shifted_points)[i][j];
        }
    }


    // frees previously shifted points, they're now garbage
    free((*shifted_points)[0]);
    // updates shifted points pointer to the new array address
    shifted_points = &new_shift;

    save_matrix((*shifted_points), iteration);

    double current_norm = norm(mean_shift_vector, NUMBER_OF_POINTS, DIMENSIONS);
    printf("Iteration n. %d, error %f \n", iteration, current_norm);

    // clean up this iteration's allocates
    free(mean_shift_vector[0]);
    free(mean_shift_vector);
    free(kernel_matrix[0]);
    free(kernel_matrix);
    free(denominator);

    /** iterate until convergence **/
    if (current_norm > opt->epsilon) {
        return meanshift(original_points, shifted_points, h, opt, ++iteration);
    }

    return iteration;
}