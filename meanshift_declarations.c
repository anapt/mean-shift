#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "meanshift_declarations.h"

void get_args(int argc, char **argv, int *h){
    if (argc != 6) {
        printf("Usage: %s h N D Pd Pl\nwhere:\n", argv[0]);
        printf("\th is the variance\n");
        printf("\tN is the the number of points\n");
        printf("\tD is the number of dimensions of each point\n");
        printf("\tPd is the path of the dataset file\n");
        printf("\tPl is the path of the labels file\n");
        exit(1);
    }

    *h = atoi(argv[1]);
    NUMBER_OF_POINTS = atoi(argv[2]);
    DIMENSIONS = atoi(argv[3]);
    POINTS_FILENAME = argv[4];
    LABELS_FILENAME = argv[5];
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

    double **kernel_matrix = alloc_2d_double(NUMBER_OF_POINTS, NUMBER_OF_POINTS);
    double *denominator = malloc(NUMBER_OF_POINTS * sizeof(double));
    // create new y vector
    double **new_shift = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);

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
    char filename[18];
    snprintf(filename, sizeof(filename), "%s%d", "output/output_", iteration);
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