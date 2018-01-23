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