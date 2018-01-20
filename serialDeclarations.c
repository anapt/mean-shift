#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "serialDeclarations.h"

void meanshift(double **originalPoints, int h, parameters *opt){

    double **y;
    y = alloc_2d_double(ROWS, COLUMNS);
    y = duplicate(originalPoints, y, ROWS, COLUMNS);

    // mean shift vector
    double **meanShiftVector;
    meanShiftVector = alloc_2d_double(ROWS, COLUMNS);
    // initialize elements of meanShiftVector to inf
    for (int i=0;i<ROWS;i++){
        for (int j=0;j<COLUMNS;j++){
            meanShiftVector[i][j] = DBL_MAX;
        }
    }

    // initialize iteration counter
    int iter = 0;

    // printf("%f \n", opt->epsilon);

    double ** kernelMatrix = alloc_2d_double(ROWS, ROWS);
    double *denominator = malloc(ROWS * sizeof(double));

    /** iterate until convergence **/
    // printf("norm : %f \n", norm(m, ROWS, COLUMNS));
    while (norm(meanShiftVector, ROWS, COLUMNS) > opt->epsilon) {
        iter = iter +1;
        // find pairwise distance matrix (inside radius)
        // [I, D] = rangesearch(x,y,h);
        for (int i=0; i<ROWS; i++){
            double sum =0;
            for (int j=0; j<ROWS; j++){
                double dist = calculateDistance(y[i],originalPoints[j]);

                if (i == j){
                    kernelMatrix[i][j] = 1;
                } else if (dist < h*h){
                    kernelMatrix[i][j] = dist * dist;
                    // compute kernel matrix
                    double pow = ((-1)*(kernelMatrix[i][j]))/(2*(h*h));
                    kernelMatrix[i][j] = exp(pow);
                } else {
                    kernelMatrix[i][j] = 0;
                }
                sum = sum + kernelMatrix[i][j];
            }
            denominator[i] = sum;
        }

        // create new y vector
        double** y_new = alloc_2d_double(ROWS, COLUMNS);

        multiply(kernelMatrix, originalPoints, y_new);
        // divide element-wise
        for (int i=0; i<ROWS; i++){
            for (int j=0; j<COLUMNS; j++){
                y_new[i][j] = y_new[i][j] / denominator[i];
                // calculate mean-shift vector
                meanShiftVector[i][j] = y_new[i][j] - y[i][j];
                // update y
                y[i][j] = y_new[i][j];
            }
        }

        save_matrix(y, iter);

        printf("Iteration n. %d, error %f \n", iter, norm(meanShiftVector, ROWS, COLUMNS));
        // TODO maybe keep y for live display later?
    }
}

// TODO check why there's is a difference in the norm calculate in matlab
double norm(double **matrix, int rows, int cols){
    double sum=0, tempMul=0;
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            tempMul = matrix[i][j] * matrix[i][j];
            sum = sum + tempMul;
        }
    }
    double norm = sqrt(sum);
    return norm;
}

void multiply(double **matrix1, double **matrix2, double **output){
    // W dims are ROWS ROWS and x dims are ROWS COLUMNS

    for (int i=0; i<ROWS; i++){
        for (int j=0; j<COLUMNS; j++){
            output[i][j] = 0;
            for (int k=0; k<ROWS; k++){
                output[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

double calculateDistance(double *y, double *x){
    double sum = 0, dif;
    for (int i=0; i<COLUMNS; i++){
        dif = y[i]-x[i];
        sum += dif * dif;
    }
    double distance = sqrt(sum);
    return distance;
}

// allocates a 2d array in continuous memory positions
double **alloc_2d_double(int rows, int cols) {
    double *data = (double *) malloc(rows*cols*sizeof(double));
    double **array = (double **) malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);
    return array;
}

// copy the values of a 2d double array to another
double **duplicate(double **a, double **b, int rows, int cols){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            b[i][j] = a[i][j];
        }
    }
    return b;
}

void print_matrix(double **array, int rows, int cols){
    for (int i=0; i<cols; i++){
        for (int j=0; j<rows; j++){
            printf("%f ", array[j][i]);
        }
        printf("\n");
    }
}

void save_matrix(double **matrix,int iteration){
    char filename[18];
    snprintf(filename, sizeof(filename), "%s%d", "output/output_", iteration);
    FILE *iterOutput;
    iterOutput = fopen(filename, "w");
    for (int rows=0; rows<ROWS; ++rows){
        for (int cols=0; cols<COLUMNS; ++cols){
            fprintf(iterOutput, "%f", matrix[rows][cols]);
            if (cols != COLUMNS - 1){
                fprintf(iterOutput, ",");
            }
        }
        fprintf(iterOutput, "\n");
    }
}