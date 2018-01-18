//
// Created by anapt on 17/1/2018.
//

#include<stdio.h>
#include<math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <stdbool.h>
#define X "data/X.bin"
#define Y "data/X.bin"
#define COLUMNS     2
#define ROWS        6
#define N           4


// allocates a 2d array in continuous memory positions
double **alloc_2d_double(int rows, int cols) {
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);
    return array;
}

// copy the values of a 2d double array to another
double **duplicate(double **a, double **b, int rows, int cols){
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            b[i][j] = a[i][j];
        }
    }
    return b;
}

// TODO check why there's is a difference in the norm calculate in matlab

double norm(double ** m, int rows, int cols){
    double sum=0, a=0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a = m[i][j] * m[i][j];
            sum = sum + a;
        }
    }
    double norm = sqrt(sum);
    return norm;
}

double norm2(double ** m, int rows, int cols){
    double sum=0, a=0;
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            a = m[i][j] * m[i][j];
            sum = sum + a;
        }
    }
    double norm = sqrt(sum);
    return norm;
}

void multiply(double ** matrix1, double ** matrix2, double ** output){
    // W dims are ROWS ROWS and x dims are ROWS COLUMNS
    // TODO IMPLEMENT
    int i, j, k;
    for (i=0; i<ROWS; i++){
        for (j=0; j<COLUMNS; j++){
            output[i][j] = 0;
            for (k=0; k<ROWS; k++){
                output[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

void print_matrix(double ** array, int rows, int cols){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            printf("%f ", array[i][j]);
        }
        printf("\n");
    }
}

int main()
{

    double ** vector1, **vector2, **res;
//    printf("norm : %f \n", norm(vectors, ROWS, COLUMNS));
//    printf("norm : %f \n", norm2(vectors, ROWS, COLUMNS));

    FILE *f;
    f = fopen(X, "rb");
    vector1 = alloc_2d_double(ROWS, ROWS);
    for (int i=0; i<ROWS; i++){
        int out = fread(vector1[i], sizeof(double), ROWS, f);
    }
    fclose(f);
    FILE *f2;
    f2 = fopen(Y, "rb");
    vector2 = alloc_2d_double(ROWS, COLUMNS);
    for (int i=0; i<ROWS; i++){
        int out = fread(vector2[i], sizeof(double), COLUMNS, f2);
    }
    fclose(f2);
    res = alloc_2d_double(ROWS, COLUMNS);
    multiply(vector1, vector2, res);
    print_matrix(vector1, ROWS, ROWS);
    print_matrix(vector2, ROWS, COLUMNS);
    print_matrix(res, ROWS, COLUMNS);
    return 0;
}

