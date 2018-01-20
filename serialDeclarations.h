#ifndef SERIAL_DECLARATIONS_H	/*	Include guard	*/
#define SERIAL_DECLARATIONS_H

#include <stdbool.h>

#define X "data/X.bin"
#define L "data/L.bin"
#define COLUMNS     2
#define ROWS        600

typedef struct parameters {
    double epsilon;
    bool verbose;
    bool display;
} parameters;

void meanshift(double **x, int h, struct parameters *opt);
double norm(double ** m, int rows, int cols);
void multiply(double ** matrix1, double ** matrix2, double ** output);
double calculateDistance(double *, double *);
double **alloc_2d_double(int rows, int cols);
double **duplicate(double **a, double **b, int rows, int cols);
void print_matrix(double ** array, int rows, int cols);
void save_matrix(double **matrix,int iteration);

#endif //SERIAL_DECLARATIONS_H