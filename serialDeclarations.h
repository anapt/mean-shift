#ifndef SERIAL_DECLARATIONS_H	/*	Include guard	*/
#define SERIAL_DECLARATIONS_H

#include <stdbool.h>

extern int NUMBER_OF_POINTS;
extern int DIMENSIONS;
extern char* POINTS_FILENAME;
extern char* LABELS_FILENAME;

typedef struct parameters {
    double epsilon;
    bool verbose;
    bool display;
} parameters;

void get_args(int argc, char **argv, int *h);
int meanshift(double **originalPoints, double ***shiftedPoints, int h
    , parameters *opt, int iteration);
double norm(double ** m, int rows, int cols);
void multiply(double ** matrix1, double ** matrix2, double ** output);
double calculateDistance(double *, double *);
double **alloc_2d_double(int rows, int cols);
double **duplicate(double **a, double **b, int rows, int cols);
void print_matrix(double ** array, int rows, int cols);
void save_matrix(double **matrix,int iteration);

#endif //SERIAL_DECLARATIONS_H