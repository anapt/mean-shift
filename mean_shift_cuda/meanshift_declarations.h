#ifndef SERIAL_DECLARATIONS_H    /*    Include guard    */
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

//Function get_args parses command line arguments.
void get_args(int argc, char **argv, int *h);

//Function meanshift recursively shifts original points according to th
//mean-shift algorithm saving the result to shiftedPoints. Struct opt has user
//options, h is the desirable deviation, iteration is this call's iteration
//number.
int meanshift(double **original_points, double ***shifted_points, int h
        , parameters *opt, int iteration);

//Function norm returns the second norm of matrix of dimensions rowsXcols.
//double norm(double **matrix, int rows, int cols);

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

#endif //SERIAL_DECLARATIONS_H