#ifndef SERIAL_UTILS_H    /*    Include guard    */
#define SERIAL_UTILS_H

#include <stdbool.h>

/*      Structures     */

//Parameters is used to store session specific variables in an orderly way
typedef struct parameters {
    double epsilon;
    bool verbose;
    bool display;
} Parameters;

//Function get_args parses command line arguments
void get_args(int argc, char **argv, Parameters *params);

//Function init sets up the GPU for later use, gets its properties and reads the dataset and label
//arrays from the corresponding files
void init(double ***vectors, char **labels);

//Function alloc_double allocates rows*cols bytes of continuous memory
double **alloc_double(int rows, int cols);

//Function duplicate copies the values of source array to dest array
void duplicate(double **source, int rows, int cols, double ***dest);

//Function print_matrix prints array of dimensions <rows X cols> to the console
void print_matrix(double **array, int rows, int cols);

//Function save_matrix stores matrix in a csv file with path/filename "../output/output_iteration".
//If a file already exists new lines are concatenated
void save_matrix(double **matrix, int iteration);

#endif //SERIAL_UTILS_H