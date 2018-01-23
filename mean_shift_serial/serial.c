#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "serial_declarations.h"

int DEVIATION = 1;
int NUMBER_OF_POINTS = 600;
int DIMENSIONS = 2;
char* POINTS_FILENAME = "../data/X.bin";
char* LABELS_FILENAME = "../data/L.bin";

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char **argv){
    double **vectors, **shifted_points;
    char *labels;
    parameters params;

    //get_args(argc, argv); commented out while in development
    init(&vectors, &labels, &params);

    //save_matrix(vectors, 0);

    // tic
    gettimeofday (&startwtime, NULL);

    int iterations = meanshift(vectors, &shifted_points, DEVIATION, &params);
    printf("Total iterations = %d\n", iterations);

    // toc
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("%s wall clock time = %f\n","Mean Shift", seq_time);

    //TODO write output points to file -> plot later
    //save_matrix(shifted_points, iterations);
}