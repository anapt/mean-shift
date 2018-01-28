#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "meanshift_utils.h"
#include "meanshift_gpu_utils.h"

int DEVIATION = 1;
int NUMBER_OF_POINTS = 600;
int DIMENSIONS = 2;
const char *POINTS_FILENAME = "../data/X.bin";
const char *LABELS_FILENAME = "../data/L.bin";
parameters params;

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char **argv){
    int iterations = 0;
    double **vectors, **shifted_points;
    char *labels;

    params.epsilon = 0.0001;
    params.verbose = true;
    params.display = true;
    //get_args(argc, argv, &params); //commented out while in development
    init(&vectors, &labels);

    //save_matrix(vectors, 0);

    // tic
    gettimeofday (&startwtime, NULL);
    iterations = meanshift(vectors, &shifted_points, DEVIATION, &params);

    // toc
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    

    printf("\nTotal number of iterations = %d\n", iterations);
    printf("%s wall clock time = %f\n","Mean Shift", seq_time);
//    printf("%f \n", seq_time);

    //TODO write output points to file -> plot later
    //save_matrix(shifted_points, iterations);
}
