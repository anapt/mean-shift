#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "meanshift_utils.h"
#include "meanshift_gpu_utils.h"

int DEVIATION = 31000;
int NUMBER_OF_POINTS = 5000;
int DIMENSIONS = 2;
const char* POINTS_FILENAME = "../data/s4";
const char *LABELS_FILENAME = "../data/L.bin";
parameters params;

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char **argv){
    int recursions = 0;
    double **vectors, **shifted_points;
    char *labels;

    params.epsilon = 0.0001;
    params.verbose = true;
    params.display = true;
    //get_args(argc, argv, &params); //commented out while in development
    init(&vectors, &labels);

    // tic
    gettimeofday (&startwtime, NULL);
    recursions = meanshift(vectors, &shifted_points, DEVIATION);

    // toc
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    

    printf("\nTotal number of recursions = %d\n", recursions);
    printf("%s wall clock time = %f\n","Mean Shift", seq_time);

    free(vectors[0]);
    free(vectors);
    free(shifted_points[0]);
    free(shifted_points);
}
