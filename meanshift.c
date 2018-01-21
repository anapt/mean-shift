#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "serial_declarations.h"

int NUMBER_OF_POINTS = 600;
int DIMENSIONS = 2;
char* POINTS_FILENAME = "data/X.bin";
char* LABELS_FILENAME = "data/L.bin";

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char **argv){
    int h = 1;

    //get_args(argc, argv, &h); commented out while in development

    FILE *f;
//    f = fopen(X, "rb");
//    fseek(f, 0L, SEEK_END);
//    long int pos = ftell(f);
//    fclose(f);
//    int elements = pos / sizeof(double);  // number of total elements (points*dimension)
//    int points = elements/DIMENSIONS;
//    //printf("points : %d \n", points);
    f = fopen(POINTS_FILENAME, "rb");
    double **vectors;
    vectors = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
    for (int i=0; i<NUMBER_OF_POINTS; i++){
        int out = fread(vectors[i], sizeof(double), DIMENSIONS, f);
    }

    save_matrix(vectors, 0);

    // initializing file that will contain the labels (train)
    f = fopen(LABELS_FILENAME, "rb");
    // NOTE : Labels were classified as <class 'numpy.uint8'>
    // variables of type uint8 are stored as 1-byte (8-bit) unsigned integers
    fseek(f, 0L, SEEK_END);
    long int pos = ftell(f);
    rewind(f);
    //printf("position : %ld \n", pos);
    int label_elements = pos/ sizeof(char);
    char *labels = (char*)malloc(label_elements* sizeof(char));
    fseek(f, 0L, SEEK_SET);
    int out = fread(labels, sizeof(char), label_elements, f);
    fclose(f);

    // MEAN SHIFT OPTIONS
    parameters params;
    params.epsilon = 0.0001;
    params.verbose = false;
    params.display = false;
    parameters *opt;
    opt = &params;

    double **shifted_points;
    // tic
    gettimeofday (&startwtime, NULL);

    int iterations = meanshift(vectors, &shifted_points, h, opt, 1);

    // toc
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("%s wall clock time = %f\n","Mean Shift", seq_time);

    //TODO write output points to file -> plot later
    //save_matrix(shifted_points, iterations);
}