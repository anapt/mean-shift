#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "serialDeclarations.h"

struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char **argv){

//    if (argc<2){
//        printf("%s\n", "Specify the k");
//        return 1;
//    }
//    = atoi(argv[1]);  // the k-parameter


    FILE *f;
    f = fopen(X, "rb");
    fseek(f, 0L, SEEK_END);
    long int pos = ftell(f);
    fclose(f);
    int elements = pos / sizeof(double);  // number of total elements (points*dimension)
    int points = elements/COLUMNS;
    //printf("points : %d \n", points);
    f = fopen(X, "rb");
    double ** vectors;
    vectors = alloc_2d_double(ROWS, COLUMNS);
    for (int i=0; i<ROWS; i++){
        int out = fread(vectors[i], sizeof(double), COLUMNS, f);
    }
    //printf("test : %f \n", vectors[0][0]);
    //printf("test : %f \n", vectors[ROWS-1][COLUMNS-1]);

    // initializing file that will contain the labels (train)
    f = fopen(L, "rb");
    // NOTE : Labels were classified as <class 'numpy.uint8'>
    // variables of type uint8 are stored as 1-byte (8-bit) unsigned integers
    fseek(f, 0L, SEEK_END);
    pos = ftell(f);
    rewind(f);
    //printf("position : %ld \n", pos);
    int label_elements = pos/ sizeof(char);
    char *labels = (char*)malloc(label_elements* sizeof(char));
    fseek(f, 0L, SEEK_SET);
    int out = fread(labels, sizeof(char), label_elements, f);
    fclose(f);

    // MEAN SHIFT OPTIONS
    int h = 1;
    struct parameters params;
    params.epsilom = 0.0001;
    params.verbose = false;
    params.display = false;
    struct parameters *opt;
    opt = &params;

    // tic
    gettimeofday (&startwtime, NULL);

    meanshift(vectors, h, opt);

    // toc
    gettimeofday (&endwtime, NULL);
    seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6 + endwtime.tv_sec - startwtime.tv_sec);
    printf("%s wall clock time = %f\n","Mean Shift", seq_time);

    //TODO write output points to file -> plot later

}