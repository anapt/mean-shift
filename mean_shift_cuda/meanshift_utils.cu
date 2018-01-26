#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "meanshift_utils.h"
#include "meanshift_gpu_utils.h"

#define OUTPUT_PREFIX "../output/output_"

void get_args(int argc, char **argv, parameters *params){
    if (argc < 7) {
        printf("Usage: %s h e N D Pd Pl\nwhere:\n"
        "\th is the variance\n"
        "\te is the min distance, between two points, that is taken into account in computations\n"
        "\tN is the the number of points\n"
        "\tD is the number of dimensions of each point\n"
        "\tPd is the path of the dataset file\n"
        "\tPl is the path of the labels file\n"
        "\n\t--verbose | -v is an optional flag to enable execution information output"
        "\n\t--output | -o is an optional flag to enable points output in each iteration", argv[0]);
        exit(1);
    }

    DEVIATION = atoi(argv[1]);
    params->epsilon = atof(argv[2]);
    NUMBER_OF_POINTS = atoi(argv[3]);
    DIMENSIONS = atoi(argv[4]);
    POINTS_FILENAME = argv[5];
    LABELS_FILENAME = argv[6];
    params->verbose = false;
    params->display = false;

    if (argc > 7){
        for (int index=7; index<argc; ++index){
            if (!strcmp(argv[index], "--verbose") || !strcmp(argv[index], "-v")){
                params->verbose = true;
            } else if (!strcmp(argv[index], "--output") || !strcmp(argv[index], "-o")){
                params->display = true;
            } else {
                printf("Couldn't parse argument %d: %s\n", index, argv[index]);
                exit(EXIT_FAILURE);
            }
        }
    }

    /*printf("DEVIATION = %d\n"
        "epsilon = %f\n"
        "NUMBER_OF_POINTS = %d\n"
        "DIMENSIONS = %d\n"
        "POINTS_FILENAME = %s\n"
        "LABELS_FILENAME = %s\n"
        "verbose = %d\n"
        "display = %d\n", DEVIATION, params->epsilon, NUMBER_OF_POINTS, DIMENSIONS, POINTS_FILENAME
            , LABELS_FILENAME, params->verbose, params->display);*/
}

void init(double ***vectors, char **labels){
    int bytes_read = 0;

    set_GPU();

    if (params.verbose){
        printf("Reading dataset and labels...\n");
    }

    // initializes vectors
    FILE *points_file;
    points_file = fopen(POINTS_FILENAME, "rb");
    if (points_file != NULL){
        // allocates memory for the array
        (*vectors) = alloc_double(NUMBER_OF_POINTS, DIMENSIONS);
        // reads vectors dataset from file
        for (int i=0; i<NUMBER_OF_POINTS; i++){
            bytes_read = fread((*vectors)[i], sizeof(double), DIMENSIONS, points_file);
            if ( bytes_read != DIMENSIONS ){
                if(feof(points_file)){
                    printf("Premature end of file reached.\n");
                } else{
                    printf("Error reading points file.");
                }
                fclose(points_file);
                exit(EXIT_FAILURE);
            }
        }
    } else {
        printf("Error reading dataset file.\n");
        exit(EXIT_FAILURE);
    }
    fclose(points_file);

    // initializes file that will contain the labels (train)
    FILE *labels_file;
    labels_file = fopen(LABELS_FILENAME, "rb");
    if (labels_file != NULL){
        // NOTE : Labels were classified as <class 'numpy.uint8'>
        // variables of type uint8 are stored as 1-byte (8-bit) unsigned integers
        // gets number of labels
        fseek(labels_file, 0L, SEEK_END);
        long int pos = ftell(labels_file);
        rewind(labels_file);
        int label_elements = pos/ sizeof(char);

        // allocates memory for the array
        *labels = (char*)malloc(label_elements* sizeof(char));
        fseek(labels_file, 0L, SEEK_SET);
        bytes_read = fread((*labels), sizeof(char), label_elements, labels_file);
        if ( bytes_read != label_elements ){
            if(feof(points_file)){
                printf("Premature end of file reached.\n");
            } else{
                printf("Error reading points file.");
            }
            fclose(labels_file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(labels_file);

    if (params.verbose){
        printf("Done.\n\n");
    }
}

// TODO check why there's is a difference in the norm calculate in matlab
double norm(double **matrix, int rows, int cols){
    double sum=0, temp_mul=0;
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            temp_mul = matrix[i][j] * matrix[i][j];
            sum = sum + temp_mul;
        }
    }
    double norm = sqrt(sum);
    return norm;
}

double **alloc_double(int rows, int cols) {
    double *data = (double *) malloc(rows*cols*sizeof(double));
    double **array = (double **) malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);
    return array;
}

void duplicate(double **source, int rows, int cols, double ***dest){
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            (*dest)[i][j] = source[i][j];
        }
    }
}

void print_matrix(double **array, int rows, int cols){
    for (int i=0; i<cols; i++){
        for (int j=0; j<rows; j++){
            printf("%f ", array[j][i]);
        }
        printf("\n");
    }
}

void save_matrix(double **matrix, int iteration){
    char filename[50];
    snprintf(filename, sizeof(filename), "%s%d", "../output/output_", iteration);
    FILE *file;
    file = fopen(filename, "w");
    for (int rows=0; rows<NUMBER_OF_POINTS; ++rows){
        for (int cols=0; cols<DIMENSIONS; ++cols){
            fprintf(file, "%f", matrix[rows][cols]);
            if (cols != DIMENSIONS - 1){
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
}