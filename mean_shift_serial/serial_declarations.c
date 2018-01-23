#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

#include "serial_declarations.h"

#define OUTPUT_PREFIX "../output/output_"

void get_args(int argc, char **argv){
    if (argc != 6) {
        printf("Usage: %s h N D Pd Pl\nwhere:\n", argv[0]);
        printf("\th is the variance\n");
        printf("\tN is the the number of points\n");
        printf("\tD is the number of dimensions of each point\n");
        printf("\tPd is the path of the dataset file\n");
        printf("\tPl is the path of the labels file\n");
        exit(1);
    }

    DEVIATION = atoi(argv[1]);
    NUMBER_OF_POINTS = atoi(argv[2]);
    DIMENSIONS = atoi(argv[3]);
    POINTS_FILENAME = argv[4];
    LABELS_FILENAME = argv[5];
}

void init(double ***vectors, char **labels, parameters *params){
    int bytes_read = 0;
    // initializes vectors
    FILE *points_file;
    points_file = fopen(POINTS_FILENAME, "rb");
    if (points_file != NULL){
        // allocates memory for the array
        (*vectors) = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
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

    // MEAN SHIFT OPTIONS
    params->epsilon = 0.0001;
    params->verbose = false;
    params->display = false;
}

int meanshift(double **original_points, double ***shifted_points, int deviation
    , parameters *opt){
    static int iteration = 0;

    // allocates memory and copies original points on first iteration
    if (iteration == 0 || (*shifted_points) == NULL){
        iteration = 0;
        (*shifted_points) = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
        duplicate(original_points, NUMBER_OF_POINTS, DIMENSIONS, shifted_points);
    }

    // allocates memory for mean shift vector
    double **mean_shift_vector;
    mean_shift_vector = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
    // initializes elements of mean_shift_vector to inf
    for (int i=0;i<NUMBER_OF_POINTS;i++){
        for (int j=0;j<DIMENSIONS;j++){
            mean_shift_vector[i][j] = DBL_MAX;
        }
    }

    // allocates memory for other arrays needed
    double **kernel_matrix = alloc_2d_double(NUMBER_OF_POINTS, NUMBER_OF_POINTS);
    double *denominator = malloc(NUMBER_OF_POINTS * sizeof(double));

    // finds pairwise distance matrix (inside radius)
    // [I, D] = rangesearch(x,y,h);
    for (int i=0; i<NUMBER_OF_POINTS; i++){
        double sum = 0;
        for (int j=0; j<NUMBER_OF_POINTS; j++){
            double distance = calculateDistance((*shifted_points)[i]
                , original_points[j]);

            double deviation_square = deviation*deviation;
            if (distance < deviation_square){
                // computes kernel matrix
                double pow = ((-1)*(distance * distance))/(2*(deviation_square));
                kernel_matrix[i][j] = exp(pow);
            } else {
                kernel_matrix[i][j] = 0;
            }
            if (i == j){
                kernel_matrix[i][j] += 1;
            }
            sum = sum + kernel_matrix[i][j];
        }
        denominator[i] = sum;
    }

    // creates new y vector
    double **new_shift = alloc_2d_double(NUMBER_OF_POINTS, DIMENSIONS);
    // builds nominator
    multiply(kernel_matrix, original_points, new_shift);
    // divides element-wise
    for (int i=0; i<NUMBER_OF_POINTS; i++){
        for (int j=0; j<DIMENSIONS; j++){
            new_shift[i][j] = new_shift[i][j] / denominator[i];
            // calculates mean-shift vector at the same time
            mean_shift_vector[i][j] = new_shift[i][j] - (*shifted_points)[i][j];
        }
    }

    // frees previously shifted points, they're now garbage
    free((*shifted_points)[0]);
    // updates shifted points pointer to the new array address
    shifted_points = &new_shift;

    save_matrix((*shifted_points), iteration);

    // calculates norm of the new mean shift vector
    double current_norm = norm(mean_shift_vector, NUMBER_OF_POINTS, DIMENSIONS);
    printf("Iteration n. %d, error %f \n", iteration, current_norm);

    // cleans up this iteration's allocations
    free(mean_shift_vector[0]);
    free(mean_shift_vector);
    free(kernel_matrix[0]);
    free(kernel_matrix);
    free(denominator);

    /** iterates until convergence **/
    if (current_norm > opt->epsilon) {
        ++iteration;
        return meanshift(original_points, shifted_points, deviation, opt);
    }

    return iteration;
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

void multiply(double **matrix1, double **matrix2, double **output){
    // W dims are NUMBER_OF_POINTS NUMBER_OF_POINTS
    // and x dims are NUMBER_OF_POINTS DIMENSIONS

    for (int i=0; i<NUMBER_OF_POINTS; i++){
        for (int j=0; j<DIMENSIONS; j++){
            output[i][j] = 0;
            for (int k=0; k<NUMBER_OF_POINTS; k++){
                output[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

double calculateDistance(double *y, double *x){
    double sum = 0, dif;
    for (int i=0; i<DIMENSIONS; i++){
        dif = y[i]-x[i];
        sum += dif * dif;
    }
    double distance = sqrt(sum);
    return distance;
}

double **alloc_2d_double(int rows, int cols) {
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
