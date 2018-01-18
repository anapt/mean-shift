#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <float.h>
#include <stdbool.h>
#include <sys/time.h>
#include <math.h>

#define X "data/X.bin"
#define L "data/L.bin"
#define COLUMNS     2
#define ROWS        600

struct parameters {
    double epsilom;
    bool verbose;
    bool display;
};
struct timeval startwtime, endwtime;
double seq_time;

double **alloc_2d_double(int rows, int cols);
double **duplicate(double **a, double **b, int rows, int cols);
void meanshift(double **x, int h, struct parameters *opt);
double norm(double ** m, int rows, int cols);
void multiply(double ** matrix1, double ** matrix2, double ** output);
double calculateDistance(double *, double *);
void print_matrix(double ** array, int rows, int cols);

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


void meanshift(double **x, int h, struct parameters *opt){

    double **y;
    y = alloc_2d_double(ROWS, COLUMNS);
    y = duplicate(x, y, ROWS, COLUMNS);

    // mean shift vectors
    double **m;
    m = alloc_2d_double(ROWS, COLUMNS);
    // initialize elements of m to inf
    for (int i=0;i<ROWS;i++){
        for (int j=0;j<COLUMNS;j++){
            m[i][j] = DBL_MAX;
        }
    }

    // initialize iteration counter
    int iter = 0;

    // printf("%f \n", opt->epsilom);
    // TODO ITERATION
    /** iterate until convergence **/
    // printf("norm : %f \n", norm(m, ROWS, COLUMNS));

    while (norm(m, ROWS, COLUMNS) > opt->epsilom) {
//    while (iter < 1){
        iter = iter +1;
        // find pairwise distance matrix (inside radius)
        // TODO write in C
        /** allocate memory for inside iteration arrays **/
        double ** W = alloc_2d_double(ROWS, ROWS);
        double * l = malloc(ROWS * sizeof(double));
        // [I, D] = rangesearch(x,y,h);
        for (int i=0; i<ROWS; i++){
            for (int j=0; j<ROWS; j++){
                // TODO implement function to calculate distance calculateDistance(double *, double *)
                double dist = calculateDistance(y[i],x[j]);

                // 2sparse matrix
                if (dist < h){
                    W[i][j] = dist;
                    //printf("%f \n", W[i][j]);
                }else{
                    W[i][j] = 0;
                }
            }
        }


        // for each element of W (x) do x^2
        // size of W is [600 600]
        // W is a sparse matrix -> apply to non-zero elements
        for (int i=0; i<ROWS; i++){
            double sum =0;
            for (int j=0; j < ROWS; j++){
                if (W[i][j] != 0){
                    W[i][j] = W[i][j]*W[i][j];
                    // compute kernel matrix
                    // apply function to non zero elements of a sparse matrix
                    double pow = ((-1)*(W[i][j]))/(2*(h*h));
                    W[i][j] = exp(pow);
                }
                // make sure diagonal elements are 1
                if (i==j){
                    W[i][j] = W[i][j] +1;
                }
                // calculate sum(W,2)
                sum = sum + W[i][j];
            }
            /** l array is correct**/
            l[i] = sum;
            // printf("l[%d] : %f \n", i, l[i]);
        }
        /** W is correct**/
        //print_matrix(W, ROWS, ROWS);


        // create new y vector
        double** y_new = alloc_2d_double(ROWS, COLUMNS);
        // TODO implement in C : y_new = W * x

        // TODO implement function double ** multiply(double **, double **)
        multiply(W, x, y_new);
        /** y_new is CORRECT **/
        // print_matrix(y_new, ROWS, COLUMNS);
        // divide element-wise
        for (int i=0; i<ROWS; i++){
            for (int j=0; j<COLUMNS; j++){
                y_new[i][j] = y_new[i][j] / l[i];
            }
        }

        // calculate mean-shift vector
        for (int i=0; i<ROWS; i++){
            for (int j=0; j<COLUMNS; j++){
                m[i][j] = y_new[i][j] - y[i][j];

                // update y
                y[i][j] = y_new[i][j];
            }
        }

        printf("Iteration n. %d, error %f \n", iter, norm(m, ROWS, COLUMNS));
        // TODO maybe keep y for live display later?
    };



}

// allocates a 2d array in continuous memory positions
double **alloc_2d_double(int rows, int cols) {
    double *data = (double *)malloc(rows*cols*sizeof(double));
    double **array= (double **)malloc(rows*sizeof(double*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);
    return array;
}

// copy the values of a 2d double array to another
double **duplicate(double **a, double **b, int rows, int cols){
    for (int i=0;i<rows;i++){
        for (int j=0;j<cols;j++){
            b[i][j] = a[i][j];
        }
    }
    return b;
}

// TODO check why there's is a difference in the norm calculate in matlab
double norm(double ** m, int rows, int cols){
    double sum=0, a=0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a = m[i][j] * m[i][j];
            sum = sum + a;
        }
    }
    double norm = sqrt(sum);
    return norm;
}

double calculateDistance(double *y, double *x){
    double sum = 0, dif;
    for (int i=0;i<COLUMNS;i++){
        dif = y[i]-x[i];
        sum += dif * dif;
    }
    double distance = sqrt(sum);
    return distance;
}

void multiply(double ** matrix1, double ** matrix2, double ** output){
    // W dims are ROWS ROWS and x dims are ROWS COLUMNS
    // TODO CHECK
    int i, j, k;
    for (i=0; i<ROWS; i++){
        for (j=0; j<COLUMNS; j++){
            output[i][j] = 0;
            for (k=0; k<ROWS; k++){
                output[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

void print_matrix(double ** array, int rows, int cols){
    for (int i=0; i<cols; i++){
        for (int j=0; j<rows; j++){
            printf("%f ", array[j][i]);
        }
        printf("\n");
    }
}


