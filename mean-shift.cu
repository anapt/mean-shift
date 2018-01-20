#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>

#define X "data/X.bin"
#define L "data/L.bin"
#define COLUMNS     2
#define ROWS        600

struct parameters {
    double epsilon;
    bool verbose;
    bool display;
};

double **alloc_2d_double(int rows, int cols);
double **duplicate(double **a, double **b, int rows, int cols);
void meanshift(double **x, int h, struct parameters *opt);
double norm(double ** m, int rows, int cols);
void multiply(double ** matrix1, double ** matrix2, double ** output);
double calculateDistance(double *, double *);
void print_matrix(double ** array, int rows, int cols);


struct timeval startwtime, endwtime;
double seq_time;

int main(int argc, char **argv){

//    if (argc<2){
//        printf("%s\n", "Specify the k");
//        return 1;
//    }
//    = atoi(argv[1]);  // the k-parameter


    FILE *f;
//    f = fopen(X, "rb");
//    fseek(f, 0L, SEEK_END);
//    long int pos = ftell(f);
//    fclose(f);
//    int elements = pos / sizeof(double);  // number of total elements (points*dimension)
//    int points = elements/COLUMNS;
//    //printf("points : %d \n", points);
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
    long int pos = ftell(f);
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
    params.epsilon = 0.0001;
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

    // printf("%f \n", opt->epsilon);

    /** iterate until convergence **/
    // printf("norm : %f \n", norm(m, ROWS, COLUMNS));
    /** allocate memory **/
    double ** W = alloc_2d_double(ROWS, ROWS);
    double * l = malloc(ROWS * sizeof(double));

    double * d_W;
    cudaMalloc(&d_W, ROWS * ROWS * sizeof(double));
    double * d_I;
    cudaMalloc(&d_I, ROWS * sizeof(double));
    double * d_y_new;
    cudaMalloc(&d_y_new, ROWS * COLUMNS * sizeof(double));

    double * d_y;
    cudaMalloc(&d_y, ROWS * COLUMNS * sizeof(double));
    double * d_m;
    cudaMalloc(&d_m, ROWS * COLUMNS * sizeof(double));

    //Copy vectors from host memory to device memory
    cudaMemcpy(d_y, y, ROWS * COLUMNS * sizeof(double), cudaMemcpyHostToDevice);
    // y[i][j] == d_y[COLUMNS*i + j]
    cudaMemcpy(d_m, m, ROWS * COLUMNS * sizeof(double), cudaMemcpyHostToDevice);


    while (norm(m, ROWS, COLUMNS) > opt->epsilon) {
        iter = iter +1;
        // find pairwise distance matrix (inside radius)
        /** allocate memory for inside iteration arrays **/
        double ** W = alloc_2d_double(ROWS, ROWS);
        double * l = malloc(ROWS * sizeof(double));
        // [I, D] = rangesearch(x,y,h);
        for (int i=0; i<ROWS; i++){
            for (int j=0; j<ROWS; j++){
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

__global__ void iteration (double* W, double epsilon){
    // TODO check if they also need cudamalloc
    // todo find how to keep counter
    int iter;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    while (norm > epsilon){
        // TODO ITERATION
        iter = iter +1;
        // find pairwise distance matrix (inside radius)
        /** allocate memory for inside iteration arrays **/
        // TODO ALLOCATE MEMORY BEFORE CALLING KERNEL
//        double ** W = alloc_2d_double(ROWS, ROWS);
//        double * l = malloc(ROWS * sizeof(double));
        // [I, D] = rangesearch(x,y,h);
        for (int i=0; i<ROWS; i++){
            for (int j=0; j<ROWS; j++){
                // TODO REFACTOR CALCULATE DISTANCE
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
    }
}