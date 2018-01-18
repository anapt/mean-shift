//
// Created by anapt on 18/1/2018.
//

// compute kernel matrix
//        // apply function to non zero elements of a sparse matrix
//        for (int i=0; i<ROWS; i++){
//            for (int j=0; j<ROWS; j++){
//                if (W[i][j] != 0){
//                    double pow = ((-1)*(W[i][j]))/(2*(h*h));
//                    W[i][j] = exp(pow);
//                }
//            }
//        }

//        // make sure diagonal elements are 1
//        for (int i=0; i<ROWS; i++){
//            for (int j=0; j<ROWS; j++){
//                if (i==j){
//                    W[i][j] = W[i][j] +1;
//                }
//            }
//        }


//        // normalize vector
//        // allocate memory for vector l [600 1]
//        double * l = malloc(ROWS * sizeof(double));
//        // calculate sum(W,2)

//        // W is a 600 by 600 sparse matrix
//        for (int i=0; i<ROWS; i++){
//            double sum =0;
//            for (int j = 0; j < ROWS; j++){
//                sum = sum + W[i][j];
//            }
//            l[i] = sum;
//        }