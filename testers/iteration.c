void iteration (int number_of_iterations, int NUMBER_OF_POINTS, int DIMENSIONS, int h){

    for (int iter=0; iter < number_of_iterations; iter++){
        double accum =0;
        for (int i =0; i< NUMBER_OF_POINTS; i++){
            for (int j=0; j< NUMBER_OF_POINTS; j++){
                //calculate distance between vectors x, y
                double sum=0;
                double dif;
                for (int k=0; k < DIMENSIONS; k++){
                    // TODO CHANGE NAMES
                    dif = y[k]-x[k];
                    sum += dif*dif;
                }
                double distance = sqrt(sum);
                // 2 sparse array
                if (distance < h){
                    kernel_matrix[i][j] = dist;
                }else{
                    kernel_matrix[i][j] = 0;
                }
                if (kernel_matrix[i][j]!=0){
                    kernel_matrix[i][j] = kernel_matrix[i][j]*kernel_matrix[i][j];
                    double pow = ((-1)*(kernel_matrix[i][j]))/(2*(h*h));
                    kernel_matrix[i][j] = exp(pow);
                }
                if (i==j){
                    kernel_matrix[i][j] = kernel_matrix[i][j] +1;
                }
                accum = accum + kernel_matrix[i][j];
            }
            denominator[i] = accum;

            for (int j =0; j < DIMENSIONS;j++){
                new_shift[i][j]=0;
                for (int k=0; k<NUMBER_OF_POINTS; k++){
                    new_shift[i][j] += kernel_matrix[i][k] * original_points[k][j];
                }
                new_shift[i][j] = new_shift[i][j] / denominator[i];
                mean_shift_vector[i][j] = new_shift[i][j] - (*shifted_points)[i][j];
            }
        }
        // frees previously shifted points, they're now garbage
        free((*shifted_points)[0]);
        // updates shifted points pointer to the new array address
        shifted_points = &new_shift;
        

    }// iteration end

}