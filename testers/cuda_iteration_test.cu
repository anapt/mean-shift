#include <stdio.h>
#include <assert.h>

#define N 11
#define M 3

__global__ void kernel(float * d_matrix, size_t pitch) {
    for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < N; j += blockDim.y * gridDim.y) {
        float* row_d_matrix = (float*)((char*)d_matrix + j*pitch);
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
            row_d_matrix[i] = (j * M + i) + (j * M + i);
        }
    }
}

void verify(float *h, float *d, int size) {
    for (int i = 0; i < size; i++) {
        assert(h[i] == d[i]);
    }
    printf("Results match\n");
}

int main() {

    float *h_matrix;
    float *d_matrix;
    float *dc_matrix;

    h_matrix = (float *) malloc(M * N * sizeof(float));
    dc_matrix = (float *) malloc(M * N * sizeof(float));

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            h_matrix[j * M + i] = (j * M + i) + (j * M + i);
        }
    }

    size_t pitch;
    cudaMallocPitch(&d_matrix, &pitch, M * sizeof(float), N);

    dim3 grid(1, 1, 1);
    dim3 block(3, 3, 1);

    kernel<<<grid, block>>>(d_matrix, pitch);

    cudaMemcpy2D(dc_matrix, M * sizeof(float), d_matrix, pitch, M * sizeof(float), N, cudaMemcpyDeviceToHost);

    verify(h_matrix, dc_matrix, M * N);

    free(h_matrix);
    cudaFree(d_matrix);
    free(dc_matrix);
}
