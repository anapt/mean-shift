
// Host code 
int width = 64, height = 64; 
float* devPtr; 
size_t pitch; 
cudaMallocPitch(&devPtr, &pitch, width * sizeof(float), height); 
MyKernel<<<100, 512>>>(devPtr, pitch, width, height); 


// Device code 
__global__ voidMyKernel(float* devPtr, size_t pitch, int width, int height) { 
    for (int r = 0; r < height; ++r) { 
        float* row = (float*)((char*)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c]; 
        } 
    } 
}

Read more at: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#ixzz54kTh80mZ 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook