
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "bitmap.h"

#include <cstdio>
#include <vector>

constexpr int kDim = 512;
constexpr int kBlockDim = 16;

__device__ double alpha(double rate) {
    rate = rate < 0.001 ? 0.001 : rate > 1.0 ? 1.0 : rate;
    return rate / 4.0;
}

__global__ void diffuse_kernel(double* input, double* output, int w, int h, double rate) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int m = y * w + x;
    int l = x - 1 < 0 ? 0 : x - 1;
    int r = x + 1 >= w ? w - 1 : x + 1;
    int b = y - 1 < 0 ? 0 : y - 1;
    int t = y + 1 >= h ? h - 1 : y + 1;

    double vs[4];
    vs[0] = input[b * w + l];
    vs[1] = input[b * w + r];
    vs[2] = input[t * w + l];
    vs[3] = input[t * w + r];

    output[m] = input[m] + alpha(rate) * (vs[0] + vs[1] + vs[2] + vs[3] - 4.0 * input[m]);
}


struct Array {
    std::vector<double> host;
    double* devptr = nullptr;
   
    int size() { return int(host.size()); }
    void alloc(int numValues) {
        host = std::vector<double>(numValues);
        auto status = cudaMalloc(&devptr, numValues * sizeof(double));
        if (status != cudaSuccess) {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(status));
        }
    }

    void free() {
        host.clear();        
        if (devptr) {
            auto status = cudaFree(devptr);
            if (status != cudaSuccess) {
                printf("cudaFree failed: %s\n", cudaGetErrorString(status));
            }
            devptr = nullptr;
        }

    }
    void push() {
        auto status = cudaMemcpy(devptr, host.data(), size() * sizeof(double), cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(status));
        }
    }

    void pull() {
        auto status = cudaMemcpy(host.data(), devptr, size() * sizeof(double),cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(status));
        }
    }
};

Array input, output;

void initDevice() {
    //initialize input buffer with values from an image
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("# of devices: %d\n", count);
    for (int i = 0; i < count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d - '%s'\n", i, prop.name);
        printf(" * warp size.................: %d\n", prop.warpSize);
        printf(" * max # of threads per block: %d\n", prop.maxThreadsPerBlock);
        printf(" * max grid size.............: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    auto status  = cudaSetDevice(0);
    if (status != cudaSuccess) {
        printf("cudaSetDevice failed: %s\n", cudaGetErrorString(status));
    } 
}

void syncDevice() {
    auto status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        printf("cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(status));
    }
}

void termDevice() {
    //call before exiting so profiling and tracing tools (eg. Nsight) will show complete traces
    auto status = cudaDeviceReset();
    if (status != cudaSuccess) {
        printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(status));
    }
}

void saveBitmap(Array& input, const char* bitmapName) {
    input.pull();
    std::vector<uint8_t> pixels(kDim * kDim * 3);
    for (int i = 0; i < kDim * kDim; i++) {
        uint8_t v = uint8_t(input.host[i] * 255.0);
        pixels[i * 3 + 0] = v;
        pixels[i * 3 + 1] = v;
        pixels[i * 3 + 2] = v;
    }
    exportBMP(pixels, kDim, kDim, bitmapName);
}

int main()
{
    initDevice();

    input.alloc(kDim * kDim);
    output.alloc(kDim * kDim);


    double outerRad = double(kDim) / 2.5;
    double innerRad = double(kDim) / 5.0;
    for (int y = 0; y < kDim; y++)
        for (int x = 0; x < kDim; x++) {
            output.host[y * kDim + x] = 0.0;
            int dx = x - kDim / 2;
            int dy = y - kDim / 2;
            double len = sqrt(dx * dx + dy * dy);
            if (len >= innerRad && len <= outerRad)
                input.host[y * kDim + x] = 1.0;
            else
                input.host[y * kDim + x] = 0.0;
        }
    input.push();
    output.push();

    saveBitmap(input, "input.bmp");

    dim3 block(kBlockDim, kBlockDim);  // 16x16 = 256 threads per block
    dim3 grid(kDim / kBlockDim, kDim / kBlockDim);  // 512/16 = 32, so 32x32 blocks

    for (int i = 0; i < 256; i++) {
        if (i & 0x01) {
            diffuse_kernel << <grid, block >> > (output.devptr, input.devptr, kDim, kDim, 0.6);
        }
        else
            diffuse_kernel << <grid, block >> > (input.devptr, output.devptr, kDim, kDim, 0.6);
    }
    syncDevice();   // sync not really needed since cudaMemcpy is synchronous by default

    saveBitmap(output, "output.bmp");

    input.free();
    output.free();

    termDevice();
    return 0;
}
