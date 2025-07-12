
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "bitmap.h"

#include <cstdio>
#include <vector>

#define IMAGE_DIM  1024
#define BLOCK_DIM   32
#define KERNEL_SIZE 2

#define CLAMP(x,a,b) {x = x < (a) ? (a) : x > (b) ? (b) : x; }

__device__ double3 box_blur(int x, int y, const double3* input, double scale, double offset) {
	double3 sum = { 0.0, 0.0, 0.0 };
	double count = 0.0;
	for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
		for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++)
		{
			int xi = x + i;
			int yj = y + j;
			CLAMP(xi, 0, IMAGE_DIM - 1);
			CLAMP(yj, 0, IMAGE_DIM - 1);
			double3 val = input[yj * IMAGE_DIM + xi];
			sum.x += val.x;
			sum.y += val.y;
			sum.z += val.z;
			count = count + 1.0;
		}
	}
	sum.x = scale * sum.x / count + offset;
	sum.y = scale * sum.y / count + offset;
	sum.z = scale * sum.z / count + offset;
	return sum;
}

// don't do this
__global__ void warp_test(const double3* input, double3* output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	bool flip = (threadIdx.x + threadIdx.y) & 0x01;
	if (flip) {
		output[y * IMAGE_DIM + x] = box_blur(x, y, input, +1.0, 0.0);
	}
	else {
		output[y * IMAGE_DIM + x] = box_blur(x, y, input, -1.0, 1.0);
	}
}

// do this instead
__global__ void warp_test_nodiv(const double3* input, double3* output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	bool flip = (threadIdx.x + threadIdx.y) & 0x01;
	double scale = flip ? -1.0 : +1.0;
	double offset = flip ? +1.0 : 0.0;
	output[y * IMAGE_DIM + x] = box_blur(x, y, input, scale, offset);
}


struct Double3Array {
	std::vector<double3> host;
	double3* devptr = nullptr;

	int count() { return int(host.size()); }
	int numBytes() { return int(host.size() * sizeof(double3)); }

	void alloc(int numValues) {
		host = std::vector<double3>(numValues);
		auto status = cudaMalloc(&devptr, numValues * sizeof(double3));
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
		auto status = cudaMemcpy(devptr, host.data(), numBytes(), cudaMemcpyHostToDevice);
		if (status != cudaSuccess) {
			printf("cudaMemcpy failed: %s\n", cudaGetErrorString(status));
		}
	}

	void pull() {
		auto status = cudaMemcpy(host.data(), devptr, numBytes(), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			printf("cudaMemcpy failed: %s\n", cudaGetErrorString(status));
		}
	}
};


void initDevice() {
	//initialize input buffer with values from an image
	int count = 0;
	cudaGetDeviceCount(&count);
	printf("# of devices: %d\n", count);
	for (int i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device %d - '%s'\n", i, prop.name);
	}
	auto status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		printf("cudaSetDevice failed: %s\n", cudaGetErrorString(status));
	}
}


void termDevice() {
	//call before exiting so profiling and tracing tools (eg. Nsight) will show complete traces
	auto status = cudaDeviceReset();
	if (status != cudaSuccess) {
		printf("cudaDeviceReset failed: %s\n", cudaGetErrorString(status));
	}
}

void saveBitmap(Double3Array& array, const char* bitmapName) {
	array.pull();
	std::vector<uint8_t> pixels(IMAGE_DIM * IMAGE_DIM * 3);
	for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++) {
		double3 rgb = array.host[i];
		rgb.x *= 255.0;
		rgb.y *= 255.0;
		rgb.z *= 255.0;
		pixels[i * 3 + 0] = uint8_t(rgb.z);
		pixels[i * 3 + 1] = uint8_t(rgb.y);
		pixels[i * 3 + 2] = uint8_t(rgb.x);
	}
	exportBMP(pixels, IMAGE_DIM, IMAGE_DIM, bitmapName);
}


int main()
{
	std::vector<uint8_t> image;
	int w, h;
	importBMP(image, w, h, "input.bmp");
	if (w != IMAGE_DIM || h != IMAGE_DIM) {
		printf("input image must be of size %d x %d\n", IMAGE_DIM, IMAGE_DIM);
		return 0;
	}

	initDevice();
	Double3Array input, output;

	input.alloc(IMAGE_DIM * IMAGE_DIM);
	output.alloc(IMAGE_DIM * IMAGE_DIM);

	for (int y = 0; y < IMAGE_DIM; y++)
		for (int x = 0; x < IMAGE_DIM; x++) {
			int idx = y * IMAGE_DIM + x;
			output.host[idx].x = 0.0;
			output.host[idx].y = 0.0;
			output.host[idx].z = 0.0;
			input.host[idx].x = image[3 * idx + 2] / 255.0;
			input.host[idx].y = image[3 * idx + 1] / 255.0;
			input.host[idx].z = image[3 * idx + 0] / 255.0;
		}
	input.push();
	output.push();

	dim3 block = dim3(BLOCK_DIM, BLOCK_DIM);
	dim3 grid = dim3(IMAGE_DIM / block.x, IMAGE_DIM / block.y);
	float ms = 0.0f;

	cudaEvent_t e0, e1;
	cudaEventCreate(&e0);
	cudaEventCreate(&e1);

	//+++++++++++++++++++++++++++++++++++++++++++

	cudaDeviceSynchronize();
	warp_test << < grid, block >> > (input.devptr, output.devptr);
	cudaDeviceSynchronize();

	cudaEventRecord(e0);
	for(int i = 0; i < 50; i++)
		warp_test << < grid, block >> > (input.devptr, output.devptr);
	cudaEventRecord(e1);
	cudaEventSynchronize(e1);
	ms = 0.0f;
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time (diverging): %f ms\n", ms);
	saveBitmap(output, "output.bmp");

	//+++++++++++++++++++++++++++++++++++++++++++

	cudaDeviceSynchronize();
	warp_test_nodiv << < grid, block >> > (input.devptr, output.devptr);
	cudaDeviceSynchronize();

	cudaEventRecord(e0);
	for (int i = 0; i < 50; i++)
		warp_test_nodiv << < grid, block >> > (input.devptr, output.devptr);
	cudaEventRecord(e1);
	cudaEventSynchronize(e1);
	ms = 0.0f;
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time: %f ms\n", ms);
	saveBitmap(output, "output2.bmp");

	//+++++++++++++++++++++++++++++++++++++++++++

	cudaEventDestroy(e0);
	cudaEventDestroy(e1);
	input.free();
	output.free();
	termDevice();

	return 0;
}
