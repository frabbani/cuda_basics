
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include "bitmap.h"

#include <cstdio>
#include <vector>

#define IMAGE_DIM  1024
#define BLOCK_DIM   16
#define KERNEL_SIZE   3
#define NUM_STEPS	25
#define RATE		0.001

#define CLAMP(x,a,b) {x = x < (a) ? (a) : x > (b) ? (b) : x; }


__global__ void static_guided_diffusion(double3* input, double3* output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double3 r = { 0.0, 0.0,0.0 };
	double3 state = input[y * IMAGE_DIM + x];

	// Loop over the full kernel
	for (int step = 0; step < NUM_STEPS; ++step) {
		double3 sum = { 0.0, 0.0, 0.0 };
		for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) {
			int yj = y + j;
			CLAMP(yj, 0, IMAGE_DIM - 1);
			for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
				int xi = x + i;
				CLAMP(xi, 0, IMAGE_DIM - 1);
				double3 neighbor = input[yj * IMAGE_DIM + xi];
				sum.x += exp(-fabs(state.x - neighbor.x)) * (neighbor.x - state.x);
				sum.y += exp(-fabs(state.y - neighbor.y)) * (neighbor.y - state.y);
				sum.z += exp(-fabs(state.z - neighbor.z)) * (neighbor.z - state.z);
			}
		}
		// small step size
		state.x += RATE * sum.x;
		state.y += RATE * sum.y;
		state.z += RATE * sum.z;
	}

	output[y * IMAGE_DIM + x] = state;
}


__global__ void static_guided_diffusion_fast(double3* input, double3* output) {
	const int tile_dim = BLOCK_DIM * 2;
	const int halo_dim = BLOCK_DIM / 2;
	__shared__ double3 tile[tile_dim * tile_dim];
	{
		int x = 2 * threadIdx.x;
		int y = 2 * threadIdx.y;

		int gx = blockIdx.x * BLOCK_DIM + x - halo_dim;
		int gy = blockIdx.y * BLOCK_DIM + y - halo_dim;

		CLAMP(gx, 0, IMAGE_DIM - 2);
		CLAMP(gy, 0, IMAGE_DIM - 2);
		
		tile[(y + 0) * tile_dim + (x + 0)] = input[(gy + 0) * IMAGE_DIM + (gx + 0)];
		tile[(y + 0) * tile_dim + (x + 1)] = input[(gy + 0) * IMAGE_DIM + (gx + 1)];
		tile[(y + 1) * tile_dim + (x + 0)] = input[(gy + 1) * IMAGE_DIM + (gx + 0)];
		tile[(y + 1) * tile_dim + (x + 1)] = input[(gy + 1) * IMAGE_DIM + (gx + 1)];
		__syncthreads();
	}


	int x = threadIdx.x;
	int y = threadIdx.y;
	int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

	double3 state = tile[(y + halo_dim) * tile_dim + (x + halo_dim)];
	for (int step = 0; step < NUM_STEPS; ++step) {
		double3 sum = { 0.0, 0.0, 0.0 };
		for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) {
			int o = (halo_dim + y + j) * tile_dim;
			for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
				double3 neighbor = tile[o + halo_dim + x + i];
				sum.x += exp(-fabs(state.x - neighbor.x)) * (neighbor.x - state.x);
				sum.y += exp(-fabs(state.y - neighbor.y)) * (neighbor.y - state.y);
				sum.z += exp(-fabs(state.z - neighbor.z)) * (neighbor.z - state.z);
			}
		}
		// small step size
		state.x += RATE * sum.x;
		state.y += RATE * sum.y;
		state.z += RATE * sum.z;
	}

	output[gy * IMAGE_DIM + gx] = state;
}


__device__ double3 read_input(const double3* input, const uint3* perm, int x, int y) {
	const double* vals = (const double*)input;
	int idx = y * IMAGE_DIM + x;
	uint3 ijk;
	ijk.x = perm[idx].x;
	ijk.y = perm[idx].y;
	ijk.z = perm[idx].z;

	double3 v;
	v.x = vals[ijk.x];
	v.y = vals[ijk.y];
	v.z = vals[ijk.z];
	return v;
}

__global__ void static_guided_diffusion_indirect(const double3* input, const uint3* perm, double3* output) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	double3 state = read_input(input, perm, x, y);

	// Loop over the full kernel
	for (int step = 0; step < NUM_STEPS; ++step) {
		double3 sum = { 0.0, 0.0, 0.0 };
		for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) {
			int yj = y + j;
			CLAMP(yj, 0, IMAGE_DIM - 1);
			for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
				int xi = x + i;
				CLAMP(xi, 0, IMAGE_DIM - 1);
				double3 neighbor = read_input(input, perm, xi, yj);
				sum.x += exp(-fabs(state.x - neighbor.x)) * (neighbor.x - state.x);
				sum.y += exp(-fabs(state.y - neighbor.y)) * (neighbor.y - state.y);
				sum.z += exp(-fabs(state.z - neighbor.z)) * (neighbor.z - state.z);
			}
		}
		// small step size
		state.x += RATE * sum.x;
		state.y += RATE * sum.y;
		state.z += RATE * sum.z;
	}

	output[y * IMAGE_DIM + x] = state;
}


__global__ void static_guided_diffusion_fast_indirect(const double3* input, const uint3* perm, double3* output) {
	const int tile_dim = BLOCK_DIM * 2;
	const int halo_dim = BLOCK_DIM / 2;
	__shared__ double3 tile[tile_dim * tile_dim];
	{
		int x = 2 * threadIdx.x;
		int y = 2 * threadIdx.y;

		int gx = blockIdx.x * BLOCK_DIM + x - halo_dim;
		int gy = blockIdx.y * BLOCK_DIM + y - halo_dim;

		CLAMP(gx, 0, IMAGE_DIM - 2);
		CLAMP(gy, 0, IMAGE_DIM - 2);

		tile[(y + 0) * tile_dim + (x + 0)] = read_input(input, perm, gx + 0, gy + 0);
		tile[(y + 0) * tile_dim + (x + 1)] = read_input(input, perm, gx + 1, gy + 0);
		tile[(y + 1) * tile_dim + (x + 0)] = read_input(input, perm, gx + 0, gy + 1);
		tile[(y + 1) * tile_dim + (x + 1)] = read_input(input, perm, gx + 1, gy + 1);
		__syncthreads();
	}


	int x = threadIdx.x;
	int y = threadIdx.y;
	int gx = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int gy = blockIdx.y * BLOCK_DIM + threadIdx.y;

	double3 state = tile[(y + halo_dim) * tile_dim + (x + halo_dim)];
	for (int step = 0; step < NUM_STEPS; ++step) {
		double3 sum = { 0.0, 0.0, 0.0 };
		for (int j = -KERNEL_SIZE; j <= KERNEL_SIZE; j++) {
			int o = (halo_dim + y + j) * tile_dim;
			for (int i = -KERNEL_SIZE; i <= KERNEL_SIZE; i++) {
				double3 neighbor = tile[o + halo_dim + x + i];
				sum.x += exp(-fabs(state.x - neighbor.x)) * (neighbor.x - state.x);
				sum.y += exp(-fabs(state.y - neighbor.y)) * (neighbor.y - state.y);
				sum.z += exp(-fabs(state.z - neighbor.z)) * (neighbor.z - state.z);
			}
		}
		// small step size
		state.x += RATE * sum.x;
		state.y += RATE * sum.y;
		state.z += RATE * sum.z;
	}

	output[gy * IMAGE_DIM + gx] = state;
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

struct Uint3Array {
	std::vector<uint3> host;
	uint3* devptr = nullptr;

	int count() { return int(host.size()); }
	int numBytes() { return int(host.size() * sizeof(uint3)); }

	void alloc(int numValues) {
		host = std::vector<uint3>(numValues);
		auto status = cudaMalloc(&devptr, numValues * sizeof(uint3));
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


Double3Array input, output;
Uint3Array perm;

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

void saveBitmap(Double3Array& array, const char* bitmapName) {
	array.pull();
	std::vector<uint8_t> pixels(IMAGE_DIM * IMAGE_DIM * 3);
	for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++) {
		double3 rgb = array.host[i];
		rgb.x *= 255.0;
		rgb.y *= 255.0;
		rgb.z *= 255.0;
		pixels[i * 3 + 0] = uint8_t(rgb.x);
		pixels[i * 3 + 1] = uint8_t(rgb.y);
		pixels[i * 3 + 2] = uint8_t(rgb.z);
	}
	exportBMP(pixels, IMAGE_DIM, IMAGE_DIM, bitmapName);
}

bool loadImageCoalesced(std::vector<uint8_t>& image) {
	int w, h;
	importBMP(image, w, h, "input.bmp");
	if (w != IMAGE_DIM || h != IMAGE_DIM) {
		printf("input image must be of size %d x %d\n", IMAGE_DIM, IMAGE_DIM);
		return false;
	}
}

bool loadImageScrambled(std::vector<uint8_t>& image, std::vector<uint32_t>& perm) {
	std::vector<uint8_t> bitmap;
	std::vector<uint32_t> invPerm;
	int w, h;
	importBMP(bitmap, w, h, "input.bmp");
	if (w != IMAGE_DIM || h != IMAGE_DIM) {
		printf("input image must be of size %d x %d\n", IMAGE_DIM, IMAGE_DIM);
		return 0;
	}
	const char* pass = "cuda is awesome!";
	encryptBMP(bitmap, pass, image);
	exportBMP(image, IMAGE_DIM, IMAGE_DIM, "encrypted.bmp");
	genPermTables(pass, bitmap.size(), perm, invPerm);
}

int main()
{
	std::vector<uint8_t> image;
	if (!loadImageCoalesced(image))
		return -1;

	initDevice();

	input.alloc(IMAGE_DIM * IMAGE_DIM);
	output.alloc(IMAGE_DIM * IMAGE_DIM);

	for (int y = 0; y < IMAGE_DIM; y++)
		for (int x = 0; x < IMAGE_DIM; x++) {
			int idx = y * IMAGE_DIM + x;
			output.host[idx].x = 0.0;
			output.host[idx].y = 0.0;
			output.host[idx].z = 0.0;
			input.host[idx].x = image[3 * idx + 0] / 255.0;
			input.host[idx].y = image[3 * idx + 1] / 255.0;
			input.host[idx].z = image[3 * idx + 2] / 255.0;
		}
	input.push();
	output.push();

	dim3 block = dim3(BLOCK_DIM, BLOCK_DIM);
	dim3 grid = dim3(IMAGE_DIM / block.x, IMAGE_DIM/block.y);
	float ms = 0.0f;

	cudaEvent_t e0, e1;
	cudaEventCreate(&e0);
	cudaEventCreate(&e1);

	// run once to cache
	static_guided_diffusion << < grid, block >> > (input.devptr, output.devptr);	

	syncDevice();
	cudaEventRecord(e0);
	static_guided_diffusion_fast << < grid, block >> > (input.devptr, output.devptr);
	cudaEventRecord(e1);
	syncDevice();
	ms = 0.0f;
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time using shared access: %f ms\n", ms);
	saveBitmap(output, "output.bmp");

	// block = dim3(16, 16);
	//grid = dim3(IMAGE_DIM / block.x, IMAGE_DIM / block.y);
	
	syncDevice();
	cudaEventRecord(e0);
	static_guided_diffusion << < grid, block >> > (input.devptr, output.devptr);
	cudaEventRecord(e1);
	syncDevice();
	ms = 0.0f;
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time using global access: %f ms\n", ms);
	saveBitmap(output, "output2.bmp");
	

	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

	std::vector<uint32_t> tab;
	loadImageScrambled(image, tab);
	for (int y = 0; y < IMAGE_DIM; y++)
		for (int x = 0; x < IMAGE_DIM; x++) {
			int idx = y * IMAGE_DIM + x;
			input.host[idx].x = image[3 * idx + 0] / 255.0;
			input.host[idx].y = image[3 * idx + 1] / 255.0;
			input.host[idx].z = image[3 * idx + 2] / 255.0;
		}
	input.push();

	perm.alloc(IMAGE_DIM * IMAGE_DIM);
	for (int i = 0; i < perm.count(); i++) {
		perm.host[i].x = tab[i * 3 + 0];
		perm.host[i].y = tab[i * 3 + 1];
		perm.host[i].z = tab[i * 3 + 2];
	}
	perm.push();

	block = dim3(BLOCK_DIM, BLOCK_DIM);
	grid = dim3(IMAGE_DIM / block.x, IMAGE_DIM / block.y);

	syncDevice();
	cudaEventRecord(e0);
	static_guided_diffusion_fast_indirect << < grid, block >> > (input.devptr, perm.devptr, output.devptr);
	cudaEventRecord(e1);
	syncDevice();
	ms = 0.0f;
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time using shared access (indirect): %f ms\n", ms);
	saveBitmap(output, "output3.bmp");

	syncDevice();
	cudaEventRecord(e0);
	static_guided_diffusion_indirect << < grid, block >> > (input.devptr, perm.devptr, output.devptr);
	cudaEventRecord(e1);
	syncDevice();
	ms = 0.0f;
	cudaEventElapsedTime(&ms, e0, e1);
	printf("total time using global access (indirect): %f ms\n", ms);
	saveBitmap(output, "output4.bmp");


	//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//

	cudaEventDestroy(e0);
	cudaEventDestroy(e1);

	input.free();
	output.free();
	perm.free();

	termDevice();
	return 0;
}
