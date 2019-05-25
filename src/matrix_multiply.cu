#include <iostream>
#include <vector>
#include <thrust\host_vector.h>
#include <thrust\device_vector.h>
#include <thrust\functional.h>

using namespace std;
using namespace thrust;

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// one side of the matrix
#define MAX_SIZE 8192
// the width of the matrix
#define SIZE (MAX_SIZE * MAX_SIZE)


// make a template struct for kernel function parameter
template <typename T>
struct KernelArray {
	T* _array;
	int _size;
};


// kernel funciton can't pass by device_vector for each parameters
// than we should have to convert struct or pointer value
template <typename T>
KernelArray<T> convertToKernel(thrust::device_vector<T>& dVec) {
	KernelArray<T> kArray;
	kArray._array = thrust::raw_pointer_cast(&dVec[0]);
	kArray._size = (int)dVec.size();

	return kArray;
}

// kernel function for matrix multiplication
__global__ void mat_mul_kernel(KernelArray<int> dv1, KernelArray<int> dv2, KernelArray<int> tmp) {
	int col, row;
	int res = 0;
	col = blockIdx.x * blockDim.x + threadIdx.x;
	row = blockIdx.y * blockDim.y + threadIdx.y;
	for (int i = 0; i < MAX_SIZE; i++) {
		res += dv1._array[MAX_SIZE*col + i] * dv2._array[MAX_SIZE*i + row];
	}
	tmp._array[col + MAX_SIZE*row] = res;
}

// function for matrix multiplication
// initialize & call kernel function : mat_mul_kernel()
float mat_mul(host_vector<int> v1, host_vector<int> v2, host_vector<int> v3) {
	// device vector variable for calculating 
	// matrix multiplication in GPU
	device_vector<int> dv1(SIZE);
	device_vector<int> dv2(SIZE);
	device_vector<int> tmp(SIZE);

	// variable for checking runtime
	cudaEvent_t start, stop;
	float t = 0.0;

	// initialize device variable, using host_vector
	dv1 = v1;
	dv2 = v2;
	// divide grid to block  && block to thread
	dim3 dimBlock(16, 16);
	dim3 dimGrid(16, 16);
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// in kernel function, parameters cannot be a device_vector variable
	// we must convert to it's address
	mat_mul_kernel << <dimGrid, dimBlock >> > (convertToKernel(dv1), convertToKernel(dv2), convertToKernel(tmp));
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&t, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	// restore result device to host variable
	v3 = tmp;

	// return running time
	return t;
}



int main(int argc, char **argv) {
	host_vector<int> v1(SIZE, 1);
	host_vector<int> v2(SIZE, 1);
	host_vector<int> temp(SIZE);
	cout << "CUDA PROGRAM" << endl;
	cout << MAX_SIZE << " x " << MAX_SIZE << "matrix multiplication > " << mat_mul(v1, v2, temp) << endl;

	return 0;
}