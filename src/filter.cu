
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <math.h>
#include <string>
#include <stdio.h>


#define CUDA_VERIFY(fn, msg) \
do { \
	cudaError_t err = (fn); \
	if (err != cudaSuccess) { \
        fprintf(stderr, msg); \
		return err;\
	} \
} while (0)


__constant__ float c_gaussian[64];   //gaussian array in device side

// it uses only one axis of the kernel (1,2r) instead of a matrix (2r,2r)
inline cudaError_t computeGaussianKernelCuda(const float delta, const int radius)
{
	float h_gaussian[64];
	for (int i = 0; i < 2 * radius + 1; ++i)
	{
		const float x = i - radius;
		h_gaussian[i] = expf(-(x * x) / (2.0f * delta * delta));
	}
	CUDA_VERIFY(cudaMemcpyToSymbol(c_gaussian, h_gaussian, sizeof(float)*(2 * radius + 1)), "CUDA Kernel Memcpy Host To Device Failed");
}

// it computes the euclidean distance between two points, each point a vector with 4 elements
__device__ inline float euclideanLenCuda(const float4 a, const float4 b, const float d)
{
	const float mod = (b.x - a.x) * (b.x - a.x) +
		(b.y - a.y) * (b.y - a.y) +
		(b.z - a.z) * (b.z - a.z) +
		(b.w - a.w) * (b.w - a.w);
	return expf(-mod / (2.0f * d * d));
}

__device__ inline float4 multiplyCuda(const float a, const float4 b)
{
	return{ a * b.x, a * b.y, a * b.z, a * b.w };
}

__device__ inline float4 addCuda(const float4 a, const float4 b)
{
	return{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

__global__ void bilateralFilterCudaKernel(const float4 * const d_input,
	float4 * const d_output,
	const float euclidean_delta,
	const int width, const int height,
	const int filter_radius)
{
	//2D Index of current thread
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	//Only valid threads perform memory I/O
	if ((x<width) && (y<height))
	{
		float sum = 0.0f;
		float4 t = { 0.f, 0.f, 0.f, 0.f };
		const float4 center = d_input[y * width + x];
		const int r = filter_radius;

		float domainDist = 0.0f, colorDist = 0.0f, factor = 0.0f;

		for (int i = -r; i <= r; ++i)
		{
			int crtY = y + i; //clamp the neighbor pixel, prevent overflow
			if (crtY < 0)				crtY = 0;
			else if (crtY >= height)   	crtY = height - 1;

			for (int j = -r; j <= r; ++j)
			{
				int crtX = x + j;
				if (crtX < 0) 				crtX = 0;
				else if (crtX >= width)	 	crtX = width - 1;

				const float4 curPix = d_input[crtY * width + crtX];
				domainDist = c_gaussian[r + i] * c_gaussian[r + j];
				colorDist = euclideanLenCuda(curPix, center, euclidean_delta);
				factor = domainDist * colorDist;
				sum += factor;
				t = addCuda(t, multiplyCuda(factor, curPix));
			}
		}

		d_output[y * width + x] = multiplyCuda(1.f / sum, t);
	}
}

cudaError_t bilateralFilterCuda(const float4 * const h_input,
	float4 * const h_output,
	const float euclidean_delta,
	const int width, const int height,
	const int filter_radius)
{
	// compute the gaussian kernel for the current radius and delta
	CUDA_VERIFY(computeGaussianKernelCuda(euclidean_delta, filter_radius), "Failed to compute gaussian kernel");

	// copy the input image from the CPU´s memory to the GPU´s global memory
	const int inputBytes = width * height * sizeof(float4);
	const int outputBytes = inputBytes;
	float4 *d_input, *d_output; // arrays in the GPU´s global memory
								// allocate device memory
	CUDA_VERIFY(cudaMalloc<float4>(&d_input, inputBytes), "CUDA Malloc Failed");
	CUDA_VERIFY(cudaMalloc<float4>(&d_output, outputBytes), "CUDA Malloc Failed");
	// copy data of input image to device memory
	CUDA_VERIFY(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	//GpuTimer timer;
	//timer.Start();

	// specify a reasonable grid and block sizes
	const dim3 block(16, 16);
	// calculate grid size to cover the whole image
	const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	// launch the size conversion kernel
	bilateralFilterCudaKernel << <grid, block >> >(d_input, d_output, euclidean_delta, width, height, filter_radius);

	//timer.Stop();
	//printf("Own Cuda code ran in: %f msecs.\n", timer.Elapsed());

	// synchronize to check for any kernel launch errors
	CUDA_VERIFY(cudaDeviceSynchronize(), "Kernel Launch Failed");

	//Copy back data from destination device meory to OpenCV output image
	CUDA_VERIFY(cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	//Free the device memory
	CUDA_VERIFY(cudaFree(d_input), "CUDA Free Failed");
	CUDA_VERIFY(cudaFree(d_output), "CUDA Free Failed");
	//CUDA_VERIFY(cudaDeviceReset(),"CUDA Device Reset Failed");

	return cudaSuccess;
}


void processUsingCuda(std::string input_file, std::string output_file) {
	//Read input image from the disk
	cv::Mat input = cv::imread(input_file, cv::IMREAD_UNCHANGED);
	if (input.empty())
	{
		fprintf(stderr, "Image Not Found: %s", input_file.c_str());
		return;
	}

	// convert from char(0-255) BGR to float (0.0-0.1) RGBA
	cv::Mat inputRGBA;
	cv::cvtColor(input, inputRGBA, CV_BGR2RGBA, 4);
	inputRGBA.convertTo(inputRGBA, CV_32FC4);
	inputRGBA /= 255;

	//Create output image
	cv::Mat output(input.size(), inputRGBA.type());

	const float euclidean_delta = 3.0f;
	const int filter_radius = 3;

	bilateralFilterCuda((float4*)inputRGBA.ptr<float4>(),
		(float4*)output.ptr<float4>(),
		euclidean_delta,
		inputRGBA.cols, inputRGBA.rows,
		filter_radius);

	// convert back to char (0-255) BGR
	output *= 255;
	//output.convertTo(output, CV_8UC4);
	cvtColor(output, output, CV_RGBA2BGR, 3);

	imwrite(output_file, output);
}


int main(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "expected 2 params (path to input and output file)");
		return 1;
	}

	processUsingCuda(argv[1], argv[2]);

	return 0;
}
