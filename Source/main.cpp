#pragma comment( lib, "OpenCL" )

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <iostream>
#include <string>
#include <fstream>

#include <CL/cl.hpp>
#include "oclHelper.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MyMat.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif //WIN32

#define TEST_COUNT 4
#define TMP_BUFFER_SIZE 4096

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU

int main(int argc, char* argv[])
{
	cl_int err_msg, err_msg2;

	MyMat img_source;
	img_source.loadImageFromFile("Lenna.png");

	// ToDo kontrola na�ten� souboru
	/*cv::Mat img_source = image_read("Lenna.png");
	if (!img_source.data)
	{
		std::cerr << "Input image could not be loaded." << std::endl;
		exit(1);
	}*/

	/*img_source.saveImageToFile("Lenna_o1.png");

	cl_float3 * asd = img_source.getData();

	asd[0].x = 0.0;
	asd[0].y = 0.0;
	asd[0].z = 0.0;

	asd[2].x = 1.0;
	asd[2].y = 1.0;
	asd[2].z = 1.0;

	img_source.setData(asd);

	img_source.saveImageToFile("Lenna_o2.png");*/

	cl_float3 * img_source_fl3 = img_source.getData();

	const int param_space = 3;
	const float param_range = 10.0f;

	int dest_rows = img_source.getMat().rows - param_space * 2;
	int dest_cols = img_source.getMat().cols - param_space * 2;

	MyMat img_dest1(dest_rows, dest_cols);
	cl_float3 * img_dest1_fl3 = img_dest1.getData();


	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> platform_devices;

	// Get Platforms count
	clPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");
	printf("Platforms:\n");

	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		// Print platform name
		printf(" %d. platform name: %s.\n", i, platforms[i].getInfo<CL_PLATFORM_NAME>(&err_msg).c_str());
		clPrintErrorExit(err_msg, "cl::Platform::getInfo<CL_PLATFORM_NAME>");

		// Get platform devices count
		clPrintErrorExit(platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices), "getDevices");

		for (unsigned int j = 0; j < platform_devices.size(); j++)
		{
			// Get device name
			printf("  %d. device name: %s.\n", j, platform_devices[j].getInfo<CL_DEVICE_NAME>(&err_msg).c_str());
			clPrintErrorExit(err_msg, "cl::Device::getInfo<CL_DEVICE_NAME>");
		}

		platform_devices.clear();
	}

	cl::Device selected_device;
	bool device_found = false;
	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		clPrintErrorExit(platforms[i].getDevices(SELECTED_DEVICE_TYPE, &platform_devices), "getDevices");

		if (platform_devices.size() != 0)
		{
			device_found = true;
			selected_device = platform_devices[0];
			break;
		}
	}

	if (!device_found)
	{
		clPrintErrorExit(CL_DEVICE_NOT_FOUND, "GPU device");
	}

	// check if device is correct
	if (selected_device.getInfo<CL_DEVICE_TYPE>() == SELECTED_DEVICE_TYPE)
	{
		printf("\nSelected device type: Correct\n");
	}
	else
	{
		printf("\nSelected device type: Incorrect\n");
	}

	printf("Selected device name: %s.\n", selected_device.getInfo<CL_DEVICE_NAME>().c_str());

	platforms.clear();

	cl::Context context(selected_device, NULL, NULL, NULL, &err_msg);
	clPrintErrorExit(err_msg, "cl::Context");

	cl::CommandQueue queue(context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err_msg);
	clPrintErrorExit(err_msg, "cl::CommandQueue");

	char *program_source;
	cl::Program::Sources sources;

	program_source = readFile("bilateralFilter_test.cl");
	sources.push_back(std::pair<const char *, size_t>(program_source, 0));

	program_source = readFile("bilateralFilter_basic.cl");
	sources.push_back(std::pair<const char *, size_t>(program_source, 0));

	program_source = readFile("bilateralFilter_optimized1.cl");
	sources.push_back(std::pair<const char *, size_t>(program_source, 0));

	cl::Program program(context, sources, &err_msg);
	clPrintErrorExit(err_msg, "clCreateProgramWithSource");

	// build program
	if ((err_msg = program.build(std::vector<cl::Device>(1, selected_device), "", NULL, NULL)) == CL_BUILD_PROGRAM_FAILURE)
	{
		printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, &err_msg2).c_str());
		clPrintErrorExit(err_msg2, "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
	}

	clPrintErrorExit(err_msg, "clBuildProgram");

	// create kernel functors
	cl::make_kernel<cl::Buffer&, cl::Buffer&, const cl_int&, const cl_int&, const cl_int&, const cl_float&> bilateralFilter_test =
		cl::make_kernel<cl::Buffer&, cl::Buffer&, const cl_int&, const cl_int&, const cl_int&, const cl_float&>(program, "bilateralFilter_test", &err_msg);

	cl::make_kernel<cl::Buffer&, cl::Buffer&, const cl_int&, const cl_int&, const cl_int&, const cl_float&> bilateralFilter_basic =
		cl::make_kernel<cl::Buffer&, cl::Buffer&, const cl_int&, const cl_int&, const cl_int&, const cl_float&>(program, "bilateralFilter_basic", &err_msg);
	
	cl::Buffer img_source_dev(context, CL_MEM_READ_ONLY, (size_t)img_source.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_source");
	cl::Buffer img_dest1_dev(context, CL_MEM_READ_WRITE, (size_t)img_dest1.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_dest1");

	cl::UserEvent img_source_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_source");
	cl::UserEvent img_dest1_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_dest1");

	cl::NDRange local(16, 16);
	cl::NDRange global(alignTo(img_dest1.getMat().cols, local[0]), alignTo(img_dest1.getMat().rows, local[1]));

	clPrintErrorExit(queue.enqueueWriteBuffer(img_source_dev, CL_FALSE, 0, img_source.getDataSize(), img_source_fl3, NULL, &img_source_event), "clEnqueueWriteBuffer: img_source");

	cl::Event kernel_test_event = bilateralFilter_basic(
		cl::EnqueueArgs(queue, global, local),
		img_source_dev,
		img_dest1_dev,
		img_dest1.getMat().cols,
		img_dest1.getMat().rows,
		param_space,
		param_range
	);

	clPrintErrorExit(queue.enqueueReadBuffer(img_dest1_dev, CL_FALSE, 0, img_dest1.getDataSize(), img_dest1_fl3, NULL, &img_dest1_event), "clEnqueueReadBuffer: img_dest1");

	// synchronize queue
	clPrintErrorExit(queue.finish(), "clFinish");

	// Ulo�en� v�sledku
	img_dest1.setData(img_dest1_fl3);
	img_dest1.saveImageToFile("Lenna_p2.png");



	getchar();

	return 0;
}

