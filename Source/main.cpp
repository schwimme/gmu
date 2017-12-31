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

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif //WIN32

#define TEST_COUNT 4
#define TMP_BUFFER_SIZE 4096

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU

/**
 * @brief Naète ze zadaného souboru obrázek a provede potøebné konverze.
 *
 * Tato funkce provede pøevod do barevného prostoru CIE-Lab s datovými typy float.
 * Typ float bude v rozsahu 0-1.
 *
 * @param Název souboru.
 */
cv::Mat image_read(const std::string fileName)
{
	cv::Mat im_bgr_uint8 = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);

	cv::Mat im_lab_uint8;
	cv::cvtColor(im_bgr_uint8, im_lab_uint8, cv::COLOR_BGR2Lab);

	cv::Mat im_lab_fp;
	im_lab_uint8.convertTo(im_lab_fp, CV_32FC3, 1.0/255);

	return im_lab_fp;
}

/**
* @brief Provede potøebné konverze a uloží obrázek do zadaného souboru.
*
* Tato funkce provede pøevod z barevného prostoru CIE-Lab s datovými typy float.
* Pøevádí se do barevného prostoru BGR s datovými typy UINT8.
* Typ float se pøedpokládá v rozsahu 0-1.
*
* @param Název souboru.
*/
void image_write(const cv::Mat & img, std::string fileName)
{
	cv::Mat im_lab_uint8;
	img.convertTo(im_lab_uint8, CV_8UC3, 255.0);

	cv::Mat im_bgr_uint8;
	cv::cvtColor(im_lab_uint8, im_bgr_uint8, cv::COLOR_Lab2BGR);

	cv::imwrite(fileName, im_bgr_uint8);
}

int main(int argc, char* argv[])
{
	cl_int err_msg, err_msg2;

	cv::Mat img_source = image_read("Lenna.png");
	if (!img_source.data)
	{
		std::cerr << "Input image could not be loaded." << std::endl;
		exit(1);
	}


	image_write(img_source, "Lenna_o1.png");

	/*std::ofstream ofile;
	ofile.open("o.txt");
	ofile << img_source << std::endl;
	ofile.close();*/

	cl_float3 * asd; // Zarovnává se na 4 byty.
	int s1 = sizeof(cl_float3);
	int s2 = sizeof(cl_float3 *);
	int s3 = sizeof(cl_float2);
	int s4 = sizeof(cl_float4);
	asd = (cl_float3 *)img_source.data;

	cl_float3 a1 = asd[0];
	cl_float3 a2 = asd[1];
	cl_float3 a3 = asd[2];
	cl_float3 a4 = asd[3];

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
	cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&, cl_float&> bilateralFilter_basic =
		cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&, cl_float&>(program, "bilateralFilter_basic", &err_msg);

	/*cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&> matrix_mul_basic =
		cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&>(program, "matrix_mul_basic", &err_msg);*/

		/*cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&, cl::LocalSpaceArg&, cl::LocalSpaceArg&> matrix_mul_local =
			cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&, cl::LocalSpaceArg&, cl::LocalSpaceArg&>(program, "matrix_mul_local", &err_msg);*/

			// ToDo
			/*size_t source_size = sizeof(cl_float3) * a_w * a_h;
			size_t destination_size = sizeof(cl_float3) * a_w * a_h;

			cl_int *a_host = genRandomBuffer(a_w * a_h);
			cl_int *b_host = genRandomBuffer(b_w * b_h);
			cl_int *c_host = (cl_int *)malloc(c_size);
			cl_int *c_basic_host = (cl_int *)malloc(c_size);
			cl_int *c_local_host = (cl_int *)malloc(c_size);

			cl::Buffer a_dev(context, CL_MEM_READ_ONLY, a_size, NULL, &err_msg);
			clPrintErrorExit(err_msg, "clCreateBuffer: a_dev");
			cl::Buffer b_dev(context, CL_MEM_READ_ONLY, b_size, NULL, &err_msg);
			clPrintErrorExit(err_msg, "clCreateBuffer: b_dev");
			cl::Buffer c_basic_dev(context, CL_MEM_READ_WRITE, c_size, NULL, &err_msg);
			clPrintErrorExit(err_msg, "clCreateBuffer: c_basic_dev");
			cl::Buffer c_local_dev(context, CL_MEM_READ_WRITE, c_size, NULL, &err_msg);
			clPrintErrorExit(err_msg, "clCreateBuffer: c_local_dev");

			cl::UserEvent a_event(context, &err_msg);
			clPrintErrorExit(err_msg, "clCreateUserEvent a_event");
			cl::UserEvent b_event(context, &err_msg);
			clPrintErrorExit(err_msg, "clCreateUserEvent b_event");

			cl::UserEvent c_basic_event(context, &err_msg);
			clPrintErrorExit(err_msg, "clCreateUserEvent c_basic_event");

			cl::UserEvent c_local_event(context, &err_msg);
			clPrintErrorExit(err_msg, "clCreateUserEvent c_local_event");

			cl::NDRange local(tests[i].local[0], tests[i].local[1]);
			cl::NDRange global(alignTo(c_w, local[0]), alignTo(c_h, local[1]));

			double cpu_start = getTime();
			matrix_mul(a_host, b_host, c_host, a_w, a_h, b_w);
			double cpu_end = getTime();

			clPrintErrorExit(queue.enqueueWriteBuffer(a_dev, CL_FALSE, 0, a_size, a_host, NULL, &a_event), "clEnqueueWriteBuffer: a_dev");

			clPrintErrorExit(queue.enqueueWriteBuffer(b_dev, CL_FALSE, 0, b_size, b_host, NULL, &b_event), "clEnqueueWriteBuffer: b_dev");

			cl::Event kernel_basic_event = matrix_mul_basic(cl::EnqueueArgs(queue, global, local), a_dev, b_dev, c_basic_dev, a_w, a_h, b_w);

			clPrintErrorExit(queue.enqueueReadBuffer(c_basic_dev, CL_FALSE, 0, c_size, c_basic_host, NULL, &c_basic_event), "clEnqueueWriteBuffer: c_basic_dev");

			// synchronize queue
			clPrintErrorExit(queue.finish(), "clFinish");

			cl::Event kernel_local_event = matrix_mul_local(cl::EnqueueArgs(queue, global, local), a_dev, b_dev, c_local_dev, a_w, a_h, b_w, cl::Local(sizeof(cl_int) * a_w * local[1]), cl::Local(sizeof(cl_int) * local[0] * b_h));

			clPrintErrorExit(queue.enqueueReadBuffer(c_local_dev, CL_FALSE, 0, c_size, c_local_host, NULL, &c_local_event), "clEnqueueWriteBuffer: c_local_dev");

			// synchronize queue
			clPrintErrorExit(queue.finish(), "clFinish");

			printf("\nTest %d:\n%s\n", i, tests[i].info);
			printf(" Global work dim: %dx%d\n", global[0], global[1]);
			printf(" Basic kernel:\n");
			if (memcmp(c_basic_host, c_host, c_size) == 0)
			{
				printf("  Result: Correct\n");
				printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
					(cpu_end - cpu_start) * 1000,
					(getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_basic_event) + getEventTime(kernel_basic_event)) * 1000,
					(getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_basic_event)) * 1000,
					getEventTime(kernel_basic_event) * 1000);
			}
			else
			{
				printf("  Result: Incorrect\n");
			}
			printf(" Local kernel:\n");
			if (memcmp(c_local_host, c_host, c_size) == 0)
			{
				printf("  Result: Correct\n");
				printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
					(cpu_end - cpu_start) * 1000,
					(getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_local_event) + getEventTime(kernel_local_event)) * 1000,
					(getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_local_event)) * 1000,
					getEventTime(kernel_local_event) * 1000);
			}
			else
			{
				printf("  Result: Incorrect\n");
			}


			// deallocate host data
			free(a_host);
			free(b_host);
			free(c_host);
			free(c_basic_host);
			free(c_local_host);
			*/

	getchar();

	return 0;
}

