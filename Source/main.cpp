/*
 * GMU projekt - Bilaterální filtr
 *
 * Autoøi: Tomáš Pelka (xpelka01), Karol Troška (xtrosk00).
 */

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

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU

void printHelp(void)
{
	std::cerr << "Špatné parametry spuštìní programu. Oèekávám" << std::endl <<
		"program.exe vstupniObraz radius barvy vystupniObraz" << std::endl <<
		"   vstupniObraz  Cesta ke vstupnímu obrázku." << std::endl <<
		"   radius        Parametr filtru - prostorový (radius)." << std::endl <<
		"   vstupniObraz  Parametr filtru - podobnost barev." << std::endl <<
		"   vystupniObraz Cesta k vstupnímu obrázku." << std::endl;
}

int main(int argc, char* argv[])
{
	/*
	 * Naètení parametrù programu.
	 */
	if (argc != 5)
	{
		printHelp();
		exit(1);
	}

	std::string inputFileName(argv[1]);
	std::string outputFileName(argv[4]);

	const int param_space = atoi(argv[2]);
	const float param_range = (float)atof(argv[3]);

	if (inputFileName == "" || outputFileName == "" || param_space < 0 || param_range < 0)
	{
		printHelp();
		exit(1);
	}

	std::cout <<
		"===============================" << std::endl <<
		"=== GMU - bilaterální filtr ===" << std::endl <<
		"===============================" << std::endl <<
		"Prostorový parametr (radius): " << param_space << std::endl <<
		"Barevný parametr (podobnost barev): " << param_range << std::endl;


	/*
	 * Pøíprava vstupního obrázku.
	 */
	MyMat img_source;
	try {
		img_source.loadImageFromFile(inputFileName);
	}
	catch (...)
	{
		std::cerr << "Input image could not be loaded." << std::endl;
		exit(1);
	}

	cl_float3 * img_source_fl3 = img_source.getData();


	/*
	 * Pøíprava výstupního obrázku.
	 */
	int dest_rows = img_source.getMat().rows - param_space * 2;
	int dest_cols = img_source.getMat().cols - param_space * 2;

	MyMat img_dest1(dest_rows, dest_cols);
	cl_float3 * img_dest1_fl3 = img_dest1.getData();


	/*
	 * Výbìr výpoèetní platformy.
	 */
	std::vector<cl::Platform> platforms;
	std::vector<cl::Device> platform_devices;

	cl_int err_msg, err_msg2;

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

	// Mám na notebooku 2 grafiky, tímto vynutím kartu NVIDIA.
	//platforms[1].getDevices(SELECTED_DEVICE_TYPE, &platform_devices);
	//selected_device = platform_devices[0];

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


	/*
	 * Sestavení OpenCL programu (kernelu).
	 */
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
	auto bilateralFilter_test = cl::make_kernel<
		cl::Buffer&,
		cl::Buffer&,
		const cl_int&,
		const cl_int&,
		const cl_int&,
		const cl_float&
	>(program, "bilateralFilter_test", &err_msg);

	auto bilateralFilter_basic = cl::make_kernel<
		cl::Buffer&, 
		cl::Buffer&, 
		const cl_int&, 
		const cl_int&,
		const cl_int&, 
		const cl_float&
	>(program, "bilateralFilter_basic", &err_msg);


	/*
	 * Spuštìní kernelu.
	 */
	cl::Buffer img_source_dev(context, CL_MEM_READ_ONLY, (size_t)img_source.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_source");
	cl::Buffer img_dest1_dev(context, CL_MEM_READ_WRITE, (size_t)img_dest1.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_dest1");

	cl::UserEvent img_source_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_source");
	cl::UserEvent img_dest1_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_dest1");

	cl::NDRange local(16, 16);
	cl::NDRange global(
		alignTo(img_dest1.getMat().cols, local[0]),
		alignTo(img_dest1.getMat().rows, local[1])
	);

	clPrintErrorExit(queue.enqueueWriteBuffer(
		img_source_dev,
		CL_FALSE, 
		0, 
		img_source.getDataSize(),
		img_source_fl3,
		NULL, 
		&img_source_event
	), "clEnqueueWriteBuffer: img_source");

	cl::Event kernel_test_event = bilateralFilter_basic(
		cl::EnqueueArgs(queue, global, local),
		img_source_dev,
		img_dest1_dev,
		img_dest1.getMat().cols,
		img_dest1.getMat().rows,
		param_space,
		param_range
	);

	clPrintErrorExit(queue.enqueueReadBuffer(
		img_dest1_dev,
		CL_FALSE,
		0, 
		img_dest1.getDataSize(),
		img_dest1_fl3,
		NULL,
		&img_dest1_event
	), "clEnqueueReadBuffer: img_dest1");

	// synchronize queue
	clPrintErrorExit(queue.finish(), "clFinish");

	
	/*
	 * Uložení výsledku.
	 */
	img_dest1.setData(img_dest1_fl3);
	img_dest1.saveImageToFile(outputFileName);


	/*
	 * Statistika.
	 */
	printf("Timers: ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
		(getEventTime(img_source_event) + getEventTime(img_dest1_event) + getEventTime(kernel_test_event)) * 1000,
		(getEventTime(img_source_event) + getEventTime(img_dest1_event)) * 1000,
		getEventTime(kernel_test_event) * 1000);


	getchar();

	return 0;
}

