#pragma comment( lib, "OpenCL" )

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <CL/cl.hpp>
#include "oclHelper.h"

#include <functional>


#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU


const char SOURCE_FILE[] = "bilateral_filter.cl";


struct scope_guard
{
	scope_guard(std::function<void()> f) :
		m_f(f)
	{}

	~scope_guard()
	{
		if (m_dissmis == false)
		{
			try { m_f(); }
			catch (...) {}
		}
	}

	void dissmis()
	{
		m_dissmis = true;
	}

private:
	bool m_dissmis = false;
	std::function<void()> m_f;
};


#define _CONCAT_1( x, y ) x##y
#define _CONCAT_2( x, y ) _CONCAT_1(x, y)
#define MAKE_SCOPE_GUARD(f) scope_guard _CONCAT_2(guard_, __COUNTER__)(f);


cl_int * read_image(const char* file, size_t size)
{
	// KTTODO impl
	return genRandomBuffer(size);
}


void store_image(const char* file, cl_int *data, size_t size)
{
	// KTTODO impl
}


int main(int argc, char* argv[])
{
	// Do params:
	char* params_filename = nullptr;
	int params_widht = 0;
	int params_height = 0;
	int params_euclidean_delta = 0;
	int params_filter_radius = 0;
	bool correct_params = false;
	do
	{
		if (argc != 6) break;
		params_filename = argv[1];

		auto string_to_int = [] (std::string str, int& val) -> bool 
		{
			try
			{
				std::string::size_type p;
				val = std::stoi(str, &p);
				return p == str.size();
			}
			catch (...)
			{
				return false;
			}
		};
		
		if (string_to_int(argv[2], params_widht) == false) break;
		if (string_to_int(argv[3], params_height) == false) break;
		if (string_to_int(argv[4], params_euclidean_delta) == false) break;
		if (string_to_int(argv[5], params_filter_radius) == false) break;

		correct_params = true;
	} while (0);

	if (correct_params == false)
	{
		printf("Params error, expected format");
		printf("./proj <input> <width> <height> <euclidean_delta> <filter_radius>");
		return 1;
	}

	cl_int err_msg;

	// Get platforms:
	std::vector<cl::Platform> platforms;
	clPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");

	// Get device:
	cl::Device selected_device;
	bool device_found = false;
	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		std::vector<cl::Device> platform_devices;
		clPrintErrorExit(platforms[i].getDevices(SELECTED_DEVICE_TYPE, &platform_devices), "getDevices");
		if (platform_devices.size() != 0)
		{
			device_found = true;
			selected_device = platform_devices[0];
			break;
		}
	}

	// Do we have device?
	if (!device_found)
	{
		clPrintErrorExit(CL_DEVICE_NOT_FOUND, "GPU device");
	}

	printf("Selected device name: %s.\n", selected_device.getInfo<CL_DEVICE_NAME>().c_str());

	// Create context:
	cl::Context context(selected_device, NULL, NULL, NULL, &err_msg);
	clPrintErrorExit(err_msg, "cl::Context");

	// Create command queue:
	cl::CommandQueue queue(context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err_msg);
	clPrintErrorExit(err_msg, "cl::CommandQueue");

	// Build program from source:
	char *program_source = readFile(SOURCE_FILE);
	cl::Program::Sources sources;
	sources.push_back(std::pair<const char *, size_t>(program_source, 0));

	cl::Program program(context, sources, &err_msg);
	clPrintErrorExit(err_msg, "clCreateProgramWithSource");

	if ((err_msg = program.build(std::vector<cl::Device>(1, selected_device), "", NULL, NULL)) != CL_SUCCESS)
	{
		if (err_msg == CL_BUILD_PROGRAM_FAILURE)
		{
			printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, nullptr).c_str());
		}

		clPrintErrorExit(err_msg, "clBuildProgram");
	}

	// Create kernels:
	cl::make_kernel<
		cl::Buffer&, 
		cl::Buffer&, 
		cl_int, 
		cl_int/*, 
		cl::LocalSpaceArg&, 
		cl::LocalSpaceArg&*/
	> bilateral_filter_kernel
		= 
		cl::make_kernel<
			cl::Buffer&, 
			cl::Buffer&, 
			cl_int,
			cl_int/*,
			cl::LocalSpaceArg&,
			cl::LocalSpaceArg&*/
		>(program, "bilateral_filter", &err_msg);
	clPrintErrorExit(err_msg, "create kernel bilateral_filter");

	cl::make_kernel<
		cl::Buffer&,
		cl::Buffer&,
		cl_int,
		cl_int /*,
		cl::LocalSpaceArg&,
		cl::LocalSpaceArg&*/
	> optimized_bilateral_filter_kernel
		=
		cl::make_kernel<
			cl::Buffer&,
			cl::Buffer&,
			cl_int,
			cl_int/*,
			cl::LocalSpaceArg&,
			cl::LocalSpaceArg&*/
		>(program, "optimized_bilateral_filter", &err_msg);
	clPrintErrorExit(err_msg, "create kernel optimized_bilateral_filter");


	// Open source file:
	size_t image_size = params_widht * params_height; // KTTODO should be here sizeof (cl_int)?

	cl_int *readed_image = read_image(params_filename, image_size);
	MAKE_SCOPE_GUARD([readed_image] { free(readed_image); });

	// Prepare output:
	cl_int *readed_image_filtered = (cl_int *)malloc(image_size * sizeof(cl_int));
	MAKE_SCOPE_GUARD([readed_image_filtered] { free(readed_image_filtered); });

	cl_int *readed_image_filtered_with_optimization = (cl_int *)malloc(image_size * sizeof(cl_int));
	MAKE_SCOPE_GUARD([readed_image_filtered_with_optimization] { free(readed_image_filtered_with_optimization); });

	// Create buffers:
	// - input:
	cl::Buffer image_buffer(context, CL_MEM_READ_ONLY, image_size, NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: image_buffer");

	// - output:
	cl::Buffer image_buffer_filtered(context, CL_MEM_READ_WRITE, image_size, NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: image_buffer_filtered");
	cl::Buffer image_buffer_filtered_with_optimization(context, CL_MEM_READ_WRITE, image_size, NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: image_buffer_filtered_with_optimization");

	// Create events:
	cl::UserEvent write_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent write_event");
	cl::UserEvent filtered_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent filtered_event");
	cl::UserEvent filtered_optimized_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent filtered_optimized_event");

	// Ranges: KTTODO
	cl::NDRange local_range(1, 2);
	cl::NDRange global_range(alignTo(1, 2), alignTo(3, 4));

	// Write data to buffer:
	err_msg = queue.enqueueWriteBuffer(image_buffer, CL_FALSE, 0, image_size, readed_image, NULL, &write_event);
	clPrintErrorExit(err_msg, "clEnqueueWriteBuffer: raeded_image");

	// Run kernel:
	cl::Event no_opt_start_event
		= bilateral_filter_kernel(
			cl::EnqueueArgs(queue, global_range, local_range), 
			image_buffer, 
			image_buffer_filtered, 
			640, 
			480 /*, 
			cl::Local(sizeof(cl_int) * 10), 
			cl::Local(sizeof(cl_int) * 10) */
		);

	// Read output data:
	err_msg = queue.enqueueReadBuffer(image_buffer_filtered, CL_FALSE, 0, image_size, readed_image_filtered, NULL, &filtered_event);
	clPrintErrorExit(err_msg, "clEnqueueWriteBuffer: c_basic_dev");

	// Synchronize queue:
	clPrintErrorExit(queue.finish(), "clFinish");

	// Run kernel:
	cl::Event opt_start_event 
		= optimized_bilateral_filter_kernel(
			cl::EnqueueArgs(queue, global_range, local_range),
			image_buffer,
			image_buffer_filtered_with_optimization,
			640,
			480/*, 
			cl::Local(sizeof(cl_int) * a_w * local[1]),
			cl::Local(sizeof(cl_int) * local[0] * b_h) */
		);

	// Read output data:
	err_msg = queue.enqueueReadBuffer(image_buffer_filtered_with_optimization, CL_FALSE, 0, image_size, readed_image_filtered_with_optimization, NULL, &filtered_optimized_event);
	clPrintErrorExit(err_msg, "clEnqueueWriteBuffer: c_local_dev");

	// Synchronize queue
	clPrintErrorExit(queue.finish(), "clFinish");

	// Write output to file:
	store_image("filtered.png", readed_image_filtered, image_size);
	store_image("filtered_optimized.png", readed_image_filtered_with_optimization, image_size);

	return 0;
}

