#pragma comment( lib, "OpenCL" )

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <CL/cl.hpp>
#include "oclHelper.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

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


cv::Mat read_image(const char* file)
{
	cv::Mat img = cv::imread(file, CV_LOAD_IMAGE_COLOR);
	
	// KTTODO - check color model
	cv::cvtColor

	return img;
}


void store_image(const char* file, const cv::Mat& img)
{
	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9); // KTTODO ??

	cv::imwrite(file, img, compression_params);
}


int main(int argc, char* argv[])
{
	// Do params:
	char* params_filename = nullptr;
	float params_euclidean_delta = 0;
	int params_filter_radius = 0;
	bool correct_params = false;
	do
	{
		if (argc != 4) break;
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

		auto string_to_float = [](std::string str, float& val) -> bool
		{
			try
			{
				std::string::size_type p;
				val = std::stof(str, &p);
				return p == str.size();
			}
			catch (...)
			{
				return false;
			}
		};

		if (string_to_float(argv[2], params_euclidean_delta) == false) break;
		if (string_to_int(argv[3], params_filter_radius) == false) break;

		correct_params = true;
	} while (0);

	if (correct_params == false)
	{
		printf("Params error, expected format");
		printf("./proj <input> <euclidean_delta> <filter_radius>");
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

	if ((err_msg = program.build({ selected_device }, "", NULL, NULL)) != CL_SUCCESS)
	{
		if (err_msg == CL_BUILD_PROGRAM_FAILURE)
		{
			printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, nullptr).c_str());
		}

		clPrintErrorExit(err_msg, "clBuildProgram");
	}

	// Create kernels:
	auto bilateral_filter_kernel = 
		cl::make_kernel<
			cl::Buffer&, 
			cl::Buffer&, 
			cl_int,
			cl_int,
			cl_float,
			cl_int/*,
			cl::LocalSpaceArg&,
			cl::LocalSpaceArg&*/
		>(program, "bilateral_filter", &err_msg);
	clPrintErrorExit(err_msg, "create kernel bilateral_filter");

	auto optimized_bilateral_filter_kernel =
		cl::make_kernel<
			cl::Buffer&,
			cl::Buffer&,
			cl_int,
			cl_int,
			cl_float,
			cl_int/*,
			cl::LocalSpaceArg&,
			cl::LocalSpaceArg&*/
		>(program, "optimized_bilateral_filter", &err_msg);
	clPrintErrorExit(err_msg, "create kernel optimized_bilateral_filter");

	// Open source file:
	cv::Mat readed_image = read_image(params_filename);
	size_t image_size_in_pixels = readed_image.cols * readed_image.rows;
	size_t image_size_in_bytes = image_size_in_pixels * sizeof(8);

	// Prepare output:
	uchar *readed_image_filtered = (uchar *)malloc(image_size_in_bytes);
	MAKE_SCOPE_GUARD([readed_image_filtered] { free(readed_image_filtered); });

	uchar *readed_image_filtered_with_optimization = (uchar *)malloc(image_size_in_bytes);
	MAKE_SCOPE_GUARD([readed_image_filtered_with_optimization] { free(readed_image_filtered_with_optimization); });

	// Create buffers:
	// - input:
	cl::Buffer image_buffer(context, CL_MEM_READ_ONLY, image_size_in_bytes, NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: image_buffer");

	// - output:
	cl::Buffer image_buffer_filtered(context, CL_MEM_READ_WRITE, image_size_in_bytes, NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: image_buffer_filtered");

	cl::Buffer image_buffer_filtered_with_optimization(context, CL_MEM_READ_WRITE, image_size_in_bytes, NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: image_buffer_filtered_with_optimization");

	// Define range: // KTTODO reduce working groups:
	cl::NDRange global_range(readed_image.rows, readed_image.cols);

	// Write data from matrix to buffer:
	err_msg = queue.enqueueWriteBuffer(image_buffer, CL_FALSE, 0, image_size_in_bytes, readed_image.data);
	clPrintErrorExit(err_msg, "clEnqueueWriteBuffer: raeded_image");

	// Run kernel:
	bilateral_filter_kernel(
		cl::EnqueueArgs(queue, global_range), 
		image_buffer, 
		image_buffer_filtered, 
		readed_image.rows, 
		readed_image.cols,
		params_euclidean_delta,
		params_filter_radius
		/*,
			cl::Local(sizeof(cl_int) * 10), 
			cl::Local(sizeof(cl_int) * 10) */
	);

	// Read output data:
	err_msg = queue.enqueueReadBuffer(image_buffer_filtered, CL_FALSE, 0, image_size_in_bytes, readed_image_filtered);
	clPrintErrorExit(err_msg, "clEnqueueWriteBuffer: image_buffer_filtered");

	// Synchronize queue:
	clPrintErrorExit(queue.finish(), "clFinish");

	// Run kernel:
	optimized_bilateral_filter_kernel(
		cl::EnqueueArgs(queue, global_range),
		image_buffer,
		image_buffer_filtered_with_optimization,
		readed_image.rows,
		readed_image.cols,
		params_euclidean_delta,
		params_filter_radius
		/*,
			cl::Local(sizeof(cl_int) * a_w * local[1]),
			cl::Local(sizeof(cl_int) * local[0] * b_h) */
	);

	// Read output data:
	err_msg = queue.enqueueReadBuffer(image_buffer_filtered_with_optimization, CL_FALSE, 0, image_size_in_bytes, readed_image_filtered_with_optimization);
	clPrintErrorExit(err_msg, "clEnqueueWriteBuffer: image_buffer_filtered");

	// Synchronize queue
	clPrintErrorExit(queue.finish(), "clFinish");
	// Write output to file:
	cv::Mat tmp = readed_image;
	tmp.data = readed_image_filtered;
	store_image("filtered.png", tmp);

	tmp.data = readed_image_filtered_with_optimization;
	store_image("filtered_optimized.png", tmp);

	return 0;
}

