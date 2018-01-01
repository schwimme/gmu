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
#include <sstream>
#include <iomanip>

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

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_GPU



namespace cv_extend {

	void bilateralFilter(cv::InputArray src, cv::OutputArray dst,
		double sigmaColor, double sigmaSpace);

	void bilateralFilterImpl(cv::Mat1d src, cv::Mat1d dst,
		double sigmaColor, double sigmaSpace);


	template<typename T, typename T_, typename T__>
	inline
		T clamp(const T_ min, const T__ max, const T x)
	{
		return
			(x < static_cast<T>(min)) ? static_cast<T>(min) :
			(x < static_cast<T>(max)) ? static_cast<T>(x) :
			static_cast<T>(max);
	}

	template<typename T>
	inline
		T
		trilinear_interpolation(const cv::Mat mat,
			const double y,
			const double x,
			const double z)
	{
		const size_t height = mat.size[0];
		const size_t width = mat.size[1];
		const size_t depth = mat.size[2];

		const size_t y_index = clamp(0, height - 1, static_cast<size_t>(y));
		const size_t yy_index = clamp(0, height - 1, y_index + 1);
		const size_t x_index = clamp(0, width - 1, static_cast<size_t>(x));
		const size_t xx_index = clamp(0, width - 1, x_index + 1);
		const size_t z_index = clamp(0, depth - 1, static_cast<size_t>(z));
		const size_t zz_index = clamp(0, depth - 1, z_index + 1);
		const double y_alpha = y - y_index;
		const double x_alpha = x - x_index;
		const double z_alpha = z - z_index;

		return
			(1.0 - y_alpha) * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<T>(y_index, x_index, z_index) +
			(1.0 - y_alpha) * x_alpha       * (1.0 - z_alpha) * mat.at<T>(y_index, xx_index, z_index) +
			y_alpha       * (1.0 - x_alpha) * (1.0 - z_alpha) * mat.at<T>(yy_index, x_index, z_index) +
			y_alpha       * x_alpha       * (1.0 - z_alpha) * mat.at<T>(yy_index, xx_index, z_index) +
			(1.0 - y_alpha) * (1.0 - x_alpha) * z_alpha       * mat.at<T>(y_index, x_index, zz_index) +
			(1.0 - y_alpha) * x_alpha       * z_alpha       * mat.at<T>(y_index, xx_index, zz_index) +
			y_alpha       * (1.0 - x_alpha) * z_alpha       * mat.at<T>(yy_index, x_index, zz_index) +
			y_alpha       * x_alpha       * z_alpha       * mat.at<T>(yy_index, xx_index, zz_index);

	}


	/**
	* Implementation
	*/

	void bilateralFilter(cv::InputArray _src, cv::OutputArray _dst,
		double sigmaColor, double sigmaSpace)
	{
		cv::Mat src = _src.getMat();
		auto c = src.channels();
//		CV_Assert(c == 1);

		// bilateralFilterImpl runs with double depth, single channel
		if (src.depth() != CV_64FC1) {
			src = cv::Mat(_src.size(), CV_64FC1);
			_src.getMat().convertTo(src, CV_64FC1);
		}

		cv::Mat dst_tmp = cv::Mat(_src.size(), CV_64FC1);
		bilateralFilterImpl(src, dst_tmp, sigmaColor, sigmaSpace);

		_dst.create(dst_tmp.size(), _src.type());
		dst_tmp.convertTo(_dst.getMat(), _src.type());

	}

	void bilateralFilterImpl(cv::Mat1d src, cv::Mat1d dst,
		double sigma_color, double sigma_space)
	{
		using namespace cv;
		const size_t height = src.rows, width = src.cols;
		const size_t padding_xy = 2, padding_z = 2;
		double src_min, src_max;
		cv::minMaxLoc(src, &src_min, &src_max);

		const size_t small_height = static_cast<size_t>((height - 1) / sigma_space) + 1 + 2 * padding_xy;
		const size_t small_width = static_cast<size_t>((width - 1) / sigma_space) + 1 + 2 * padding_xy;
		const size_t small_depth = static_cast<size_t>((src_max - src_min) / sigma_color) + 1 + 2 * padding_xy;

		int data_size[] = { small_height, small_width, small_depth };
		cv::Mat data(3, data_size, CV_64FC2);
		data.setTo(0);

		// down sample
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const size_t small_x = static_cast<size_t>(x / sigma_space + 0.5) + padding_xy;
				const size_t small_y = static_cast<size_t>(y / sigma_space + 0.5) + padding_xy;
				const double z = src.at<double>(y, x) - src_min;
				const size_t small_z = static_cast<size_t>(z / sigma_color + 0.5) + padding_z;

				cv::Vec2d v = data.at<cv::Vec2d>(small_y, small_x, small_z);
				v[0] += src.at<double>(y, x);
				v[1] += 1.0;
				data.at<cv::Vec2d>(small_y, small_x, small_z) = v;
			}
		}

		// convolution
		cv::Mat buffer(3, data_size, CV_64FC2);
		buffer.setTo(0);
		int offset[3];
		offset[0] = &(data.at<cv::Vec2d>(1, 0, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
		offset[1] = &(data.at<cv::Vec2d>(0, 1, 0)) - &(data.at<cv::Vec2d>(0, 0, 0));
		offset[2] = &(data.at<cv::Vec2d>(0, 0, 1)) - &(data.at<cv::Vec2d>(0, 0, 0));

		for (int dim = 0; dim < 3; ++dim) { // dim = 3 stands for x, y, and depth
			const int off = offset[dim];
			for (int ittr = 0; ittr < 2; ++ittr) {
				cv::swap(data, buffer);

				for (int y = 1; y < small_height - 1; ++y) {
					for (int x = 1; x < small_width - 1; ++x) {
						cv::Vec2d *d_ptr = &(data.at<cv::Vec2d>(y, x, 1));
						cv::Vec2d *b_ptr = &(buffer.at<cv::Vec2d>(y, x, 1));
						for (int z = 1; z < small_depth - 1; ++z, ++d_ptr, ++b_ptr) {
							cv::Vec2d b_prev = *(b_ptr - off), b_curr = *b_ptr, b_next = *(b_ptr + off);
							*d_ptr = (b_prev + b_next + 2.0 * b_curr) / 4.0;
						} // z
					} // x
				} // y

			} // ittr
		} // dim

		  // upsample

		for (cv::MatIterator_<cv::Vec2d> d = data.begin<cv::Vec2d>(); d != data.end<cv::Vec2d>(); ++d)
		{
			(*d)[0] /= (*d)[1] != 0 ? (*d)[1] : 1;
		}

		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const double z = src.at<double>(y, x) - src_min;
				const double px = static_cast<double>(x) / sigma_space + padding_xy;
				const double py = static_cast<double>(y) / sigma_space + padding_xy;
				const double pz = static_cast<double>(z) / sigma_color + padding_z;
				dst.at<double>(y, x) = trilinear_interpolation<cv::Vec2d>(data, py, px, pz)[0];
			}
		}

	}

} // end of namespace cv_extend


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
	bool benchmark = false;

	/*
	 * Naètení parametrù programu.
	 */
	if (argc != 5)
	{
		// Benchmark
		// - Do názvu souboru se vloží doba zpracování.
		// - Použije se výkonná GK.
		if (argc == 6 && std::string(argv[5]) == "-b")
		{
			benchmark = true;
		}
		else
		{
			printHelp();
			exit(1);
		}
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

	MyMat img_dest_opt(dest_rows, dest_cols);
	cl_float3 * img_dest_opt_fl3 = img_dest_opt.getData();

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
	if (benchmark)
	{
		//platforms[1].getDevices(SELECTED_DEVICE_TYPE, &platform_devices);
		//selected_device = platform_devices[0];
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
	clPrintErrorExit(err_msg, "_test");

	auto bilateralFilter_basic = cl::make_kernel<
		cl::Buffer&, 
		cl::Buffer&, 
		const cl_int&, 
		const cl_int&,
		const cl_int&, 
		const cl_float&
	>(program, "bilateralFilter_basic", &err_msg);
	clPrintErrorExit(err_msg, "_basic");

	auto bilateralFilter_optimized = cl::make_kernel<
		cl::Buffer&,
		cl::Buffer&,
		const cl_int&,
		const cl_int&,
		const cl_int&,
		const cl_float&
	>(program, "bilateralFilter_optimized", &err_msg);
	clPrintErrorExit(err_msg, "_optimized");

	MyMat opt_output;
	cv::Mat input_gray_scale, tmp;

	cv::Mat im_lab_uint8;
	img_source.getMat().convertTo(im_lab_uint8, CV_8UC3, 255.0);

	cv::cvtColor(im_lab_uint8, tmp, 57 /*cl::COLOR_Lab2RGB*/);
	cv::cvtColor(tmp, input_gray_scale, 7 /*cl::COLOR_RGB2GRAY*/);
	cv::imwrite("origin_gray_scale.png", input_gray_scale);
	cv_extend::bilateralFilter(input_gray_scale, opt_output.getMat(), param_range, param_space);
	cv::imwrite("origin_gray_scale_filtered_optimized.png", opt_output.getMat());

	/*
	 * Spuštìní kernelu.
	 */
	cl::Buffer img_source_dev(context, CL_MEM_READ_ONLY, (size_t)img_source.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_source");
	cl::Buffer img_dest1_dev(context, CL_MEM_READ_WRITE, (size_t)img_dest1.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_dest1");
	cl::Buffer img_dest_opt_dev(context, CL_MEM_READ_WRITE, (size_t)img_dest_opt.getDataSize(), NULL, &err_msg);
	clPrintErrorExit(err_msg, "clCreateBuffer: img_dest_opt");

	cl::UserEvent img_source_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_source");
	cl::UserEvent img_dest1_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_dest1");
	cl::UserEvent img_dest_opt_event(context, &err_msg);
	clPrintErrorExit(err_msg, "clCreateUserEvent img_dest_opt");

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

	cl::Event kernel_optimized_event = bilateralFilter_optimized(
		cl::EnqueueArgs(queue, global, local),
		img_source_dev,
		img_dest_opt_dev,
		img_dest1.getMat().cols,
		img_dest1.getMat().rows,
		param_space,
		param_range
	);

	clPrintErrorExit(queue.enqueueReadBuffer(
		img_dest_opt_dev,
		CL_FALSE,
		0,
		img_dest_opt.getDataSize(),
		img_dest_opt_fl3,
		NULL,
		&img_dest_opt_event
	), "clEnqueueReadBuffer: img_dest1");

	// synchronize queue
	clPrintErrorExit(queue.finish(), "clFinish");


	/*
	 * Statistika.
	 */
	printf("Timers: ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
		(getEventTime(img_source_event) + getEventTime(img_dest1_event) + getEventTime(kernel_test_event)) * 1000,
		(getEventTime(img_source_event) + getEventTime(img_dest1_event)) * 1000,
		getEventTime(kernel_test_event) * 1000);

	/*
	* Uložení výsledku.
	*/
	if (benchmark)
	{
		// Odhodíme pøíponu
		std::string prefix = outputFileName.substr(0, outputFileName.length() - 4);
		
		unsigned int time = (unsigned int)(getEventTime(kernel_test_event) * 1000);
		
		std::stringstream ss;
		ss << prefix << "_t" << time << ".png";
		outputFileName = ss.str();
	}

	img_dest1.setData(img_dest1_fl3);
	img_dest1.saveImageToFile(outputFileName);


	if (!benchmark)
	{
		getchar();
	}

	exit(0);
	//return 0; // Na NVIDIA se neukonèilo, ale èekalo.
}

