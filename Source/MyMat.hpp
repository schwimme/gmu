#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CL/cl.hpp>

#include <string>

#pragma once
class MyMat
{
protected:
	cv::Mat mat;
	cl_float3 * flData;

private:
	void freeFlData(void);

public:
	MyMat();
	MyMat(int rows, int cols);
	~MyMat();

	cv::Mat& getMat(void);

	void loadImageFromFile(std::string fileName);
	void saveImageToFile(std::string fileName);

	cl_float3 * getData(void);
	void setData(cl_float3 * data);
	int getDataSize();
};

