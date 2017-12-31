/*
 * GMU projekt - Bilaterální filtr
 *
 * Autoøi: Tomáš Pelka (xpelka01).
 */

#include "MyMat.hpp"


MyMat::MyMat() : flData(NULL)
{
	mat = cv::Mat(0, 0, CV_32FC3);
}


MyMat::MyMat(int rows, int cols) : flData(NULL)
{
	mat = cv::Mat(rows, cols, CV_32FC3);
}


MyMat::~MyMat()
{
	freeFlData();
}


cv::Mat& MyMat::getMat(void)
{
	return mat;
}


void MyMat::loadImageFromFile(std::string fileName)
{
	/*
     * Naète ze zadaného souboru obrázek a provede potøebné konverze.
	 *
	 * Tato funkce provede pøevod do barevného prostoru CIE-Lab s datovými typy float.
	 * Typ float bude v rozsahu 0-1.
	 */

	cv::Mat im_bgr_uint8 = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);

	if (!im_bgr_uint8.data)
	{
		throw;
	}

	cv::Mat im_lab_uint8;
	cv::cvtColor(im_bgr_uint8, im_lab_uint8, cv::COLOR_BGR2Lab);

	im_lab_uint8.convertTo(mat, CV_32FC3, 1.0 / 255);
}


void MyMat::saveImageToFile(std::string fileName)
{
	/*
	 * Provede potøebné konverze a uloží obrázek do zadaného souboru.
	 *
	 * Tato funkce provede pøevod z barevného prostoru CIE-Lab s datovými typy float.
	 * Pøevádí se do barevného prostoru BGR s datovými typy UINT8.
	 * Typ float se pøedpokládá v rozsahu 0-1.
	 */
	cv::Mat im_lab_uint8;
	mat.convertTo(im_lab_uint8, CV_8UC3, 255.0);

	cv::Mat im_bgr_uint8;
	cv::cvtColor(im_lab_uint8, im_bgr_uint8, cv::COLOR_Lab2BGR);

	cv::imwrite(fileName, im_bgr_uint8);
}


cl_float3 * MyMat::getData(void)
{
	freeFlData();

	flData = (cl_float3 *)malloc(getDataSize());

	cv::Vec3f temp;

	for (int row = 0; row < mat.rows; row++)
	{
		for (int col = 0; col < mat.cols; col++)
		{
			temp = mat.at<cv::Vec3f>(cv::Point(col, row));

			flData[row * mat.cols + col].x = temp.val[0]; // L
			flData[row * mat.cols + col].y = temp.val[1]; // A
			flData[row * mat.cols + col].z = temp.val[2]; // B
		}
	}

	return flData;
}


void MyMat::setData(cl_float3 * data)
{
	cl_float3 temp;

	for (int row = 0; row < mat.rows; row++)
	{
		for (int col = 0; col < mat.cols; col++)
		{
			temp = flData[row * mat.cols + col];

			mat.at<cv::Vec3f>(cv::Point(col, row)) = cv::Vec3f(temp.x, temp.y, temp.z);;
		}
	}
}


int MyMat::getDataSize()
{
	return sizeof(cl_float3) * mat.rows * mat.cols;
}


void MyMat::freeFlData(void)
{
	if (flData)
	{
		free(flData);
		flData = NULL;
	}
}