#include "MyMat.hpp"


MyMat::MyMat() : flData(NULL)
{
}


MyMat::~MyMat()
{
	freeFlData();
}


/**
* @brief Na�te ze zadan�ho souboru obr�zek a provede pot�ebn� konverze.
*
* Tato funkce provede p�evod do barevn�ho prostoru CIE-Lab s datov�mi typy float.
* Typ float bude v rozsahu 0-1.
*
* @param N�zev souboru.
*/
void MyMat::loadImageFromFile(std::string fileName)
{
	cv::Mat im_bgr_uint8 = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);

	cv::Mat im_lab_uint8;
	cv::cvtColor(im_bgr_uint8, im_lab_uint8, cv::COLOR_BGR2Lab);

	im_lab_uint8.convertTo(mat, CV_32FC3, 1.0 / 255);
}

/**
* @brief Provede pot�ebn� konverze a ulo�� obr�zek do zadan�ho souboru.
*
* Tato funkce provede p�evod z barevn�ho prostoru CIE-Lab s datov�mi typy float.
* P�ev�d� se do barevn�ho prostoru BGR s datov�mi typy UINT8.
* Typ float se p�edpokl�d� v rozsahu 0-1.
*
* @param N�zev souboru.
*/
void MyMat::saveImageToFile(std::string fileName)
{
	cv::Mat im_lab_uint8;
	mat.convertTo(im_lab_uint8, CV_8UC3, 255.0);

	cv::Mat im_bgr_uint8;
	cv::cvtColor(im_lab_uint8, im_bgr_uint8, cv::COLOR_Lab2BGR);

	cv::imwrite(fileName, im_bgr_uint8);
}

cl_float3 * MyMat::getData(void)
{
	freeFlData();

	flData = (cl_float3 *)malloc(sizeof(cl_float3) * mat.rows * mat.cols);

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

void MyMat::freeFlData(void)
{
	if (flData)
	{
		free(flData);
		flData = NULL;
	}
}