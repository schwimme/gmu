/*
 * GMU projekt - Bilater�ln� filtr
 *
 * Auto�i: Tom� Pelka (xpelka01).
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CL/cl.hpp>

#include <string>

 /**
  * T��da zabaluj�c� OpenCV::Mat
  * + dopln�n� konverzn�ch funkc�.
  */
class MyMat
{
protected:
	cv::Mat mat;
	cl_float3 * flData;

private:
	/**
	 * Uvoln� pam� alokovanou pro flData.
	 */
	void freeFlData(void);

public:
	MyMat();
	MyMat(int rows, int cols);
	~MyMat();

	/**
	 * Vr�t� referenci na objekt mat.
	 * Aby bylo mo�n� i z venku prov�d�t OpenCV operace.
	 */
	cv::Mat& getMat(void);

	/**
	 * Na�te obr�zek ze souboru.
	 * Obrazov� data jsou na�tena do mat.
	 */
	void loadImageFromFile(std::string fileName);

	/**
	 * Ulo�� obr�zek do zadan�ho souboru.
	 * Obrazov� data se berou z mat.
	 */
	void saveImageToFile(std::string fileName);

	/**
	 * Nasype data z obrazov� matice mat do pole flData, kter� alokuje.
	 */
	cl_float3 * getData(void);

	/**
	 * Nasype data z pole flData do obrazov� matice mat.
	 */
	void setData(cl_float3 * data);

	/**
	 * Vr�t� velikost flData v bajtech.
	 */
	int getDataSize();
};

