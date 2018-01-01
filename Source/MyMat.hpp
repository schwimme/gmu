/*
 * GMU projekt - Bilaterální filtr
 *
 * Autoøi: Tomáš Pelka (xpelka01).
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CL/cl.hpp>

#include <string>

 /**
  * Tøída zabalující OpenCV::Mat
  * + doplnìní konverzních funkcí.
  */
class MyMat
{
protected:
	cv::Mat mat;
	cl_float3 * flData;

private:
	/**
	 * Uvolní pamì alokovanou pro flData.
	 */
	void freeFlData(void);

public:
	MyMat();
	MyMat(int rows, int cols);
	~MyMat();

	/**
	 * Vrátí referenci na objekt mat.
	 * Aby bylo moné i z venku provádìt OpenCV operace.
	 */
	cv::Mat& getMat(void);

	/**
	 * Naète obrázek ze souboru.
	 * Obrazová data jsou naètena do mat.
	 */
	void loadImageFromFile(std::string fileName);

	/**
	 * Uloí obrázek do zadaného souboru.
	 * Obrazová data se berou z mat.
	 */
	void saveImageToFile(std::string fileName);

	/**
	 * Nasype data z obrazové matice mat do pole flData, které alokuje.
	 */
	cl_float3 * getData(void);

	/**
	 * Nasype data z pole flData do obrazové matice mat.
	 */
	void setData(cl_float3 * data);

	/**
	 * Vrátí velikost flData v bajtech.
	 */
	int getDataSize();
};

