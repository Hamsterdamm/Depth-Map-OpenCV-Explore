#include <iostream>
#include <opencv2\calib3d.hpp>
#include <opencv\cv.h>
#include <opencv2\imgcodecs.hpp>
#include <opencv2\highgui.hpp>

#include <stdio.h>
#include <stdarg.h>

using namespace cv;

const char *windowDisparity = "Disparity";

int main()
{

	cv::String image_Name_1("data/im0.png"); //расположение 1го кадра
	cv::String image_Name_2("data/im1.png"); //расположение 2го кадра
	cv::Mat imgLeft = imread(image_Name_1, cv::IMREAD_GRAYSCALE);
	cv::Mat imgRight = imread(image_Name_2, cv::IMREAD_GRAYSCALE);


	//-- And create the image in which we will save our disparities
	Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
	Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

	if (imgLeft.empty() || imgRight.empty())
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- 2. Call the constructor for StereoBM
	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 9; /**< Size of the block window. Must be odd */

	Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);


	//-- 3. Calculate the disparity image
	sbm->compute(imgLeft, imgRight, imgDisparity16S);

	//-- Check its extreme values
	double minVal; double maxVal;

	minMaxLoc(imgDisparity16S, &minVal, &maxVal);

	//-- 4. Display it as a CV_8UC1 image
	imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255 / (maxVal - minVal));

//	namedWindow(windowDisparity, WINDOW_NORMAL);

//	cv::ShowManyImages("Images", 2, img1, img2);
	imshow("imgLeft", imgLeft);
	imshow("imgRight", imgRight);
	imshow("disp", imgDisparity8U);

	waitKey(0);

	return 0;
}