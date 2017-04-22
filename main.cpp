#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"
#include <iostream>
#include <string>

#include <stdio.h>
#include <stdarg.h>

using namespace cv;
using namespace cv::ximgproc;
using namespace std;

const char *windowDisparity = "Disparity";

int main()
{

	cv::String image_Name_1("data/im0.png"); //расположение 1го кадра
	cv::String image_Name_2("data/im1.png"); //расположение 2го кадра
	cv::Mat imgLeft = imread(image_Name_1/*, cv::IMREAD_GRAYSCALE*/);
	cv::Mat imgRight = imread(image_Name_2/*, cv::IMREAD_GRAYSCALE*/);

	double lambda = 8000.0;
	double sigma = 1.5;

	


	//-- And create the image in which we will save our disparities
	Mat imgDisparity16SL/* = Mat(imgLeft.rows, imgLeft.cols, CV_16S)*/;
	Mat imgDisparity16SL_prefiter;
	Mat imgDisparity8UL/* = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1)*/;
	Mat imgDisparity16SR/* = Mat(imgLeft.rows, imgLeft.cols, CV_16S)*/;
	Mat imgDisparity8UR/* = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1)*/;
	Mat filtered_disp16S/* = Mat(imgLeft.rows, imgLeft.cols, CV_16S)*/;
	Mat filtered_disp8U/* = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1)*/;

	Ptr<DisparityWLSFilter> wls_filter;
	

	if (imgLeft.empty() || imgRight.empty())
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- 2. Call the constructor for StereoBM
	int ndisparities = 16 * 5;   /**< Range of disparity */
	int SADWindowSize = 7; /**< Size of the block window. Must be odd */

	Ptr<StereoBM> sbmL = StereoBM::create(ndisparities, SADWindowSize);
	Ptr<StereoBM> sbmL_prefiter = StereoBM::create(ndisparities, SADWindowSize);
	Ptr<StereoBM> sbmR = StereoBM::create(ndisparities, SADWindowSize);
	

	//Mat conf_map = Mat(imgLeft.rows, imgLeft.cols, CV_8U);
	//conf_map = Scalar(255);

	cvtColor(imgLeft, imgLeft, COLOR_BGR2GRAY);
	cvtColor(imgRight, imgRight, COLOR_BGR2GRAY);


	//-- 3. Calculate the disparity image

	
	wls_filter = createDisparityWLSFilter(sbmL_prefiter);

	sbmL->compute(imgLeft, imgRight, imgDisparity16SL);
	sbmL_prefiter->compute(imgLeft, imgRight, imgDisparity16SL_prefiter);
	sbmR->compute(imgRight, imgLeft, imgDisparity16SR);

	

	//filter



	wls_filter->setLambda(lambda);
	wls_filter->setSigmaColor(sigma);
	wls_filter->filter(imgDisparity16SL_prefiter, imgLeft, filtered_disp16S, imgDisparity16SR);
	//conf_map = wls_filter->getConfidenceMap();

	//-- Check its extreme values
	double minVal; double maxVal;

	minMaxLoc(imgDisparity16SL, &minVal, &maxVal);

	//-- 4. Display it as a CV_8UC1 image
	imgDisparity16SL.convertTo(imgDisparity8UL, CV_8UC1, 255 / (maxVal - minVal));

	/*minMaxLoc(filtered_disp16S, &minVal, &maxVal);*/

	//-- 4. Display it as a CV_8UC1 image
	filtered_disp16S.convertTo(filtered_disp8U, CV_8UC1, 255 / (maxVal - minVal));

//	namedWindow(windowDisparity, WINDOW_NORMAL);

//	cv::ShowManyImages("Images", 2, img1, img2);
	imshow("imgLeft", imgLeft);
	imshow("imgRight", imgRight);
	imshow("disp", imgDisparity8UL);
	imshow("filtered disp", filtered_disp8U);

	waitKey(0);

	return 0;
}