#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <sstream>
#include <iostream>
#include <ostream>
#include <fstream>
#include <stdio.h>

using namespace std;
using namespace cv;
//Load the dictionary that was used to generate the markers
Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

void createArucoMarkers(bool create)
{
	if (create)
	{
		cv::Mat markerImage;
		char buffer[50];
		for (int i = 0; i < 50; i++)
		{
			aruco::drawMarker(dictionary, i, 200, markerImage, 1);
			sprintf(buffer, "aruco/DICT_6X6_250_%d.png", i);
			imwrite(buffer, markerImage);
		}
	}
}
int main(int argc, char **argv)
{
	//Create markers
	createArucoMarkers(false);
	Mat frame;
	Mat im_src = imread("656336.png");
	VideoCapture vid(0);
	if (!vid.isOpened())
	{
		return 0;
	}
	namedWindow("Webcam", WINDOW_NORMAL);
	namedWindow("Out", WINDOW_NORMAL);
	//Initialize the detector parameters using default values
	Ptr<aruco::DetectorParameters> parameters = aruco::DetectorParameters::create();
	//Declare the vectors that would contain the detected marker corners and the rejected marker candidates
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	// The ids of the detected markers are stored in vector
	vector<int> markerIds;

	while (true)
	{
		if (!vid.read(frame))
			break;
		//vid >> frame;
		vid.retrieve(frame);
		imshow("Webcam", frame);

		//Detect the markers in the image
		aruco::detectMarkers(frame, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
		cv::Mat outputImage = frame.clone();
		cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds, Scalar(0, 255, 0));
		if (markerIds.size()==4)
		{
			vector<Point> pts_dst;
			float scalingFac = 0.02;
			Point refPt1, refPt2, refPt3, refPt4;
			//top left corner of the target quadrilateral
			vector<int>::iterator it = std::find(markerIds.begin(), markerIds.end(), 1);
			int index = std::distance(markerIds.begin(), it);
			refPt1 = markerCorners.at(index).at(0);

			//top right corner of the target quadrilateral
			it = std::find(markerIds.begin(), markerIds.end(), 2);
			index = std::distance(markerIds.begin(), it);
			refPt2 = markerCorners.at(index).at(1);

			float distance = norm(refPt1 - refPt2);
			pts_dst.push_back(Point(refPt1.x - round(scalingFac * distance), refPt1.y - round(scalingFac * distance)));
			pts_dst.push_back(Point(refPt2.x + round(scalingFac * distance), refPt2.y - round(scalingFac * distance)));

			//bottom right corner of the target quadrilateral
			it = std::find(markerIds.begin(), markerIds.end(), 3);
			index = std::distance(markerIds.begin(), it);
			refPt3 = markerCorners.at(index).at(2);
			pts_dst.push_back(Point(refPt3.x + round(scalingFac * distance), refPt3.y + round(scalingFac * distance)));

			//bottom left corner of the target quadrilateral
			it = std::find(markerIds.begin(), markerIds.end(), 4);
			index = std::distance(markerIds.begin(), it);
			refPt4 = markerCorners.at(index).at(3);
			pts_dst.push_back(Point(refPt4.x - round(scalingFac * distance), refPt4.y + round(scalingFac * distance)));

			//corner points of the new scene image
			vector<Point> pts_src;
			pts_src.push_back(Point(0, 0));
			pts_src.push_back(Point(im_src.cols, 0));
			pts_src.push_back(Point(im_src.cols, im_src.rows));
			pts_src.push_back(Point(0, im_src.rows));

			//Compute homography from source and destination points
			Mat h = cv::findHomography(pts_src, pts_dst);
			//Warped image;
			Mat warpedImage;
			//Warp source image to destination based on homogarphy
			warpPerspective(im_src, warpedImage, h, frame.size(), INTER_CUBIC);
			//Prepare a mask representing region to copy from the warped image into the original frame
			Mat mask = Mat::zeros(frame.rows, frame.cols, CV_8UC1);
			fillConvexPoly(mask, pts_dst, Scalar(255, 255, 255));
			//Erode the mask to not copy the boundary effects from the warping
			Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
			erode(mask, mask, element);
			//Copy the masked warped image into the original frame in the mask region
			Mat imOut = frame.clone();
			warpedImage.copyTo(imOut, mask);
			//Showing the original image and the new output image side by side
			Mat concatenatedOutput;
			hconcat(outputImage, imOut, concatenatedOutput);

			imshow("Out", concatenatedOutput);
		}

		char character = waitKey(1);
		if (character == 27)
			break;
	}

	return 0;
}