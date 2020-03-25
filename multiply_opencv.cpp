#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>

#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
const float calibrationSquareDimension = 0.031f; //meters
const float arucoSquareDimension = 0.1016f; //meters
const Size chessboardDimensions = Size(8,5);

void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners){
	for(int i=0;i<boardSize.height;i++){
		for(int j=0;j<boardSize.width; j++){
			corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));
		}
	}
}
void getChessboardCorners(vector<Mat> images, vector<vector<Point2f>>& allFoundCorners, bool showResults){
	for(vector<Mat>::iterator iter = images.begin(); iter!=images.end();iter++){
		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(*iter, chessboardDimensions, pointBuf, CALIB_CB_ADAPTIVE_THRESH| CALIB_CB_NORMALIZE_IMAGE);
		if(found){
			allFoundCorners.push_back(pointBuf);
		}
		if(showResults){
			drawChessboardCorners(*iter, chessboardDimensions,pointBuf, found);
			imshow("Looking for Corners", *iter);
			waitKey(0);
		}
	}
}


int main(int argc, char** argv){
	Mat frame;
	Mat drawToFrame;
	Mat cameraMatrix = Mat::eye(3,3,CV_64F);
	Mat distanceCoefficients;
	vector<Mat> savedImages;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;

	VideoCapture vid(0);
	if(!vid.isOpened()){
		return 0;
	}
	int framesPerSecond = 50;
	namedWindow("Webcam", WINDOW_NORMAL);
	while(true){
		if(!vid.read(frame))
			break;
		vector<Vec2f> foundPoints;
		bool found = false;
		found = findChessboardCorners(frame, chessboardDimensions, foundPoints, CALIB_CB_ADAPTIVE_THRESH|CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame,chessboardDimensions,foundPoints,found);
		if(found)
			imshow("Webcam",drawToFrame);
		else
			imshow("Webcam",frame);
		char character = waitKey(1000/framesPerSecond);
		if (character == 27) break;
	}
	return 0;
}