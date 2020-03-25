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

const float calibrationSquareDimension = 0.01905f; //meters
const float arucoSquareDimension = 0.1016f; //meters
const Size chessboardDimensions = Size(8,5);

void createArucoMarkers(){
	Mat outputMarker;
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
	for(int i = 0;i<50;i++){
		aruco::drawMarker(markerDictionary, i, 500, outputMarker, 1);
		ostringstream convert;
		string imageName = "aruco/4x4Marker_";
		convert << imageName <<i<<".jpg";
		imwrite(convert.str(), outputMarker);
	}
}
void createKnownBoardPosition(Size boardSize, float squareEdgeLength, vector<Point3f>& corners){
	for(int i=0;i<boardSize.height;i++){
		for(int j=0;j<boardSize.width; j++){
			corners.push_back(Point3f(j*squareEdgeLength, i*squareEdgeLength, 0.0f));
		}
	}
}
// for processing set of images
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

void cameraCalibration(vector<Mat> calibrationImages, Size boardSize, float squareEdgeLength, Mat& cameraMatrix, Mat& distanceCoefficients){
	vector<vector<Point2f>> checkerboardImageSpacePoints;
	getChessboardCorners(calibrationImages, checkerboardImageSpacePoints,false);
	vector<vector<Point3f>> worldSpaceCornerPoints(1);
	createKnownBoardPosition(boardSize, squareEdgeLength, worldSpaceCornerPoints[0]);
	worldSpaceCornerPoints.resize(checkerboardImageSpacePoints.size(), worldSpaceCornerPoints[0]);
	vector<Mat> rVectors, tVectors;// radial, tangential
	distanceCoefficients = Mat::zeros(8,1,CV_64F);
	calibrateCamera(worldSpaceCornerPoints,checkerboardImageSpacePoints, boardSize, cameraMatrix, distanceCoefficients, rVectors,tVectors);
}

bool saveCameraCalibration(string name, Mat cameraMatrix, Mat distanceCoefficients){
	ofstream outStream(name);
	if(outStream){
		uint16_t rows = cameraMatrix.rows;
		uint16_t columns = cameraMatrix.cols;
		outStream << rows<< endl;
		outStream << columns <<endl;
		//cameraMatrix
		for(int r = 0;r<rows;r++){
			for(int c = 0;c<columns;c++){
				double value = cameraMatrix.at<double>(r,c);
				outStream << value <<endl;
			}
		}
		// distanceCoefficients
		rows = distanceCoefficients.rows;
		columns = distanceCoefficients.cols;
		for(int r = 0;r<rows;r++){
			for(int c = 0;c<columns;c++){
				double value = distanceCoefficients.at<double>(r,c);
				outStream << value <<endl;
			}
		}
		outStream.close();
		return true;
	}
	return false;
}

bool loadCameraCalibration(string name, Mat& cameraMatrix, Mat& distanceCoefficients){
	ifstream inStream(name);
	if(inStream){
		uint16_t rows;
		uint16_t columns;

		inStream >> rows;
		inStream >> columns;

		cameraMatrix = Mat(Size(columns, rows), CV_64F);
		for(int r=0;r<rows;r++){
			for(int c=0;c<columns;c++){
				double read = 0.0f;
				inStream >> read;
				cameraMatrix.at<double>(r,c) = read;
				cout<<cameraMatrix.at<double>(r,c) <<"\n";
			}
		}
		//Distance Coefficients
		inStream >> rows;
		inStream >> columns;
		distanceCoefficients = Mat::zeros(rows, columns, CV_64F);
		for(int r=0;r<rows;r++){
			for(int c=0;c<columns;c++){
				double read = 0.0f;
				inStream >> read;
				distanceCoefficients.at<double>(r,c) = read;
				cout<< distanceCoefficients.at<double>(r,c) << "\n";
			}
		}
		inStream.close();
		return true;
	}
	return false;
}

int startWebcamMonitoring(const Mat& cameraMatrix, const Mat& distanceCoefficients, float arucoSquareDimension){
	Mat frame;
	vector<int> markerIds;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	aruco::DetectorParameters parameters;
	Ptr<aruco::Dictionary> markerDictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50);
	VideoCapture vid(0);
	if(!vid.isOpened()){
		return -1;
	}
	namedWindow("Webcam", WINDOW_NORMAL);
	vector<Vec3d> rotationVectors, translationVectors;
	while(true){
		if(!vid.read(frame))break;
		aruco::detectMarkers(frame, markerDictionary, markerCorners, markerIds);
		aruco::estimatePoseSingleMarkers(markerCorners, arucoSquareDimension, cameraMatrix, distanceCoefficients, rotationVectors, translationVectors);
		for(int i = 0;i<markerIds.size();i++){
			aruco::drawAxis(frame, cameraMatrix, distanceCoefficients, rotationVectors[i],translationVectors[i], 0.1f);
		}
		imshow("Webcam",frame);
		if(waitKey(30)>=0)break;
	}
}

int cameraCalibrationProcess(Mat& cameraMatrix, Mat& distanceCoefficients){
	Mat frame;
	Mat drawToFrame;
	vector<Mat> savedImages;
	vector<vector<Point2f>> markerCorners, rejectedCandidates;
	VideoCapture vid(0);
	if(!vid.isOpened()){
		return 0;
	}
	int framePerSecond = 20;
	namedWindow("Webcam", WINDOW_NORMAL);
	while(true){
		if(!vid.read(frame))
			break;
		vector<Vec2f> foundPoints;
		bool found = false;
		found = findChessboardCorners(frame,chessboardDimensions,foundPoints, CALIB_CB_ADAPTIVE_THRESH| CALIB_CB_NORMALIZE_IMAGE);
		frame.copyTo(drawToFrame);
		drawChessboardCorners(drawToFrame, chessboardDimensions,foundPoints, found);
		if(found)
			imshow("Webcam", drawToFrame);
		else
			imshow("Webcam", frame);
		char character = waitKey(1000/framePerSecond);
		switch(character){
			case ' ':
			//saving image
			if(found){
				Mat temp;
				frame.copyTo(temp);
				savedImages.push_back(temp);
			}
			break;
			case 13:// space
			//start calibration
			if(savedImages.size()>15){
				cameraCalibration(savedImages,chessboardDimensions,calibrationSquareDimension,cameraMatrix,distanceCoefficients);
				saveCameraCalibration("ILoveCameraCalibration", cameraMatrix, distanceCoefficients);
			}	
			break;
			case 27:// esc
			//exit
			return 0;
			break;

		}
	}
}
int main(int argv, char** argc){
	Mat cameraMatrix = Mat::eye(3,3,CV_64F);
	Mat distanceCoefficients;
	//calibration
/*	cameraCalibrationProcess(cameraMatrix, distanceCoefficients);
*/	
	// monitoring
	loadCameraCalibration("ILoveCameraCalibration",cameraMatrix,distanceCoefficients);
	startWebcamMonitoring(cameraMatrix, distanceCoefficients, 0.099f);

	return 0;
}