#include <opencv2/opencv.hpp>
#include <stdint.h>

using namespace cv;
using namespace std;

int main(int argv, char** argc){
	Mat frame;

	//VideoCapture vid(0); // use webcam
	VideoCapture vid("rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov");

	namedWindow("Webcam", WINDOW_AUTOSIZE);

	int fps =(int)vid.get(CAP_PROP_FPS);
	if(!vid.isOpened()){
		return -1;
	}

	while(vid.read(frame)){
		imshow("Webcam", frame);
		if(waitKey(1000/fps)>=0){
			break;
		}
	}
	return 1;
}