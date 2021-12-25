#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
int main() {
	cv::Mat src = cv::imread("4.jpg"); // lena.png 2peop.jpg
	//imshow("src1", src);
	Mat test;
	cvtColor(src, test, COLOR_BGR2GRAY);
	//imshow("src2", test);
	threshold(test, test, 147, 255,THRESH_BINARY);
	//imshow("threshold", test);
	cv::CascadeClassifier c;
	c.load("haarcascade_frontalface_alt.xml");
	std::vector<cv::Rect> faces;
	c.detectMultiScale(src, faces, 1.1, 2,0| CASCADE_SCALE_IMAGE, cv::Size(30, 30));
	for (int i = 0; i < faces.size(); i++)
	{
		rectangle(src, faces[i], Scalar(0, 255, 0), 1, 8, 0);
		/*Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		ellipse(src, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);*/
	}
	cv::namedWindow("FaceDetection", 1);
	cv::imshow("FaceDetection", src);
	cv::waitKey(0);
	return 0;
}