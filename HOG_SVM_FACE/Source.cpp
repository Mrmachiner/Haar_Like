//#ifndef SVMTRAINING_H
//#define SVMTRAINING_H
//#ifndef SVMDEFINES_H
//#define SVMDEFINES_H
//
//#define RANDOM_PATCH_COUNT 200
//#define SVM_ITERATIONS 100000
//#define SVM_OUTPUT_NAME "SVM_MARC.yaml"
//#define WINDOW_SIZE 64
////#define DESCRIPTOR_SIZE 1764
//#define DESCRIPTOR_SIZE 3780
//// Standard = 0.8
//#define DOWNSCALE_FACTOR 0.91
//// Standard = 5
//#define PATCH_PIXEL_MOVEMENT 23
//
//#endif // SVMDEFINES_H
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/ml/ml.hpp"
//#include <time.h>
//#include <ctime>
//#include <cstdio>
//#include <cstdlib>
//#include <vector>
//
//int main(int argc, const char** argv);
//
//bool trainSVM(cv::String* positiveTrainPath, cv::String* negativeTrainPath);
//
//#endif // SVMTRAINING_H
//using namespace cv;
//bool trainSVM(String* positiveTrainPath, String* negativeTrainPath)
//{
//	// Finding all images in both pathes
//	std::vector<String> positiveFileNames, negativeFileNames;
//	glob(*positiveTrainPath, positiveFileNames);
//	glob(*negativeTrainPath, negativeFileNames);
//
//	Mat trainingLabel = Mat_<int>(1, positiveFileNames.size() + negativeFileNames.size() * RANDOM_PATCH_COUNT);
//	Mat trainingData = Mat_<float>(DESCRIPTOR_SIZE, positiveFileNames.size() + negativeFileNames.size() * RANDOM_PATCH_COUNT);
//	int trainingCount = 0;
//	/*std::vector<Mat>DataTrain;
//	std::vector<Mat>LabelTrain;*/
//
//	HOGDescriptor hogD;
//	//HOGDescriptor hog(
//	//	Size(64, 128), //winSize
//	//	Size(10, 10), //blocksize
//	//	Size(5, 5), //blockStride,
//	//	Size(10, 10), //cellSize,
//	//	9, //nbins,
//	//	1, //derivAper,
//	//	-1, //winSigma,
//	//	HOGDescriptor::L2Hys, //histogramNormType,
//	//	0.2, //L2HysThresh,
//	//	1,//gammal correction,
//	//	64,//nlevels=64
//	//	1);//Use signed gradients 
//
//	//hogD.winSize = Size(WINDOW_SIZE, WINDOW_SIZE);
//	hogD.winSize = Size(64, 128);
//	std::vector<float> descriptorsValues;
//	std::vector<Point> locations;
//
//	clock_t beginTime = clock();
//
//#pragma endregion
//
//#pragma region Positive HOG Descriptors
//
//	// Converting the positve images and calculating the HOG
//	std::cout << "Calculate positive HOG Descriptors (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
//	for (std::vector<String>::iterator fileName = positiveFileNames.begin(); fileName != positiveFileNames.end(); ++fileName)
//	{
//		Mat actualImage = imread(*fileName);
//
//		// Testing if the file is an image
//		if (actualImage.empty())
//		{
//			printf("Couldn't read the image %s\n", *fileName);
//			return false;
//		}
//		cvtColor(actualImage, actualImage, COLOR_BGR2GRAY);
//		resize(actualImage, actualImage, Size(64, 128));
//
//		// Calculating the HOG
//		hogD.compute(actualImage, descriptorsValues, Size(16, 16), Size(0, 0), locations);
//
//		Mat descriptorsVector = Mat_<float>(descriptorsValues, true);
//		descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
//		trainingLabel.at<int>(0, trainingCount) = 1;
//		trainingCount++;
//	}
//	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;
//
//#pragma endregion
//
//#pragma region Negative HOG Descriptors
//
//	// Calculating the HOG of the negativ images
//	std::cout << "Calculate negative HOG Descriptors (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
//	for (std::vector<String>::iterator fileName = negativeFileNames.begin() + 1; fileName != negativeFileNames.end(); ++fileName)
//	{
//		Mat actualImage = imread(*fileName);
//
//		// Testing if the file is an image
//		if (actualImage.empty())
//		{
//			printf("Couldn't read the image %s\n", *fileName);
//			return false;
//		}
//		cvtColor(actualImage, actualImage, COLOR_BGR2GRAY);
//
//		// Choose the random windows and theire size
//		for (int c = 0; c < RANDOM_PATCH_COUNT; c++)
//		{
//			int rWidth = (rand() % 191) + 10;
//			Point rPoint = Point(rand() % (actualImage.cols - rWidth),
//				rand() % (actualImage.rows - rWidth));
//			// Pick the window out of the image
//			Mat actualWindow;
//
//			resize(actualImage(Range(rPoint.y, rPoint.y + rWidth), Range(rPoint.x, rPoint.x + rWidth)), actualWindow, Size(64, 128));
//
//			// Calculating the HOG
//			hogD.compute(actualWindow, descriptorsValues, Size(16, 16), Size(0, 0), locations);
//
//			Mat descriptorsVector = Mat_<float>(descriptorsValues, true);
//			descriptorsVector.col(0).copyTo(trainingData.col(trainingCount));
//			trainingLabel.at<int>(0, trainingCount) = -1;
//			trainingCount++;
//		}
//	}
//	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;
//
//#pragma endregion
//
//#pragma region SVM Training
//	// Set up SVM's parameters
//	Ptr<ml::SVM> svm = ml::SVM::create();
//	svm->setType(ml::SVM::C_SVC);
//	svm->setKernel(ml::SVM::LINEAR);
//	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//	// Create the Trainingdata
//	std::vector<std::vector<Mat>> dataaa;
//	Ptr<ml::TrainData> tData = ml::TrainData::create(trainingData, ml::SampleTypes::COL_SAMPLE, trainingLabel);
//	std::cout << "Start SVM training (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ") ...";
//	svm->train(tData);
//	std::cout << " Finished (" << (clock() - beginTime) / (float)CLOCKS_PER_SEC << ")" << std::endl;
//
//	svm->save("SVM_train.xml");
//
//#pragma endregion
//
//	return true;
//}
//int main(int argc, const char** argv)
//{
//	String positivePath = "D:\\Data\\faces";
//	String negativePath = "D:\\Data\\coast";
//
//	bool train = trainSVM(&positivePath, &negativePath);
//	if (!train)
//		//return -1;
//		std::cout << "-1";
//	getchar();
//	//cv::Mat src = cv::imread("lena.png"); // lena.png 2peop.jpg
//	//imshow("src1", src);
//	//resize(src, src, Size(64, 128));
//	//cv::CascadeClassifier c;
//	//c.load("SVM_MARC.xml");
//	//std::vector<cv::Rect> faces;
//	//c.detectMultiScale(src, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE);
//	//for (int i = 0; i < faces.size(); i++)
//	//{
//	//	rectangle(src, faces[i], Scalar(255, 0, 255), 1, 8, 0);
//	//	/*Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
//	//	ellipse(src, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 2, 8, 0);*/
//	//}
//	//cv::namedWindow("FaceDetection", 1);
//	//cv::imshow("FaceDetection", src);
//	//cv::waitKey(0);
//	return 0;
//
//}