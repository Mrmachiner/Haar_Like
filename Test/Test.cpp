#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <iostream>
#include<vector>
#include <time.h>
#include<fstream>
#include <numeric>
#include <opencv2/dnn.hpp>

//using namespace cv;
//using namespace cv::ml;
//using namespace std;
//using namespace cv::dnn;
int RANDOM_PATCH_COUNT = 10;
std::vector< float > get_svm_detector(const cv::Ptr< cv::ml::SVM >& svm);
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData);
void load_images(const cv::String & dirname, std::vector< cv::Mat > & img_lst, bool showImages);
void sample_neg(const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size);
void computeHOGs(const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip);

std::vector< float > get_svm_detector(const cv::Ptr< cv::ml::SVM >& svm)
{
	// get the support vectors
	cv::Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	cv::Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);

	std::vector< float > hog_detector(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
	hog_detector[sv.cols] = (float)-rho;
	return hog_detector;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a cv::Matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);
	cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = cv::Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < train_samples.size(); ++i)
	{
		CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);

		if (train_samples[i].cols == 1)
		{
			transpose(train_samples[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (train_samples[i].rows == 1)
		{
			train_samples[i].copyTo(trainData.row((int)i));
		}
	}
}

void load_images_test(const cv::String & dirname, std::vector< cv::Mat > & img_lst, bool showImages = false)
{
	std::vector< cv::String > files;
	cv::glob(dirname, files);
	for (size_t i = 0; i < files.size(); ++i)
	{
		cv::Mat img = cv::imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			std::cout << files[i] << " is invalid!" << std::endl;
			continue;
		}

		if (showImages)
		{
			cv::imshow("image", img);
			cv::waitKey(1);
		}
		resize(img, img,cv:: Size((int)img.size().width /2, (int)img.size().height / 2));
		img_lst.push_back(img);
	}
}
void load_images(const cv::String & dirname, std::vector< cv::Mat > & img_lst, bool showImages = false)
{
	std::vector< cv::String > files;
	cv::glob(dirname, files);
	random_shuffle(files.begin(), files.end());
	for (size_t i = 1; i < files.size(); ++i)
	{
		cv::Mat img = cv::imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			std::cout << files[i] << " is invalid!" << std::endl;
			continue;
		}

		if (showImages)
		{
			cv::imshow("image", img);
			cv::waitKey(1);
		}
		//resize(img, img,cv:: Size(40, 56));
		img_lst.push_back(img);
	}
}
void sample_neg(const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size)
{
	cv::Rect box;
	box.width = size.width;
	box.height = size.height;

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));

	for (size_t i = 0; i < full_neg_lst.size(); i++)
		if (full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height)
		{
			box.x = rand() % (full_neg_lst[i].cols - size_x);
			box.y = rand() % (full_neg_lst[i].rows - size_y);
			cv::Mat roi = full_neg_lst[i](box);
			neg_lst.push_back(roi.clone());
		}
}

void computeHOGs(const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip)
{
	cv::HOGDescriptor hog(wsize,cv:: Size(24, 24),cv:: Size(8, 8),cv:: Size(8, 8), 9);
	//HOGDescriptor hog;
	//HOGDescriptor hog(
	//	Size(64, 128), //winSize
	//	Size(10, 10), //blocksize
	//	Size(5, 5), //blockStride,
	//	Size(10, 10), //cellSize,
	//	9, //nbins,
	//	1, //derivAper,
	//	-1, //winSigma,
	//	HOGDescriptor::L2Hys, //histogramNormType,
	//	0.2, //L2HysThresh,
	//	1,//gammal correction,
	//	64,//nlevels=64
	//	1);//Use signed gradients 
	//hog.winSize = wsize;
	/*hog.compute(inputImg, Descriptors, WinStrideSize, PaddingSize, VecPointLocations);*/
	std::vector< float > descriptors;
	//std::vector< Point> locations;
	cv::Mat gray;
	for (size_t i = 1; i < img_lst.size(); i++)
	{
		cvtColor(img_lst[i], gray, cv::COLOR_BGR2GRAY);
		hog.compute(gray, descriptors,cv:: Size(8, 8),cv:: Size(0, 0));
		gradient_lst.push_back(cv::Mat(descriptors).clone());
		/*if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height)
		{
			Rect r = Rect((img_lst[i].cols - wsize.width) / 2,
				(img_lst[i].rows - wsize.height) / 2,
				wsize.width,
				wsize.height);
			cvtColor(img_lst[i](r), gray, cv::COLOR_BGR2GRAY);
			hog.compute(gray, descriptors,cv:: Size(8, 8),cv:: Size(0, 0));
			gradient_lst.push_back(cv::Mat(descriptors).clone());
		}*/
	}
}
float Calu_predic_test(std::vector<cv::Mat> Vec_Mat, cv::Ptr< cv::ml::SVM > svm, int choise)
{
	std::vector<float> lst_predic;
	if (choise == 0)
	{
		for (int i = 0; i < Vec_Mat.size(); i++)
		{
			float Prec_end = svm->predict(Vec_Mat[i]);
			std::cout << std::endl << "Predic_Faces" + std::to_string(i) << "\t" << Prec_end * 1.0;
			lst_predic.push_back(Prec_end);
		}
	}
	else
	{
		for (int i = 0; i < Vec_Mat.size(); i++)
		{
			float Prec_end = svm->predict(Vec_Mat[i]);
			std::cout << std::endl << "Predic_Non-Faces" + std::to_string(i) << "\t" << Prec_end * 1.0;
			lst_predic.push_back(Prec_end);
		}
	}
	float sum = 0;
	for (int i = 0; i < lst_predic.size(); i++) {
		sum += lst_predic[i];
	}
	return sum / lst_predic.size()*1.0;
}
float Calu_predic(std::vector<cv::Mat> Vec_Mat, cv::Ptr< cv::ml::SVM > svm)
{
	std::vector<float> lst_predic;
	for (int i = 0; i < Vec_Mat.size(); i++)
	{
		float Prec_end = svm->predict(Vec_Mat[i]);
		lst_predic.push_back(Prec_end);
	}
	float sum = 0;
	for (int i = 0; i < lst_predic.size(); i++) {
		sum += lst_predic[i];
	}
	return sum / lst_predic.size()*1.0;
}
std::vector<cv::Mat> Vec_Mat_predic(std::vector<cv::Mat> vec_Mat) {
	std::vector<cv::Mat> Mat_predic;
	for (int i = 0; i < vec_Mat.size(); i++)
	{
		std::vector<cv::Mat> lst_gra_test;
		cv::Mat test_pre;
		lst_gra_test.push_back(vec_Mat[i]);
		convert_to_ml(lst_gra_test, test_pre);
		Mat_predic.push_back(test_pre);
	}
	return Mat_predic;
}
std::vector<std::vector<float>> PredictSVM(cv::String path) {
	std::vector<cv::Mat> lst_img;
	std::vector<std::vector<float>> lst_pre;
	load_images(path, lst_img, false);
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm = cv::ml::SVM::load("C:\\Users\\MinhHoang\\source\\repos\\Haar_Like\\Test\\my_detector_4h57SVM.xml");
	std::vector<float> lst_preTrue;
	std::vector<float> lst_preFalse;
	//for (int i = 0; i < lst_img.size(); i++) {
	//	cv::Mat gray;
	//	resize(lst_img[i], lst_img[i],cv:: Size((int)lst_img[i].size().width / 5, (int)lst_img[i].size().height / 5));
	//	cvtColor(lst_img[i], gray, cv::COLOR_BGR2GRAY);
	//	cv::Mat sample = cv::Mat_<float>(1, 1215) << lst_img[i];
	//	//sample.convertTo(gray, CV_32FC1);
	//	float pre = svm->predict(sample);
	//	if (pre == 1) {
	//		lst_preTrue.push_back(pre);
	//	}
	//	else
	//	{
	//		lst_preFalse.push_back(pre);
	//	}
	//}
	lst_pre.push_back(lst_preTrue);
	lst_pre.push_back(lst_preFalse);
	return lst_pre;
}
void computeNagativeHOGs(const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip)
{
	cv::HOGDescriptor hog;
	//HOGDescriptor hog(
	//	Size(64, 128), //winSize
	//	Size(10, 10), //blocksize
	//	Size(5, 5), //blockStride,
	//	Size(10, 10), //cellSize,
	//	9, //nbins,
	//	1, //derivAper,
	//	-1, //winSigma,
	//	HOGDescriptor::L2Hys, //histogramNormType,
	//	0.2, //L2HysThresh,
	//	1,//gammal correction,
	//	64,//nlevels=64
	//	1);//Use signed gradients 
	hog.winSize = wsize;
	cv::Mat gray;
	//cv::Mat rsize;
	std::vector< float > descriptors;

	for (size_t i = 1; i < img_lst.size(); i++)
	{
		//resize(img_lst[i], rsize,cv:: Size(32, 32));
		if (img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height)
		{
			cv::Rect r = cv::Rect((img_lst[i].cols - wsize.width) / 2,
				(img_lst[i].rows - wsize.height) / 2,
				wsize.width,
				wsize.height);
			cvtColor(img_lst[i](r), gray, cv::COLOR_BGR2GRAY);
			hog.compute(gray, descriptors,cv:: Size(8, 8),cv:: Size(0, 0));
			gradient_lst.push_back(cv::Mat(descriptors).clone());
			if (use_flip)
			{
				flip(gray, gray, 1);
				hog.compute(gray, descriptors,cv:: Size(8, 8),cv:: Size(0, 0));
				gradient_lst.push_back(cv::Mat(descriptors).clone());
			}
		}
	}
}
int Loca(std::vector<double> foundWeights, std::vector< cv::Rect > detections_test, int minW, int minH)
{
	//int vt = -1;
	//bool br = false;
	//for (int i = 0; i < foundWeights.size(); i++)
	//{
	//	if (detections_test[i].width > minW && detections_test[i].height > minH && foundWeights[i] > 0.5) {
	//		vt = i;
	//		return vt;
	//	}
	//}
	//return -1;
	int vt = -1;
	bool br = false;
	for (int i = 0; i < foundWeights.size(); i++)
	{
		if (foundWeights[i] > 0.5) {
			vt = i;
			return vt;
		}
	}
	return -1;
}
int Loca_Max_weight(std::vector<double> &foundWeights, std::vector< cv::Rect > &detections_test, int minWidth, int minHeight)
{
	int vtMax = -1;
	double maxAcreage = foundWeights[0];
	for (int i = 0; i < detections_test.size() - 1; i++)
	{
		int acreageI = detections_test[i].height*detections_test[i].width;
		for (int j = i + 1; j < detections_test.size(); j++) {
			int acreageJ = detections_test[j].height*detections_test[j].width;
			if (acreageI < acreageJ)
			{
				cv::Rect swR;
				swR = detections_test[i];
				detections_test[i] = detections_test[j];
				detections_test[j] = swR;
				double swW;
				swW = foundWeights[i];
				foundWeights[i] = foundWeights[j];
				foundWeights[j] = swW;
			}
		}
	}
	vtMax = Loca(foundWeights, detections_test, minWidth, minHeight);
	return vtMax;
}
std::vector<std::vector<float>> Swap_info(std::vector< std::vector<float> > array)
{
	std::vector<std::vector<float>> Lst_Swap_info;

	for (int i = 0; i < array[2].size(); i++)
	{
		std::vector<float> inf;
		inf.push_back(array[0][i]);
		inf.push_back(array[1][i]);
		inf.push_back(array[2][i]);
		inf.push_back(array[3][i]);
		inf.push_back(array[4][i]);
		inf.push_back(array[5][i]);
		inf.push_back(array[6][i]);
		inf.push_back(array[7][i]);
		Lst_Swap_info.push_back(inf);
	}
	return Lst_Swap_info;
}
std::vector<std::vector<float>> GetInforFromPath(cv::String pathID)
{

	std::ifstream in(pathID);

	cv::String line, field;

	std::vector< std::vector<float> > array;  // the 2D array
	std::vector< std::vector<float> > lst_info;
	std::vector<float> v;                // array of values for one line only
	int countV = 0;
	int countA = 0;
	while (getline(in, line))    // get next line in file
	{
		//v.clear();

		std::stringstream ss(line);
		while (std::getline(ss, field, ','))  // break line into comma delimitted fields
		{
			countV++;
			v.push_back(::atof(field.c_str())/2.0);
			//v.push_back(::atof(field.c_str()));
			// add each field to the 1D array
			if (countV == 450)
			{
				array.push_back(v);
				countV = 0;
				v.clear();
			}
		}
		//array.push_back(v);  // add the 1D array to the 2D array
	}
	in.close();
	lst_info = Swap_info(array);
	return lst_info;
}
float compe_intersection_over_union(std::vector<float> Getinfo, cv::Rect detections)
{

	int xA = std::max(detections.x, (int)Getinfo[2]);
	int yA = std::max((int)(detections.y), (int)Getinfo[3]);
	
	int xB = std::min((int)(detections.x + detections.width), (int)Getinfo[6]);
	int yB = std::min((int)(detections.y + detections.height), (int)Getinfo[7]);
	int interArea = (xB - xA)*(yB - yA);
	int BoxAArea = detections.width*detections.height;
	int BoxBArea = ((int)Getinfo[6] - (int)Getinfo[2])*((int)Getinfo[7] - (int)Getinfo[3]);
	float Iou = (float)interArea / (float)(BoxBArea + BoxAArea - interArea)*1.0;
	return Iou;
}
std::vector<cv::Mat> lst_Draw_pic(std::vector<cv::Mat> lst_img, std::vector<std::vector<float>> Getinfo)
{
	std::vector<cv::Mat> draw_lst_pic;
	std::vector< std::vector<float>> lst_potin;
	for (int i = 1; i < lst_img.size(); i++)
	{
		std::vector<cv::Point2f> point;
		cv::Mat pic = lst_img[i];
		cv::Point a(Getinfo[i][0], Getinfo[i][1]);
		cv::Point b(Getinfo[i][2], Getinfo[i][3]);
		cv::Point c(Getinfo[i][4], Getinfo[i][5]);
		cv::Point d(Getinfo[i][6], Getinfo[i][7]);
		point.push_back(a);
		line(pic, a, b, cv::Scalar(255, 0, 255));
		line(pic, b, c, cv::Scalar(255, 0, 255));
		line(pic, c, d, cv::Scalar(255, 0, 255));
		line(pic, d, a, cv::Scalar(255, 0, 255));
		draw_lst_pic.push_back(pic);
		imwrite("D:\\Data\\Real_detec_pic\\test\\abcd" + std::to_string(i) + ".jpg", pic);
	}
	return draw_lst_pic;
}
cv::Mat DrawTest(cv::Mat pic, std::vector<float> Getinfo)
{
	cv::Mat test = pic;
	cv::Point a(Getinfo[0], Getinfo[1]);
	cv::Point b(Getinfo[2], Getinfo[3]);
	cv::Point c(Getinfo[4], Getinfo[5]);
	cv::Point d(Getinfo[6], Getinfo[7]);
	
	cv::line(test, a, b, cv::Scalar(255, 0, 255));
	cv::line(test, b, c, cv::Scalar(255, 0, 255));
	cv::line(test, c, d, cv::Scalar(255, 0, 255));
	cv::line(test, d, a, cv::Scalar(255, 0, 255));
	return test;
}
float Sum_vec(std::vector<float> number)
{
	float sum = 0;
	for (int i = 0; i < number.size(); i++)
	{
		sum += number[i];
	}
	return sum;
}
std::vector<float> Convert_double_to_float(std::vector<double> vec_double)
{
	std::vector<float> fl;
	for (int i = 0; i < vec_double.size(); i++) {
		fl.push_back((float)vec_double[i]);
	}
	return fl;
}
void Test_lst_Img_NMSbox_IoU(std::vector<cv::Mat> lst_img, cv::HOGDescriptor hog, std::vector<std::vector<float>> info)
{ 
	bool br;
	std::vector<int> count;
	std::vector<float> vec_IoU;
	for (int i = 0; i < lst_img.size(); i++) {
		cv::Mat drawing = cv::Mat::zeros(lst_img[i].size(), CV_8UC3);
		std::cout << i << std::endl;
		br = false;
		std::vector< cv::Rect > detections_test;
		std::vector< double > foundWeights_test;
		int BoxAArea = -1;
		int BoxBArea = -1;
		float IOU = -1.0;
		hog.detectMultiScale(lst_img[i], detections_test, foundWeights_test, 0,cv:: Size(8, 8),cv:: Size(0, 0), 1.3);
		std::vector<int> indices;
		std::vector<float> foundWeights_Float;
		foundWeights_Float = Convert_double_to_float(foundWeights_test);
		cv::dnn::NMSBoxes(detections_test, foundWeights_Float, 0.5, 0.8, indices);
		/*for (int j = 0; j < indices.size(); j++) 
		{
			cv::Scalar color = cv::Scalar(0, 255, 0);
			int idx = indices[indices.size()-1];
			rectangle(lst_img[i], detections_test[idx], color, lst_img[i].cols / 400 + 1);
			BoxAArea = detections_test[idx].width*detections_test[idx].height;
			BoxBArea = ((int)info[i][6] - (int)info[i][2])*((int)info[i][7] - (int)info[i][3]);
			IOU = compe_intersection_over_union(info[i], detections_test[idx]);
			vec_IoU.push_back(IOU);
			count.push_back(1);
			br = true;
			break;
		}*/
		if (indices.size() != 0)
		{
			cv::Scalar color = cv::Scalar(0, 255, 0);
			int idx = indices[indices.size() - 1];
			rectangle(lst_img[i], detections_test[idx], color, lst_img[i].cols / 400 + 1);
			BoxAArea = detections_test[idx].width*detections_test[idx].height;
			BoxBArea = ((int)info[i][6] - (int)info[i][2])*((int)info[i][7] - (int)info[i][3]);
			IOU = compe_intersection_over_union(info[i], detections_test[idx]);
			vec_IoU.push_back(IOU);
			count.push_back(1); 
		}
		//std::ostd::Stringstream ss;
		//ss << IOU;
		//std::std::String s(ss.str());
		//std::String A = s +"  "+ to_std::String(BoxAArea)+"  "+ to_std::String(BoxBArea);
		//Point pointA(20, 20);
		//putText(drawing, A, pointA, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 8);
		//lst_img[i] = DrawTest(lst_img[i], info[i]);
		//drawing.copyTo(lst_img[i], drawing);
		//imwrite("D:\\Data\\Real_detec_pic\\Test_imt_NMS_IOU\\" + to_std::String(i) + ".jpg", lst_img[i]);
		////imwrite("D:\\Data\\Test\\abcd" + to_std::String(i) +to_std::String(detections_test[vt].height)+"x"+to_std::String(detections_test[vt].width)+ ".jpg", lst_img[i]);
	}
	float sum = Sum_vec(vec_IoU);
	std::cout << std::endl << "ty le" << sum / vec_IoU.size()*1.0;
	std::cout << std::endl << "Image True:  " << count.size() << std::endl;
	getchar();
}
void Test_lst_Img(std::vector<cv::Mat> lst_img, cv::HOGDescriptor hog, std::vector<std::vector<float>> info)
{
	bool br;
	std::vector<int> count;
	std::vector<float> vec_IoU;
	for (int i = 0; i < lst_img.size(); i++) {
		cv::Mat drawing = cv::Mat::zeros(lst_img[i].size(), CV_8UC3);
		std::cout << i << std::endl;
		br = false;
		std::vector< cv::Rect > detections_test;
		std::vector< double > foundWeights_test;
		int minWidth = (int)lst_img[i].size().width*0.15;
		int minHeight = (int)lst_img[i].size().height*0.15;
		int BoxAArea = -1;
		int BoxBArea = -1;
		float IOU = -1.0;
		hog.detectMultiScale(lst_img[i], detections_test, foundWeights_test, 0,cv:: Size(8, 8),cv:: Size(0, 0), 1.3);
		for (int j = 0; j < foundWeights_test.size(); j++) {
			int vt = Loca_Max_weight(foundWeights_test, detections_test, minWidth, minHeight);
			if (vt != -1)
			{
				cv::Scalar color = cv::Scalar(0, foundWeights_test[vt] * foundWeights_test[vt] * 200, 0);
				rectangle(lst_img[i], detections_test[vt], color, lst_img[i].cols / 400 + 1);
				BoxAArea = detections_test[vt].width*detections_test[vt].height;
				BoxBArea = ((int)info[i][6] - (int)info[i][2])*((int)info[i][7] - (int)info[i][3]);
				IOU = compe_intersection_over_union(info[i], detections_test[vt]);
				vec_IoU.push_back(IOU);
				//imwrite("D:\\Data\\Test\\A Size20\\A Size20 Full\\abcd" + to_std::String(i) +"-"+ to_std::String(detections_test[vt].height) + "x" + to_std::String(detections_test[vt].width) + ".jpg", lst_img[i]);
				//imwrite("D:\\Data\\Test\\abcd" + to_std::String(i) + ".jpg", lst_img[i]);
				/*cv::imshow("detections_test", lst_img[i]);
				cv::waitKey(50);*/
				count.push_back(1);
				br = true;
				break;
			}
			if (br == true) break;
		}
		/*std::ostd::Stringstream ss;
		ss << IOU;
		std::std::String s(ss.str());
		std::String A = s +"  "+ to_std::String(BoxAArea)+"  "+ to_std::String(BoxBArea);
		Point pointA(20, 20);
		putText(drawing, A, pointA, FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, 8);
		lst_img[i] = DrawTest(lst_img[i], info[i]);
		drawing.copyTo(lst_img[i], drawing);
		imwrite("D:\\Data\\Real_detec_pic\\test_img2_1.3\\" + to_std::String(i) + ".jpg", lst_img[i]);*/	
	}
	float sum = Sum_vec(vec_IoU);
	std::cout << std::endl << "ty le" << sum / vec_IoU.size()*1.0;
	std::cout << std::endl << "Image True:  " << count.size() << std::endl;
	getchar();
}
void IoU_lst_Img_train(std::vector<cv::Mat> lst_Img, cv::HOGDescriptor hog)
{
	bool br;
	std::vector<float> vec_IoU;
	for (int i = 0; i < lst_Img.size(); i++) {
		br = false;
		std::vector< cv::Rect > detections_test;
		std::vector< double > foundWeights_test;
		int minWidth = lst_Img[i].size().width*0.2;;
		int minHeight = lst_Img[i].size().height*0.2;;
		hog.detectMultiScale(lst_Img[i], detections_test, foundWeights_test);
		for (int j = 0; j < foundWeights_test.size(); j++) {
			int vt = Loca_Max_weight(foundWeights_test, detections_test, minWidth, minHeight);
			if (vt != -1)
			{
				int V_img = lst_Img[i].size().width*lst_Img[i].size().height;
				vec_IoU.push_back((detections_test[vt].height*detections_test[vt].width) / (lst_Img[i].size().width*lst_Img[i].size().height*1.0));
				br = true;
				break;
			}
			if (br == true) break;
		}
	}
}

void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat& frame)
{
	rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0));
	cv::String label = cv::format("%.2f", conf);
	int baseLine;
	cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = std::max(top, labelSize.height);
	rectangle(frame, cv::Point(left, top - labelSize.height),
		cv::Point(left + labelSize.width, top + baseLine), cv::Scalar::all(255), cv::FILLED);
	putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
}
int main(int argc, char** argv)
{
	//////const char* keys =
	//////{
	//////	"{help h|     | show help message}"
	//////	"{pd    |     | path of directory contains positive images}"
	//////	"{nd    |     | path of directory contains negative images}"
	//////	"{td    |     | path of directory contains test images}"
	//////	"{tv    |     | test video file name}"
	//////	"{dw    |     | width of the detector}"
	//////	"{dh    |     | height of the detector}"
	//////	"{f     |false| indicates if the program will generate and use mirrored samples or not}"
	//////	"{d     |false| train twice}"
	//////	"{t     |false| test a trained detector}"
	//////	"{v     |false| visualize training steps}"
	//////	"{fn    |my_detector.yml| file name of trained SVM}"
	//////};

	//std::String pos_dir = "D:\\Data\\positive\\positive";
	//std::String neg_dir = "D:\\Data\\negative\\negative";
	////std::String test_dir = parser.get< std::String >("td");
	//std::String obj_det_filename ="my_detector_16h00_20_8.xml";
	////std::String videofilename = parser.get< std::String >("tv");
	//int detector_width=40;
	//int detector_height=56;
	//bool test_detector = true;
	//bool train_twice = false;
	//bool visualization = false;
	//bool flip_samples = false;

	//std::vector< cv::Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
	//std::vector< int > labels;
	//std::vector<cv::Mat> pre_posi, pre_nega,full_lst_posi, full_lst_nega;
	//clog << "Positive images are being loaded...";
	//load_images(pos_dir, full_lst_posi, visualization);
	//int SL_train_posi = full_lst_posi.size()*0.8;
	//int SL_pre_posi = full_lst_posi.size() - SL_train_posi;
	//random_shuffle(full_lst_posi.begin(),full_lst_posi.end());
	////SL image train
	//for (int i = 0; i < SL_train_posi; i++) 
	//{
	//	pos_lst.push_back(full_lst_posi[i]);
	//}
	////SL image pre
	//for (int i = SL_train_posi; i < full_lst_posi.size(); i++) 
	//{
	//	pre_posi.push_back(full_lst_posi[i]);
	//}
	//if (pos_lst.size() > 0)
	//{
	//	clog << "...[done]" << endl;
	//}
	//else
	//{
	//	clog << "no image in " << pos_dir << endl;
	//	return 1;
	//}

	//Size pos_image_size = pos_lst[0].size();

	//if (detector_width && detector_height)
	//{
	//	pos_image_size =cv:: Size(detector_width, detector_height);
	//}
	//else
	//{
	//	for (size_t i = 1; i < pos_lst.size(); ++i)
	//	{
	//		if (pos_lst[i].size() != pos_image_size)
	//		{
	//			std::cout << "All positive images should be same size!" << endl;
	//			exit(1);
	//		}
	//	}
	//	pos_image_size = pos_image_size / 8 * 8;
	//}

	//clog << "Negative images are being loaded...";
	//load_images(neg_dir, full_lst_nega, visualization);
	//int SL_train_nega= full_lst_nega.size()*0.8;
	//int SL_pre_nega = full_lst_nega.size() - SL_train_nega;
	////SL image train nega
	//for (int i = 0; i < SL_train_nega; i++) {
	//	neg_lst.push_back(full_lst_nega[i]);
	//}
	////SL image pre nega
	//for (int i = SL_train_nega; i < full_lst_nega.size(); i++) {
	//	pre_nega.push_back(full_lst_nega[i]);
	//}
	////sample_neg(full_neg_lst, neg_lst, pos_image_size);
	//clog << "...[done]" << endl;
	//std::vector<cv::Mat> dt_train_predic_posi_test;
	//clog << "Histogram of Gradients are being calculated for positive images...";
	//computeHOGs(pos_image_size, pos_lst, gradient_lst, flip_samples);
	//computeHOGs(pos_image_size, pos_lst, dt_train_predic_posi_test, flip_samples);
	//size_t positive_count = gradient_lst.size();
	//labels.assign(positive_count, +1);
	//clog << "...[done] ( positive count : " << positive_count << " )" << endl;
	//std::vector<cv::Mat> gradien_lst_pre_posi;
	//std::cout << "Histogram of Gradients are being calculated for Pre_positive images...";
	//computeHOGs(pos_image_size, pre_posi, gradien_lst_pre_posi, flip_samples);
	//clog << "...[done] ( positive count : " << pre_posi.size() << " )" << endl;

	//std::vector<cv::Mat> dt_train_predic_nega_test;
	//clog << "Histogram of Gradients are being calculated for negative images...";
	//computeHOGs(pos_image_size, neg_lst, gradient_lst, flip_samples);
	//computeHOGs(pos_image_size, neg_lst, dt_train_predic_nega_test, flip_samples);
	//size_t negative_count = gradient_lst.size() - positive_count;
	//labels.insert(labels.end(), negative_count, -1);
	//CV_Assert(positive_count < labels.size());
	//clog << "...[done] ( negative count : " << negative_count << " )" << endl;
	//std::vector<cv::Mat> gradien_lst_pre_enga;
	//std::cout << "Histogram of Gradients are being calculated for Pre_negative images...";
	//computeHOGs(pos_image_size, pre_nega, gradien_lst_pre_enga, flip_samples);
	//std::cout << "...[done] ( positive count : " << pre_nega.size() << " )" << endl;
	///// test predict
	////cv::Mat sample = cv::imread("D:\\Data\\positive\\positive\\imp100.jpg");
	////cv::Mat sample1 = cv::imread("D:\\Data\\positive\\positive\\imp12845.jpg");
	////cv::Mat sample2 = cv::imread("D:\\Data\\positive\\positive\\imp12683.jpg");
	////cv::Mat sample3 = cv::imread("D:\\Data\\positive\\positive\\imp462.jpg");
	////cv::Mat sample4 = cv::imread("D:\\Data\\positive\\positive\\imp12679.jpg");
	////cv::Mat sample5 = cv::imread("D:\\Data\\positive\\positive\\imp12579.jpg");
	////std::vector<cv::Mat> lst_sample,lst_gra_test;
	////lst_sample.push_back(sample);
	////lst_sample.push_back(sample1);
	////lst_sample.push_back(sample2);
	////lst_sample.push_back(sample3);
	////lst_sample.push_back(sample4);
	////lst_sample.push_back(sample5);

	////
	////std::vector<cv::Mat> lst_xample, lst_gra_testx;
	////cv::Mat xample = cv::imread("D:\\Data\\negative\\negative\\im330.jpg");
	////cv::Mat xample1 = cv::imread("D:\\Data\\negative\\negative\\im346.jpg");
	////cv::Mat xample2 = cv::imread("D:\\Data\\negative\\negative\\im28675.jpg");
	////cv::Mat xample3 = cv::imread("D:\\Data\\negative\\negative\\im27867.jpg");
	////cv::Mat xample4 = cv::imread("D:\\Data\\negative\\negative\\im24992.jpg");
	////cv::Mat xample5 = cv::imread("D:\\Data\\negative\\negative\\im14992.jpg");
	////lst_sample.push_back(xample);
	////lst_sample.push_back(xample1);
	////lst_sample.push_back(xample2);
	////lst_sample.push_back(xample3);
	////lst_sample.push_back(xample4);
	////lst_sample.push_back(xample5);
	//////// 
	////computeHOGs(pos_image_size, lst_sample, lst_gra_test, flip_samples);
	////computeHOGs(pos_image_size, lst_xample, lst_gra_testx, flip_samples);
	////std::vector<cv::Mat> vec_pre_posi_sample = Vec_cv::Mat_predic(lst_gra_test);
	////std::vector<cv::Mat> vec_pre_nega_xample = Vec_cv::Mat_predic(lst_gra_testx);
	///// test vector cv::Mat
	////2
	//std::vector<cv::Mat>vec_pre_posi_train = Vec_cv::Mat_predic(dt_train_predic_posi_test);
	//std::vector<cv::Mat>vec_pre_nega_train = Vec_cv::Mat_predic(dt_train_predic_nega_test);
	////end 2
	////1
	//std::vector<cv::Mat> vec_pre_posi = Vec_cv::Mat_predic(gradien_lst_pre_posi);
	//std::vector<cv::Mat> vec_pre_nega = Vec_cv::Mat_predic(gradien_lst_pre_enga);
	////end 1
	//////
	//cv::Mat train_data;
	//convert_to_ml(gradient_lst, train_data);
	//std::vector<int> pre_lable;
	//clog << "Training SVM...";
	//cv::Ptr< SVM > svm = SVM::create();
	//
	///* Default values to train SVM */
	//svm->setCoef0(0.0);
	//svm->setDegree(3);
	//svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3));
	//svm->setGamma(0);
	//svm->setKernel(SVM::LINEAR);
	//svm->setNu(0.5);
	//svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	//svm->setC(0.01); // From paper, soft classifier
	//svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	//svm->train(train_data, ROW_SAMPLE, labels);
	////2
	//float dt_pre_positive = Calu_predic(vec_pre_posi_train, svm);
	//float dt_pre_negative = Calu_predic(vec_pre_nega_train, svm);
	////end2
	////1
	//float pre_positive = Calu_predic(vec_pre_posi, svm);
	//float pre_negative = Calu_predic(vec_pre_nega, svm);
	////end 1 
	/////
	////float Float_pre_posi_sample_face = Calu_predic_test(vec_pre_posi_sample, svm,0);
	////float Float_pre_nega_xample_nonface = Calu_predic_test(vec_pre_nega_xample, svm,1);
	/////
	////float Prec_end = svm->predict(test_pre);
	//svm->save("my_detector_4h00_20_8_SVMtest.xml");
	//clog << "...[done]" << endl;
	//HOGDescriptor hog(pos_image_size,cv:: Size(24, 24),cv:: Size(8, 8),cv:: Size(8, 8), 9);
	////hog.save(obj_det_filename);
	////hog.winSize = pos_image_size;
	//hog.setSVMDetector(get_svm_detector(svm));
	//hog.save(obj_det_filename);
	////2
	//std::cout <<endl<<"datatrain_Predic positive=:\t"<< dt_pre_positive;
	//std::cout << endl<<"datatrain_Predic neagtive=:\t" << dt_pre_negative;
	////end 2
	////1
	//std::cout <<endl<<"dataTestPredic positive=:\t"<< pre_positive;
	//std::cout << endl<<"dataTestPredic neagtive=:\t" << pre_negative;
	////end 1
	//std::cout << endl << pre_lable.size();
	//std::cout << endl << "DOne";
	//getchar();
	////test_trained_detector(obj_det_filename, test_dir, videofilename);



////// TEST

	//std::String pathID = "inFo_image.csv";
	//std::vector<std::vector<float>> Getinfo = GetInforFromPath(pathID);
	//std::String path = "D:\\faces";
	//std::vector<cv::Mat> lst_img;
	//std::cout << "Load IMG...";
	//load_images_test(path, lst_img, false);
	//std::cout << endl << "Load Done ";
	//std::cout << endl << "draw ...";
	//lst_Draw_pic(lst_img, Getinfo);
	//std::cout << endl << "done";
	//getchar();
	//cv::Mat testDawr = DrawTest(lst_img[1], Getinfo[1]);
	//cv::imshow("abcd", testDawr);
	//cv::waitKey();


	cv::String obj_det_filename = "my_detector_16h00_20_8.xml"; //my_detector_10h41 my_detector_11h24
	//HOGDescriptor hog;
	cv::HOGDescriptor hog(cv::Size(40, 60), cv::Size(24, 24), cv::Size(8, 8), cv::Size(8, 8), 9);
	//std::cout << endl << hog.winSize << endl;
	hog.load(obj_det_filename);
	std::vector< cv::Rect > detections;
	std::vector< double > foundWeights;
	std::vector<float> foundWeights_Float;

	//IoU_lst_Img_train(lst_img, hog);
	//Test_lst_Img_NMSbox_IoU(lst_img, hog, Getinfo);
	//
	//Test_lst_Img(lst_img, hog, Getinfo);
	//
	//getchar();

	std::vector<int> indices;
	//cv::Mat img = cv::imread("D:\\faces\\image_0008.jpg");
	cv::Mat img = cv::imread("C:\\Users\\T450s\\source\\repos\\Haar_Like\\FaceTest\\3ng.jpg");
	cv::imshow("Original", img);
	//resize(img, img,cv:: Size((int)img.size().width / 5, (int)img.size().height / 5));
	hog.detectMultiScale(img, detections, foundWeights, 0,cv:: Size(8, 8),cv:: Size(0, 0), 1.3);
	foundWeights_Float = Convert_double_to_float(foundWeights);
	float score_threshold = 0.5, nms_threshold = 0.4;
	cv::dnn::NMSBoxes(detections, foundWeights_Float, score_threshold, nms_threshold, indices);
	//test 1 image

	for (int i = 0; i < indices.size(); i++) {
		int idx = indices[i];
		cv::Rect box = detections[idx];
		cv::Scalar color = cv::Scalar(0, 255, 0);
		rectangle(img, box, color, img.cols / 400 + 1);
	}
	cv::imshow("img", img);
	cv::waitKey();
	//for (size_t j = 0; j < detections.size(); j++)
	//{
	//	//cv::Scalar color = cv::Scalar(255, 0, 255);
	//	cv::Scalar color = cv::Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
	//	rectangle(img, detections[j], color, img.cols / 400 + 1);
	//	//cv::imshow("img", img);
	//	//cv::waitKey(300);
	//	//if (foundWeights[j] >= 0.85)
	//	////if (foundWeights[j] >= 0.5&&foundWeights[j] <0.58) 
	//	//{
	//	//	//Rect recFaces;
	//	//	//recFaces.x = detections[j].x;
	//	//	//recFaces.y = detections[j].y;
	//	//	//int a = (int)(img.size().width / 40);
	//	//	//recFaces.width = detections[j].width + (int)(img.size().width/40);
	//	//	//recFaces.height = detections[j].height + (int)(img.size().height / 56);
	//	//	cv::Scalar color = cv::Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
	//	//	//rectangle(img, recFaces, color, img.cols / 400 + 1);
	//	//	rectangle(img, detections[j], color, img.cols / 400 + 1);
	//	//	//rectangle(img, detections[j], cv::Scalar(0, 255, 0), 1, 8, 0);
	//	//}
	//}
	//cv::imshow("img", img);
	//cv::waitKey();

	return 0;
}