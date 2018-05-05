#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <vector>
#include "vfc.h"

using namespace std;
using namespace cv;

bool convertEllipseKptsToStandardKpts(const vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, vector<KeyPoint>& kpts);

void draw_ellipse(Mat& image, vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, Mat& image_rgb);

int main(void)
{

	Mat img1 = imread("../HarrisAffine/image/adam_zoom1_front.png", 1);
	Mat img2 = imread("../HarrisAffine/image/adam_zoom1_45deg.png", 1);

	if (img1.empty() || img2.empty())
	{
		cout << "Reading the input image false!" << endl;
		return 0;
	}
	Mat img1gray, img2gray;   // Color convert to Gray Image for detection and descriptor
	if (img1.channels() == 3) {
		cvtColor(img1, img1gray, CV_RGB2GRAY);
	}
	else {
		img1.copyTo(img1gray);
	}
	if (img2.channels() == 3) {
		cvtColor(img2, img2gray, CV_RGB2GRAY);
	}
	else {
		img2.copyTo(img2gray);
	}
	//-- ��ʼ��HarrisLaplace��������
	Ptr<FeatureDetector> feature2D = xfeatures2d::HarrisLaplaceFeatureDetector::create();  
	//-- ��ʼ��SIFT ����������																				  
	Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::SIFT::create();  

    //-- Step 1: ��ȡ������
	vector<xfeatures2d::Elliptic_KeyPoint> aff_kpts1, aff_kpts2;
	//-- ��ʼ�����䲻��������   feature2D: ��������ȡ��ʽ  descriptor2D: ��������ȡ��ʽ
	Ptr<xfeatures2d::AffineFeature2D> affineFeature2D = xfeatures2d::AffineFeature2D::create(feature2D, descriptor2D);
	//-- ������������
	affineFeature2D->detect(img1gray, aff_kpts1);
	affineFeature2D->detect(img2gray, aff_kpts2);
	//-- ��⵽�������㼯��Ϣ���ݽṹ Elliptic_KeyPoint ת�� KeyPoint
	vector<KeyPoint> kpts1, kpts2;
	convertEllipseKptsToStandardKpts(aff_kpts1, kpts1);
	convertEllipseKptsToStandardKpts(aff_kpts2, kpts2);

	//-- Step 2: ������������������
	Mat desc1, desc2;
	//-- �Ҷ�ͼ�������������������   ������true/false true��ʾ�ⲿ�ṩ������(false���ʾ�෴)
	affineFeature2D->detectAndCompute(img1gray, Mat(), aff_kpts1, desc1, true);
	affineFeature2D->detectAndCompute(img2gray, Mat(), aff_kpts2, desc2, true);

	//-- Step 3: ƥ������������
	FlannBasedMatcher matcher;
	vector<DMatch> matches;
	matcher.match(desc1, desc2, matches);

	//-- Step 4: ͨ��VFC�Ƴ�����ƥ����
	vector<Point2f> X;  vector<Point2f> Y;
	X.clear();   Y.clear();

	for (unsigned int i = 0; i < matches.size(); i++) {
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		X.push_back(kpts1[idx1].pt);
		Y.push_back(kpts2[idx2].pt);
	}
	//-- VFC process
	double t = (double)getTickCount();
	VFC myvfc;
	myvfc.setData(X, Y);
	myvfc.optimize();
	vector<int> matchIdx = myvfc.obtainCorrectMatch();
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "VFC Times (s): " << t << endl;

	vector< DMatch > correctMatches;
	correctMatches.clear();
	for (unsigned int i = 0; i < matchIdx.size(); i++) {
		int idx = matchIdx[i];
		correctMatches.push_back(matches[idx]);
	}

	//--Step 5: ��ȡ��ȷƥ��㼯����Բ������Ϣ
	vector<xfeatures2d::Elliptic_KeyPoint> inliers_kpts1, inliers_kpts2;
	for (unsigned int i = 0; i < correctMatches.size(); ++i)
	{
		int idx1 = correctMatches[i].queryIdx;
		int idx2 = correctMatches[i].trainIdx;
		inliers_kpts1.push_back(aff_kpts1[idx1]);
		inliers_kpts2.push_back(aff_kpts2[idx2]);
	}
	//-- ���ƾ�ƥ������������Բ
	Mat img1_rgb, img2_rgb;
	draw_ellipse(img1, inliers_kpts1, img1_rgb);
	draw_ellipse(img2, inliers_kpts2, img2_rgb);
	//-- ���Ƽ���������Բ
	Mat img1_rgbs, img2_rgbs;
	draw_ellipse(img1, aff_kpts1, img1_rgbs);
	draw_ellipse(img2, aff_kpts2, img2_rgbs);
	//-- ����ƥ����ͼ
	Mat img_correctMatches;
	drawMatches(img1_rgb, kpts1, img2_rgb, kpts2, correctMatches, img_correctMatches, Scalar::all(-1), \
		        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow("PreciseMatchWithVFC");
	imshow("PreciseMatchWithVFC", img_correctMatches);
	imwrite("../HarrisAffine/image/harrisAffine_match.png", img_correctMatches);
	imwrite("../HarrisAffine/image/harris_img1.png", img1_rgbs);
	imwrite("../HarrisAffine/image/harris_img5.png", img2_rgbs);

	waitKey(0);

	return 0;
}



void draw_ellipse(Mat& image, vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, Mat& image_rgb)
{   // ��ɫͼ���Ͻ����ж������
	if (1 == image.channels()) {
		image_rgb = Mat(Size(image.cols, image.rows), CV_8UC3);
		cvtColor(image, image_rgb, COLOR_GRAY2BGR);
	}
	else {
		image.copyTo(image_rgb);
	}
	Point center; // ���ĵ�����
	Size axes;    // ��Բ������
				  // ������������Բ����
	for (int i = 0; i<elliptic_keypoints.size(); ++i)
	{
		center.x = elliptic_keypoints[i].pt.x;
		center.y = elliptic_keypoints[i].pt.y;
		axes.width = elliptic_keypoints[i].axes.width;
		axes.height = elliptic_keypoints[i].axes.height;
		double angle = elliptic_keypoints[i].angle;  // �Ƕ�
													 // ������Բͼ��
		ellipse(image_rgb, center, axes, angle * 180 / CV_PI, 0, 360, Scalar(255, 255, 255), 2, 8);
		// �������ĵ�����
		circle(image_rgb, center, 1, Scalar(0, 0, 255));
	}
}

bool convertEllipseKptsToStandardKpts(const vector<xfeatures2d::Elliptic_KeyPoint>& elliptic_keypoints, vector<KeyPoint>& kpts)
{
	if (0 == elliptic_keypoints.size())
		return false;
	for (int i = 0; i < elliptic_keypoints.size(); ++i)
	{
		KeyPoint kpt;
		kpt.pt.x = elliptic_keypoints[i].pt.x;
		kpt.pt.y = elliptic_keypoints[i].pt.y;
		kpt.angle = elliptic_keypoints[i].angle;
		float diam = elliptic_keypoints[i].axes.height*elliptic_keypoints[i].axes.width;
		kpt.size = sqrt(diam);
		kpts.push_back(kpt);
	}
	return true;
}



//Mat img1s, img2s;
//img1s = Mat(Size(img1.cols, img1.rows), CV_8UC3);
//img2s = Mat(Size(img2.cols, img2.rows), CV_8UC3);
//cvtColor(img1, img1s, CV_GRAY2BGR);
//cvtColor(img2, img2s, CV_GRAY2BGR);
//imwrite("adam_zoom1_front.png", img1s);
//imwrite("adam_zoom1_65deg.png", img2s);

//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::SURF::create();  // SURF ����������
//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::DAISY::create();  // DAISY������
//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::BoostDesc::create(); // Learning Image Descriptor with Boosting
//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::FREAK::create();  // FREAK������
//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::BriefDescriptorExtractor::create(); //BRIEF ������
//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::LATCH::create(); // LATCH ������
//Ptr<DescriptorExtractor> descriptor2D = xfeatures2d::LUCID::create(); // LUCID ������