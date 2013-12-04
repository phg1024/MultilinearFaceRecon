#ifndef LSALIGNMENT_H
#define LSALIGNMENT_H
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include "GRandom.h"
using namespace cv;

class LSSquare{
public:
	void align(Mat &input,Mat &ref,vector<int>&,float &scale,float &theta,float &tx,float &ty);

	bool refineAlign(Mat &input,Mat &ref, int i,int j,float &distance,Mat *img=NULL);

	bool refineAlign(Mat &input,Mat &ref, vector<int> &initialInlier,float &distance,Mat *img=NULL);

	bool refineAlign_noChange(Mat &input,Mat &ref, vector<int> &initialInlier,float &distance,Mat *img=NULL);

	bool getInlier(Mat &input,Mat &ref, int i,int j,vector<int> &ind,Mat *img=NULL);

	Mat alignedShape;
};

#endif