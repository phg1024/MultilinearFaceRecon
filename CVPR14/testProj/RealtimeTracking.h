#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv.h"      
#include "highgui.h" 
#include <iostream>
#include "TwoLevelRegression.h"
#include "LocalGlobalRegression.h"
using namespace cv;
using namespace std;


#include "FaceFounder.h"

class FaceTracker
{
	public:
		FaceTracker(TwoLevelRegression *reg=NULL,bool isCVPR14=true);
		FaceTracker(LocalGlobalRegression *reg=NULL);

		bool isCVPR14;
		void setSampleNum(int);
		int sampleNum;
		FaceDetector d;
		TwoLevelRegression *reg;

		LocalGlobalRegression* reg_cvpr14;
		void start();

		void end();
		
};