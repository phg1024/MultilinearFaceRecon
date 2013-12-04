#ifndef DETECTIONAAMCOMBINATION_H
#define DETECTIONAAMCOMBINATION_H

#include <iostream>
#include "AAM_RealGlobal_GPU.h"
#include "randtree.h"

using namespace std;


class DetectionWithAAM
{
public:
	DetectionWithAAM(float targetPrecison=0.995);

	float *host_depthImage,*host_colorImage;
	RandTree *rt;
	AAM_RealGlobal_GPU *AAM_exp;

	void searchPics(string );
	void track(Mat &colorIMG,Mat &depthIMG);

	int sampleNumberFromTrainingSet;
	int sampleNumberFromProbMap;
	int sampleNumberFromFeature;
	int window_threshold_small;
	int window_threshold_large;
	int bestFitTrainingSampleInd;
	
	int criticalIndNum;
	//vector<int> criticalIndList;
	int w_critical;
	void ransac(int **,int width,int,vector<int> &);
	void ransac_optimized(int **,int width,int,vector<int> &);
	void ransac_noSampling(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb);

	void ransac_noSampling_multiModes(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb);

	void track_NoSampling(Mat &colorIMG,Mat &depthIMG);
	void track_sampling(Mat &colorIMG,Mat &depthIMG);

	void track_sampling_CPU(Mat &colorIMG,Mat &depthIMG);

	int labelNum;
	float *hostDetectionResult;
	double *meanShapeCenter;

	int ptsNum;
	int shapeDim;
	int totalShapeNum;
	Mat shapes;
	Mat eigenVectors;

	int fullIndNum;
	//float getNN();
	void getTransformationInfo(vector<int> &usedInd,int **finalPos,Mat &shapeList,int shapeInd);
	void getTransformationInfo_optimize(vector<int> &usedInd,int **finalPos,Mat &shapeList,int shapeInd);
	void getVisibleInd(int **finalPos,vector<int> &outputInd,Mat &shapeList,int shapeInd);
	Mat globalTransformation;
	Mat globalTransformation_optimal;

	Mat fullDetectedPos;
	Mat *fullTrainingPos;

	float targetPrecison;

	int round(float num)
	{
		if (num-floor(num)<=0.4)
		{
			return floor(num);
		}
		else
			return ceil(num);
	}

	bool isFindModes;
};



#endif