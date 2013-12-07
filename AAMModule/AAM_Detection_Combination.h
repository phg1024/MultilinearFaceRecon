#ifndef AAM_DETECTION_COMBINATION_H
#define AAM_DETECTION_COMBINATION_H

#include "ImageRenderer.h"

#include "AAM_RealGlobal_GPU.h"
#include "randtree.h"
#include <iostream>
#include "geoHashing.h"


using namespace std;

class AAM_Detection_Combination
{
public:
	int currentShapePtsNum;
	vector<float> trackWithData(const unsigned char* cimg, const unsigned char* dimg, int w, int h) {

		cv::Mat m_img, gimg, depthImg;
		m_img.create(h, w, CV_8UC4);
		memcpy(m_img.ptr<BYTE>(), cimg, sizeof(unsigned char)*w*h*4);
		// convert to gray image
		cvtColor(m_img, gimg,CV_BGR2GRAY);

		depthImg = cv::Mat::zeros(h, w, CV_32FC1);
		const float standardDepth = 750.0;
		for (int i=0, idx=0;i<depthImg.rows;i++)
		{
			for (int j=0;j<depthImg.cols;j++,idx+=4)
			{
				int tmp=(dimg[idx]<<16|dimg[idx+1]<<8|dimg[idx+2]<<0);
				depthImg.at<float>(i,j)=tmp/standardDepth;
			}
		}

		int curStatus=0;

		currentShapePtsNum = 0;
		bool isSucceed=track_combine(gimg,depthImg,curStatus, 0,640,0,480, true);

		// return the content in currentShape
		if( isSucceed )
			return vector<float>(currentShape, currentShape + currentShapePtsNum * 2);
		else
			return vector<float>();
	}

public:
	AAM_Detection_Combination(double _AAMWeight=1,double _RTWeight=0,double _PriorWeight=0,double _localWeight=0,string colorDir="",string depthDir="",string aammodelPath="",string alignedShapeDir="",bool _isAdpt=false);
	//the absolute index of visible feature points
	float *absInd;

	string curPureName;

	//face detection
	CvHaarClassifierCascade *face_cascade;
	CvMemStorage*  faces_storage;

	void prepareModel(bool isApt=false);

	float *host_depthImage,*host_colorImage;

	void track_AAM(Mat &colorImg,Mat &depthImg,float *currentShape,int startInd=0,char*str1=NULL,char*str2=NULL);
	void ransac_MeanShape(float *curShape);


	int state;
	bool track_combine(Mat &colorImg,Mat &depthImg,int sx=0,int ex=0,int sy=0,int ey=0,bool initialPara=true);

	//with status
	bool track_combine(Mat &colorImg,Mat &depthImg,int &status,int sx=0,int ex=0,int sy=0,int ey=0,bool initialPara=true);
	void track_AAM_GPU(Mat &colorImg,Mat &depthImg,float *currentShape);

	


	void searchPics(string );

	float *hostDetectionResult;

	///////////////////AAM detection on GPU////////////////////////////
	void calculateData_onrun_AAM();
	///////////////////////////////////////////////

	//for models on CPU
	RandTree *rt;

	////////////NewADD for missing depth/////////////////////////
	RandTree *rt_colorRef;
	////////////NewADD for missing depth/////////////////////////

	AAM_RealGlobal_GPU *AAM_exp;

	float **finalPos;
	float *maximumProb;

	Mat tmpImg;
	//cpu
	void getTransformationInfo(vector<int> &InputInd,int **finalPos,Mat &shapeList,int shapeInd);
	void getTransformationInfo(vector<int> &InputInd,float **finalPos,Mat &shapeList,int shapeInd);
	void ransac_noSampling(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb);
	void ransac_noSampling_parrllel(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,Mat *img=NULL);

	void ransac_noSampling_Candidates(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,char *name="",Mat *img=NULL,Mat *depthImg=NULL);



	///////////geoHashing////////////////
	//geoHashing
	GeoHashing *geohashSearch;
	void buildHashTabel(string name);

	void buildHashTabel(Mat &shape);

	//using all inliers
	void geoHashing_Candidates(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,char *name="",Mat *img=NULL,Mat *depthImg=NULL,vector<Point2f> *candidatePts=NULL);

	//using only nearest inliers
	bool geoHashing_Candidates_nearestInlier(float **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,char *name="",Mat *img=NULL,Mat *depthImg=NULL,vector<Point2f> *candidatePts=NULL);


	vector<Point2f> candidatePoints[50];
	//void findSecondModes(int **samplePos,int width,int height, vector<int>&,float *,float *,Mat *img=NULL);
	void findSecondModes(int **samplePos,int width,int height, vector<int>&,float *,float *,Mat *img=NULL,int startX=0,int endX=0,int startY=0,int endY=0);
	void findSecondModes_Meanshift(int **samplePos,int width,int height, vector<int>&,float *,float *,Mat *img=NULL,int startX=0,int endX=0,int startY=0,int endY=0);
	void findSecondModes_Maxima(int **samplePos,int width,int height, vector<int>&,float *,float *,Mat *img=NULL,int startX=0,int endX=0,int startY=0,int endY=0);

	void findSecondModes_localMaxima(int **samplePos,int width,int height, vector<int>&,float *,float *,Mat *img=NULL,int startX=0,int endX=0,int startY=0,int endY=0);
	int totalShapeNum;
	int fullIndNum;
	Mat shapes;
	int sampleNumberFromTrainingSet;
	int sampleNumberFromProbMap,sampleNumberFromFeature;
	Mat *fullTrainingPos;
	Mat globalTransformation,globalTransformation_optimal;
	int window_threshold_small,window_threshold_large;
	int bestFitTrainingSampleInd;
	int ptsNum;
	int w_critical;
	double *meanShapeCenter;
	Mat eigenVectors;
	int round(float num)
	{
		if (num-floor(num)<=0.4)
		{
			return floor(num);
		}
		else
			return ceil(num);
	}

	float *host_preCalculatedConv;
	
	double AAMWeight,RTWeight,priorWeight,localWeight;

	float currentShape[200];
	float currentGT[200];

	float currentDetection[200];
	int startX,endX,startY,endY;
	char prefix[50];

	string colorRT_dir;
	string depthRT_dir;
	string AAMModelPath;

	bool hasVelocity;
	float veolocity[50],pridictedPts[50],lastPts[50];


	bool isAAMOnly,showNN;

	bool TemporalTracking;

	bool showProbMap;

	bool initial;//wheter we need to initialize the search window

	//ImageRenderer*          m_pDrawColor;

	float lastTheta;
	float lastlastTheta;
	float thetaVeolosity;
};






#endif