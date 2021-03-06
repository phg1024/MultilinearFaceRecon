#pragma once
//#define EIGEN_USE_LAPACKE 
//#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS
#include <iostream>
#include <fstream>
#include <Eigen/Eigen>
using namespace Eigen;
using Eigen::MatrixXf;

#define NOMINMAX
#include<windows.h>
#include<stdio.h>
#include<process.h>

using namespace std;
//using namespace cv;
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
using namespace cv;

#include "Utils/Timer.h"

static PhGUtils::Timer tUpdate;

class Adp_PCA_float
{
public:
	static unsigned int __stdcall threadProc(void* pV) { 
		//cout<<"updating model\n";
		Adp_PCA_float* p = (Adp_PCA_float*)pV;
		//cout<<p->blockNum<<endl;
		tUpdate.tic();
		p->updateModel();
		tUpdate.toc("update model");
		//cout<<"finished updating model\n";
		return 0;
	}

	//multi-thread
	bool readyToTransfer;
	void setNewMeanVec(float *);
	float *newMeanVec_GPU;
	void updateModel();

	bool isrunning;

	Adp_PCA_float(int dim,int maximumDim,bool outputData=false,int blockNum=10);
	void updateModel(MatrixXf &data,bool isNormalize=false);

	void normalizeMean();
	void initialModel(MatrixXf &data);

	void setModel(Mat &model,Mat &eigenVec,Mat &m,int dataNum,int modelDim);

	void setModel(float *model,float *eigenVec,float *m,int dataNum,int modelDim);
	void updateModel(float *data,int _sampleNum,bool isNormalize=false);

	void getMeanAndModel(float *meanVec);

	void checkReconError(MatrixXf &data);
	MatrixXf meanA;
	int n;//number of samples
	int dim;//dim of data
	int maximumDim;
	MatrixXf eigenVector;

	MatrixXf eigenValueMat;

	bool outputData;

	int blockNum;
	MatrixXf dataForUse;

};

