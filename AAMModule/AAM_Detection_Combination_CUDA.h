#ifndef AAM_DETECTION_COMBINATION_CUDA_H
#define AAM_DETECTION_COMBINATION_CUDA_H

#include "AAM_RealGlobal_CUDA.h"
#include "RandomizedTreeGPU.h"


int MAXIMUMPOINTDIM_COM=MAX_PIXEL_NUM;

class AAM_Detection_Combination_CUDA
{
public:
	AAM_Detection_Combination_CUDA(double _AAMWeight=1,double _RTWeight=0,double _priorWeight=0)
	{
		init();

		CUDA_CALL(cudaMalloc((void **)&cu_colorImage,MAXIMUMPOINTDIM_COM*sizeof(float)));
		CUDA_CALL(cudaMalloc((void **)&cu_depthImage,MAXIMUMPOINTDIM_COM*sizeof(float)));

	

		


		CUDA_CALL(cudaMalloc((void **)&cu_precalculatedConv,MAXIMUMPOINTDIM_COM*4*MAX_LABEL_NUMBER*sizeof(float)));
		

		//CUDA_CALL(cudaMalloc((void **)&cu_visibleIndex,MAX_LABEL_NUMBER*3*sizeof(float)));
		CUDA_CALL(cudaMalloc((void **)&cu_absVisibleIndex,MAX_LABEL_NUMBER*4*sizeof(float)));
		cu_visibleIndex=cu_absVisibleIndex+MAX_LABEL_NUMBER;
		cu_detectedFeatureLocations=cu_visibleIndex+MAX_LABEL_NUMBER;
		//CUDA_CALL(cudaMalloc((void **)&cu_detectedFeatureLocations,2*MAX_LABEL_NUMBER*sizeof(float)));
		//visibleIndex=new float[MAX_LABEL_NUMBER];
		//host_detectedFeatureLocations=new float[MAX_LABEL_NUMBER*2];
		CUDA_CALL(cudaHostAlloc((void **)&absInd,MAX_LABEL_NUMBER*4*sizeof(float),cudaHostAllocMapped));
		//CUDA_CALL(cudaHostAlloc(&visibleIndex,MAX_LABEL_NUMBER*3*sizeof(float),cudaHostAllocMapped));
		visibleIndex=absInd+MAX_LABEL_NUMBER;
		host_detectedFeatureLocations=visibleIndex+MAX_LABEL_NUMBER;

		
	


		CUDA_CALL(cudaMalloc((void **)&cu_conv,MAX_LABEL_NUMBER*2*MAX_LABEL_NUMBER*2*sizeof(float)));
		//CUDA_CALL(cudaMalloc((void **)&cu_JConv,MAX_LABEL_NUMBER*2**sizeof(float)));

		float *allZeroConv=new float[MAX_LABEL_NUMBER*2*MAX_LABEL_NUMBER*2];
		for (int i=0;i<MAX_LABEL_NUMBER*2*MAX_LABEL_NUMBER*2;i++)
		{
			allZeroConv[i]=0;
		}
		CUDA_CALL(cudaMemcpy(cu_conv,allZeroConv,MAX_LABEL_NUMBER*2*MAX_LABEL_NUMBER*2*sizeof(float),cudaMemcpyHostToDevice));
		delete []allZeroConv;

		//CUDA_CALL(cudaMalloc((void **)&cu_fullJacobian,(MAX_LABEL_NUMBER+MAXIMUMPOINTDIM_COM)**sizeof(float)));

		AAM_ENGINE=new AAM_Search_RealGlobal_CUDA(BlockNumGlobal);
		randomizedTrees_Color=new RandmizedTree_CUDA();
		randomizedTrees_Depth=new RandmizedTree_CUDA();

		AAM_ENGINE->cu_inputImg=cu_colorImage;
		randomizedTrees_Color->cu_colorImage=cu_colorImage;
		randomizedTrees_Color->cu_depthImage=cu_depthImage;

		randomizedTrees_Depth->cu_colorImage=cu_colorImage;
		randomizedTrees_Depth->cu_depthImage=cu_depthImage;

		AAMWeight=_AAMWeight;
		RTWeight=_RTWeight;
		priorWeight=_priorWeight;

		

		

	/*	host_colorImage=new float[MPN];
		host_depthImage=new float[MPN];*/

		//hostDetectionResult=new float[MPN*(1+MAX_LABEL_NUMBER)];

	}

	//for initial input transfer
	float *absInd;
	float *host_colorImage,*host_depthImage;
	float *p_mean,*p_sigma;


	float *cu_colorImage,*cu_depthImage;

	float *cu_precalculatedConv;

	float *cu_conv;

	float *cu_detectedFeatureLocations;
	float *host_detectedFeatureLocations;

	float *cu_visibleIndex;
	float *cu_absVisibleIndex;
	float *visibleIndex;
//	float *cu_fullJacobian;
	int visibleNum;
	

	AAM_Search_RealGlobal_CUDA* AAM_ENGINE;
	RandmizedTree_CUDA* randomizedTrees_Color;

	RandmizedTree_CUDA* randomizedTrees_Depth;

	

	float AAMWeight,RTWeight,priorWeight,localWeight;
	float RTWeight_backup,priorWeight_backup;

	//for prior
	float *cuPriorMean;
	float *cu_priorDifference;
	float *cuHessianPrior;

	float *cu_maximumPtsPos,*host_maximumPtsPos;

	int *cu_usedIndex;//int fullIndNum;
	//float *hostDetectionResult;

	//float *host_depthImage,*host_colorImage;

	//void track_combine(Mat &colorImg,Mat &depthImg);

	//for models on CPU
	//RandTree *rt;
	//AAM_RealGlobal_GPU *AAM_exp;
};


#endif