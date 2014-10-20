// testProj.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include "testCode_14.h"

Mat fastAdd(Mat W,vector<int> &onesInd,int l, int r)
{
	cout<<l<<" "<<r<<endl;
	if(l==r)
		return W.col(onesInd[l]);
	else if(r==l+1)
		return W.col(onesInd[l])+W.col(onesInd[r]);
	else
	{
		int mid=(l+r)/2;
		if(mid>l&&mid<r)
		{
			return fastAdd(W,onesInd,l,mid)+fastAdd(W, onesInd,mid+1,r);
		}
	}

}

int _tmain(int argc, _TCHAR* argv[])
{
	//Mat W=Mat::zeros(5,10,CV_32FC1);
	//for(int i=0;i<W.rows;i++)
	//{
	//	for(int j=0;j<W.cols;j++)
	//		W.at<float>(i,j)=i*W.cols+j;
	//}

	//Mat res=Mat::zeros(W.rows,1,CV_32FC1);
	//for(int i=0;i<W.cols;i++)
	//	res+=W.col(i);
	//vector<int> onesInd;
	//for(int i=0;i<W.cols;i++)
	//	onesInd.push_back(i);
	//Mat res1=fastAdd(W,onesInd,0,onesInd.size()-1);

	//cout<<res.t()<<endl<<res1.t()<<endl;

	//return 0;

	//testLearnW("testW.txt");
	//return 0;
	
	evaluation_cvpr14();
	return 0;
//
	int curVal;
	cout<<"1: training 2: detect images 3: realtime perframe detect\n";
	cin>>curVal;

	if(curVal==1)
		train("model_LocalGlobal_RandLarge_WRCV_Advanced_fast.bin");
	else if(curVal==2)
		pridict("model_LocalGlobal_meanShape_WCV_fast.bin");
	else if(curVal==3)
	{
		realtimeFace("model_LocalGlobal_RandLarge_WRCV_Advanced_fast.bin");
	}
	//model_LocalGlobal_RandLarge_WCV_fast model_LocalGlobal_meanShape_WCV_fast
	//model_LocalGlobal_RandLarge_WCV_fast model_LocalGlobal_RandLarge_WRCV_fast
	//model_LocalGlobal_RandLarge_WRCV_moreComplete_fast
	//train("model_LocalGlobal_RandLarge_WRCV_moreComplete_fast.bin");
	//pridict("model_LocalGlobal_meanShape_WCV_fast.bin");
	//model_LocalGlobal_MeanShaoe_WCV model_LocalGlobal_Full5_WCV  model_LocalGlobal_MeanShape_WCV_fast
	//pridict_direct14("model_LocalGlobal_MeanShape_WCV_fast.bin");
	//treeSLTest();
	//pridict("modelFullyLocal.bin");
	//runEXE("D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\windows\\train.exe -s 12 -p 0 -B 1 D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\heart_scale D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\heart_scal_model_test");

	//loadW("W_Train\\modelX_10.dat");

	//loadW();
	return 0;
}

