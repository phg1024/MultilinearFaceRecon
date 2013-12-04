#include "LSAlignment.h"

#include <iostream>

using namespace std;

void LSSquare::align(Mat &input,Mat &ref,vector<int>&ind,float &scale,float &theta,float &tx,float &ty)
{
	int ptsNum=input.cols/2;

	int usedPtsNum=ind.size();
	Mat U=cv::Mat(usedPtsNum*2,1,CV_32FC1);
	Mat V=cv::Mat(usedPtsNum*2,4,CV_32FC1);

	//#pragma omp parallel for
	for (int i=0;i<usedPtsNum;i++)
	{
		U.at<float>(i,0)=input.at<double>(0,ind[i]);
		U.at<float>(usedPtsNum+i,0)=input.at<double>(0,ind[i]+ptsNum);
		//	V.at<float>(i,0)=AAM_exp->meanShape->pts[cind][0]-meanShapeCenter[0];
		//	V.at<float>(i,1)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		V.at<float>(i,0)=ref.at<double>(0,ind[i]);
		V.at<float>(i,1)=ref.at<double>(0,ind[i]+ptsNum);
		V.at<float>(i,2)=1;
		V.at<float>(i,3)=0;
		//V.at<float>(i+usedPtsNum,0)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		//V.at<float>(i+usedPtsNum,1)=-AAM_exp->meanShape->pts[cind][0]+meanShapeCenter[0];
		V.at<float>(i+usedPtsNum,0)=ref.at<double>(0,ind[i]+ptsNum);
		V.at<float>(i+usedPtsNum,1)=-ref.at<double>(0,ind[i]);
		V.at<float>(i+usedPtsNum,2)=0;
		V.at<float>(i+usedPtsNum,3)=1;
	}
	//cout<<usedPtsNum<<endl;
	Mat globalTransformation;
	solve(V,U,globalTransformation,DECOMP_SVD);
		//solve(V,U,globalTransformation,DECOMP_QR );

	scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	theta=-atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	tx=globalTransformation.at<float>(2,0);
	ty=globalTransformation.at<float>(3,0);
}

bool LSSquare::refineAlign(Mat &input,Mat &ref, int i,int j,float &distance,Mat *img)
{
	int ptsNum=ref.cols/2;

	vector<int> ind;
	ind.push_back(i);
	ind.push_back(j);

	float scale,theta,tx,ty;
	align(input,ref,ind,scale,theta,tx,ty);



	Mat ttt=ref.clone();
	ref.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;


	//Mat tmpImg=(*img).clone();

	//for (int i=0;i<input.cols/2;i++)
	//{
	//	circle(tmpImg,Point(input.at<double>(0,i),input.at<double>(0,i+ptsNum)),3,255);
	//}
	//circle(tmpImg,Point(input.at<double>(0,ind[0]),input.at<double>(0,ind[0]+ptsNum)),5,255);
	//circle(tmpImg,Point(input.at<double>(0,ind[1]),input.at<double>(0,ind[1]+ptsNum)),5,255);
	//namedWindow("1");
	//imshow("1",tmpImg);
	//waitKey();


	//alignedShape=ref.clone();

	ind.clear();
	float curDis;
	for (int i=0;i<ptsNum;i++)
	{
		curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
			(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
		if (curDis<10)
		{
			ind.push_back(i);
		}
	}

	if (ind.size()<ptsNum*0.6)
	{
		distance=1000000;
		return false;
	}

	align(input,ref,ind,scale,theta,tx,ty);
	ttt=ref.clone();
	ref.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;

	distance=0;
	int effectiveNum=0;
	for (int i=0;i<ptsNum;i++)
	{
		curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
			(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
		//if (curDis<5)
		{
			distance+=curDis;
			effectiveNum++;
		}
	}

	if (effectiveNum>18)
	{
		distance/=effectiveNum;
	}
	return true;

}

bool LSSquare::refineAlign_noChange(Mat &input,Mat &ref_in,vector<int> &initialInlier,float &distance,Mat *img)
{
	Mat ref=ref_in.clone();

	int ptsNum=ref.cols/2;

	vector<int> ind;
	for (int i=0;i<initialInlier.size();i++)
	{
		ind.push_back(initialInlier[i]);
	}

	float scale,theta,tx,ty;
	align(input,ref,ind,scale,theta,tx,ty);



	
	ref.colRange(0,ptsNum)=cos(theta)*ref_in.colRange(0,ptsNum)-sin(theta)*ref_in.colRange(ptsNum,ptsNum*2);
	ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ref_in.colRange(0,ptsNum)+cos(theta)*ref_in.colRange(ptsNum,ptsNum*2);
	ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;


	//Mat tmpImg=(*img).clone();

	//for (int i=0;i<input.cols/2;i++)
	//{
	//	circle(tmpImg,Point(input.at<double>(0,i),input.at<double>(0,i+ptsNum)),3,255);
	//}
	//circle(tmpImg,Point(input.at<double>(0,ind[0]),input.at<double>(0,ind[0]+ptsNum)),5,255);
	//circle(tmpImg,Point(input.at<double>(0,ind[1]),input.at<double>(0,ind[1]+ptsNum)),5,255);
	//namedWindow("1");
	//imshow("1",tmpImg);
	//waitKey();


	//alignedShape=ref.clone();

	ind.clear();
	float curDis;
	for (int i=0;i<ptsNum;i++)
	{
		curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
			(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
		if (curDis<5)
		{
			ind.push_back(i);
		}
	}

	if (ind.size()<ptsNum*0.6)
	{
		distance=1000000;
		return false;
	}

	align(input,ref_in,ind,scale,theta,tx,ty);

	ref.colRange(0,ptsNum)=cos(theta)*ref_in.colRange(0,ptsNum)-sin(theta)*ref_in.colRange(ptsNum,ptsNum*2);
	ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ref_in.colRange(0,ptsNum)+cos(theta)*ref_in.colRange(ptsNum,ptsNum*2);
	ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;

	distance=0;
	int effectiveNum=0;
	for (int i=0;i<ptsNum;i++)
	{
		curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
			(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
		//if (curDis<5)
		{
			distance+=curDis;
			if (curDis<5)
				effectiveNum++;
		/*	else
				distance+=100;*/
		}
	}

	//if (effectiveNum>14)
	{
		distance/=effectiveNum;
	}
	//distance=effectiveNum;
	return true;

}

bool LSSquare::refineAlign(Mat &input,Mat &ref,vector<int> &initialInlier,float &distance,Mat *img)
{
	int ptsNum=ref.cols/2;

	vector<int> ind;
	for (int i=0;i<initialInlier.size();i++)
	{
		ind.push_back(initialInlier[i]);
	}

	float scale,theta,tx,ty;
	align(input,ref,ind,scale,theta,tx,ty);



	Mat ttt=ref.clone();
	ref.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;


	//Mat tmpImg=(*img).clone();

	//for (int i=0;i<input.cols/2;i++)
	//{
	//	circle(tmpImg,Point(input.at<double>(0,i),input.at<double>(0,i+ptsNum)),3,255);
	//}
	//circle(tmpImg,Point(input.at<double>(0,ind[0]),input.at<double>(0,ind[0]+ptsNum)),5,255);
	//circle(tmpImg,Point(input.at<double>(0,ind[1]),input.at<double>(0,ind[1]+ptsNum)),5,255);
	//namedWindow("1");
	//imshow("1",tmpImg);
	//waitKey();


	//alignedShape=ref.clone();

	ind.clear();
	float curDis;
	for (int i=0;i<ptsNum;i++)
	{
		curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
			(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
		if (curDis<10)
		{
			ind.push_back(i);
		}
	}

	if (ind.size()<ptsNum*0.6)
	{
		distance=1000000;
		return false;
	}

	align(input,ref,ind,scale,theta,tx,ty);
	ttt=ref.clone();
	ref.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
	ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;

	distance=0;
	int effectiveNum=0;
	for (int i=0;i<ptsNum;i++)
	{
		curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
			(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
		//if (curDis<5)
		{
			distance+=curDis;
			effectiveNum++;
		}
	}

	if (effectiveNum>16)
	{
		distance/=effectiveNum;
	}
	return true;
	
}

bool LSSquare::getInlier(Mat &input,Mat &ref, int i,int j,vector<int> &inlierInd,Mat *img)
{
	int ptsNum=ref.cols/2;

	vector<int> ind;
	ind.push_back(i);
	ind.push_back(j);

	float scale,theta,tx,ty;

	bool ok=true;
	int times=0;
	while(1)
	{
		align(input,ref,ind,scale,theta,tx,ty);
		Mat ttt=ref.clone();
		float sinTheta=sin(theta);
		float cosTheta=cos(theta);
		ref.colRange(0,ptsNum)=cosTheta*ttt.colRange(0,ptsNum)-sinTheta*ttt.colRange(ptsNum,ptsNum*2);
		ref.colRange(ptsNum,ptsNum*2)=sinTheta*ttt.colRange(0,ptsNum)+cosTheta*ttt.colRange(ptsNum,ptsNum*2);
		ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
		ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;


	


		//alignedShape=ref.clone();

		ind.clear();
		float curDis;
		for (int i=0;i<ptsNum;i++)
		{
			curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
				(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
			if (curDis<5)
			{
				ind.push_back(i);
			}
		}

		if (ind.size()<6)
		{
			RandSample_V1(ptsNum,2,ind);
		}
		else
		{
			inlierInd.clear();
			for (int i=0;i<ind.size();i++)
			{
				inlierInd.push_back(ind[i]);
			}
		/*	Mat tmpImg=(*img).clone();

			for (int i=0;i<input.cols/2;i++)
			{
				circle(tmpImg,Point(input.at<double>(0,i),input.at<double>(0,i+ptsNum)),3,255);
			}
			for (int i=0;i<inlierInd.size();i++)
			{
				circle(tmpImg,Point(input.at<double>(0,inlierInd[i]),
					input.at<double>(0,inlierInd[i]+ptsNum)),5,Scalar(255,255,0));
			}
			
			namedWindow("initial inliers");
			imshow("initial inliers",tmpImg);
			waitKey();*/
			break;
		}

		times++;
		if (times>15)
		{
			ok=false;
			break;
		}
	}
	
	return ok;

	//align(input,ref,ind,scale,theta,tx,ty);
	//ttt=ref.clone();
	//ref.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
	//ref.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
	//ref.colRange(0,ptsNum)=(ref.colRange(0,ptsNum))*scale+tx;
	//ref.colRange(ptsNum,ptsNum*2)=(ref.colRange(ptsNum,ptsNum*2))*scale+ty;

	//distance=0;
	//int effectiveNum=0;
	//for (int i=0;i<ptsNum;i++)
	//{
	//	curDis=sqrtf((input.at<double>(0,i)-ref.at<double>(0,i))*(input.at<double>(0,i)-ref.at<double>(0,i))+
	//		(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i))*(input.at<double>(0,ptsNum+i)-ref.at<double>(0,ptsNum+i)));
	//	//if (curDis<5)
	//	{
	//		distance+=curDis;
	//		effectiveNum++;
	//	}
	//}

	//if (effectiveNum>18)
	//{
	//	distance/=effectiveNum;
	//}
	//return true;

}