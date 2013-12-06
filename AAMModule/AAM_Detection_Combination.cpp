#include "AAM_Detection_Combination.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <iostream>
#include "definationCPU.h"
#include <tchar.h>
#include <conio.h>
#include "CodeTimer.h"


//#include "CUDA_basic.h"
using namespace std;
//int combineIndList[]={2,3  ,  10  ,  11   , 22 ,   23    , 0  ,   1  ,   8,     9  ,  24 ,   27   ,  12   , 13  ,  14   , 15  ,  16  ,  17 ,   18,
//	19 ,   20   , 21   , 25 ,   26};
const float e=2.7173;
//5 13 21 29 43 49
//1 3 6 10 12 14
int combineIndList[]={0,4,8,12,16,18,20,22,24,26,28,30,42,45,48,51,56,61,76,35,38};
int specialInd[]={1,3, 6, 10, 12, 14};
Mat colorImgBackUP;

string nameList[5000];


float *host_colorImage_global,*host_depthImage_global;
float *p_mean_global,*p_sigma_global;
float *absInd_Global;
void setColorDepthData(float *colorImgData,float *depthImgData)
{
	host_colorImage_global=colorImgData;
	host_depthImage_global=depthImgData;
}



extern "C" void setPMeanHessian(float *p_mean,float *p_sigma)
{
	p_mean_global=p_mean;
	p_sigma_global=p_sigma;
}

extern "C" void setAbsIndex(float *absInd)
{
	absInd_Global=absInd;
}

bool comparator ( const distancePir& l, const distancePir& r)
{ return l.first < r.first; }

AAM_Detection_Combination::AAM_Detection_Combination(double _AAMWeight,double _RTWeight,double _PriorWeight,double _localWeight,string colorDir,string depthDir,string aammodelPath,string alignedShapeDir,bool isAdp)
{
		face_cascade=(CvHaarClassifierCascade*)cvLoad("d:/opencv 2.4/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
		//face_cascade=(CvHaarClassifierCascade*)cvLoad("d:/fuhao/OpenCV2.1/data/haarcascades/haarcascade_frontalface_alt.xml");
		faces_storage=cvCreateMemStorage(0);

		AAMWeight=_AAMWeight;
		RTWeight=_RTWeight;
		priorWeight=_PriorWeight;
		localWeight=_localWeight;
		AAMModelPath=aammodelPath;


		colorRT_dir=colorDir;
		depthRT_dir=depthDir;

		fullIndNum=sizeof(combineIndList)/sizeof(int);

		prepareModel(isAdp);

		host_colorImage=new float[MPN];
		host_depthImage=new float[MPN];

		hostDetectionResult=new float[MPN*(1+MAX_LABEL_NUMBER)];

		finalPos=new float *[MAX_LABEL_NUMBER];
		for (int i=0;i<MAX_LABEL_NUMBER;i++)
		{
			finalPos[i]=new float [2];
		}

		//CPU
		ptsNum=AAM_exp->meanShape->ptsNum;
		maximumProb=new float[MAX_LABEL_NUMBER];
		sampleNumberFromTrainingSet=ceil(log(1-0.95)/log(5.0f/6.0f));
		sampleNumberFromProbMap=1;
		sampleNumberFromFeature=ceil(log(1-0.95)/log(5.0f/9.0f));
		window_threshold_small=10;
		window_threshold_large=20;
		window_threshold_small*=window_threshold_small;
		window_threshold_large*=window_threshold_large;
		//ifstream in("D:\\Fuhao\\AAM model\\allignedshape.txt",ios::in);
		//ifstream in("D:\\Fuhao\\face dataset\\train_all_final\\allignedshape.txt",ios::in);
//		ifstream in("D:\\Fuhao\\face dataset\\train_larger database\\allignedshape.txt",ios::in);
		//ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_91_90.txt",ios::in);
		ifstream in(alignedShapeDir.c_str(),ios::in);
		//ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_90_90.txt",ios::in);
		float x,y;
		totalShapeNum=0;
		while(in)
		{
			for (int i=0;i<ptsNum;i++)
			{
				in>>x>>y;
			}
			totalShapeNum++;		
		}
		totalShapeNum--;
		in.clear();
		in.seekg(0);
		shapes=Mat::zeros(ptsNum*2,totalShapeNum+1,CV_64FC1);
		for (int j=0;j<ptsNum;j++)
		{
			shapes.at<double>(j,0)=AAM_exp->meanShape->pts[j][0]-meanShapeCenter[0];
			shapes.at<double>(j+ptsNum,0)=AAM_exp->meanShape->pts[j][1]-meanShapeCenter[1];
		}
		for (int i=1;i<=totalShapeNum;i++)
		{
			for (int j=0;j<ptsNum;j++)
			{ 
				in>>x>>y;
				shapes.at<double>(j,i)=x*AAM_exp->shape_scale;
				shapes.at<double>(j+ptsNum,i)=y*AAM_exp->shape_scale;
			}
		}


		

	
	/*	ofstream out("D:\\Fuhao\\face dataset\\train_larger database\\allignedshape_feature.txt",ios::out);
		out<<totalShapeNum<<" "<<ptsNum<<endl;
		for (int i=0;i<shapes.cols;i++)
		{
			for (int j=0;j<shapes.rows;j++)
			{
				out<<shapes.at<double>(j,i)<<" ";
			}
			out<<endl;
		}
		out.close();*/

	/*	ofstream out("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_feature_90_90.txt",ios::out);
		out<<totalShapeNum<<" "<<ptsNum<<endl;
		for (int i=0;i<shapes.cols;i++)
		{
			for (int j=0;j<shapes.rows;j++)
			{
				out<<shapes.at<double>(j,i)<<" ";
			}
			out<<endl;
		}
		out.close();*/



		w_critical=ceil((float)11*0.6);


		Mat shapeCenter=shapes.clone();
		for (int i=1;i<=totalShapeNum;i++)
		{
			for (int j=0;j<ptsNum*2;j++)
			{
				shapeCenter.at<double>(j,i)=shapes.at<double>(j,0);
			}

		}
		cout<<AAM_exp->m_s_vec.rows<<" "<<AAM_exp->m_s_vec.cols<<" "<<AAM_exp->shapeDim<<endl;
		cout<<shapes.rows<<" "<<shapes.cols<<" "<<shapeCenter.rows<<" "<<shapeCenter.cols<<endl;
		eigenVectors=AAM_exp->m_s_vec.rowRange(Range(0,AAM_exp->shapeDim))*(shapes-shapeCenter);

		host_preCalculatedConv=new float [MAX_LABEL_NUMBER*MPN*4];

		cout << "building hash table ..." << endl;
		buildHashTabel(shapes);
		cout << "done" << endl;

		state=1;
		hasVelocity=false;
		isAAMOnly=false;
		showNN=false;
		TemporalTracking=false;
		showProbMap=false;
		for (int i=0;i<fullIndNum*2;i++)
		{
			pridictedPts[i]=0;
		}

		initial=true;//true means need to be initialized



			totalShapeNum=shapes.cols;
		fullTrainingPos=new Mat[totalShapeNum];
		int cind;
		for (int j=0;j<totalShapeNum;j++)
		{
			fullTrainingPos[j].create(fullIndNum*2,4,CV_32FC1);
			for (int i=0;i<fullIndNum;i++)
			{
				cind=combineIndList[i];

				fullTrainingPos[j].at<float>(i,0)=shapes.at<double>(cind,j);
				fullTrainingPos[j].at<float>(i,1)=shapes.at<double>(cind+ptsNum,j);
				fullTrainingPos[j].at<float>(i,2)=1;
				fullTrainingPos[j].at<float>(i,3)=0;

				fullTrainingPos[j].at<float>(i+fullIndNum,0)=shapes.at<double>(cind+ptsNum,j);
				fullTrainingPos[j].at<float>(i+fullIndNum,1)=-shapes.at<double>(cind,j);;
				fullTrainingPos[j].at<float>(i+fullIndNum,2)=0;
				fullTrainingPos[j].at<float>(i+fullIndNum,3)=1;
			}
		}

		absInd=new float[MAX_LABEL_NUMBER];
		lastTheta=0;
		/*m_pDrawColor = new ImageRenderer();
		m_pDrawColor->setPtsInfo(currentShape,ptsNum);

		m_pDrawColor->Initialize(NULL, m_pD2DFactory, 640, 480, 640 * sizeof(long));
		if (FAILED(hr))
		{
			return;
		}*/
}


void AAM_Detection_Combination::searchPics(string listName)
{
	////read in the name
	//char cname[800];
	//ifstream inn("D:\\Fuhao\\face dataset\\train_larger database\\imgList.txt",ios::in);
	//int totalNum;
	//inn>>totalNum;
	//inn.getline(cname,800);
	//for (int i=0;i<totalNum;i++)
	//{
	//	inn.getline(cname,800);
	//	nameList[i]=cname;
	//}


	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	cout<<_imgNum<<" "<<name<<endl;
	//muscle 11675, david 11705,rock 11941
	//int startNum=49-49+11361-11361;//+11811-11361;//+11378-11361;//+11410-11361;//+714-361;//477-49;//63-49;//381-49;//459-49;//7353-49;//428-49;

	int curFrameNum=0;

	int startNum=0;//725;//725;//650
	//zain: 10146
	//int startNum=10146-10146;
	//int startNum=49-49+10958-10529;//for kaitlin
	//int startNum=0;//for syn data testing
	AAM_exp->currentFrame=startNum;
	AAM_exp->setGlobalStartNum(startNum);

	for (int ll=0;ll<_imgNum;ll++)
	{
		
		//readin the image name first
		in.getline(name,500,'\n');
		

		if (ll<startNum)//||ll>459-49)
		{
			continue;
		}

		/*if (ll>=155)
		{
			break;
		}*/

		if (ll==349) //253,350 for babe, 357-222 for me
		{
			setShowSingle(true);
			AAM_exp->showSingleStep=true;
			state=1;
		}

		//if (ll==265) //253,350 for babe, 357-222 for me
		//{
		//	setShowSingle(false);
		//	AAM_exp->showSingleStep=false;
		//	state=1;
		//}
		
		//cout<<"processing "<<ll<<" th image\n";
		//AAM_exp->setGlobalStartNum(AAM_exp->startNum);

		string curName=name;
		curName=curName.substr(0,curName.length()-4);
		curPureName=curName;

		int cStart=11361;
		char *refDir="D:/Fuhao/Facial feature points detection/Final_X64_1.2 Full GPU - code optimization - RandomizedTrees - AAM -KNN finalization-More Points/Final_X64/AAM+detection_pridicted";

		//char *refDir="D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\david\\1\\kinectColor\\AAM Sequence\\";

		//char *refDir="D:\\Fuhao\\face dataset\\kinect sequences\\MSRA data\\Muscle\\kinectColor\\AAM Sequence\\";

		/*LONGLONG   t1,t2; 
		LONGLONG   persecond; 
		QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
		QueryPerformanceCounter((LARGE_INTEGER   *)&t1);*/

	/*	char name[500];
		sprintf(name, "AAM+detection/colordepth/%s_%dModes.txt", AAM_exp->prefix,11361+AAM_exp->currentFrame);
		ifstream inModes(name,ios::in);
		if (inModes)
		{
			inModes.close();
			AAM_exp->currentFrame++;
			continue;
		}*/

		//cout<<ll<<" "<<name<<endl;

		string saveName=name;
		saveName=saveName.substr(0,saveName.length()-4);
		string prob_name=saveName;
		saveName+=".txt";
		cout<<saveName<<endl;

	//////////////read in the data and copy to groundtruth directory//////////////////////
	//	char nameee[500];
	//	sprintf(nameee, "groundtruth/%s_%d.txt", AAM_exp->prefix,11361+AAM_exp->currentFrame);

	//	ifstream f1(saveName.c_str(), fstream::binary);
	//	ofstream f2(nameee, fstream::trunc|fstream::binary);
	//	f2 << f1.rdbuf();

	//	string imgCpy=nameee;
	//	saveName=name;
	//	imgCpy=imgCpy.substr(0,imgCpy.length()-4);
	//	imgCpy+=saveName.substr(saveName.length()-4,saveName.length());
	//	cout<<imgCpy<<endl;
	//	ifstream f3(name, fstream::binary);
	//	ofstream f4(imgCpy.c_str(), fstream::trunc|fstream::binary);
	//	f4 << f3.rdbuf();

	//	imgCpy=nameee;
	//	saveName=name;
	//	saveName=saveName.substr(0,saveName.length()-4)+"_depth.png";
	//	imgCpy=imgCpy.substr(0,imgCpy.length()-4);
	//	imgCpy+="_depth.png";
	//	cout<<imgCpy<<endl;
	//	ifstream f5(saveName.c_str(), fstream::binary);
	//	if (f5)
	//	{
	//		ofstream f6(imgCpy.c_str(), fstream::trunc|fstream::binary);
	//		f6 << f5.rdbuf();
	//	}
	//	

	//	AAM_exp->currentFrame++;
	//	//CopyFile(_TEXT(saveName.c_str()), _TEXT(nameee), true);;
	///*	ifstream in_g(saveName.c_str(),ios::in);
	//	int cwidth,cheight;
	//	int cnum;
	//	float tx,ty;
	//	if (in_g)
	//	{
	//		in_g>>cwidth>>cheight;
	//		in_g>>cnum>>cnum;
	//		for (int i=0;i<ptsNum;i++)
	//		{
	//			in>>currentShape[i]>>currentShape[i+ptsNum];
	//			currentShape[i]*=cwidth;
	//			currentShape[i+ptsNum]*=cheight;
	//		}
	//	}*/
	//	continue;
		/////////////////////////////////////////////////////////////////
	/*	if (ll>_imgNum-8)
		{
			continue;
		}*/
		
		Mat m_img=imread(name,0);

		colorImgBackUP=imread(name);

	

		Mat depthImg=Mat::ones(colorImgBackUP.rows,colorImgBackUP.cols,CV_32FC1);


		//namedWindow("1");
		//imshow("1",depthImg);
		//waitKey();

		depthImg*=0;

		///////////////////////read in the depth data//////////////////////////////

		//depth
		string depthFileName=name;
		depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		depthFileName+="_depth.txt";
		int x,y;
		double depthValue;

		string depthPNGName=name;
		depthPNGName=depthPNGName.substr(0,depthPNGName.length()-4);
		depthPNGName+="_depth.png";

		string depthPNGPureName=name;
		depthPNGPureName=depthPNGPureName.substr(0,depthPNGPureName.length()-4);
		depthPNGPureName+=".png";
		//ifstream in22(depthFileName.c_str(),ios::in);

		//while(in22)
		//{
		//	in22>>x>>y>>depthValue;
		//	if (abs(depthValue)<=1)
		//	{
		//		depthImg.at<float>(y,x)=depthValue;
		//	}
		//}

		double standardDepth=750;
		ifstream in22(depthFileName.c_str(),ios::in);
		ifstream inPNG(depthPNGName.c_str(),ios::in);
		ifstream inPNGPure(depthPNGPureName.c_str(),ios::in);

		////save image and return
		//char name[500];
		//sprintf(name, "D:\\Fuhao\\Facial feature points detection\\Final_X64_1.2 Full GPU - code optimization - RandomizedTrees - AAM -KNN finalization-More Points\\Final_X64\\originalImg\\%s_tracked_%d.png",
		//	AAM_exp->prefix,11361+AAM_exp->currentFrame);
		////imwrite(name,colorImg);
		//if (inPNG)
		//{
		//	Mat img=imread(depthPNGName);
		//	imwrite(name,img);
		//}
		//else if (inPNGPure)
		//{
		//	Mat img=imread(depthPNGPureName);
		//	imwrite(name,img);
		//}
		//AAM_exp->currentFrame++;

		//continue;


		if (in22)
		{
			in22>>y>>x>>depthValue;
			in22.clear(); 
			in22.seekg(0);


			if (y==0&&x==0&&depthValue==0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						in22>>depthValue;
						if (depthValue!=0)
						{
							//depthImg.at<float>(i,j)=2980-depthValue;
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}
						else
						{
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}

					}
				}
			}
			else
			{
				double depthScale;
				string scaleFileName="G:\\face database\\facegen database depth only";
				scaleFileName+=depthFileName.substr(depthFileName.find_last_of('\\'),depthFileName.find_first_of('_')-depthFileName.find_last_of('\\'));
				scaleFileName+="_scale.txt";

				ifstream inscale(scaleFileName.c_str(),ios::in);
				inscale>>depthScale;
				inscale.close();

				//for synthesized data, need to be adjusted
				while(in22)
				{
					in22>>y>>x>>depthValue;
					if (depthValue==-1)
						depthValue=0;
					else
					{
						depthValue=(5-depthValue)/depthScale/standardDepth;
						//cout<<depthValue<<endl;
					}
					//if (abs(depthValue)<=1)
					{
						depthImg.at<float>(x,y)=depthValue;
					}
				}

			}
		}
		else if (inPNG)
		{
			Mat depthPNGImg=imread(depthPNGName);
			//colorImg.create(depthPNGImg.rows,depthPNGImg.cols,CV_32FC1);

			////////////NewADD for missing depth: depth inpainting/////////////////////////
		/*	namedWindow("old depth");
			imshow("old depth",depthPNGImg);
			waitKey();*/

			//Mat mask=Mat::zeros(depthPNGImg.rows,depthPNGImg.cols,CV_8UC1);
			//for (int k=startY;k<endY;k++)
			//{
			//	for (int p=startX;p<endX;p++)
			//	{
			//		float tmptmp=(depthPNGImg.at<Vec3b>(k,p)[0]<<16|depthPNGImg.at<Vec3b>(k,p)[1]<<8|depthPNGImg.at<Vec3b>(k,p)[2]<<0);
			//		if (tmptmp==0)
			//		{
			//			mask.at<uchar>(k,p)=1;
			//		}
			//		
			//	}
			//}
			////Mat depthPNG_new;
			//inpaint(depthPNGImg,mask,depthPNGImg,5,INPAINT_TELEA);

			/*namedWindow("inpainted depth");
			imshow("inpainted depth",depthPNGImg);
			waitKey();*/
			///////////////////////////////////////////////////////////////////////////////

			float tmp;
			for (int i=0;i<depthPNGImg.rows;i++)
			{
				for (int j=0;j<depthPNGImg.cols;j++)
				{
					//tmp=depthPNGImg.at<Vec3b>(i,j)[0]*256*256+depthPNGImg.at<Vec3b>(i,j)[1]*256+depthPNGImg.at<Vec3b>(i,j)[2];
					tmp=(depthPNGImg.at<Vec3b>(i,j)[0]<<16|depthPNGImg.at<Vec3b>(i,j)[1]<<8|depthPNGImg.at<Vec3b>(i,j)[2]<<0);
					/*	if (tmp!=0)
					{
					cout<<tmp<<" "<<(depthPNGImg.at<Vec3b>(i,j)[0]<<16|depthPNGImg.at<Vec3b>(i,j)[1]<<8|depthPNGImg.at<Vec3b>(i,j)[2]<<0)<<endl;

					}*/
					depthImg.at<float>(i,j)=tmp/standardDepth;

				}
			}
		}
		else if (inPNGPure)
		{
			Mat depthPNGImg=imread(depthPNGPureName);
			//colorImg.create(depthPNGImg.rows,depthPNGImg.cols,CV_32FC1);

			float tmp;
			for (int i=0;i<depthPNGImg.rows;i++)
			{
				for (int j=0;j<depthPNGImg.cols;j++)
				{
					//tmp=depthPNGImg.at<Vec3b>(i,j)[0]*256*256+depthPNGImg.at<Vec3b>(i,j)[1]*256+depthPNGImg.at<Vec3b>(i,j)[2];
					tmp=(depthPNGImg.at<Vec3b>(i,j)[0]<<16|depthPNGImg.at<Vec3b>(i,j)[1]<<8|depthPNGImg.at<Vec3b>(i,j)[2]<<0);
					/*	if (tmp!=0)
					{
					cout<<tmp<<" "<<(depthPNGImg.at<Vec3b>(i,j)[0]<<16|depthPNGImg.at<Vec3b>(i,j)[1]<<8|depthPNGImg.at<Vec3b>(i,j)[2]<<0)<<endl;

					}*/
					depthImg.at<float>(i,j)=tmp/standardDepth;

				}
			}
		}
		else //if it is in xml format, it should be exactly the depth. If there are minus values, then it should be 1200-depth
		{
			depthFileName=name;
			depthFileName=depthFileName.substr(0,depthFileName.length()-4);
			depthFileName+=".xml";
			CvMat* tmpDepth = (CvMat*)cvLoad( depthFileName.c_str());
			depthImg.release();
			depthImg=cvarrToMat(tmpDepth).clone();


			///*	Mat tmp=cvarrToMat(tmpDepth);*/
			//	for (int i=0;i<depthImg.rows;i++)
			//	{
			//		for (int j=0;j<depthImg.cols;j++)
			//		{
			//			if (depthImg.at<float>(i,j)<0)
			//			{
			//				cout<<depthImg.at<float>(i,j)<<" ";
			//			}
			//			
			//		}
			//		cout<<endl;
			//	}
			double testMaxval=500;
			int maxIdx[3]; 
			minMaxIdx(depthImg, &testMaxval, 0, maxIdx,0);
			if (testMaxval<0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)=1200-depthImg.at<float>(i,j);
						}

					}
				}
			}

			minMaxIdx(depthImg,0, &testMaxval, 0, maxIdx);
			if (testMaxval>10)	//else, already divided by the standard depth
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0) {
							depthImg.at<float>(i,j)/=standardDepth;
							////bad values
							//if (colorImg.at<float>(i,j)<0.1||colorImg.at<float>(i,j)>10)
							//{
							//	cout<<colorImg.at<float>(i,j)<<" ";
							//	colorImg.at<float>(i,j)=0;
							//}
						}
					}
				}
			}
		}
		cout<<"after reading depth\n";
		//depthImg-=46.0f/standardDepth;

		
		////face detection here
		int startX,endX,startY,endY;

		if (ll==startNum||initial)
		{

			
			//IplImage *img=&((IplImage)colorImgBackUP);
			//CvSeq* faces = cvHaarDetectObjects( img, face_cascade, faces_storage,
			//	1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
			//	cvSize(30, 30) );

			////printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
			//for( int i = 0; i < (faces ? faces->total : 0); i++ )
			//{
			//	CvRect* r = (CvRect*)cvGetSeqElem( faces, i );

			//	if (r->width>50)
			//	{
			//		startX=r->x;
			//		startY=r->y;
			//		endX=startX+r->width;
			//		endY=startY+r->width+15;
			//	}
			//}
			

			startX=0;endX=500;
			startY=0;endY=500;

			////08
			//startX=297;endX=497;
			//startY=152;endY=333;

			////11
			//startX=273;endX=446;
			//startY=123;endY=278;

			////17
			//startX=264;endX=419;
			//startY=233;endY=369;

			//21
			//startX=223;endX=420;
			//startY=253;endY=409;

			if (AAM_exp->showSingleStep&&initial)
			{
				////277
				//startX=326;endX=439;
				//startY=166;endY=283;

				////350
				//startX=320;endX=404;
				//startY=173;endY=261;

				////30
				//startX=315;endX=405;
				//startY=182;endY=274;

				//64
				//startX=268;endX=379;
				//startY=275;endY=367;
			}
		}
		else
		{
			startX=startY=100000;
			endX=endY=-1;
			int cptsNum=AAM_exp->meanShape->ptsNum;
			for (int i=0;i<cptsNum;i++)
			{
				if (currentShape[i]<startX)
				{
					startX=currentShape[i];
				}
				if (currentShape[i]>endX)
				{
					endX=currentShape[i];
				}

				if (currentShape[i+cptsNum]<startY)
				{
					startY=currentShape[i+cptsNum];
				}
				if (currentShape[i+cptsNum]>endY)
				{
					endY=currentShape[i+ptsNum];
				}
			}
			startX-=10;
			startY-=10;
			endX+=10;
			endY+=10;
			//startX=274;
			//endY=358;
		}

		//cout << "Check point 1" << endl;

		//use the groundtruth
		if(AAM_exp->showSingleStep)
		{
			//check if we have ground truth, if yes, use it to limit depth data
			/*char name[500];
			sprintf(name, "%s/%s_tracked_%d.txt", refDir,prefix,cStart+AAM_exp->currentFrame);*/
			string name=curPureName;
			string dirName=curPureName.substr(0,curPureName.find_last_of('\\')+1);
			name=curPureName.substr(curPureName.find_last_of('\\')+1,curPureName.length()-curPureName.find_last_of('\\'));

			string gtName=dirName+"AAM FINAL\\"+name+"_AAM_Sin.txt";

		/*	string gtName=curPureName+".txt";*/
			//cout<<name<<endl;
			//strcat(name);
			//cout<<name<<endl;
			ifstream in(gtName.c_str(),ios::in);
			int cwidth,cheight;
			int cnum;
			float tx,ty;
			if (in)
			{
				in>>cwidth>>cheight;
				in>>cnum>>cnum;
				for (int i=0;i<ptsNum;i++)
				{
					in>>currentGT[i]>>currentGT[i+ptsNum];
					currentGT[i]*=cwidth;
					currentGT[i+ptsNum]*=cheight;
				}

			/*	int cx,cy;
				endX=endY=0;
				startX=startY=1000;
				for (int i=0;i<ptsNum;i++)
				{
					cx=currentShape[i];
					cy=currentShape[i+ptsNum];
					if (cx>endX)
					{
						endX=cx;
					}
					if (cx<startX)
					{
						startX=cx;
					}
					if (cy>endY)
					{
						endY=cy;
					}
					if (cy<startY)
					{
						startY=cy;
					}
				}*/

			/*	for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (i>=startY&&i<=endY&&j>=startX&&j<=endX)
						{
							continue;
						}
						depthImg.at<float>(i,j)=0;
					}
				}*/
				//track_AAM(m_img,depthImg,currentShape,cStart,"AAMPrior",AAM_exp->prefix);
			}
			/*else
			{
				startX=0;endX=10000;
				startY=0;endY=10000;
			}*/
		}


		/*Mat curImg=colorImgBackUP.clone();
		for (int i=0;i<curImg.rows;i++)
		{
			for (int j=0;j<curImg.cols;j++)
			{
				if (j<startX||j>endX||i<startY||i>endY)
				{
					curImg.at<Vec3b>(i,j)=0;
				}
			}
		}*/
		/*string faceRegionName;
		faceRegionName=curPureName+AAM_exp->prefix+"_faceRegion.jpg";
		imwrite(faceRegionName.c_str(),curImg);*/

		/*for (int i=0;i<depthImg.rows;i++)
		{
			for (int j=0;j<depthImg.cols;j++)
			{
				if (j<startX||j>endX||i<startY||i>endY)
				{
					depthImg.at<float>(i,j)=0;
				}
			}
		}*/

		/*Mat curImg=colorImgBackUP.clone();
		for (int i=0;i<curImg.rows;i++)
		{
			for (int j=0;j<curImg.cols;j++)
			{
				if (j<startX||j>endX||i<startY||i>endY)
				{
					curImg.at<Vec3b>(i,j)=0;
				}
			}
		}
		namedWindow("face region");
		imshow("face region",curImg);
		waitKey(1);*/

		/*for (int i=0;i<depthImg.rows;i++)
		{
			for (int j=0;j<depthImg.cols;j++)
			{
				if (depthImg.at<float>(i,j)*standardDepth>1200)
				{
					depthImg.at<float>(i,j)=0;
				}
			}
		}

		namedWindow("depth");
		imshow("depth",depthImg);
		waitKey();*/
//		
//		startX=287;
//		startY=99;
//		endX=402;
//		endY=250;
//
//
//		/*startX=200;
//		startY=99;
//		endX=402;
//		endY=300;
//*/
	//	cout<<startX<<" "<<startY<<" "<<endX<<" "<<endY<<endl;
		//for (int i=0;i<depthImg.rows;i++)
		//{
		//	for (int j=0;j<depthImg.cols;j++)
		//	{
		//		if (j<startX||j>endX||i<startY||i>endY)
		//		{
		//			depthImg.at<float>(i,j)=0;
		//		}
		//	}
		//}

	/*	ofstream out("D:\\Fuhao\\face dataset\\images for combination debug\\Fuhao_424.txt",ios::out);
		for (int i=0;i<depthImg.rows;i++)
		{
			for (int j=0;j<depthImg.cols;j++)
			{
				out<<depthImg.at<float>(i,j)*standardDepth<<" ";
			}
			out<<endl;
		}
		out.close();*/
		
		//and then detect
		//track(m_img,depthImg);
		//track_sampling(m_img,depthImg);

		//for AAM only with prior
		
	/*	char name[500];
		sprintf(name, "AAM+detection/colordepth/%s_%dModes.txt", AAM_exp->prefix,11361+AAM_exp->currentFrame);
		ifstream inModes(name,ios::in);
		if (inModes)
		{
			inModes.close();
			AAM_exp->currentFrame++;
			continue;
		}*/
		/*char name1[500];
		sprintf(name1, "AAM+detection/colordepth/%s_%dModesNew.txt", AAM_exp->prefix,11361+AAM_exp->currentFrame);
		ifstream inModes(name,ios::in);
		ofstream outModesWithDepth(name1,ios::out);

		float tid,tx,ty;
		float tdepth;
		while(inModes)
		{
			inModes>>tid>>tx>>ty;
			tdepth=depthImg.at<float>(ty,tx)*standardDepth;
			outModesWithDepth<<tid<<" "<<tx<<" "<<ty<<" "<<tdepth<<endl;
		}
		inModes.close();
		outModesWithDepth.close();*/
		

	/*	DWORD HessianStart, HessianStop,d_totalstart;  
		HessianStart=GetTickCount();*/

		//set up the name

		//output	initial parameters only
		//track_AAM_GPU(m_img,depthImg,currentShape);
		
		//////////////////full method////////////////////////
		int curStatus=0;
		char cccname[500];
		sprintf(cccname, "%s_%s", curName.c_str(),AAM_exp->prefix);
		cout<<cccname<<endl;
		AAM_exp->setSaveName(cccname);

		LONGLONG   t1,t2; 
		LONGLONG   persecond; 
		double time;

		QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);

		QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	
		bool isSucceed=false;

		isSucceed=track_combine(m_img,depthImg,curStatus,startX,endX,startY,endY,ll>startNum&&curFrameNum>0);
		if (initial&&isSucceed)
		{
			initial=false;

		}
		curFrameNum++;
		if (!isSucceed)//reset all the parameters to initial status
		{
			initial=true;
			curFrameNum=0;
			hasVelocity=false;
		}

		/*if (1||AAM_exp->currentFrame==startNum)
		{
			track_combine(m_img,depthImg,startX,endX,startY,endY,false);
		}
		else
		{
			track_combine(m_img,depthImg,startX,endX,startY,endY,true);
		}*/
		QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
		time=(t2-t1)*1000/persecond; 
		cout<<"total time: "<<time<<"ms "<<endl;
		cout<<"begin tracking...\n";
		/////////////////////////////////////////////////
		
		/*HessianStop=GetTickCount();
		cout<<"\n************************************************\n";
		cout<<"frame  time: "<<(HessianStop-HessianStart)<<" ms"<<endl;
		cout<<"\n************************************************\n";*/
		AAM_exp->currentFrame++;
		}
}

void AAM_Detection_Combination::track_AAM_GPU(Mat &colorImg,Mat &depthImg,float *cur_Shape)
{
	//LONGLONG   t1,t2; 
	//LONGLONG   persecond; 
	//double time;

	//QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	
	//ransac_noSampling(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb);
	ransac_MeanShape(cur_Shape);
	//ransac_noSampling_parrllel(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb);

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	///time=(t2-t1)*1000/persecond; 
	//cout<<"----------------ransac and knn time: "<<time<<"------------------------------"<<endl;

	//return;

	//finally, AAM with detection on GPU

	//cout<<bestFitTrainingSampleInd<<endl;
	int minInd=bestFitTrainingSampleInd;
	globalTransformation=globalTransformation_optimal;
	for (int i=0;i<AAM_exp->shapeDim;i++)
	{
		AAM_exp->s_weight[i]=0;//eigenVectors.at<double>(i,minInd);
	}

	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	//AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	//AAM_exp->setInitialScale(scale);
	//AAM_exp->setInitialTheta(theta);
	
	//output the initial parameters

	string outName=curPureName+"_initialParameters.txt";
	ofstream out(outName.c_str(),ios::out);
	int totalDim=AAM_exp->shape_dim+AAM_exp->texture_dim+4;
	out<<scale<<" "<<theta<<" "<< globalTransformation.at<float>(2,0)<<" "<<
		globalTransformation.at<float>(3,0)<<endl;
	out.close();
	return;	
}

void AAM_Detection_Combination::track_AAM(Mat &colorImg,Mat &depthImg,float *cur_Shape,int sID,char* dirName,char*preFix)
{
	//LONGLONG   t1,t2; 
	//LONGLONG   persecond; 
	//double time;

	//QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	int width=depthImg.cols;
	int height=depthImg.rows;

	//Step 1: transfer the data to gpu
	for (int i=0;i<depthImg.rows;i++)
	{
		for (int j=0;j<depthImg.cols;j++)
		{
			host_depthImage[i*depthImg.cols+j]=depthImg.at<float>(i,j);
			host_colorImage[i*depthImg.cols+j]=colorImg.at<uchar>(i,j);
		}
	}
	/*QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	time=(t2-t1)*1000/persecond; 
	cout<<"----------------prepare CPU data time: "<<time<<"------------------------------"<<endl;
	
	QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	setData_onrun_shared(host_colorImage,host_depthImage,width, height);
	QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	time=(t2-t1)*1000/persecond; 
	cout<<"----------------copy CPU data to GPU time: "<<time<<"------------------------------"<<endl;*/
	
	//step 2: detection
	//////////crutial part 1///////////
	//1: detect face region only
	//2: optimize to speed up 
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	//predict_GPU_withDepth_combination(depthImg.cols,depthImg.rows,hostDetectionResult,rt->trainStyle,finalPos,maximumProb);
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	//time=(t2-t1)*1000/persecond; 
	//cout<<"----------------randomized tree and mode find time: "<<time<<"------------------------------"<<endl;

	//return;
	

	//Mat tmp=colorImg.clone();
	//Point c;
	//for (int i=0;i<rt->labelNum-1;i++)
	//{
	//	//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
	//	c.x=finalPos[i][0];
	//	c.y=finalPos[i][1];
	//	circle(tmp,c,5,255);

	//}
	//namedWindow("feature locations");
	//imshow("feature locations",tmp);
	//waitKey();
	
	//return;


	////////////////////////////////////////////////////
	////then, KNN. Currently, on CPU
	//time consuming

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 

	vector<int> finalInd;
	//ransac_noSampling(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb);
	ransac_MeanShape(cur_Shape);
	//ransac_noSampling_parrllel(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb);

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	///time=(t2-t1)*1000/persecond; 
	//cout<<"----------------ransac and knn time: "<<time<<"------------------------------"<<endl;

	//return;

	//finally, AAM with detection on GPU

	//cout<<bestFitTrainingSampleInd<<endl;
	int minInd=bestFitTrainingSampleInd;
	globalTransformation=globalTransformation_optimal;
	for (int i=0;i<AAM_exp->shapeDim;i++)
	{
		AAM_exp->s_weight[i]=0;//eigenVectors.at<double>(i,minInd);
	}

	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	AAM_exp->setInitialScale(scale);
	AAM_exp->setInitialTheta(theta);
	


	//aam in cpu





	rt->setupProbMaps(depthImg.cols,depthImg.rows,hostDetectionResult,finalInd);

//	return;

	//lastyly, AAM with detection
	//1. initialization setting
	//2. iteration






	////////////////////////////////////////////////////////////////
	//set up the sampled feature locations
	for (int ll=0;ll<finalInd.size();ll++)
	{
		AAM_exp->shapeSample[ll][0]=finalPos[finalInd[ll]][0];
		AAM_exp->shapeSample[ll][1]=finalPos[finalInd[ll]][1];
	}
	
	AAM_exp->iterate_clean_CPU(colorImg,rt,1);
	
}

void AAM_Detection_Combination::ransac_MeanShape(float *curShape)
{
	vector<int> tmp;
	for (int l=0;l<fullIndNum;l++)
		tmp.push_back(l);

	float **sampledPos=new float *[fullIndNum];

	for (int l=0;l<fullIndNum;l++)
	{
		sampledPos[l]=new float [2];
		sampledPos[l][0]=curShape[combineIndList[l]];
		sampledPos[l][1]=curShape[combineIndList[l]+ptsNum];
	}
	getTransformationInfo(tmp,sampledPos,shapes,0);

	globalTransformation_optimal=globalTransformation.clone();

	for (int l=0;l<fullIndNum;l++)
	{
		delete []sampledPos[l];
	}
	delete []sampledPos;
}
void AAM_Detection_Combination::getTransformationInfo(vector<int> &InputInd,int **finalPos,Mat &shapeList,int shapeInd)
{
	int usedPtsNum=InputInd.size();
	Mat U=cv::Mat(usedPtsNum*2,1,CV_32FC1);
	Mat V=cv::Mat(usedPtsNum*2,4,CV_32FC1);
	//Mat globalTransformationLocal=cv::Mat(4,1,CV_32FC1);
	int cind;
	for (int i=0;i<usedPtsNum;i++)
	{
		cind=combineIndList[InputInd[i]];
		U.at<float>(i,0)=finalPos[InputInd[i]][0];
		U.at<float>(usedPtsNum+i,0)=finalPos[InputInd[i]][1];
		//	V.at<float>(i,0)=AAM_exp->meanShape->pts[cind][0]-meanShapeCenter[0];
		//	V.at<float>(i,1)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		V.at<float>(i,0)=shapeList.at<double>(cind,shapeInd);
		V.at<float>(i,1)=shapeList.at<double>(cind+ptsNum,shapeInd);
		V.at<float>(i,2)=1;
		V.at<float>(i,3)=0;
		//V.at<float>(i+usedPtsNum,0)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		//V.at<float>(i+usedPtsNum,1)=-AAM_exp->meanShape->pts[cind][0]+meanShapeCenter[0];
		V.at<float>(i+usedPtsNum,0)=shapeList.at<double>(cind+ptsNum,shapeInd);
		V.at<float>(i+usedPtsNum,1)=-shapeList.at<double>(cind,shapeInd);;
		V.at<float>(i+usedPtsNum,2)=0;
		V.at<float>(i+usedPtsNum,3)=1;
	}
	//cout<<usedPtsNum<<endl;
	solve(V,U,globalTransformation,DECOMP_SVD);

	//for (int i=0;i<globalTransformationLocal.rows;i++)
	//{
	//	globalTransformation.at<float>(i,0)=globalTransformationLocal.at<float>(i,0);
	//}
}


void AAM_Detection_Combination::getTransformationInfo(vector<int> &InputInd,float **finalPos,Mat &shapeList,int shapeInd)
{
	int usedPtsNum=InputInd.size();
	Mat U=cv::Mat(usedPtsNum*2,1,CV_32FC1);
	Mat V=cv::Mat(usedPtsNum*2,4,CV_32FC1);
	//Mat globalTransformationLocal=cv::Mat(4,1,CV_32FC1);
	int cind;
	for (int i=0;i<usedPtsNum;i++)
	{
		cind=combineIndList[InputInd[i]];
		U.at<float>(i,0)=finalPos[InputInd[i]][0];
		U.at<float>(usedPtsNum+i,0)=finalPos[InputInd[i]][1];
		//	V.at<float>(i,0)=AAM_exp->meanShape->pts[cind][0]-meanShapeCenter[0];
		//	V.at<float>(i,1)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		V.at<float>(i,0)=shapeList.at<double>(cind,shapeInd);
		V.at<float>(i,1)=shapeList.at<double>(cind+ptsNum,shapeInd);
		V.at<float>(i,2)=1;
		V.at<float>(i,3)=0;
		//V.at<float>(i+usedPtsNum,0)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		//V.at<float>(i+usedPtsNum,1)=-AAM_exp->meanShape->pts[cind][0]+meanShapeCenter[0];
		V.at<float>(i+usedPtsNum,0)=shapeList.at<double>(cind+ptsNum,shapeInd);
		V.at<float>(i+usedPtsNum,1)=-shapeList.at<double>(cind,shapeInd);;
		V.at<float>(i+usedPtsNum,2)=0;
		V.at<float>(i+usedPtsNum,3)=1;
	}
	//cout<<usedPtsNum<<endl;
	solve(V,U,globalTransformation,DECOMP_SVD);

	//for (int i=0;i<globalTransformationLocal.rows;i++)
	//{
	//	globalTransformation.at<float>(i,0)=globalTransformationLocal.at<float>(i,0);
	//}
}

//typedef std::pair<float,int> distancePir;
int NNNum=40;
distancePir *distanceVec=NULL;


void AAM_Detection_Combination::ransac_noSampling_parrllel(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,Mat *img)
{
		srand(time(NULL));
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	float e=2.7173;


	vector<int> traningSapleInd;
	//RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

	//use all the training sample
	//traningSapleInd.resize(totalShapeNum);
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	traningSapleInd[i]=i;
	//}

	Mat newU;

	//vector<int> *traningSapleInd;
	//go over every traing example
	//for (int i=0;i<sampleNumberFromTrainingSet;i++)

	for(int i=0;i<fullIndNum;i++)
		cout<<i<<" "<<maximumProb[i]<<endl;
	//#pragma omp parallel for
	traningSapleInd.push_back(0);
	for(int i=0;i<traningSapleInd.size();i++)
	{
		//sample from probability map, currently we use global maxima only: sampledPos
		for (int j=0;j<sampleNumberFromProbMap;j++)
		{
			//sample from the feature locations
			for (int k=0;k<sampleNumberFromFeature*2;k++)
			{
				vector<int> currentInd;
				RandSample_V1(fullIndNum,2,currentInd);	//sample the current used indices
				if (k==0)
				{
					currentInd[0]=18;currentInd[1]=16;
				}
				else if (k==1)
				{
					currentInd[0]=18;currentInd[1]=17;
				}
				getTransformationInfo(currentInd,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters

				newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;



				//cout<<"the distances are: ";
				vector<int> tmp;
				float distance;
				int cind;
				for (int l=0;l<fullIndNum;l++)
				{
					distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
						(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);

					int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));
					//cout<<round(newU.at<float>(l+fullIndNum,0))<<" "<<round(newU.at<float>(l,0))<<" "<<currentIndex<<endl;

					float probAll=powf(e,-distance/200);
					if (currentIndex<0||currentIndex>=width*height)
					{
						probAll=-1;
					}
					else
						probAll*=maximumProb[l];//hostDetectionResult[currentIndex+l*width*height];
					//cout<<distance<<" ";
					//if (distance<window_threshold_small)
					if(probAll>0.00001&&maximumProb[l]>0)
					{
						tmp.push_back(l);
					}
				}
			
				//check if there are enough inliers
				if (tmp.size()<w_critical)
				{
					continue;
				}

				getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters
				newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;

				tmp.clear();
				for (int l=0;l<fullIndNum;l++)
				{
					distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
						(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
					int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));
					//cout<<round(newU.at<float>(l+fullIndNum,0))<<" "<<round(newU.at<float>(l,0))<<" "<<currentIndex<<endl;
					

					float probAll=powf(e,-distance/50);
					if (currentIndex<0||currentIndex>=width*height)
					{
						probAll=-1;
					}
					else
						probAll*=maximumProb[l];
					//if (distance<window_threshold_large)//hostDetectionResult[currentIndex+l*width*height]>=0.2)*/
					if(probAll>=0.0001&&maximumProb[l]>0.0)
					{
						tmp.push_back(l);
					}
				}

				if (tmp.size()<=finalInd.size())
				{
					continue;
				}					

				float prob=0;
				for (int jj=0;jj<tmp.size();jj++)
				{
					int currentIndex=round(newU.at<float>(tmp[jj]+fullIndNum,0))*width+round(newU.at<float>(tmp[jj],0));
					prob+=hostDetectionResult[currentIndex+tmp[jj]*width*height];///maximumProb[tmp[jj]];
				}
				prob/=tmp.size();

				//cout<<"tmp size: "<<tmp.size()<<endl;

				if (prob>bestProb)
				{
					bestProb=prob;
					bestFitTrainingSampleInd=traningSapleInd[i];
					finalInd.clear();
					for (int kkk=0;kkk<tmp.size();kkk++)
					{
						finalInd.push_back(tmp[kkk]);
					}
					globalTransformation_optimal=globalTransformation.clone();
				}
				
			}
		}
	}
	int bestIndNum=finalInd.size();
	vector<int> tmpInd;
	for (int i=0;i<totalShapeNum;i++)
	{
		tmpInd.clear();
		getTransformationInfo(finalInd,sampledPos,shapes,i);
		newU=fullTrainingPos[i]*globalTransformation;

		//int inNum=0;
		for (int l=0;l<fullIndNum;l++)
		{
			float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
				(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

			if (distance<window_threshold_large&&maximumProb[l]>0)
			{
				tmpInd.push_back(l);
			}
		}
		if (tmpInd.size()>=finalInd.size())
		{
			getTransformationInfo(tmpInd,sampledPos,shapes,i);	//get transformation parameters
			newU=fullTrainingPos[i]*globalTransformation;
			tmpInd.clear();
			for (int l=0;l<fullIndNum;l++)
			{
				float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
					(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

				int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));
				
				if (distance<window_threshold_small&&maximumProb[l]>0)
				{
					tmpInd.push_back(l);
				}

			
			}	
			if (tmpInd.size()>bestIndNum)
			{
				bestIndNum=tmpInd.size();
				bestFitTrainingSampleInd=i;
			}
		}
	}
	for (int i=0;i<30;i++)
	{
		AAM_exp->probForEachFeature[i]=0;
	}
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;

	tmpInd.clear();
	for (int ii=0;ii<fullIndNum;ii++)
	{
		float currentDis=(((newU.at<float>(ii,0)-sampledPos[ii][0])*
			(newU.at<float>(ii,0)-sampledPos[ii][0])+
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])*
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])));	
		if (currentDis<window_threshold_large)
		{
			tmpInd.push_back(ii);
		}
	}
	getTransformationInfo(tmpInd,sampledPos,shapes,bestFitTrainingSampleInd);	//get transformation parameters
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	finalInd.clear();
	for (int l=0;l<fullIndNum;l++)
	{
		float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
			(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
		//cout<<combineIndList[l]<<" "<<distance<<endl;
		if (distance<window_threshold_small&&maximumProb[l]>0)
		{
			finalInd.push_back(l);
		}
	}

	globalTransformation_optimal=globalTransformation.clone();

	/*Mat tmpIMg=(*img).clone();
	
			newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
			Point c;
			for (int j=0;j<fullIndNum;j++)
			{
				c.x=newU.at<float>(j,0);
				c.y=newU.at<float>(j+fullIndNum,0);
				circle(tmpIMg,c,5,255);
			}
			namedWindow("KNN visualization");
				imshow("KNN visualization",tmpIMg);
				waitKey();*/

	float bestDistance=1000000;
	float currentDis;
	//float 
	//now check distance

	//search KNN

	if (distanceVec==NULL)
	{
		distanceVec=new distancePir[totalShapeNum];
	}


	float prob_used[50];
	float sigma=0.2;
	for (int ii=0;ii<finalInd.size();ii++)
	{
		

		//float probAll=powf(e,-currentDis*currentDis/200);
		prob_used[ii]=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
			(2*sigma*sigma));
		//prob_used[ii]=maximumProb[ii];
		//	prob_used[ii]=1;
		//int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));
		//int currentIndex=(int)(sampledPos[finalInd[ii]][1]*width+sampledPos[finalInd[ii]][0]);
		//cout<<ii<<" "<<maximumProb[ii]<<" "<<hostDetectionResult[currentIndex+finalInd[ii]*width*height]<<endl;
		/*if (hostDetectionResult[currentIndex+finalInd[ii]*width*height]>0.1)
		{
			prob_used[ii]=1;
		}
		else
			prob_used[ii]=0;*/
	}
	

	for (int i=0;i<totalShapeNum;i++)
	//for (int i=22;i<23;i++)
	{
		getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
		//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
		//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
		newU=fullTrainingPos[i]*globalTransformation;

		//newU=fullTrainingPos[i]*globalTransformation_optimal;

	/*	for (int ii=0;ii<finalInd.size();ii++)
		{
			cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
		}
		cout<<endl;*/

		float currentDis=0;
		for (int ii=0;ii<finalInd.size();ii++)
		{
			currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
				(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]))*prob_used[ii];	
		}

		distanceVec[i].first=currentDis;
		distanceVec[i].second=i;
		/*if (currentDis<bestDistance)
		{
			bestDistance=currentDis;
			bestFitTrainingSampleInd=i;
			globalTransformation_optimal=globalTransformation.clone();
		}*/
		//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
		//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	}
	sort(distanceVec,distanceVec+totalShapeNum,comparator);


	bestFitTrainingSampleInd=distanceVec[0].second;
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	globalTransformation_optimal=globalTransformation.clone();

	//update the inlier again
	tmpInd.clear();
	for (int ii=0;ii<fullIndNum;ii++)
	{
		float currentDis=(((newU.at<float>(ii,0)-sampledPos[ii][0])*
			(newU.at<float>(ii,0)-sampledPos[ii][0])+
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])*
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])));	
		if (currentDis<window_threshold_large)
		{
			tmpInd.push_back(ii);
		}
	}
	getTransformationInfo(tmpInd,sampledPos,shapes,bestFitTrainingSampleInd);	//get transformation parameters
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	finalInd.clear();
	for (int l=0;l<fullIndNum;l++)
	{
		float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
			(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
		//cout<<combineIndList[l]<<" "<<distance<<endl;
		if (distance<window_threshold_small&&maximumProb[l]>0)
		{
			finalInd.push_back(l);
		}
	}
	globalTransformation_optimal=globalTransformation.clone();
	//Mat tmpIMg=(*img).clone();
	//
	//		newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
	//		Point c;
	//		for (int j=0;j<finalInd.size();j++)
	//		{
	//			c.x=newU.at<float>(finalInd[j],0);
	//			c.y=newU.at<float>(finalInd[j]+fullIndNum,0);
	//			circle(tmpIMg,c,5,255);
	//		}
	//		namedWindow("KNN visualization");
	//			imshow("KNN visualization",tmpIMg);
	//			waitKey();

	//visualize the image data
	//if (AAM_exp->showSingleStep&&img!=NULL)
	{
		Mat tmpIMg=(*img).clone();
		for (int i=0;i<NNNum;i++)
		{
			int tmpInd=distanceVec[i].second;
			getTransformationInfo(finalInd,sampledPos,shapes,tmpInd);
			newU=fullTrainingPos[tmpInd]*globalTransformation;
			Point c;
			for (int j=0;j<fullIndNum;j++)
			{
				c.x=newU.at<float>(j,0);
				c.y=newU.at<float>(j+fullIndNum,0);
				circle(tmpIMg,c,5,255);
			}

			
		}

		namedWindow("KNN visualization");
		imshow("KNN visualization",tmpIMg);
		waitKey();
		
	}
	

	float sigma1=sigma;
	for (int ii=0;ii<finalInd.size();ii++)
	{
		float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

		AAM_exp->distanceKNN[ii]=currentDis;

		int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

		float probAll=powf(e,-currentDis*currentDis/50);
		//probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
		/*probAll*=powf(e,-(hostDetectionResult[currentIndex+finalInd[ii]*width*height]-1)*(hostDetectionResult[currentIndex+finalInd[ii]*width*height]-1)/
			(2*sigma1*sigma1));*/
		//probAll*=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
			//(2*sigma1*sigma1));;
		AAM_exp->probForEachFeature[ii]=probAll;
		//cout<<finalInd[ii]<<" "<<AAM_exp->probForEachFeature[ii]<<endl;
		//AAM_exp->probForEachFeature[ii]=1;
	}

	/*cout<<"feature probabilities:\n";
	for (int i=0;i<finalInd.size();i++)
	{
		cout<<i<< " "<<maximumProb[finalInd[i]]<<" "<<AAM_exp->probForEachFeature[i]<<endl;
	}*/

	//then do the KNN fit
	//eigenVectors.at<double>(i,minInd);
	//cout<<eigenVectors.cols<<" "<<eigenVectors.rows<<endl;
	
	Mat KNNVec=Mat::zeros(eigenVectors.rows,NNNum,CV_64FC1);
	for (int i=0;i<NNNum;i++)
	{
		KNNVec.col(i)+=eigenVectors.col(distanceVec[i].second);
	}
	
	//if (localWeight>0)
	{
	//update the mean
	Mat mean_KNN=KNNVec.col(0)*0;
	for (int i=0;i<KNNVec.cols;i++)
	{
		mean_KNN+=KNNVec.col(i);
	}
	mean_KNN/=KNNVec.cols;
	for (int i=0;i<mean_KNN.rows;i++)
	{
		AAM_exp->priorMean[i]=mean_KNN.at<double>(i,0);
	}

	//update the conv
	for (int i=0;i<KNNVec.cols;i++)
	{
		KNNVec.col(i)-=mean_KNN;
	}
	Mat KNNVec_tran;
	transpose(KNNVec,KNNVec_tran);
	Mat convKNN=KNNVec*KNNVec_tran/KNNVec.cols;

	Mat conv_inv_KNN=convKNN.inv();

//	cout<<AAM_exp->priorSigma.cols<<" "<<AAM_exp->priorSigma.rows<<endl;
	for (int i=0;i<conv_inv_KNN.rows;i++)
	{
		for (int j=0;j<conv_inv_KNN.cols;j++)
		{
			AAM_exp->priorSigma.at<double>(i,j)=conv_inv_KNN.at<double>(i,j);
		}
	}


	if (localWeight>0)
	{


		//train the local PCA model
		int s_dim=eigenVectors.rows;
		//use KNNVec_tran to train
		CvMat *pData=cvCreateMat(KNNVec_tran.rows,KNNVec_tran.cols,CV_64FC1);
		for (int i=0;i<KNNVec_tran.rows;i++)
		{
			for (int j=0;j<KNNVec_tran.cols;j++)
			{
				//CV_MAT_ELEM(*pData,double,i,j)=shape[i]->ptsForMatlab[j];

				//here,we keep the shape in the same scale with the meanshape
				CV_MAT_ELEM(*pData,double,i,j)=KNNVec_tran.at<double>(i,j);
			}

		}
		if (AAM_exp->local_s_mean==NULL)
		{
			AAM_exp->local_s_mean = cvCreateMat(1, KNNVec_tran.cols, CV_64FC1);
			AAM_exp->m_local_s_mean=cvarrToMat(AAM_exp->local_s_mean);
		}

		if (AAM_exp->local_s_vec==NULL)
		{
			AAM_exp->local_s_vec=cvCreateMat( min(KNNVec_tran.rows,KNNVec_tran.cols), KNNVec_tran.cols, CV_64FC1); 
		}

		CvMat* s_value = cvCreateMat(1, min(KNNVec_tran.cols,KNNVec_tran.rows), CV_64FC1);
		//CvMat *s_PCAvec = cvCreateMat( min(KNNVec_tran.rows,KNNVec_tran.cols), KNNVec_tran.cols, CV_64FC1); 
		cvCalcPCA( pData, AAM_exp->local_s_mean, s_value, AAM_exp->local_s_vec, CV_PCA_DATA_AS_ROW );
		AAM_exp->m_local_mean=cvarrToMat(AAM_exp->local_s_mean);

		double sumEigVal=0;
		for (int i=0;i<s_value->cols;i++)
		{
			sumEigVal+=CV_MAT_ELEM(*s_value,double,0,i);
		}

		double sumCur=0;
		for (int i=0;i<s_value->cols;i++)
		{
			sumCur+=CV_MAT_ELEM(*s_value,double,0,i);
			if (sumCur/sumEigVal>=0.98)
			{
				AAM_exp->local_shape_dim=i+1;
				break;
			}
		}
		//cout<<"local dim: "<<AAM_exp->local_shape_dim<<endl;

		Mat curEigenVec=cvarrToMat(AAM_exp->local_s_vec);
		Mat usedEigenVec=curEigenVec.rowRange(Range(0,AAM_exp->local_shape_dim));
		Mat usedEigenVec_tran;
		transpose(usedEigenVec,usedEigenVec_tran);
		Mat localHessian=usedEigenVec_tran*usedEigenVec;
		localHessian=Mat::eye(localHessian.rows,localHessian.cols,CV_64FC1)-localHessian;

		Mat localHessian_tran;
		transpose(localHessian,localHessian_tran);
		localHessian=localHessian_tran*localHessian;
		for (int i=0;i<localHessian.rows;i++)
		{
			for (int j=0;j<localHessian.cols;j++)
			{
				AAM_exp->m_localHessian.at<double>(i,j)=localHessian.at<double>(i,j);
			}
		}
	}

	}
	//delete []distance;
	

	//find the best one
	//for (int i=0;i<totalShapeNum;i++)
	////for (int i=22;i<23;i++)
	//{
	//	getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
	//	//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
	//	//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
	//	newU=fullTrainingPos[i]*globalTransformation;

	//	//newU=fullTrainingPos[i]*globalTransformation_optimal;

	///*	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
	//	}
	//	cout<<endl;*/

	//	float currentDis=0;
	//	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]));	
	//	}

	//	if (currentDis<bestDistance)
	//	{
	//		bestDistance=currentDis;
	//		bestFitTrainingSampleInd=i;
	//		globalTransformation_optimal=globalTransformation.clone();
	//	}
	//	//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
	//	//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	//}


	return;

	//return;

	//////////////////KNN on GPU////////////////////////////////
	//input: shapes and initial global transformation and inlier set
	//output: the final inlier set and the best example
	//float *rt_parameters=new float[4];
	//for (int i=0;i<4;i++)
	//{
	//	rt_parameters[i]=globalTransformation_optimal.at<float>(i,0);
	//}
	//float *currentShape=new float[fullIndNum*2];
	//for (int i=0;i<fullIndNum;i++)
	//{
	//	currentShape[i]=sampledPos[i][0];
	//	currentShape[i+fullIndNum]=sampledPos[i][1];
	//}
	////return;
	//KNN_search(rt_parameters,currentShape,totalShapeNum,fullIndNum);

	//return;
	///////////////////////////////////////////////////
	//LONGLONG   t1,t2; 
	//LONGLONG   persecond; 

	//QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 

	// #pragma omp parallel for
	//for (int i=0;i<10000;i++)
	//{
	//	getTransformationInfo_toVector(finalInd,sampledPos,shapes,i%totalShapeNum);	//get transformation parameters
	//}

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	//double   time=(t2-t1)*1000/persecond; 


	//cout<<"\n************************************************\n";
	//cout<<"transofrmation  time: "<<time<<" ms"<<endl;
	//cout<<"\n************************************************\n";

	//then, check the two lower mouth points 11 and 23
	//vector<int> tmp;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	tmp.push_back(finalInd[i]);
	//}
	//if (find(finalInd.begin(),finalInd.end(),11)==finalInd.end())
	//{
	//	tmp.push_back(11);
	//	getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[bestFitTrainingSampleInd]);	//get transformation parameters
	//
	//	float scale1=globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
	//		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0);

	//	float scale2=globalTransformation_optimal.at<float>(0,0)*globalTransformation_optimal.at<float>(0,0)+
	//		globalTransformation_optimal.at<float>(1,0)*globalTransformation_optimal.at<float>(1,0);

	//	if (abs(scale1-scale2)<0.05&&abs(globalTransformation.at<float>(2,0)-globalTransformation_optimal.at<float>(2,0))<5&&
	//		abs(globalTransformation.at<float>(3,0)-globalTransformation_optimal.at<float>(3,0))<5)
	//	{
	//		finalInd.push_back(11);
	//	}
	//	//globalTransformation_optimal.at<float>(0,0)=globalTransformation.at<float>(0,0);
	//	//globalTransformation_optimal.at<float>(1,0)=globalTransformation.at<float>(1,0);
	//	//globalTransformation_optimal.at<float>(2,0)=globalTransformation.at<float>(2,0);
	//	//globalTransformation_optimal.at<float>(3,0)=globalTransformation.at<float>(3,0);
	//	tmp.pop_back();
	//}

	//if (find(finalInd.begin(),finalInd.end(),23)==finalInd.end())
	//{
	//	tmp.push_back(23);
	//	getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[bestFitTrainingSampleInd]);	//get transformation parameters
	//	
	//	float scale1=globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
	//		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0);

	//	float scale2=globalTransformation_optimal.at<float>(0,0)*globalTransformation_optimal.at<float>(0,0)+
	//		globalTransformation_optimal.at<float>(1,0)*globalTransformation_optimal.at<float>(1,0);

	//	if (abs(scale1-scale2)<0.05&&abs(globalTransformation.at<float>(2,0)-globalTransformation_optimal.at<float>(2,0))<5&&
	//		abs(globalTransformation.at<float>(3,0)-globalTransformation_optimal.at<float>(3,0))<5)
	//	{
	//		finalInd.push_back(23);
	//	}
	//	//globalTransformation_optimal.at<float>(0,0)=globalTransformation.at<float>(0,0);
	//	//	globalTransformation_optimal.at<float>(1,0)=globalTransformation.at<float>(1,0);
	//	//globalTransformation_optimal.at<float>(2,0)=globalTransformation.at<float>(2,0);
	//	//globalTransformation_optimal.at<float>(3,0)=globalTransformation.at<float>(3,0);
	//}

	/////////////////////////new version, do not estimate gtP for every example//////////////////////////////////
	//globalTransformation=globalTransformation_optimal;
//bestProb=0;
//finalInd.clear();
//for (int i=0;i<fullIndNum;i++)
//{
//	if (maximumProb[i]>0.95)
//	{
//		finalInd.push_back(i);
//		bestProb+=maximumProb[i];
//	}
//}
//bestProb/=finalInd.size();
	Mat *newUList=new Mat[totalShapeNum];
	bool *usedLabel=new bool[fullIndNum];
	//vector <int >tmpInd;
	bestProb=0;
	bool needC=false;
	for (int i=0;i<totalShapeNum;i++)
	{
		tmpInd.clear();
		getTransformationInfo(finalInd,sampledPos,shapes,i);
		newUList[i]=fullTrainingPos[i]*globalTransformation;
		newU=newUList[i];

		//estimate the prob
		float prob=0;
		for (int jj=0;jj<fullIndNum;jj++)
		{
			int currentIndex=round(newU.at<float>(jj+fullIndNum,0))*width+round(newU.at<float>(jj,0));
			prob+=hostDetectionResult[currentIndex+jj*width*height];///maximumProb[tmpInd[jj]];
		}
		prob/=fullIndNum;
		if (prob>bestProb)
		{
			bestProb=prob;
			bestFitTrainingSampleInd=i;
		}
		continue;
		//int inNum=0;
		for (int l=0;l<fullIndNum;l++)
		{
			float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
				(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
			
			//cout<<l<<endl;
			//{
			//	Mat tmp=tmpImg.clone();
			//	Point c;
			//	for (int j=0;j<finalInd.size();j++)
			//	{
			//	
			//		//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
			//		c.x=newU.at<float>(finalInd[j],0);
			//		c.y=newU.at<float>(finalInd[j]+fullIndNum,0);
			//		circle(tmp,c,5,255);
			//	}
			//	for (int j=0;j<fullIndNum;j++)
			//	{
			//		//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
			//		if (j!=l)
			//		{
			//			continue;
			//		}
			//		c.x=sampledPos[j][0];
			//		c.y=sampledPos[j][1];
			//		circle(tmp,c,2,255);
			//	}
			//	namedWindow("aligned shape");
			//	imshow("aligned shape",tmp);
			//	waitKey();

			//}
			//cout<<distance<<endl;
			int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));

			float probAll=powf(e,-distance/50);
			probAll*=hostDetectionResult[currentIndex+l*width*height];
			//cout<<l<<" "<<probAll<<endl;
		//	cout<<probAll<<" "<<powf(e,-distance/50)<<" "<<hostDetectionResult[currentIndex+l*width*height]<<endl;
			//if (distance<window_threshold_small&&hostDetectionResult[currentIndex+l*width*height]>=0.2)
			if(probAll>0.00001)
			{
		
				tmpInd.push_back(l);
				//inNum++;
			}

		}
		if (needC)
		{
			if (tmpInd.size()<=finalInd.size())
			{
				continue;
			}
		}
		//if (needC&&tmpInd.size()>finalInd.size())
		{
			float prob=0;
			for (int jj=0;jj<tmpInd.size();jj++)
			{
				int currentIndex=round(newU.at<float>(tmpInd[jj]+fullIndNum,0))*width+round(newU.at<float>(tmpInd[jj],0));
				prob+=hostDetectionResult[currentIndex+tmpInd[jj]*width*height];///maximumProb[tmpInd[jj]];
			}
			prob/=tmpInd.size();

			//cout<<"tmp size: "<<tmp.size()<<endl;

			if (prob>bestProb)
			{
				bestProb=prob;
				bestFitTrainingSampleInd=traningSapleInd[i];
				finalInd.clear();
				for (int kkk=0;kkk<tmpInd.size();kkk++)
				{
					finalInd.push_back(tmpInd[kkk]);
				}
				/*getTransformationInfo(tmpInd,sampledPos,shapes,i);
				globalTransformation_optimal=globalTransformation.clone();
				newUList[i]=fullTrainingPos[i]*globalTransformation;*/
				needC=true;
			}
		}
		//cout<<inNum<<endl;
	}
	
	for (int i=0;i<30;i++)
	{
		AAM_exp->probForEachFeature[i]=0;
	}
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;

	finalInd.clear();
	for (int ii=0;ii<fullIndNum;ii++)
	{
		float currentDis=(((newU.at<float>(ii,0)-sampledPos[ii][0])*
			(newU.at<float>(ii,0)-sampledPos[ii][0])+
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])*
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])));	
		if (currentDis<window_threshold_large)
		{
			finalInd.push_back(ii);
		}
	}

	for (int ii=0;ii<finalInd.size();ii++)
	{
		float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

		AAM_exp->distanceKNN[ii]=currentDis;

		int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

		float probAll=powf(e,-currentDis*currentDis/200);
		probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
		AAM_exp->probForEachFeature[ii]=probAll;
	}
	return;
	/*finalInd.clear();
	for (int i=0;i<fullIndNum;i++)
	{
		if (usedLabel[i])
		{
			finalInd.push_back(i);
		}
	}*/

	//

	////return;
	////cout<<"----------------inlier num on CPU: "<<finalInd.size()<<endl;
	//float bestDistance=1000000;
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	float currentDis=0;
	//	newU=newUList[i];

	//

	//	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		currentDis+=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	
	//	}

	//	//cout<<currentDis<<endl;
	//	//Mat tmp=tmpImg.clone();
	//	//Point c;
	//	//for (int j=0;j<finalInd.size();j++)
	//	//{
	//	//	//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
	//	//	c.x=newU.at<float>(finalInd[j],0);
	//	//	c.y=newU.at<float>(finalInd[j]+fullIndNum,0);
	//	//	circle(tmp,c,5,255);
	//	//}
	//	//for (int j=0;j<finalInd.size();j++)
	//	//{
	//	//	//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
	//	//	c.x=sampledPos[finalInd[j]][0];
	//	//	c.y=sampledPos[finalInd[j]][1];
	//	//	circle(tmp,c,2,255);
	//	//}
	//	//namedWindow("aligned shape");
	//	//imshow("aligned shape",tmp);
	//	//waitKey();

	//	if (currentDis<bestDistance)
	//	{
	//		bestDistance=currentDis;
	//		bestFitTrainingSampleInd=i;
	//	}
	//}

	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);

	//for (int i=0;i<30;i++)
	//{
	//	AAM_exp->probForEachFeature[i]=0;
	//}
	//newU=newUList[bestFitTrainingSampleInd];
	//for (int ii=0;ii<finalInd.size();ii++)
	//{
	//	float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//		(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//		(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//		(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

	//	AAM_exp->distanceKNN[ii]=currentDis;

	//	int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

	//	float probAll=powf(e,-currentDis*currentDis/50);
	//	probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
	//	AAM_exp->probForEachFeature[ii]=probAll;
	//}

	//globalTransformation_optimal=globalTransformation.clone();


	///////////////////////////////////////////////////////////////////////////////////

	///////////////////////old version, re-estimate global transforation every time///////////////////////////////
//	float bestDistance=1000000;
//	float currentDis;
	//float 
	//now check distance
	for (int i=0;i<totalShapeNum;i++)
	//for (int i=22;i<23;i++)
	{
		//cout<<i<<endl;
		getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
		//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
		//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
		newU=fullTrainingPos[i]*globalTransformation;

		//newU=fullTrainingPos[i]*globalTransformation_optimal;

	/*	for (int ii=0;ii<finalInd.size();ii++)
		{
			cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
		}
		cout<<endl;*/

		float currentDis=0;
		for (int ii=0;ii<finalInd.size();ii++)
		{
			currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
				(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]));	
		}

		if (currentDis<bestDistance)
		{
			bestDistance=currentDis;
			bestFitTrainingSampleInd=i;
			globalTransformation_optimal=globalTransformation.clone();
		}
		//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
		//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

	}

	for (int i=0;i<30;i++)
	{
		AAM_exp->probForEachFeature[i]=0;
	}
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	for (int ii=0;ii<finalInd.size();ii++)
	{
		float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

		AAM_exp->distanceKNN[ii]=currentDis;

		int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

		float probAll=powf(e,-currentDis*currentDis/200);
		probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
		AAM_exp->probForEachFeature[ii]=probAll;
	}
	return;
	for (int l=0;l<fullIndNum;l++)
	{
		float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
			(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
		/*if (combineIndList[l]==56||combineIndList[l]==61)
		{
			cout<<combineIndList[l]<<" "<<sqrt(distance)<<endl;
		}*/
		if (distance<window_threshold_large)
		{
			if (find(finalInd.begin(),finalInd.end(),l)==finalInd.end())
			{
				finalInd.push_back(l);
			}
			int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));

			float probAll=powf(e,-currentDis/200);
			probAll*=hostDetectionResult[currentIndex+l*width*height];
			AAM_exp->probForEachFeature[finalInd.size()-1]=probAll;
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////

	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);	
	//newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	//finalInd.clear();
	//for (int l=0;l<fullIndNum;l++)
	//{
	//	float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
	//		(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	//	if (distance<window_threshold_large)
	//	{
	//		finalInd.push_back(l);
	//	}
	//}
	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);	
	//globalTransformation_optimal=globalTransformation.clone();
}

void AAM_Detection_Combination::ransac_noSampling_Candidates(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,char *saveName,Mat *img,Mat *depthImg)
{
		srand(time(NULL));
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	
	float prob_threshold=0;

	vector<int> traningSapleInd;
	//RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

	//use all the training sample
	//traningSapleInd.resize(totalShapeNum);
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	traningSapleInd[i]=i;
	//}

	Mat newU;

	int selectedPointIndex[50];		//save index of the selected point
	int curIndex[50];

	int sampleNum=1;
	for (int i=0;i<fullIndNum;i++)
	{
		sampleNum+=candidatePoints[i].size();
	}
	sampleNum*=3;

	bool usedPosition[50][50];
	for (int i=0;i<fullIndNum;i++)
	{
		for (int j=0;j<50;j++)
		{
			usedPosition[i][j]=false;
		}
	}

	//vector<int> *traningSapleInd;
	//go over every traing example
	//for (int i=0;i<sampleNumberFromTrainingSet;i++)

//	for(int i=0;i<fullIndNum;i++)
		//cout<<i<<" "<<maximumProb[i]<<endl;
	//#pragma omp parallel for
	traningSapleInd.push_back(0);
	for(int i=0;i<traningSapleInd.size();i++)
	{
		//sample from probability map, currently we use global maxima only: sampledPos
		for (int j=0;j<sampleNum;j++)
		{
			//sample the feature candidates
			for (int p=0;p<fullIndNum;p++)
			{
				int k;
				for (k=0;k<candidatePoints[p].size();k++)
				{
					if (usedPosition[p][k]==false)
					{
						curIndex[p]=k;
						usedPosition[p][k]=true;
						break;
					}
				}
				if (k==candidatePoints[p].size())
				{
					curIndex[p]=RandInt_cABc(0,candidatePoints[p].size()-1);
				}
			//	if (combineIndList[p]==51||combineIndList[p]==48)
			//	{
			//		curIndex[p]=1;
			//	}
				sampledPos[p][0]=candidatePoints[p][curIndex[p]].x;
				sampledPos[p][1]=candidatePoints[p][curIndex[p]].y;
			}
			//sample from the feature locations
			for (int k=0;k<sampleNumberFromFeature*2;k++)
			{
				vector<int> currentInd;

				bool found=false;
				while(!found)
				{
					RandSample_V1(fullIndNum,2,currentInd);	//sample the current used indices
					if ((sampledPos[currentInd[0]][0]-sampledPos[currentInd[1]][0])*(sampledPos[currentInd[0]][0]-sampledPos[currentInd[1]][0])+
						(sampledPos[currentInd[0]][1]-sampledPos[currentInd[1]][1])*(sampledPos[currentInd[0]][1]-sampledPos[currentInd[1]][1])>1)
					{
						found=true;
					}
				}
				
				if (k==0)
				{
					currentInd[0]=18;currentInd[1]=16;
				}
				else if (k==1)
				{
					currentInd[0]=18;currentInd[1]=17;
				}
			//	else
				//{
					//currentInd[0]=14;currentInd[1]=15;
				//}
				getTransformationInfo(currentInd,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters

				newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;



				//cout<<"the distances are: ";
				vector<int> tmp;
				float distance;
				int cind;
				for (int l=0;l<fullIndNum;l++)
				{
					distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
						(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);

					int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));
					//cout<<round(newU.at<float>(l+fullIndNum,0))<<" "<<round(newU.at<float>(l,0))<<" "<<currentIndex<<endl;

					float probAll=powf(e,-distance/200);
					if (currentIndex<0||currentIndex>=width*height)
					{
						probAll=-1;
					}
					else
						probAll*=maximumProb[l];//hostDetectionResult[currentIndex+l*width*height];
					//cout<<distance<<" ";
					//if (distance<window_threshold_small)
					if(probAll>0.00001&&maximumProb[l]>prob_threshold)
					{
						tmp.push_back(l);
					}
				}
			
				//check if there are enough inliers
				if (tmp.size()<w_critical)
				{
					continue;
				}

				getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters
				newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;

				tmp.clear();
				for (int l=0;l<fullIndNum;l++)
				{
					distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
						(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
					int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));
					//cout<<round(newU.at<float>(l+fullIndNum,0))<<" "<<round(newU.at<float>(l,0))<<" "<<currentIndex<<endl;
					

					float probAll=powf(e,-distance/50);
					if (currentIndex<0||currentIndex>=width*height)
					{
						probAll=-1;
					}
					else
						probAll*=maximumProb[l];
					//if (distance<window_threshold_large)//hostDetectionResult[currentIndex+l*width*height]>=0.2)*/
					if(probAll>=0.0001&&maximumProb[l]>prob_threshold)
					{
						tmp.push_back(l);
					}
				}

				if (tmp.size()<=finalInd.size())
				{
					continue;
				}					

				float prob=0;
				for (int jj=0;jj<tmp.size();jj++)
				{
					int currentIndex=round(newU.at<float>(tmp[jj]+fullIndNum,0))*width+round(newU.at<float>(tmp[jj],0));
					prob+=hostDetectionResult[currentIndex+tmp[jj]*width*height];///maximumProb[tmp[jj]];
				}
				prob/=tmp.size();

				//cout<<"tmp size: "<<tmp.size()<<endl;

				if (prob>bestProb)
				{
					bestProb=prob;
					bestFitTrainingSampleInd=traningSapleInd[i];
					finalInd.clear();
					for (int kkk=0;kkk<tmp.size();kkk++)
					{
						finalInd.push_back(tmp[kkk]);
					}
					globalTransformation_optimal=globalTransformation.clone();

					//then, update the index again
					for (int ii=0;ii<fullIndNum;ii++)
					{
						if (candidatePoints[ii].size()<=1)
						{
							selectedPointIndex[ii]=0;
							continue;
						}
						int cindex;
						float minDis=1000000000000000;
						for (int jj=0;jj<candidatePoints[ii].size();jj++)
						{
							float currentDis=(((newU.at<float>(ii,0)-candidatePoints[ii][jj].x)*
								(newU.at<float>(ii,0)-candidatePoints[ii][jj].x)+
								(newU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][jj].y)*
								(newU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][jj].y)));
						
							if (currentDis<minDis)
							{
								minDis=currentDis;
								cindex=jj;
							}
						}	
					/*	if (ii==8)
						{
							cout<<cindex<<endl;
						}*/
						sampledPos[ii][0]=candidatePoints[ii][cindex].x;
						sampledPos[ii][1]=candidatePoints[ii][cindex].y;
						selectedPointIndex[ii]=cindex;
					}

				/*	for (int p=0;p<fullIndNum;p++)
					{
						selectedPointIndex[p]=curIndex[p];
					}*/
				}
				
			}
		}
	}

	//	Mat tmpIMg1=(*img).clone();
	//
	//	//	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
	//		Point c1;
	//	/*	for (int j=0;j<finalInd.size();j++)
	//		{
	//			c.x=sampledPos[finalInd[j]][0];
	//			c.y=sampledPos[finalInd[j]][1];
	//			circle(tmpIMg,c,5,255);
	//			cout<<finalInd[j]<<" "<<c.x<<" "<<c.y<<endl;
	//			namedWindow("KNN visualization");
	//			imshow("KNN visualization",tmpIMg);
	//			waitKey();
	//		}*/
	//		

	////then, use the best index set
	//for (int p=0;p<fullIndNum;p++)
	//{
	//	sampledPos[p][0]=candidatePoints[p][selectedPointIndex[p]].x;
	//	sampledPos[p][1]=candidatePoints[p][selectedPointIndex[p]].y;

	//	c1.x=sampledPos[p][0];
	//	c1.y=sampledPos[p][1];
	//	circle(tmpIMg1,c1,5,255);
	//	//cout<<selectedPointIndex[p]<<" "<<candidatePoints[p].size()<<endl;
	//	//cout<<p<<" "<<c.x<<" "<<c.y<<endl;
	//
	//}
	//namedWindow("KNN visualization");
	//	imshow("KNN visualization",tmpIMg1);
	//	waitKey();



	int bestIndNum=finalInd.size();
	vector<int> tmpInd;
	for (int i=0;i<totalShapeNum;i++)
	{
		tmpInd.clear();
		getTransformationInfo(finalInd,sampledPos,shapes,i);
		newU=fullTrainingPos[i]*globalTransformation;

		//int inNum=0;
		for (int l=0;l<fullIndNum;l++)
		{
			float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
				(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

			if (distance<window_threshold_large&&maximumProb[l]>prob_threshold)
			{
				tmpInd.push_back(l);
			}
		}
		if (tmpInd.size()>=finalInd.size())
		{
			getTransformationInfo(tmpInd,sampledPos,shapes,i);	//get transformation parameters
			newU=fullTrainingPos[i]*globalTransformation;
			tmpInd.clear();
			for (int l=0;l<fullIndNum;l++)
			{
				float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
					(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

				int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));
				
				if (distance<window_threshold_small&&maximumProb[l]>prob_threshold)
				{
					tmpInd.push_back(l);
				}

			
			}	
			if (tmpInd.size()>bestIndNum)
			{
				bestIndNum=tmpInd.size();
				bestFitTrainingSampleInd=i;
			}
		}
	}
	for (int i=0;i<30;i++)
	{
		AAM_exp->probForEachFeature[i]=0;
	}
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;


	
	////////////////////update the index again//////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////

	tmpInd.clear();
	for (int ii=0;ii<fullIndNum;ii++)
	{
		float currentDis=(((newU.at<float>(ii,0)-sampledPos[ii][0])*
			(newU.at<float>(ii,0)-sampledPos[ii][0])+
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])*
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])));	
		if (currentDis<window_threshold_large&&maximumProb[ii]>prob_threshold)
		{
			tmpInd.push_back(ii);
		}
	}
	getTransformationInfo(tmpInd,sampledPos,shapes,bestFitTrainingSampleInd);	//get transformation parameters
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	finalInd.clear();
	for (int l=0;l<fullIndNum;l++)
	{
		float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
			(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
		//cout<<combineIndList[l]<<" "<<distance<<endl;
		if (distance<window_threshold_small&&maximumProb[l]>prob_threshold)
		{
			finalInd.push_back(l);
		}
	}

	globalTransformation_optimal=globalTransformation.clone();

	/*Mat tmpIMg=(*img).clone();
	
			newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
			Point c;
			for (int j=0;j<fullIndNum;j++)
			{
				c.x=newU.at<float>(j,0);
				c.y=newU.at<float>(j+fullIndNum,0);
				circle(tmpIMg,c,5,255);
			}
			namedWindow("KNN visualization");
				imshow("KNN visualization",tmpIMg);
				waitKey();*/

	float bestDistance=1000000;
	float currentDis;
	//float 
	//now check distance

	//search KNN

	if (distanceVec==NULL)
	{
		distanceVec=new distancePir[totalShapeNum];
	}


	float prob_used[50];
	float sigma=0.3;
	for (int ii=0;ii<finalInd.size();ii++)
	{
		

		//float probAll=powf(e,-currentDis*currentDis/200);
		prob_used[ii]=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
			(2*sigma*sigma));
		//prob_used[ii]=maximumProb[ii];
		//prob_used[ii]=1;
		//int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));
		//int currentIndex=(int)(sampledPos[finalInd[ii]][1]*width+sampledPos[finalInd[ii]][0]);
		//cout<<ii<<" "<<maximumProb[ii]<<" "<<hostDetectionResult[currentIndex+finalInd[ii]*width*height]<<endl;
		/*if (hostDetectionResult[currentIndex+finalInd[ii]*width*height]>0.1)
		{
			prob_used[ii]=1;
		}
		else
			prob_used[ii]=0;*/
	}
	

	for (int i=0;i<totalShapeNum;i++)
	//for (int i=22;i<23;i++)
	{
		getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
		//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
		//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
		newU=fullTrainingPos[i]*globalTransformation;

		//newU=fullTrainingPos[i]*globalTransformation_optimal;

	/*	for (int ii=0;ii<finalInd.size();ii++)
		{
			cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
		}
		cout<<endl;*/

		float currentDis=0;
		for (int ii=0;ii<finalInd.size();ii++)
		{
			currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
				(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]))*prob_used[ii];	
		}

		distanceVec[i].first=currentDis;
		distanceVec[i].second=i;
		/*if (currentDis<bestDistance)
		{
			bestDistance=currentDis;
			bestFitTrainingSampleInd=i;
			globalTransformation_optimal=globalTransformation.clone();
		}*/
		//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
		//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	}
	sort(distanceVec,distanceVec+totalShapeNum,comparator);

	//{
	//	Mat meanU=newU.clone();
	//	meanU*=0;
	//	Mat tmpIMg=(*img).clone();
	//	for (int i=0;i<NNNum;i++)
	//	{
	//		int tmpInd=distanceVec[i].second;
	//		getTransformationInfo(finalInd,sampledPos,shapes,tmpInd);
	//		Mat newU1=fullTrainingPos[tmpInd]*globalTransformation;
	//		meanU+=newU1;
	//		Point c;
	//		for (int j=0;j<fullIndNum;j++)
	//		{
	//			c.x=newU1.at<float>(j,0);
	//			c.y=newU1.at<float>(j+fullIndNum,0);
	//			circle(tmpIMg,c,5,255);
	//		}
	//	}

	//	namedWindow("KNN visualization");
	//	imshow("KNN visualization",tmpIMg);
	//	waitKey();
	//}


	ofstream outttt("D:\\Fuhao\\Facial feature points detection\\GeoHash\\result comparison\\KNN.txt",ios::out);
	int nnnum=30;
	outttt<<nnnum<<endl;
	for (int i=0;i<nnnum;i++)
	{
		if (distanceVec[i].second>0)
		{
			outttt<<nameList[distanceVec[i].second-1]<<endl;
		}

	}
	outttt.close();

	////visualize the first N neighbors
	//for (int kk=0;kk<20;kk++)
	//{
	//	if (distanceVec[kk].second==0)
	//	{
	//		continue;
	//	}
	//	Mat imgtmp=imread(nameList[distanceVec[kk].second-1]);
	//	namedWindow("1");
	//	imshow("1",imgtmp);
	//	waitKey();
	//}

	bestFitTrainingSampleInd=distanceVec[0].second;
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	globalTransformation_optimal=globalTransformation.clone();

	//////////////////////do the KNN again/////////////////////////////////////////////
	//Mat KNN_NewU;
	//vector<int> tmpInd_KNN;
	//int bestNUm_beforeKNN=finalInd.size();
	//for (int i=0;i<NNNum;i++)
	////for (int i=22;i<23;i++)
	//{
	//	tmpInd_KNN.clear();
	//	int cind=distanceVec[i].second;
	//	KNN_NewU=fullTrainingPos[cind]*globalTransformation;

	//	//newU=fullTrainingPos[i]*globalTransformation_optimal;

	///*	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
	//	}
	//	cout<<endl;*/

	//	float currentDis=0;
	//	for (int ii=0;ii<fullIndNum;ii++)
	//	{
	//		for (int jj=0;jj<candidatePoints[ii].size();jj++)
	//		{
	//			currentDis=((KNN_NewU.at<float>(ii,0)-candidatePoints[ii][jj].x)*
	//				(KNN_NewU.at<float>(ii,0)-candidatePoints[ii][jj].x)+
	//				(KNN_NewU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][jj].y)*
	//				(KNN_NewU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][jj].y));	
	//			if (currentDis<window_threshold_small)
	//			{
	//				tmpInd_KNN.push_back(ii);
	//				break;
	//			}
	//		}	
	//	}

	//	if (tmpInd.size()>bestNUm_beforeKNN)
	//	{
	//		bestNUm_beforeKNN=tmpInd.size();
	//		bestFitTrainingSampleInd=cind;
	//	}
	//	/*if (currentDis<bestDistance)
	//	{
	//		bestDistance=currentDis;
	//		bestFitTrainingSampleInd=i;
	//		globalTransformation_optimal=globalTransformation.clone();
	//	}*/
	//	//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
	//	//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	//}

	////update the inlier and finalIND
	//tmpInd_KNN.clear();
	//KNN_NewU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;

	////newU=fullTrainingPos[i]*globalTransformation_optimal;

	///*	for (int ii=0;ii<finalInd.size();ii++)
	//{
	//cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
	//}
	//cout<<endl;*/

	//float currentDis_KNN=0;
	//for (int ii=0;ii<fullIndNum;ii++)
	//{
	//	for (int jj=0;jj<candidatePoints[ii].size();jj++)
	//	{
	//		currentDis_KNN=((KNN_NewU.at<float>(ii,0)-candidatePoints[ii][jj].x)*
	//			(KNN_NewU.at<float>(ii,0)-candidatePoints[ii][jj].x)+
	//			(KNN_NewU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][jj].y)*
	//			(KNN_NewU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][jj].y));	
	//		if (currentDis<window_threshold_small)
	//		{
	//			tmpInd_KNN.push_back(ii);
	//			sampledPos[ii][0]=candidatePoints[ii][jj].x;
	//			sampledPos[ii][1]=candidatePoints[ii][jj].y;
	//			break;
	//		}
	//	}	
	//}
	//finalInd.clear();
	//for (int ii=0;ii<tmpInd_KNN.size();ii++)
	//{
	//	finalInd.push_back(tmpInd_KNN[ii]);
	//}

	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	//newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	//globalTransformation_optimal=globalTransformation.clone();

			Mat tmpIMg=colorImgBackUP.clone();
	
			newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
			Point c;
			for (int j=0;j<fullIndNum;j++)
			{
				c.x=newU.at<float>(j,0);
				c.y=newU.at<float>(j+fullIndNum,0);
				circle(tmpIMg,c,3,Scalar(0,0,255));
			}
		/*	for (int j=0;j<finalInd.size();j++)
			{
				cout<<sampledPos[finalInd[j]][0]<<" "<<sampledPos[finalInd[j]][1]<<endl;
				c.x=sampledPos[finalInd[j]][0];
				c.y=sampledPos[finalInd[j]][1];
				circle(tmpIMg,c,2,Scalar(0,0,255));
			}*/

			//for (int j=0;j<fullIndNum;j++)
			//{
			//	for (int k=0;k<candidatePoints[j].size();j++)
			//	{
			//		circle(tmpIMg,candidatePoints[j][k],5,Scalar(255,0,0));
			//	}

			//}

			//char name_withSize[50];
			//sprintf(name_withSize, "Red: nearest neighbor");

			//char name_withSize1[50];
			//sprintf(name_withSize1, "Blue: detected modes");

			//int fontFace = FONT_HERSHEY_PLAIN;
			//putText(tmpIMg,name_withSize,Point(tmpIMg.cols-300,tmpIMg.rows-50),fontFace,1.2,Scalar(0,255,0));
			//putText(tmpIMg,name_withSize1,Point(tmpIMg.cols-300,tmpIMg.rows-25),fontFace,1.2,Scalar(0,255,0));
			//imwrite(saveName,tmpIMg);
			////namedWindow("KNN visualization");
			//	//imshow("KNN visualization",tmpIMg);
			//	//waitKey(2);

			//string nameTxt=saveName;
			//nameTxt=nameTxt.substr(0,nameTxt.length()-3);
			//nameTxt+="txt";

			//ofstream out_txt(nameTxt,ios::out);
			//for (int j=0;j<fullIndNum;j++)
			//{
			//	out_txt<<newU.at<float>(j,0)<<" "<<newU.at<float>(j+fullIndNum,0)<<endl;
			//}
			//out_txt.close();

			//nameTxt=saveName;
			//nameTxt=nameTxt.substr(0,nameTxt.length()-4);
			//nameTxt+="Modes.txt";
			//ofstream out_mode(nameTxt,ios::out);
			//for (int j=0;j<fullIndNum;j++)
			//{
			//	for (int k=0;k<candidatePoints[j].size();j++)
			//	{
			//		out_mode<<j<<" "<<candidatePoints[j][k].x<<" "<<candidatePoints[j][k].y<<" "<<(*depthImg).at<float>(candidatePoints[j][k].y,candidatePoints[j][k].x)<<endl;
			//	}
			//}
			//out_mode.close();
			//	return;
	///////////////////////////////////////////////////////////////////////////////////



	//update the inlier again
	tmpInd.clear();
	for (int ii=0;ii<fullIndNum;ii++)
	{
		float currentDis=(((newU.at<float>(ii,0)-sampledPos[ii][0])*
			(newU.at<float>(ii,0)-sampledPos[ii][0])+
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])*
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])));	
		if (currentDis<window_threshold_large)
		{
			tmpInd.push_back(ii);
		}
	}
	getTransformationInfo(tmpInd,sampledPos,shapes,bestFitTrainingSampleInd);	//get transformation parameters
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	finalInd.clear();
	for (int l=0;l<fullIndNum;l++)
	{
		float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
			(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
		//cout<<combineIndList[l]<<" "<<distance<<endl;
		if (distance<window_threshold_small&&maximumProb[l]>prob_threshold)
		{
			finalInd.push_back(l);
		}
	}
	globalTransformation_optimal=globalTransformation.clone();

	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;

	//then, update the index again
	for (int ii=0;ii<fullIndNum;ii++)
	{
		if (candidatePoints[ii].size()<=1)
		{
			continue;
		}
		int cindex;
		float minDis=1000000;
		for (int j=0;j<candidatePoints[ii].size();j++)
		{
			float currentDis=(((newU.at<float>(ii,0)-candidatePoints[ii][j].x)*
				(newU.at<float>(ii,0)-candidatePoints[ii][j].x)+
				(newU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][j].y)*
				(newU.at<float>(ii+fullIndNum,0)-candidatePoints[ii][j].y)));
			if (currentDis<minDis)
			{
				minDis=currentDis;
				cindex=j;
			}
		}
		selectedPointIndex[ii]=cindex;
		sampledPos[ii][0]=candidatePoints[ii][cindex].x;
		sampledPos[ii][1]=candidatePoints[ii][cindex].y;
	}

	///////////////////do the KNN again//////////////////////////////////////
	//for (int i=0;i<totalShapeNum;i++)
	////for (int i=22;i<23;i++)
	//{
	//	getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
	//	//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
	//	//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
	//	newU=fullTrainingPos[i]*globalTransformation;

	//	//newU=fullTrainingPos[i]*globalTransformation_optimal;

	///*	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
	//	}
	//	cout<<endl;*/

	//	float currentDis=0;
	//	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]))*prob_used[ii];	
	//	}

	//	distanceVec[i].first=currentDis;
	//	distanceVec[i].second=i;
	//	/*if (currentDis<bestDistance)
	//	{
	//		bestDistance=currentDis;
	//		bestFitTrainingSampleInd=i;
	//		globalTransformation_optimal=globalTransformation.clone();
	//	}*/
	//	//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
	//	//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	//}
	//sort(distanceVec,distanceVec+totalShapeNum,comparator);

	//bestFitTrainingSampleInd=distanceVec[0].second;
	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	//newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	//globalTransformation_optimal=globalTransformation.clone();

	///////////////////////////////////////////////////////////////////////////////////////


		float sigma1=sigma;
	//for (int ii=0;ii<finalInd.size();ii++)
	//{
	//	float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//		(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//		(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//		(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

	//	AAM_exp->distanceKNN[ii]=currentDis;

	//	int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

	//	float probAll=powf(e,-currentDis*currentDis/50);
	//	//probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
	//	/*probAll*=powf(e,-(hostDetectionResult[currentIndex+finalInd[ii]*width*height]-1)*(hostDetectionResult[currentIndex+finalInd[ii]*width*height]-1)/
	//		(2*sigma1*sigma1));*/
	//	probAll*=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
	//		(2*sigma1*sigma1));;
	//	AAM_exp->probForEachFeature[ii]=probAll;
	//	cout<<finalInd[ii]<<" "<<AAM_exp->probForEachFeature[ii]<<endl;
	//	//AAM_exp->probForEachFeature[ii]=1;
	//}

		//visualize the image data
		if (AAM_exp->showSingleStep&&img!=NULL)
		{
			ofstream out("D:\\Fuhao\\exp results\\flow chart\\KNN.txt",ios::out);
		
			//out.close();
			Mat meanU=newU.clone();
			meanU*=0;
			Mat tmpIMg=(*img).clone();
			for (int i=0;i<NNNum;i++)
			{
				int tmpInd=distanceVec[i].second;
				getTransformationInfo(finalInd,sampledPos,shapes,tmpInd);
				Mat newU1=fullTrainingPos[tmpInd]*globalTransformation;
				meanU+=newU1;
				Point c;
				for (int j=0;j<fullIndNum;j++)
				{
					c.x=newU1.at<float>(j,0);
					c.y=newU1.at<float>(j+fullIndNum,0);
					circle(tmpIMg,c,5,255);
				}
				for (int j=0;j<fullIndNum;j++)
				{
					out<<" "<<newU1.at<float>(j,0)<<" "<<
					newU1.at<float>(j+fullIndNum,0)<<" ";
				//	circle(tmpIMg,c,5,255);
				}
				out<<endl;
			}
			
			out.close();
			meanU/=NNNum;
			for (int j=0;j<fullIndNum;j++)
			{
				Point c;
				c.x=meanU.at<float>(j,0);
				c.y=meanU.at<float>(j+fullIndNum,0);
				//	circle(tmpIMg,c,5,255);
			}
			namedWindow("KNN visualization");
			imshow("KNN visualization",tmpIMg);
			waitKey();

		}

		//using all the index
		finalInd.clear();
		for (int i=0;i<fullIndNum;i++)
		{
			finalInd.push_back(i);
		}
	for (int ii=0;ii<finalInd.size();ii++)
	{
		int cindex=finalInd[ii];
		for (int j=0;j<candidatePoints[cindex].size();j++)
		{
			float currentDis=sqrt(((newU.at<float>(cindex,0)-candidatePoints[cindex][j].x)*
				(newU.at<float>(cindex,0)-candidatePoints[cindex][j].x)+
				(newU.at<float>(cindex+fullIndNum,0)-candidatePoints[cindex][j].y)*
				(newU.at<float>(cindex+fullIndNum,0)-candidatePoints[cindex][j].y)));
			float probAll=powf(e,-currentDis*currentDis/50);
			//probAll*=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
						//(2*sigma1*sigma1));
			AAM_exp->probForEachFeatureCandidates[ii][j]=probAll;

		/*	if (finalInd[ii]==2)
			{
				cout<<j<<" "<<currentDis<<" "<<probAll<<" "<<candidatePoints[cindex][j].x<<" "<<candidatePoints[cindex][j].y<<endl;
				
			}*/
			
		}
		AAM_exp->candidateNum[ii]=candidatePoints[cindex].size();
		//AAM_exp->probForEachFeature[ii]=1;
	}

	//use all the feature candidates
	for (int i=0;i<finalInd.size();i++)
	{
		AAM_exp->candidatePoints[i].clear();
		for (int j=0;j<candidatePoints[finalInd[i]].size();j++)
		{
			AAM_exp->candidatePoints[i].push_back(candidatePoints[finalInd[i]][j]);
		}
	}

	//Mat tmpIMg=(*img).clone();
	//
	//		newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
	//		Point c;
	//		for (int j=0;j<finalInd.size();j++)
	//		{
	//			c.x=newU.at<float>(finalInd[j],0);
	//			c.y=newU.at<float>(finalInd[j]+fullIndNum,0);
	//			circle(tmpIMg,c,5,255);
	//		}
	//		namedWindow("KNN visualization");
	//			imshow("KNN visualization",tmpIMg);
	//			waitKey();


	//



	////use only the selected features
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	AAM_exp->candidatePoints[i].clear();
	//	//for (int j=0;j<candidatePoints[finalInd[i]].size();j++)
	//	{
	//		AAM_exp->candidatePoints[i].push_back(candidatePoints[finalInd[i]][selectedPointIndex[i]]);
	//		int cindex=finalInd[i];
	//		float currentDis=sqrt(((newU.at<float>(cindex,0)-candidatePoints[cindex][selectedPointIndex[i]].x)*
	//			(newU.at<float>(cindex,0)-candidatePoints[cindex][selectedPointIndex[i]].x)+
	//			(newU.at<float>(cindex+fullIndNum,0)-candidatePoints[cindex][selectedPointIndex[i]].y)*
	//			(newU.at<float>(cindex+fullIndNum,0)-candidatePoints[cindex][selectedPointIndex[i]].y)));
	//		float probAll=powf(e,-currentDis*currentDis/50);
	//		AAM_exp->probForEachFeatureCandidates[i][0]=probAll;
	//	}
	//}
	/*cout<<"feature probabilities:\n";
	for (int i=0;i<finalInd.size();i++)
	{
		cout<<i<< " "<<maximumProb[finalInd[i]]<<" "<<AAM_exp->probForEachFeature[i]<<endl;
	}*/

	//then do the KNN fit
	//eigenVectors.at<double>(i,minInd);
	//cout<<eigenVectors.cols<<" "<<eigenVectors.rows<<endl;
	
	Mat KNNVec=Mat::zeros(eigenVectors.rows,NNNum,CV_64FC1);
	for (int i=0;i<NNNum;i++)
	{
		KNNVec.col(i)+=eigenVectors.col(distanceVec[i].second);
	}
	
	//if (localWeight>0)
	{
	//update the mean
	Mat mean_KNN=KNNVec.col(0)*0;
	for (int i=0;i<KNNVec.cols;i++)
	{
		mean_KNN+=KNNVec.col(i);
	}
	mean_KNN/=KNNVec.cols;
	for (int i=0;i<mean_KNN.rows;i++)
	{
		AAM_exp->priorMean[i]=mean_KNN.at<double>(i,0);
	}

	//update the conv
	for (int i=0;i<KNNVec.cols;i++)
	{
		KNNVec.col(i)-=mean_KNN;
	}
	Mat KNNVec_tran;
	transpose(KNNVec,KNNVec_tran);
	Mat convKNN=KNNVec*KNNVec_tran/KNNVec.cols;

	Mat conv_inv_KNN=convKNN.inv();

//	cout<<AAM_exp->priorSigma.cols<<" "<<AAM_exp->priorSigma.rows<<endl;
	for (int i=0;i<conv_inv_KNN.rows;i++)
	{
		for (int j=0;j<conv_inv_KNN.cols;j++)
		{
			AAM_exp->priorSigma.at<double>(i,j)=conv_inv_KNN.at<double>(i,j);
		}
	}


	if (localWeight>0)
	{


		//train the local PCA model
		int s_dim=eigenVectors.rows;
		//use KNNVec_tran to train
		CvMat *pData=cvCreateMat(KNNVec_tran.rows,KNNVec_tran.cols,CV_64FC1);
		for (int i=0;i<KNNVec_tran.rows;i++)
		{
			for (int j=0;j<KNNVec_tran.cols;j++)
			{
				//CV_MAT_ELEM(*pData,double,i,j)=shape[i]->ptsForMatlab[j];

				//here,we keep the shape in the same scale with the meanshape
				CV_MAT_ELEM(*pData,double,i,j)=KNNVec_tran.at<double>(i,j);
			}

		}
		if (AAM_exp->local_s_mean==NULL)
		{
			AAM_exp->local_s_mean = cvCreateMat(1, KNNVec_tran.cols, CV_64FC1);
			AAM_exp->m_local_s_mean=cvarrToMat(AAM_exp->local_s_mean);
		}

		if (AAM_exp->local_s_vec==NULL)
		{
			AAM_exp->local_s_vec=cvCreateMat( min(KNNVec_tran.rows,KNNVec_tran.cols), KNNVec_tran.cols, CV_64FC1); 
		}

		CvMat* s_value = cvCreateMat(1, min(KNNVec_tran.cols,KNNVec_tran.rows), CV_64FC1);
		//CvMat *s_PCAvec = cvCreateMat( min(KNNVec_tran.rows,KNNVec_tran.cols), KNNVec_tran.cols, CV_64FC1); 
		cvCalcPCA( pData, AAM_exp->local_s_mean, s_value, AAM_exp->local_s_vec, CV_PCA_DATA_AS_ROW );
		AAM_exp->m_local_mean=cvarrToMat(AAM_exp->local_s_mean);

		double sumEigVal=0;
		for (int i=0;i<s_value->cols;i++)
		{
			sumEigVal+=CV_MAT_ELEM(*s_value,double,0,i);
		}

		double sumCur=0;
		for (int i=0;i<s_value->cols;i++)
		{
			sumCur+=CV_MAT_ELEM(*s_value,double,0,i);
			if (sumCur/sumEigVal>=0.98)
			{
				AAM_exp->local_shape_dim=i+1;
				break;
			}
		}
		//cout<<"local dim: "<<AAM_exp->local_shape_dim<<endl;

		Mat curEigenVec=cvarrToMat(AAM_exp->local_s_vec);
		Mat usedEigenVec=curEigenVec.rowRange(Range(0,AAM_exp->local_shape_dim));
		Mat usedEigenVec_tran;
		transpose(usedEigenVec,usedEigenVec_tran);
		Mat localHessian=usedEigenVec_tran*usedEigenVec;
		localHessian=Mat::eye(localHessian.rows,localHessian.cols,CV_64FC1)-localHessian;

		Mat localHessian_tran;
		transpose(localHessian,localHessian_tran);
		localHessian=localHessian_tran*localHessian;
		for (int i=0;i<localHessian.rows;i++)
		{
			for (int j=0;j<localHessian.cols;j++)
			{
				AAM_exp->m_localHessian.at<double>(i,j)=localHessian.at<double>(i,j);
			}
		}
	}

	}
	//delete []distance;
	

	//find the best one
	//for (int i=0;i<totalShapeNum;i++)
	////for (int i=22;i<23;i++)
	//{
	//	getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
	//	//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
	//	//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
	//	newU=fullTrainingPos[i]*globalTransformation;

	//	//newU=fullTrainingPos[i]*globalTransformation_optimal;

	///*	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
	//	}
	//	cout<<endl;*/

	//	float currentDis=0;
	//	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]));	
	//	}

	//	if (currentDis<bestDistance)
	//	{
	//		bestDistance=currentDis;
	//		bestFitTrainingSampleInd=i;
	//		globalTransformation_optimal=globalTransformation.clone();
	//	}
	//	//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
	//	//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	//}


	return;

	//return;

	//////////////////KNN on GPU////////////////////////////////
	//input: shapes and initial global transformation and inlier set
	//output: the final inlier set and the best example
	//float *rt_parameters=new float[4];
	//for (int i=0;i<4;i++)
	//{
	//	rt_parameters[i]=globalTransformation_optimal.at<float>(i,0);
	//}
	//float *currentShape=new float[fullIndNum*2];
	//for (int i=0;i<fullIndNum;i++)
	//{
	//	currentShape[i]=sampledPos[i][0];
	//	currentShape[i+fullIndNum]=sampledPos[i][1];
	//}
	////return;
	//KNN_search(rt_parameters,currentShape,totalShapeNum,fullIndNum);

	//return;
	///////////////////////////////////////////////////
	//LONGLONG   t1,t2; 
	//LONGLONG   persecond; 

	//QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 

	// #pragma omp parallel for
	//for (int i=0;i<10000;i++)
	//{
	//	getTransformationInfo_toVector(finalInd,sampledPos,shapes,i%totalShapeNum);	//get transformation parameters
	//}

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	//double   time=(t2-t1)*1000/persecond; 


	//cout<<"\n************************************************\n";
	//cout<<"transofrmation  time: "<<time<<" ms"<<endl;
	//cout<<"\n************************************************\n";

	//then, check the two lower mouth points 11 and 23
	//vector<int> tmp;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	tmp.push_back(finalInd[i]);
	//}
	//if (find(finalInd.begin(),finalInd.end(),11)==finalInd.end())
	//{
	//	tmp.push_back(11);
	//	getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[bestFitTrainingSampleInd]);	//get transformation parameters
	//
	//	float scale1=globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
	//		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0);

	//	float scale2=globalTransformation_optimal.at<float>(0,0)*globalTransformation_optimal.at<float>(0,0)+
	//		globalTransformation_optimal.at<float>(1,0)*globalTransformation_optimal.at<float>(1,0);

	//	if (abs(scale1-scale2)<0.05&&abs(globalTransformation.at<float>(2,0)-globalTransformation_optimal.at<float>(2,0))<5&&
	//		abs(globalTransformation.at<float>(3,0)-globalTransformation_optimal.at<float>(3,0))<5)
	//	{
	//		finalInd.push_back(11);
	//	}
	//	//globalTransformation_optimal.at<float>(0,0)=globalTransformation.at<float>(0,0);
	//	//globalTransformation_optimal.at<float>(1,0)=globalTransformation.at<float>(1,0);
	//	//globalTransformation_optimal.at<float>(2,0)=globalTransformation.at<float>(2,0);
	//	//globalTransformation_optimal.at<float>(3,0)=globalTransformation.at<float>(3,0);
	//	tmp.pop_back();
	//}

	//if (find(finalInd.begin(),finalInd.end(),23)==finalInd.end())
	//{
	//	tmp.push_back(23);
	//	getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[bestFitTrainingSampleInd]);	//get transformation parameters
	//	
	//	float scale1=globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
	//		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0);

	//	float scale2=globalTransformation_optimal.at<float>(0,0)*globalTransformation_optimal.at<float>(0,0)+
	//		globalTransformation_optimal.at<float>(1,0)*globalTransformation_optimal.at<float>(1,0);

	//	if (abs(scale1-scale2)<0.05&&abs(globalTransformation.at<float>(2,0)-globalTransformation_optimal.at<float>(2,0))<5&&
	//		abs(globalTransformation.at<float>(3,0)-globalTransformation_optimal.at<float>(3,0))<5)
	//	{
	//		finalInd.push_back(23);
	//	}
	//	//globalTransformation_optimal.at<float>(0,0)=globalTransformation.at<float>(0,0);
	//	//	globalTransformation_optimal.at<float>(1,0)=globalTransformation.at<float>(1,0);
	//	//globalTransformation_optimal.at<float>(2,0)=globalTransformation.at<float>(2,0);
	//	//globalTransformation_optimal.at<float>(3,0)=globalTransformation.at<float>(3,0);
	//}

	/////////////////////////new version, do not estimate gtP for every example//////////////////////////////////
	//globalTransformation=globalTransformation_optimal;
//bestProb=0;
//finalInd.clear();
//for (int i=0;i<fullIndNum;i++)
//{
//	if (maximumProb[i]>0.95)
//	{
//		finalInd.push_back(i);
//		bestProb+=maximumProb[i];
//	}
//}
//bestProb/=finalInd.size();
	Mat *newUList=new Mat[totalShapeNum];
	bool *usedLabel=new bool[fullIndNum];
	//vector <int >tmpInd;
	bestProb=0;
	bool needC=false;
	for (int i=0;i<totalShapeNum;i++)
	{
		tmpInd.clear();
		getTransformationInfo(finalInd,sampledPos,shapes,i);
		newUList[i]=fullTrainingPos[i]*globalTransformation;
		newU=newUList[i];

		//estimate the prob
		float prob=0;
		for (int jj=0;jj<fullIndNum;jj++)
		{
			int currentIndex=round(newU.at<float>(jj+fullIndNum,0))*width+round(newU.at<float>(jj,0));
			prob+=hostDetectionResult[currentIndex+jj*width*height];///maximumProb[tmpInd[jj]];
		}
		prob/=fullIndNum;
		if (prob>bestProb)
		{
			bestProb=prob;
			bestFitTrainingSampleInd=i;
		}
		continue;
		//int inNum=0;
		for (int l=0;l<fullIndNum;l++)
		{
			float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
				(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
			
			//cout<<l<<endl;
			//{
			//	Mat tmp=tmpImg.clone();
			//	Point c;
			//	for (int j=0;j<finalInd.size();j++)
			//	{
			//	
			//		//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
			//		c.x=newU.at<float>(finalInd[j],0);
			//		c.y=newU.at<float>(finalInd[j]+fullIndNum,0);
			//		circle(tmp,c,5,255);
			//	}
			//	for (int j=0;j<fullIndNum;j++)
			//	{
			//		//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
			//		if (j!=l)
			//		{
			//			continue;
			//		}
			//		c.x=sampledPos[j][0];
			//		c.y=sampledPos[j][1];
			//		circle(tmp,c,2,255);
			//	}
			//	namedWindow("aligned shape");
			//	imshow("aligned shape",tmp);
			//	waitKey();

			//}
			//cout<<distance<<endl;
			int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));

			float probAll=powf(e,-distance/50);
			probAll*=hostDetectionResult[currentIndex+l*width*height];
			//cout<<l<<" "<<probAll<<endl;
		//	cout<<probAll<<" "<<powf(e,-distance/50)<<" "<<hostDetectionResult[currentIndex+l*width*height]<<endl;
			//if (distance<window_threshold_small&&hostDetectionResult[currentIndex+l*width*height]>=0.2)
			if(probAll>0.00001)
			{
		
				tmpInd.push_back(l);
				//inNum++;
			}

		}
		if (needC)
		{
			if (tmpInd.size()<=finalInd.size())
			{
				continue;
			}
		}
		//if (needC&&tmpInd.size()>finalInd.size())
		{
			float prob=0;
			for (int jj=0;jj<tmpInd.size();jj++)
			{
				int currentIndex=round(newU.at<float>(tmpInd[jj]+fullIndNum,0))*width+round(newU.at<float>(tmpInd[jj],0));
				prob+=hostDetectionResult[currentIndex+tmpInd[jj]*width*height];///maximumProb[tmpInd[jj]];
			}
			prob/=tmpInd.size();

			//cout<<"tmp size: "<<tmp.size()<<endl;

			if (prob>bestProb)
			{
				bestProb=prob;
				bestFitTrainingSampleInd=traningSapleInd[i];
				finalInd.clear();
				for (int kkk=0;kkk<tmpInd.size();kkk++)
				{
					finalInd.push_back(tmpInd[kkk]);
				}
				/*getTransformationInfo(tmpInd,sampledPos,shapes,i);
				globalTransformation_optimal=globalTransformation.clone();
				newUList[i]=fullTrainingPos[i]*globalTransformation;*/
				needC=true;
			}
		}
		//cout<<inNum<<endl;
	}
	
	for (int i=0;i<30;i++)
	{
		AAM_exp->probForEachFeature[i]=0;
	}
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;

	finalInd.clear();
	for (int ii=0;ii<fullIndNum;ii++)
	{
		float currentDis=(((newU.at<float>(ii,0)-sampledPos[ii][0])*
			(newU.at<float>(ii,0)-sampledPos[ii][0])+
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])*
			(newU.at<float>(ii+fullIndNum,0)-sampledPos[ii][1])));	
		if (currentDis<window_threshold_large)
		{
			finalInd.push_back(ii);
		}
	}

	for (int ii=0;ii<finalInd.size();ii++)
	{
		float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

		AAM_exp->distanceKNN[ii]=currentDis;

		int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

		float probAll=powf(e,-currentDis*currentDis/200);
		probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
		AAM_exp->probForEachFeature[ii]=probAll;
	}
	return;
	/*finalInd.clear();
	for (int i=0;i<fullIndNum;i++)
	{
		if (usedLabel[i])
		{
			finalInd.push_back(i);
		}
	}*/

	//

	////return;
	////cout<<"----------------inlier num on CPU: "<<finalInd.size()<<endl;
	//float bestDistance=1000000;
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	float currentDis=0;
	//	newU=newUList[i];

	//

	//	for (int ii=0;ii<finalInd.size();ii++)
	//	{
	//		currentDis+=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	
	//	}

	//	//cout<<currentDis<<endl;
	//	//Mat tmp=tmpImg.clone();
	//	//Point c;
	//	//for (int j=0;j<finalInd.size();j++)
	//	//{
	//	//	//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
	//	//	c.x=newU.at<float>(finalInd[j],0);
	//	//	c.y=newU.at<float>(finalInd[j]+fullIndNum,0);
	//	//	circle(tmp,c,5,255);
	//	//}
	//	//for (int j=0;j<finalInd.size();j++)
	//	//{
	//	//	//cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
	//	//	c.x=sampledPos[finalInd[j]][0];
	//	//	c.y=sampledPos[finalInd[j]][1];
	//	//	circle(tmp,c,2,255);
	//	//}
	//	//namedWindow("aligned shape");
	//	//imshow("aligned shape",tmp);
	//	//waitKey();

	//	if (currentDis<bestDistance)
	//	{
	//		bestDistance=currentDis;
	//		bestFitTrainingSampleInd=i;
	//	}
	//}

	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);

	//for (int i=0;i<30;i++)
	//{
	//	AAM_exp->probForEachFeature[i]=0;
	//}
	//newU=newUList[bestFitTrainingSampleInd];
	//for (int ii=0;ii<finalInd.size();ii++)
	//{
	//	float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
	//		(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
	//		(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
	//		(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

	//	AAM_exp->distanceKNN[ii]=currentDis;

	//	int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

	//	float probAll=powf(e,-currentDis*currentDis/50);
	//	probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
	//	AAM_exp->probForEachFeature[ii]=probAll;
	//}

	//globalTransformation_optimal=globalTransformation.clone();


	///////////////////////////////////////////////////////////////////////////////////

	///////////////////////old version, re-estimate global transforation every time///////////////////////////////
//	float bestDistance=1000000;
//	float currentDis;
	//float 
	//now check distance
	for (int i=0;i<totalShapeNum;i++)
	//for (int i=22;i<23;i++)
	{
		//cout<<i<<endl;
		getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
		//globalTransformation.at<float>(0,0)=globalTransformation_optimal.at<float>(0,0);
		//globalTransformation.at<float>(1,0)=globalTransformation_optimal.at<float>(1,0);
		newU=fullTrainingPos[i]*globalTransformation;

		//newU=fullTrainingPos[i]*globalTransformation_optimal;

	/*	for (int ii=0;ii<finalInd.size();ii++)
		{
			cout<<newU.at<float>(finalInd[ii],0)<<" "<<newU.at<float>(finalInd[ii]+fullIndNum,0)<<" ";
		}
		cout<<endl;*/

		float currentDis=0;
		for (int ii=0;ii<finalInd.size();ii++)
		{
			currentDis+=sqrtf((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
				(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]));	
		}

		if (currentDis<bestDistance)
		{
			bestDistance=currentDis;
			bestFitTrainingSampleInd=i;
			globalTransformation_optimal=globalTransformation.clone();
		}
		//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
		//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

	}

	for (int i=0;i<30;i++)
	{
		AAM_exp->probForEachFeature[i]=0;
	}
	newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	for (int ii=0;ii<finalInd.size();ii++)
	{
		float currentDis=sqrt(((newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
			(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
			(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])));	

		AAM_exp->distanceKNN[ii]=currentDis;

		int currentIndex=round(newU.at<float>(finalInd[ii]+fullIndNum,0))*width+round(newU.at<float>(finalInd[ii],0));

		float probAll=powf(e,-currentDis*currentDis/200);
		probAll*=hostDetectionResult[currentIndex+finalInd[ii]*width*height];
		AAM_exp->probForEachFeature[ii]=probAll;
	}
	return;
	for (int l=0;l<fullIndNum;l++)
	{
		float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
			(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
		/*if (combineIndList[l]==56||combineIndList[l]==61)
		{
			cout<<combineIndList[l]<<" "<<sqrt(distance)<<endl;
		}*/
		if (distance<window_threshold_large)
		{
			if (find(finalInd.begin(),finalInd.end(),l)==finalInd.end())
			{
				finalInd.push_back(l);
			}
			int currentIndex=round(newU.at<float>(l+fullIndNum,0))*width+round(newU.at<float>(l,0));

			float probAll=powf(e,-currentDis/200);
			probAll*=hostDetectionResult[currentIndex+l*width*height];
			AAM_exp->probForEachFeature[finalInd.size()-1]=probAll;
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////

	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);	
	//newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	//finalInd.clear();
	//for (int l=0;l<fullIndNum;l++)
	//{
	//	float distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
	//		(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
	//	if (distance<window_threshold_large)
	//	{
	//		finalInd.push_back(l);
	//	}
	//}
	//getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);	
	//globalTransformation_optimal=globalTransformation.clone();
}

void AAM_Detection_Combination::ransac_noSampling(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb)
{
	srand(time(NULL));
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	


	vector<int> traningSapleInd;
	//RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

	//use all the training sample
	traningSapleInd.resize(totalShapeNum);
	for (int i=0;i<totalShapeNum;i++)
	{
		traningSapleInd[i]=i;
	}

	Mat newU;
	//go over every traing example
	//for (int i=0;i<sampleNumberFromTrainingSet;i++)
	for(int i=0;i<traningSapleInd.size();i++)
	{
		//sample from probability map, currently we use global maxima only: sampledPos
		for (int j=0;j<sampleNumberFromProbMap;j++)
		{
			//sample from the feature locations
			for (int k=0;k<sampleNumberFromFeature*2;k++)
			{
				vector<int> currentInd;
				RandSample_V1(fullIndNum,2,currentInd);	//sample the current used indices
				getTransformationInfo(currentInd,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters

				newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;



				//cout<<"the distances are: ";
				vector<int> tmp;
				float distance;
				int cind;
				for (int l=0;l<fullIndNum;l++)
				{
					distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
						(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);

					//cout<<distance<<" ";
					if (distance<window_threshold_small)
					{
						tmp.push_back(l);
					}
				}
				//cout<<endl;
				
		/*		if (tmp.size()>=w_critical)
				{
					Mat img=imread("G:\\face database\\kinect data\\test real data_original image_compact sampling\\Fuhao_583.jpg");
					Point c;
					for (int oo=0;oo<newU.rows/2;oo++)
					{
						c.x=newU.at<float>(oo);
						c.y=newU.at<float>(oo+newU.rows/2);
						circle(img,c,5,255);
						c.x=sampledPos[oo][0];
						c.y=sampledPos[oo][1];
						circle(img,c,2,255);
						namedWindow("1");
						imshow("1",img);
						waitKey();

					}
				}*/
				
			/*	for (int oo=0;oo<labelNum-1;oo++)
				{
					c.x=sampledPos[oo][0];
					c.y=sampledPos[oo][1];
					circle(img,c,2,255);
				}*/
			
				//check if there are enough inliers
				if (tmp.size()<w_critical)
				{
					continue;
				}

				//int inlierNum=0;
				//for (int l=0;l<criticalIndNum;l++)
				//{
				//	if (find(tmp.begin(),tmp.end(),criticalList[l])!=tmp.end())
				//	{
				//		inlierNum++;
				//	}
				//}

				////if there are enough inliers, then check its goodness
				//if (inlierNum>w_critical)
				{
					getTransformationInfo(tmp,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters
					newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;

					tmp.clear();
					for (int l=0;l<fullIndNum;l++)
					{
						distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
							(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;
						if (distance<window_threshold_large)
						{
							tmp.push_back(l);
						}
					}

					if (tmp.size()<finalInd.size())
					{
						continue;
					}

					//Mat img=imread("G:\\face database\\kinect data\\test real data_original image_compact sampling\\Fuhao_583.jpg");

					//Mat newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation_optimal;
				/*	Point c;
					for (int oo=0;oo<newU.rows/2;oo++)
					{
						c.x=newU.at<float>(oo);
						c.y=newU.at<float>(oo+newU.rows/2);
						circle(img,c,5,255);
					}	
					for (int oo=0;oo<labelNum-1;oo++)
					{
						c.x=sampledPos[oo][0];
						c.y=sampledPos[oo][1];
						circle(img,c,2,255);
					}
					namedWindow("1");
					imshow("1",img);
					waitKey();*/

					
					
					float prob=0;
					for (int jj=0;jj<tmp.size();jj++)
					{
						int currentIndex=round(newU.at<float>(tmp[jj]+fullIndNum,0))*width+round(newU.at<float>(tmp[jj],0));
						prob+=hostDetectionResult[currentIndex+tmp[jj]*width*height]/maximumProb[tmp[jj]];
					}
					prob/=tmp.size();

					//cout<<"tmp size: "<<tmp.size()<<endl;

					if (prob>bestProb)
					{
						bestProb=prob;
						bestFitTrainingSampleInd=traningSapleInd[i];
						finalInd.clear();
						for (int kkk=0;kkk<tmp.size();kkk++)
						{
							finalInd.push_back(tmp[kkk]);
						}
						globalTransformation_optimal=globalTransformation.clone();
					}
				}
			}
		}
	}

	//return;

	float bestDistance=1000000;
	//float 
	//now check distance
	for (int i=0;i<totalShapeNum;i++)
	//for (int i=22;i<23;i++)
	{
		//cout<<i<<endl;
		getTransformationInfo(finalInd,sampledPos,shapes,i);	//get transformation parameters
		newU=fullTrainingPos[i]*globalTransformation;
		float currentDis=0;
		for (int ii=0;ii<finalInd.size();ii++)
		{
			currentDis+=(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])*
				(newU.at<float>(finalInd[ii],0)-sampledPos[finalInd[ii]][0])+
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1])*
				(newU.at<float>(finalInd[ii]+fullIndNum,0)-sampledPos[finalInd[ii]][1]);	
		}

		if (currentDis<bestDistance)
		{
			bestDistance=currentDis;
			bestFitTrainingSampleInd=i;
			globalTransformation_optimal=globalTransformation.clone();
		}
		//distance=(newU.at<float>(l,0)-sampledPos[l][0])*(newU.at<float>(l,0)-sampledPos[l][0])+
		//(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1])*(newU.at<float>(l+fullIndNum,0)-sampledPos[l][1]);		//cout<<distance<<endl;

	}


//	cout<<"the best probility: "<<bestProb<<endl;

	//Mat img=imread("G:\\face database\\kinect data\\test real data_original image_compact sampling\\Fuhao_583.jpg");

	//Mat newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation_optimal;
	//Point c;
	//for (int oo=0;oo<newU.rows/2;oo++)
	//{
	//	c.x=newU.at<float>(oo);
	//	c.y=newU.at<float>(oo+newU.rows/2);
	//	circle(img,c,5,255);
	//}				
	//namedWindow("1");
	//imshow("1",img);
	//waitKey();
	//
}


void AAM_Detection_Combination::findSecondModes_Maxima(int **samplePos,int width,int height, vector<int>& Ind,float *probMap,float *maxProb,Mat *img,int sx,int ex,int sy,int ey)
{

	//Mat tmpImg=(*img).clone();
	//ofstream out("D:\\Fuhao\\exp results\\Final\\mode process\\modes.txt",ios::out);
//
	//ofstream out1("D:\\Fuhao\\exp results\\Final\\mode process\\prob_map.txt",ios::out);

	for (int i=0;i<Ind.size();i++)
	{
		candidatePoints[i].clear();

	/*	if (maxProb[Ind[i]]<0.35)
		{
			candidatePoints[i].push_back(Point(samplePos[Ind[i]][0],samplePos[Ind[i]][1]));
			continue;
		}*/
		int cIndex=Ind[i];

		Mat tmpProb=Mat::zeros(height,width,CV_64FC1);
		Mat usageMap=Mat::zeros(height,width,CV_64FC1);

		//set up the prob image and map
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				float cProb=probMap[j*width+k+cIndex*width*height];
				if (cProb>=0.7*maxProb[cIndex])
				{
					tmpProb.at<double>(j,k)=cProb;
					usageMap.at<double>(j,k)=1;
				}
				//continue;
			}
		}

		//if (Ind[i]==14)
		//{
		//	for (int j=0;j<height;j++)
		//	{
		//		for (int k=0;k<width;k++)
		//		{
		//			//if (j>=sy&&j<=ey&&k>=sx&&k<=ex)
		//			if(probMap[j*width+k+cIndex*width*height]>0)
		//			{
		//				out1<<k<<" "<<j<<" "<<probMap[j*width+k+cIndex*width*height]<<endl;
		//			}
		//			//float cProb=probMap[j*width+k+cIndex*width*height];
		//		
		//			//continue;
		//		}
		//	}
		//	
		//}

		//kill the connected region
		vector<Point> totalPointList;
		vector<Point> pointList;
		Point c,c_cur;
		c.x=samplePos[cIndex][0];
		c.y=samplePos[cIndex][1];
		pointList.push_back(c);
		while (pointList.size()!=0)
		{
			totalPointList.clear();
			c_cur=pointList[pointList.size()-1];

			//candidatePoints[i].push_back(c_cur);
			pointList.pop_back();
			//usageMap.at<double>(c_cur.y,c_cur.x)=0;
			//cout<<tmpProb.at<double>(c_cur.y,c_cur.x)<<endl;
			tmpProb.at<double>(c_cur.y,c_cur.x)=0;
			totalPointList.push_back(c_cur);
			//then check the 8 neighbors
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x-1)>0)
			{
				pointList.push_back(Point(c_cur.x-1,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x)>0)
			{
				pointList.push_back(Point(c_cur.x,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x+1)>0)
			{
				pointList.push_back(Point(c_cur.x+1,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y,c_cur.x-1)>0)
			{
				pointList.push_back(Point(c_cur.x-1,c_cur.y));
			}
			if (tmpProb.at<double>(c_cur.y,c_cur.x+1)>0)
			{
				pointList.push_back(Point(c_cur.x+1,c_cur.y));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x-1)>0)
			{
				pointList.push_back(Point(c_cur.x-1,c_cur.y+1));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x)>0)
			{
				pointList.push_back(Point(c_cur.x,c_cur.y+1));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x+1)>0)
			{
				pointList.push_back(Point(c_cur.x+1,c_cur.y+1));
			}

		}
		////calculate the estimation
		//double expectation[]={0,0};
		//double sumProb=0;
		//for (int l=0;l<totalPointList.size();l++)
		//{
		//	c_cur=totalPointList[l];
		//	expectation[0]+=c_cur.x*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
		//	expectation[1]+=c_cur.y*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
		//	sumProb+=probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
		//}
		//expectation[0]/=sumProb;
		//expectation[1]/=sumProb;

		////if ((expectation[0]-samplePos[cIndex][0])*(expectation[0]-samplePos[cIndex][0])+
		////(expectation[1]-samplePos[cIndex][1])*(expectation[1]-samplePos[cIndex][1])>9)
		//{
		//	candidatePoints[i].push_back(Point(expectation[0],expectation[1]));
		//}
		
		//use the maxima
		float maxProb=0;
		int maxInd;
		for (int l=0;l<totalPointList.size();l++)
		{
			c_cur=totalPointList[l];
			if (maxProb<probMap[c_cur.y*width+c_cur.x+cIndex*width*height])
			{
				maxProb=probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
				maxInd=l;
			}
		}
		candidatePoints[i].push_back(totalPointList[maxInd]);
		/*if (Ind[i]==14)
		{
			for (int l=0;l<totalPointList.size();l++)
			{
				out<<candidatePoints[i].size()<<" "<<totalPointList[l].x<<" "<<totalPointList[l].y<<endl;
			}
		}*/

		//candidatePoints[i].push_back(c);

		totalPointList.clear();
		//then check the remaining region, only remain region which size>3
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				if (tmpProb.at<double>(j,k)>0)
				{
					//check the connected region
					pointList.clear();
					totalPointList.clear();
					pointList.push_back(Point(k,j));
					while (pointList.size()!=0)
					{
						c_cur=pointList[pointList.size()-1];
						totalPointList.push_back(c_cur);
						pointList.pop_back();
						//usageMap.at<double>(c_cur.y,c_cur.x)=0;
						//cout<<tmpProb.at<double>(c_cur.y,c_cur.x)<<endl;
						tmpProb.at<double>(c_cur.y,c_cur.x)=0;

						//then check the 8 neighbors
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x-1)>0)
						{
							pointList.push_back(Point(c_cur.x-1,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x)>0)
						{
							pointList.push_back(Point(c_cur.x,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x+1)>0)
						{
							pointList.push_back(Point(c_cur.x+1,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y,c_cur.x-1)>0)
						{
							pointList.push_back(Point(c_cur.x-1,c_cur.y));
						}
						if (tmpProb.at<double>(c_cur.y,c_cur.x+1)>0)
						{
							pointList.push_back(Point(c_cur.x+1,c_cur.y));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x-1)>0)
						{
							pointList.push_back(Point(c_cur.x-1,c_cur.y+1));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x)>0)
						{
							pointList.push_back(Point(c_cur.x,c_cur.y+1));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x+1)>0)
						{
							pointList.push_back(Point(c_cur.x+1,c_cur.y+1));
						}

					}

					if (totalPointList.size()>3)
					{
						//double expectation[]={0,0};
						//double sumProb=0;
						//for (int l=0;l<totalPointList.size();l++)
						//{
						//	c_cur=totalPointList[l];
						//	expectation[0]+=c_cur.x*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
						//	expectation[1]+=c_cur.y*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
						//	sumProb+=probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
						//}
						//expectation[0]/=sumProb;
						//expectation[1]/=sumProb;
						////if ((expectation[0]-samplePos[cIndex][0])*(expectation[0]-samplePos[cIndex][0])+
						//	//(expectation[1]-samplePos[cIndex][1])*(expectation[1]-samplePos[cIndex][1])>9)
						//if(expectation[0]>=sx&&expectation[0]<=ex&&expectation[1]>=sy&&expectation[1]<=ey)
						////if(expectation[0]>50&&expectation[0]<590&&expectation[1]>50&&expectation[1]<430)
						//{
						//	candidatePoints[i].push_back(Point(expectation[0],expectation[1]));

						//	/*if (Ind[i]==14)
						//	{
						//		for (int l=0;l<totalPointList.size();l++)
						//		{
						//			out<<candidatePoints[i].size()<<" "<<totalPointList[l].x<<" "<<totalPointList[l].y<<endl;
						//		}
						//	}*/
						//}

						for (int l=0;l<totalPointList.size();l++)
						{
							c_cur=totalPointList[l];
							if (maxProb<probMap[c_cur.y*width+c_cur.x+cIndex*width*height])
							{
								maxProb=probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
								maxInd=l;
							}
						}
						candidatePoints[i].push_back(totalPointList[maxInd]);
					
					}
					else
					{
						/*for (int l=0;l<totalPointList.size();l++)
						{
							c_cur=totalPointList[l];
							tmpProb.at<double>(c_cur.y,c_cur.x)=0;
						}*/
					}
				}
			}
		}
		
	
	/*	if (img!=NULL&&(combineIndList[cIndex]==42||combineIndList[cIndex]==48))
		{
			cout<<combineIndList[cIndex]<<" "<<candidatePoints[i].size()<<endl;
			Mat tmpImg=(*img).clone();
			for (int l=0;l<candidatePoints[i].size();l++)
			{
				circle(tmpImg,candidatePoints[i][l],2,255);
			}
			namedWindow("ProbMap");
			imshow("ProbMap",tmpImg);
			waitKey();
		}*/
		

	}
		//out.close();
	//	out1.close();
}

void AAM_Detection_Combination::findSecondModes_localMaxima(int **samplePos,int width,int height, vector<int>& Ind,float *probMap,float *maxProb,Mat *img,int sx,int ex,int sy,int ey)
{

	//Mat tmpImg=(*img).clone();
	//ofstream out("D:\\Fuhao\\exp results\\Final\\mode process\\modes.txt",ios::out);
//
	//ofstream out1("D:\\Fuhao\\exp results\\Final\\mode process\\prob_map.txt",ios::out);

	for (int i=0;i<Ind.size();i++)
	{
		candidatePoints[i].clear();

		if (i!=18)
		{
			continue;
		}

	/*	if (maxProb[Ind[i]]<0.35)
		{
			candidatePoints[i].push_back(Point(samplePos[Ind[i]][0],samplePos[Ind[i]][1]));
			continue;
		}*/
		int cIndex=Ind[i];

		/*if (maxProb[cIndex]<0.1)
		{
			candidatePoints[i].push_back(Point(samplePos[cIndex][0],samplePos[cIndex][1]));
			continue;
		}*/

		Mat tmpProb=Mat::zeros(height,width,CV_64FC1);
		Mat usageMap=Mat::zeros(height,width,CV_64FC1);

		//set up the prob image and map
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				float cProb=probMap[j*width+k+cIndex*width*height];
				//if (cProb>=0.7*maxProb[cIndex])
				{
					tmpProb.at<double>(j,k)=cProb;
					usageMap.at<double>(j,k)=1;
				}
				//continue;
			}
		}

		Mat tmp=tmpProb.clone()*0;
		int windowSize=8;
		for (int j=windowSize/2+1;j<height-windowSize/2-1;j++)
		{
			for (int k=windowSize/2+1;k<width-windowSize/2-1;k++)
			{
				if (j<sy||j>ey||k<sx||k>ex)
				{
					continue;
				}

				bool ok=true;
				float curValue=tmpProb.at<double>(j,k);
				for (int m=-windowSize/2;m<=windowSize/2;m++)
				{
					for (int n=-windowSize/2;n<=windowSize/2;n++)
					{
						if (tmpProb.at<double>(j+m,k+n)>curValue)
						{
							ok=false;
							break;
						}
					}
				}

				if (ok&&curValue>0)
				{
					cout<<"localMaxima: "<<j<<" "<<k<<endl;
					tmp.at<double>(j,k)=255;
				}
			}
		}
		namedWindow("localMaxima");
		imshow("localMaxima",tmp);
		
		waitKey();
	/*	ofstream out("D:\\Fuhao\\cpu gpu validation\\mapExp.txt",ios::out);
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				float cProb=probMap[j*width+k+cIndex*width*height];
				out<<cProb<<" ";
			}
			out<<endl;
		}
		out.close();*/
		//namedWindow("prob");
		//imshow("prob",tmpProb);
		//cout<<i<<" "<<cIndex<<endl;
		//waitKey();




	}
	//out.close();
	//	out1.close();
}


void AAM_Detection_Combination::findSecondModes_Meanshift(int **samplePos,int width,int height, vector<int>& Ind,float *probMap,float *maxProb,Mat *img,int sx,int ex,int sy,int ey)
{

	//Mat tmpImg=(*img).clone();
	//ofstream out("D:\\Fuhao\\exp results\\Final\\mode process\\modes.txt",ios::out);
//
	//ofstream out1("D:\\Fuhao\\exp results\\Final\\mode process\\prob_map.txt",ios::out);

	for (int i=0;i<Ind.size();i++)
	{
		candidatePoints[i].clear();

		

	/*	if (maxProb[Ind[i]]<0.35)
		{
			candidatePoints[i].push_back(Point(samplePos[Ind[i]][0],samplePos[Ind[i]][1]));
			continue;
		}*/
		int cIndex=Ind[i];

		/*if (maxProb[cIndex]<0.1)
		{
			candidatePoints[i].push_back(Point(samplePos[cIndex][0],samplePos[cIndex][1]));
			continue;
		}*/

		Mat tmpProb=Mat::zeros(height,width,CV_64FC1);
		Mat usageMap=Mat::zeros(height,width,CV_64FC1);

		//set up the prob image and map
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				float cProb=probMap[j*width+k+cIndex*width*height];
				if (cProb>=0.7*maxProb[cIndex])
				{
					tmpProb.at<double>(j,k)=cProb;
					usageMap.at<double>(j,k)=1;
				}
				//continue;
			}
		}

	/*	if (i==0)
		{
			namedWindow("1");
			imshow("1",tmpProb);
			waitKey();
		}*/

		//if (Ind[i]==14)
		//{
		//	for (int j=0;j<height;j++)
		//	{
		//		for (int k=0;k<width;k++)
		//		{
		//			//if (j>=sy&&j<=ey&&k>=sx&&k<=ex)
		//			if(probMap[j*width+k+cIndex*width*height]>0)
		//			{
		//				out1<<k<<" "<<j<<" "<<probMap[j*width+k+cIndex*width*height]<<endl;
		//			}
		//			//float cProb=probMap[j*width+k+cIndex*width*height];
		//		
		//			//continue;
		//		}
		//	}
		//	
		//}

		//kill the connected region
		vector<Point2f> totalPointList;
		vector<Point2f> pointList;
		Point2f c,c_cur;
		c.x=samplePos[cIndex][0];
		c.y=samplePos[cIndex][1];
		//cout<<cIndex<<" "<<c.x<<" "<<c.y<<endl;
		pointList.push_back(c);
		totalPointList.clear();
		while (pointList.size()!=0)
		{
			c_cur=pointList[pointList.size()-1];

			//candidatePoints[i].push_back(c_cur);
			pointList.pop_back();
			//usageMap.at<double>(c_cur.y,c_cur.x)=0;
			//cout<<tmpProb.at<double>(c_cur.y,c_cur.x)<<endl;
			tmpProb.at<double>(c_cur.y,c_cur.x)=0;
			totalPointList.push_back(c_cur);
			//then check the 8 neighbors
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x-1)>0)
			{
				pointList.push_back(Point2f(c_cur.x-1,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x)>0)
			{
				pointList.push_back(Point2f(c_cur.x,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x+1)>0)
			{
				pointList.push_back(Point2f(c_cur.x+1,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y,c_cur.x-1)>0)
			{
				pointList.push_back(Point2f(c_cur.x-1,c_cur.y));
			}
			if (tmpProb.at<double>(c_cur.y,c_cur.x+1)>0)
			{
				pointList.push_back(Point2f(c_cur.x+1,c_cur.y));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x-1)>0)
			{
				pointList.push_back(Point2f(c_cur.x-1,c_cur.y+1));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x)>0)
			{
				pointList.push_back(Point2f(c_cur.x,c_cur.y+1));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x+1)>0)
			{
				pointList.push_back(Point2f(c_cur.x+1,c_cur.y+1));
			}

		}
		//calculate the estimation
		double expectation[]={0,0};
		double sumProb=0;
		for (int l=0;l<totalPointList.size();l++)
		{
			c_cur=totalPointList[l];
			//cout<<c_cur.y+c_cur.x;
			expectation[0]+=c_cur.x*probMap[(int)(c_cur.y*width+c_cur.x+cIndex*width*height)];
			expectation[1]+=c_cur.y*probMap[(int)(c_cur.y*width+c_cur.x+cIndex*width*height)];
			sumProb+=probMap[(int)(c_cur.y*width+c_cur.x+cIndex*width*height)];
			//cout<<cIndex<<" "<<l<<" "<<probMap[c_cur.y*width+c_cur.x+cIndex*width*height]<<" ";
		}
		expectation[0]/=sumProb;
		expectation[1]/=sumProb;

		//if ((expectation[0]-samplePos[cIndex][0])*(expectation[0]-samplePos[cIndex][0])+
		//(expectation[1]-samplePos[cIndex][1])*(expectation[1]-samplePos[cIndex][1])>9)
		{
			candidatePoints[i].push_back(Point2f(expectation[0],expectation[1]));
			//cout<<sumProb<<" "<<expectation[0]<<" "<<expectation[1]<<endl;
		}
		
		/*if (Ind[i]==14)
		{
			for (int l=0;l<totalPointList.size();l++)
			{
				out<<candidatePoints[i].size()<<" "<<totalPointList[l].x<<" "<<totalPointList[l].y<<endl;
			}
		}*/

		//candidatePoints[i].push_back(c);
	/*	if (i==12)
		{
			namedWindow("1");
			imshow("1",tmpProb);
			waitKey();
		}*/

		totalPointList.clear();
		//then check the remaining region, only remain region which size>3
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				if (tmpProb.at<double>(j,k)>0)
				{
					//check the connected region
					pointList.clear();
					totalPointList.clear();
					pointList.push_back(Point2f(k,j));
					while (pointList.size()!=0)
					{
						c_cur=pointList[pointList.size()-1];
						totalPointList.push_back(c_cur);
						pointList.pop_back();
						//usageMap.at<double>(c_cur.y,c_cur.x)=0;
						//cout<<tmpProb.at<double>(c_cur.y,c_cur.x)<<endl;
						tmpProb.at<double>(c_cur.y,c_cur.x)=0;

						//then check the 8 neighbors
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x-1)>0)
						{
							pointList.push_back(Point2f(c_cur.x-1,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x)>0)
						{
							pointList.push_back(Point2f(c_cur.x,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x+1)>0)
						{
							pointList.push_back(Point2f(c_cur.x+1,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y,c_cur.x-1)>0)
						{
							pointList.push_back(Point2f(c_cur.x-1,c_cur.y));
						}
						if (tmpProb.at<double>(c_cur.y,c_cur.x+1)>0)
						{
							pointList.push_back(Point2f(c_cur.x+1,c_cur.y));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x-1)>0)
						{
							pointList.push_back(Point2f(c_cur.x-1,c_cur.y+1));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x)>0)
						{
							pointList.push_back(Point2f(c_cur.x,c_cur.y+1));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x+1)>0)
						{
							pointList.push_back(Point2f(c_cur.x+1,c_cur.y+1));
						}

					}

					if (totalPointList.size()>3)//||(i==12&&totalPointList.size()>1))
					{
						double expectation[]={0,0};
						double sumProb=0;
						for (int l=0;l<totalPointList.size();l++)
						{
							c_cur=totalPointList[l];
							expectation[0]+=c_cur.x*probMap[(int)c_cur.y*width+(int)c_cur.x+cIndex*width*height];
							expectation[1]+=c_cur.y*probMap[(int)c_cur.y*width+(int)c_cur.x+cIndex*width*height];
							sumProb+=probMap[(int)c_cur.y*width+(int)c_cur.x+cIndex*width*height];
						}
						expectation[0]/=sumProb;
						expectation[1]/=sumProb;
						//if ((expectation[0]-samplePos[cIndex][0])*(expectation[0]-samplePos[cIndex][0])+
							//(expectation[1]-samplePos[cIndex][1])*(expectation[1]-samplePos[cIndex][1])>9)
						if(expectation[0]>=sx&&expectation[0]<=ex&&expectation[1]>=sy&&expectation[1]<=ey)
						//if(expectation[0]>50&&expectation[0]<590&&expectation[1]>50&&expectation[1]<430)
						{
							candidatePoints[i].push_back(Point2f(expectation[0],expectation[1]));

							/*if (Ind[i]==14)
							{
								for (int l=0;l<totalPointList.size();l++)
								{
									out<<candidatePoints[i].size()<<" "<<totalPointList[l].x<<" "<<totalPointList[l].y<<endl;
								}
							}*/
						}

					
					}
					else
					{
						/*for (int l=0;l<totalPointList.size();l++)
						{
							c_cur=totalPointList[l];
							tmpProb.at<double>(c_cur.y,c_cur.x)=0;
						}*/
					}

					/*if (i==12)
					{
						cout<<candidatePoints[i].size()<<endl;
						namedWindow("1");
						imshow("1",tmpProb);
						waitKey();
					}*/

				}
			}
		}
		
	
	/*	if (img!=NULL&&(combineIndList[cIndex]==42||combineIndList[cIndex]==48))
		{
			cout<<combineIndList[cIndex]<<" "<<candidatePoints[i].size()<<endl;
			Mat tmpImg=(*img).clone();
			for (int l=0;l<candidatePoints[i].size();l++)
			{
				circle(tmpImg,candidatePoints[i][l],2,255);
			}
			namedWindow("ProbMap");
			imshow("ProbMap",tmpImg);
			waitKey();
		}*/
		
		/*if (i==12)
		{
			cout<<"size "<<i<<" "<<candidatePoints[i].size()<<endl;
		}*/
	}
	//return;
	//using meanshift to refine the position

	/*LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);*/
	for (int i=0;i<Ind.size();i++)
	{
		//cout<<i<<endl;

		int cind=Ind[i];
		Mat tmpProb=Mat::zeros(height,width,CV_64FC1);

		//set up the prob image and map
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				float cProb=probMap[j*width+k+cind*width*height];

				tmpProb.at<double>(j,k)=cProb;
			}
		}


		for (int j=0;j<candidatePoints[cind].size();j++)
		{
			//meanshift
			bool isstop=false;
			//float lastP[2];
			float curP[2];
			
			curP[0]=candidatePoints[cind][j].x;
			curP[1]=candidatePoints[cind][j].y;

			float tmpP[2]; float sumProb;float curDis;float ttProb;

			int localWindowSize=20;
			float sigma=3;
			
			int times=0;
			while(!isstop)
			{
				tmpP[0]=tmpP[1]=0;
				sumProb=0;
				for (int m=curP[1]-localWindowSize/2;m<=curP[1]+localWindowSize/2;m++)
				{
					for (int n=curP[0]-localWindowSize/2;n<=curP[0]+localWindowSize/2;n++)
					{
						if (tmpProb.at<double>(m,n)>0)
						{
							curDis=((float)m-curP[1])*((float)m-curP[1])+
								((float)n-curP[0])*((float)n-curP[0]);
							ttProb=powf(e,-curDis/(2*sigma*sigma))*tmpProb.at<double>(m,n);
							tmpP[0]+=ttProb*(float)n;
							tmpP[1]+=ttProb*(float)m;
							sumProb+=ttProb;
						}
						
					}
				}
				tmpP[0]/=sumProb;tmpP[1]/=sumProb;

				float distance=(tmpP[0]-curP[0])*(tmpP[0]-curP[0])+(tmpP[1]-curP[1])*(tmpP[1]-curP[1]);
				//cout<<distance<<endl;
				if (distance<0.00001)
				{
					
					isstop=true;
				}
				else
				{
					curP[0]=tmpP[0];curP[1]=tmpP[1];
				}

				times++;
			}

			candidatePoints[cind][j].x=curP[0];
			candidatePoints[cind][j].y=curP[1];
			if (times>50)
			{
				cout<<i<<" "<<times<<endl;

			}
			
			////Mat tmpImg=tmpProb.clone();
			//Mat tmpImg=(*img).clone();
			//circle(tmpImg,candidatePoints[cind][j],5,255);
			//circle(tmpImg,Point(curP[0],curP[1]),2,255);

			//namedWindow("MeanShift");
			//imshow("MeanShift",tmpImg);
			//waitKey();
		
		}		
	}
//	QueryPerformanceCounter((LARGE_INTEGER   *)&t2);
//	double   time=(t2-t1)*1000/persecond; 
//	cout<<"meanshift time: "<<time<<"ms "<<endl;
	//update the first candidate
	for (int i=0;i<fullIndNum;i++)
	{
		samplePos[i][0]=candidatePoints[i][0].x;
		samplePos[i][1]=candidatePoints[i][0].y;
	}

		//out.close();
	//	out1.close();
}


void AAM_Detection_Combination::findSecondModes(int **samplePos,int width,int height, vector<int>& Ind,float *probMap,float *maxProb,Mat *img,int sx,int ex,int sy,int ey)
{

	//Mat tmpImg=(*img).clone();
	//ofstream out("D:\\Fuhao\\exp results\\Final\\mode process\\modes.txt",ios::out);
//
	//ofstream out1("D:\\Fuhao\\exp results\\Final\\mode process\\prob_map.txt",ios::out);

	for (int i=0;i<Ind.size();i++)
	{
		candidatePoints[i].clear();

	/*	if (maxProb[Ind[i]]<0.35)
		{
			candidatePoints[i].push_back(Point(samplePos[Ind[i]][0],samplePos[Ind[i]][1]));
			continue;
		}*/
		int cIndex=Ind[i];

		Mat tmpProb=Mat::zeros(height,width,CV_64FC1);
		Mat usageMap=Mat::zeros(height,width,CV_64FC1);

		//set up the prob image and map
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				float cProb=probMap[j*width+k+cIndex*width*height];
				if (cProb>=0.7*maxProb[cIndex])
				{
					tmpProb.at<double>(j,k)=cProb;
					usageMap.at<double>(j,k)=1;
				}
				//continue;
			}
		}

	/*	if (i==0)
		{
			namedWindow("1");
			imshow("1",tmpProb);
			waitKey();
		}*/

		//if (Ind[i]==14)
		//{
		//	for (int j=0;j<height;j++)
		//	{
		//		for (int k=0;k<width;k++)
		//		{
		//			//if (j>=sy&&j<=ey&&k>=sx&&k<=ex)
		//			if(probMap[j*width+k+cIndex*width*height]>0)
		//			{
		//				out1<<k<<" "<<j<<" "<<probMap[j*width+k+cIndex*width*height]<<endl;
		//			}
		//			//float cProb=probMap[j*width+k+cIndex*width*height];
		//		
		//			//continue;
		//		}
		//	}
		//	
		//}

		//kill the connected region
		vector<Point> totalPointList;
		vector<Point> pointList;
		Point c,c_cur;
		c.x=samplePos[cIndex][0];
		c.y=samplePos[cIndex][1];
		//cout<<cIndex<<" "<<c.x<<" "<<c.y<<endl;
		pointList.push_back(c);
		totalPointList.clear();
		while (pointList.size()!=0)
		{
			c_cur=pointList[pointList.size()-1];

			//candidatePoints[i].push_back(c_cur);
			pointList.pop_back();
			//usageMap.at<double>(c_cur.y,c_cur.x)=0;
			//cout<<tmpProb.at<double>(c_cur.y,c_cur.x)<<endl;
			tmpProb.at<double>(c_cur.y,c_cur.x)=0;
			totalPointList.push_back(c_cur);
			//then check the 8 neighbors
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x-1)>0)
			{
				pointList.push_back(Point(c_cur.x-1,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x)>0)
			{
				pointList.push_back(Point(c_cur.x,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y-1,c_cur.x+1)>0)
			{
				pointList.push_back(Point(c_cur.x+1,c_cur.y-1));
			}
			if (tmpProb.at<double>(c_cur.y,c_cur.x-1)>0)
			{
				pointList.push_back(Point(c_cur.x-1,c_cur.y));
			}
			if (tmpProb.at<double>(c_cur.y,c_cur.x+1)>0)
			{
				pointList.push_back(Point(c_cur.x+1,c_cur.y));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x-1)>0)
			{
				pointList.push_back(Point(c_cur.x-1,c_cur.y+1));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x)>0)
			{
				pointList.push_back(Point(c_cur.x,c_cur.y+1));
			}
			if (tmpProb.at<double>(c_cur.y+1,c_cur.x+1)>0)
			{
				pointList.push_back(Point(c_cur.x+1,c_cur.y+1));
			}

		}
		//calculate the estimation
		double expectation[]={0,0};
		double sumProb=0;
		for (int l=0;l<totalPointList.size();l++)
		{
			c_cur=totalPointList[l];
			expectation[0]+=c_cur.x*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
			expectation[1]+=c_cur.y*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
			sumProb+=probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
			//cout<<cIndex<<" "<<l<<" "<<probMap[c_cur.y*width+c_cur.x+cIndex*width*height]<<" ";
		}
		expectation[0]/=sumProb;
		expectation[1]/=sumProb;

		//if ((expectation[0]-samplePos[cIndex][0])*(expectation[0]-samplePos[cIndex][0])+
		//(expectation[1]-samplePos[cIndex][1])*(expectation[1]-samplePos[cIndex][1])>9)
		{
			candidatePoints[i].push_back(Point(expectation[0],expectation[1]));
			//cout<<sumProb<<" "<<expectation[0]<<" "<<expectation[1]<<endl;
		}
		
		/*if (Ind[i]==14)
		{
			for (int l=0;l<totalPointList.size();l++)
			{
				out<<candidatePoints[i].size()<<" "<<totalPointList[l].x<<" "<<totalPointList[l].y<<endl;
			}
		}*/

		//candidatePoints[i].push_back(c);
	/*	if (i==12)
		{
			namedWindow("1");
			imshow("1",tmpProb);
			waitKey();
		}*/

		totalPointList.clear();
		//then check the remaining region, only remain region which size>3
		for (int j=0;j<height;j++)
		{
			for (int k=0;k<width;k++)
			{
				if (tmpProb.at<double>(j,k)>0)
				{
					//check the connected region
					pointList.clear();
					totalPointList.clear();
					pointList.push_back(Point(k,j));
					while (pointList.size()!=0)
					{
						c_cur=pointList[pointList.size()-1];
						totalPointList.push_back(c_cur);
						pointList.pop_back();
						//usageMap.at<double>(c_cur.y,c_cur.x)=0;
						//cout<<tmpProb.at<double>(c_cur.y,c_cur.x)<<endl;
						tmpProb.at<double>(c_cur.y,c_cur.x)=0;

						//then check the 8 neighbors
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x-1)>0)
						{
							pointList.push_back(Point(c_cur.x-1,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x)>0)
						{
							pointList.push_back(Point(c_cur.x,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y-1,c_cur.x+1)>0)
						{
							pointList.push_back(Point(c_cur.x+1,c_cur.y-1));
						}
						if (tmpProb.at<double>(c_cur.y,c_cur.x-1)>0)
						{
							pointList.push_back(Point(c_cur.x-1,c_cur.y));
						}
						if (tmpProb.at<double>(c_cur.y,c_cur.x+1)>0)
						{
							pointList.push_back(Point(c_cur.x+1,c_cur.y));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x-1)>0)
						{
							pointList.push_back(Point(c_cur.x-1,c_cur.y+1));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x)>0)
						{
							pointList.push_back(Point(c_cur.x,c_cur.y+1));
						}
						if (tmpProb.at<double>(c_cur.y+1,c_cur.x+1)>0)
						{
							pointList.push_back(Point(c_cur.x+1,c_cur.y+1));
						}

					}

					if (totalPointList.size()>3)//||(i==12&&totalPointList.size()>1))
					{
						double expectation[]={0,0};
						double sumProb=0;
						for (int l=0;l<totalPointList.size();l++)
						{
							c_cur=totalPointList[l];
							expectation[0]+=c_cur.x*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
							expectation[1]+=c_cur.y*probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
							sumProb+=probMap[c_cur.y*width+c_cur.x+cIndex*width*height];
						}
						expectation[0]/=sumProb;
						expectation[1]/=sumProb;
						//if ((expectation[0]-samplePos[cIndex][0])*(expectation[0]-samplePos[cIndex][0])+
							//(expectation[1]-samplePos[cIndex][1])*(expectation[1]-samplePos[cIndex][1])>9)
						if(expectation[0]>=sx&&expectation[0]<=ex&&expectation[1]>=sy&&expectation[1]<=ey)
						//if(expectation[0]>50&&expectation[0]<590&&expectation[1]>50&&expectation[1]<430)
						{
							candidatePoints[i].push_back(Point(expectation[0],expectation[1]));

							/*if (Ind[i]==14)
							{
								for (int l=0;l<totalPointList.size();l++)
								{
									out<<candidatePoints[i].size()<<" "<<totalPointList[l].x<<" "<<totalPointList[l].y<<endl;
								}
							}*/
						}

					
					}
					else
					{
						/*for (int l=0;l<totalPointList.size();l++)
						{
							c_cur=totalPointList[l];
							tmpProb.at<double>(c_cur.y,c_cur.x)=0;
						}*/
					}

					/*if (i==12)
					{
						cout<<candidatePoints[i].size()<<endl;
						namedWindow("1");
						imshow("1",tmpProb);
						waitKey();
					}*/

				}
			}
		}
		
	
	/*	if (img!=NULL&&(combineIndList[cIndex]==42||combineIndList[cIndex]==48))
		{
			cout<<combineIndList[cIndex]<<" "<<candidatePoints[i].size()<<endl;
			Mat tmpImg=(*img).clone();
			for (int l=0;l<candidatePoints[i].size();l++)
			{
				circle(tmpImg,candidatePoints[i][l],2,255);
			}
			namedWindow("ProbMap");
			imshow("ProbMap",tmpImg);
			waitKey();
		}*/
		
		/*if (i==12)
		{
			cout<<"size "<<i<<" "<<candidatePoints[i].size()<<endl;
		}*/
	}
		//out.close();
	//	out1.close();
}

bool AAM_Detection_Combination::track_combine(Mat &colorImg,Mat &depthImg,int &status,int sx,int ex,int sy,int ey,bool initialPara)
{

	//step 2: detection
	//GTB("S");
	//for (int ll=0;ll<200;ll++)
	//{

	status=0;

	int width=depthImg.cols;
	int height=depthImg.rows;

	

	//Step 1: transfer the data to gpu
	//cout << "transferring data to gpu ..." << endl;
	#pragma omp parallel for
	for (int i=0;i<depthImg.rows;i++)
	{
		for (int j=0;j<depthImg.cols;j++)
		{
			host_depthImage_global[i*depthImg.cols+j]=depthImg.at<float>(i,j);
			host_colorImage_global[i*depthImg.cols+j]=colorImg.at<uchar>(i,j);
		}
	}
	setData_onrun_shared(host_colorImage_global,host_depthImage_global,width, height);

	


	/*Mat tmpImg;
	colorImg.convertTo(tmpImg,CV_32FC1);
	host_colorImage=(float *)tmpImg.data;
	host_depthImage=(float *)depthImg.data;	
	setData_onrun_shared(host_colorImage,host_depthImage,width, height);*/

	/*for (int i=0;i<10;i++)
	{
		cout<<host_colorImage[i]<<" ";
	}
	cout<<endl;*/

	bool bigChange=false;
	vector<int> finalInd;

	//if (state!=1||(state==1&&!hasVelocity))
		{
		//GTB("A");
			bool enoughVisibleNum;

			float curTheta=lastTheta;
			if (hasVelocity)
			{
				curTheta=lastTheta+thetaVeolosity;
				//curTheta=0;
				//cout<<"thetaV: "<<thetaVeolosity<<endl;
			}

			//not rotate if larger than 30, for demonstration only
		/*	if (curTheta>30.0f/180.0f*3.1415926f)
			{
				curTheta=30.0f/180.0f*3.1415926f;
			}
			else if (curTheta<-30.0f/180.0f*3.1415926f)
			{
				curTheta=-30.0f/180.0f*3.1415926f;
			}*/


			/*if (curTheta>15.0f/180.0f*3.1415926f)
			{
				curTheta=30.0f/180.0f*3.1415926f;
			}*/
			if (!initialPara)
			{
				enoughVisibleNum=predict_GPU_separated_combination(depthImg.cols,depthImg.rows,hostDetectionResult,finalPos,maximumProb,sx,ex,sy,ey,initialPara&&TemporalTracking,showProbMap);
			}
			else
			{
				enoughVisibleNum=predict_GPU_separated_combination(depthImg.cols,depthImg.rows,hostDetectionResult,finalPos,maximumProb,sx,ex,sy,ey,initialPara&&TemporalTracking,showProbMap,curTheta);
			}

		
			if (AAM_exp->showSingleStep)
			{
				int usedNum=fullIndNum;
				for (int i=0;i<usedNum;i++)
				{
					/*if (i<4)
					{
					continue;
					}
					if (i>17)
						continue;*/
					finalPos[i][0]=currentGT[combineIndList[i]];
					finalPos[i][1]=currentGT[combineIndList[i]+ptsNum];
				}
			}

		if (showProbMap)
		{
			//probMap[j*width+k+cIndex*width*height];
			//visualize the probMap
			Mat curProbMap=depthImg.clone()*0;
			//	cvtColor(curProbMap,curProbMap,CV_GRAY2BGR);
			int w_size=5;
			for (int i=0;i<fullIndNum;i++)
			{
				for (int j=finalPos[i][0]-w_size;j<finalPos[i][0]+w_size;j++)
				{
					for (int k=finalPos[i][1]-w_size;k<finalPos[i][1]+w_size;k++)
					{
						curProbMap.at<float>(k,j)=hostDetectionResult[k*width+j+i*width*height];
					}

				}

				//circle(curProbMap,Point(finalPos[i][0],finalPos[i][1]),3,Scalar(1));
			}
			namedWindow("probMap",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
			imshow("probMap",curProbMap(Range(153,289),Range(640-348,640-205)));
			waitKey(1);
			//return false;
		}



		/*GTE("A");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"used time per iteration: "<<time<<endl;*/

		/*if (!enoughVisibleNum)
		{
			return false;
		}*/

		//check the probability map
		/*{
		checkProbMap(hostDetectionResult);
		return false;
		}*/

		/*if (state==2)
		{
		for (int i=0;i<ptsNum*2;i++)
		{
		currentShape[i]=0;
		}
		for (int i=0;i<fullIndNum;i++)
		{
		currentShape[combineIndList[i]]=finalPos[i][0];
		currentShape[combineIndList[i]+ptsNum]=finalPos[i][1];
		}
		return true;
		}*/

		bigChange=true;
		if (state==1)
		{
			int okayNum=0;
			for (int i=0;i<fullIndNum;i++)
			{
				if (abs(lastPts[i]-finalPos[i][0])<2&&abs(lastPts[i+fullIndNum]-finalPos[i][1])<2)
				{
					okayNum++;
				}

			}
			//cout<<okayNum<<endl;
			if (okayNum>12)
			{
				bigChange=false;
			}
			if (!(abs(lastPts[18]-finalPos[18][0])<2&&abs(lastPts[18+fullIndNum]-finalPos[18][1])<2))
				bigChange=true;
			if (!(abs(lastPts[15]-finalPos[15][0])<2&&abs(lastPts[15+fullIndNum]-finalPos[15][1])<2))
				bigChange=true;

			bigChange=false;
			//else
			//	cout<<abs(lastPts[18]-finalPos[18][0])<<" "<<abs(lastPts[18+fullIndNum]-finalPos[18][1])<<endl;
			/*if (bigChange)
			{
			cout<<rand()<<" big change\n";
			}*/
		}

		//then, KNN. Currently, on CPU
		
		//ransac_noSampling(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb);
		//tmpImg=colorImg.clone();
		//ransac_noSampling_parrllel(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb,&colorImg);

		vector<int> totalInd;
		for (int i=0;i<fullIndNum;i++)
		{
			totalInd.push_back(i);
			candidatePoints[i].clear();
			/*if (hasVelocity)
			{

			if (abs(pridictedPts[i]-finalPos[i][0])<10&&abs(pridictedPts[i+fullIndNum]-finalPos[i][1])<10)
			{
			candidatePoints[i].push_back(Point2f(pridictedPts[i],pridictedPts[i+fullIndNum]));
			}

			}*/
			/*if (state==1&&!bigChange)
			{
			candidatePoints[i].push_back(Point2f(lastPts[i],lastPts[i+fullIndNum]));
			}
			else*/
			if(AAM_exp->showSingleStep&&0)
			{
				if(i>=8&&i<=11)
					candidatePoints[i].push_back(Point2f(finalPos[i][0],finalPos[i][1]-2));
				else
					candidatePoints[i].push_back(Point2f(finalPos[i][0],finalPos[i][1]));
			}
			else
			{
				candidatePoints[i].push_back(Point2f(finalPos[i][0],finalPos[i][1]));
			}
		

			if (initialPara&&state!=2)
			{

				//if (abs(pridictedPts[i]-finalPos[i][0])<10&&abs(pridictedPts[i+fullIndNum]-finalPos[i][1])<10)
				/*if((combineIndList[i]<42||combineIndList[i]>63)&&(i!=1&&i!=3&&i!=6&&i!=10&&i!=12&&i!=14))*/
				//if((combineIndList[i]<42||combineIndList[i]>63))
				{
				//	candidatePoints[i].push_back(Point2f(lastPts[i],lastPts[i+fullIndNum]));
				}

			}
		}


			if (AAM_exp->showSingleStep)
			{
				//if (AAM_exp->showSingleStep)
				{
					cout<<"last Theta: "<<lastTheta/3.1415926f*180.0f<<" "<<curTheta/3.1415926f*180.0f<<endl;
				}
				//for (int i=0;i<fullIndNum;i++)
				//{
				//	/*if(i>=4&&i<=11)
				//		continue;*/
				//	/*if(i==13||i==15||i==16||i==17)
				//		continue;*/
				//	finalPos[i][0]=currentGT[combineIndList[i]];
				//	finalPos[i][1]=currentGT[combineIndList[i]+ptsNum];
				//}
				Mat tmp=colorImgBackUP.clone();

				for (int i=0;i<fullIndNum;i++)
				{
					for(int j=0;j<candidatePoints[i].size();j++)
						//circle(tmp,Point(finalPos[i][0],finalPos[i][1]),2,Scalar(255));
						circle(tmp,candidatePoints[i][j],2,Scalar(255));
				}
				imshow("modes",tmp);
				waitKey();
			}


		//	findSecondModes(finalPos,depthImg.cols,depthImg.rows,totalInd,hostDetectionResult,maximumProb,&colorImg,startX,endX,startY,endY);

		/*LONGLONG   t1,t2; 
		LONGLONG   persecond; 
		QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
		QueryPerformanceCounter((LARGE_INTEGER   *)&t1);*/
		//cout<<"mean Shift\n";
		//	findSecondModes_Meanshift(finalPos,depthImg.cols,depthImg.rows,totalInd,hostDetectionResult,maximumProb,&colorImg,startX,endX,startY,endY);

		/*{
		for (int i=0;i<ptsNum*2;i++)
		{
		currentShape[i]=0;
		}
		for (int i=0;i<fullIndNum;i++)
		{
		currentShape[combineIndList[i]]=finalPos[i][0];
		currentShape[combineIndList[i]+ptsNum]=finalPos[i][1];
		}
		}
		return false;*/

		/*for (int i=0;i<fullIndNum;i++)
		{
		for (int j=0;j<candidatePoints[i].size();j++)
		{
		cout<<i<<" "<<candidatePoints[i][j].x<<" "<<candidatePoints[i][j].y<<endl;
		}

		}*/

		//return true;

		//LONGLONG   t1,t2; 
		//LONGLONG   persecond; 
		//double time;
		//
		//QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
		//
		//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 

		//2ms

		bool isS;
		isS=geoHashing_Candidates_nearestInlier(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb,"",NULL,NULL,candidatePoints);
		if (!isS)
		{
			cout<<"no nn found!\n";
			return false;
			if(initialPara)
				bigChange=false;
		}

		if (state==2)
		{
			for (int i=0;i<ptsNum*2;i++)
			{
				currentDetection[i]=0;
			}
			for (int i=0;i<finalInd.size();i++)
			{
				currentDetection[combineIndList[finalInd[i]]]=finalPos[finalInd[i]][0];
				currentDetection[combineIndList[finalInd[i]]+ptsNum]=finalPos[finalInd[i]][1];
				/*currentDetection[combineIndList[finalInd[i]]]=candidatePoints[finalInd[i]][0].x;
				currentDetection[combineIndList[finalInd[i]]+ptsNum]=candidatePoints[finalInd[i]][0].y;*/
			}
		}

		if(isS)
		{

			for (int i=0;i<finalInd.size();i++)
			{
				absInd_Global[i]=combineIndList[finalInd[i]];
			}
			setupConv_featurePts(colorImg.cols,colorImg.rows,20,finalPos,finalInd,absInd_Global);


			//finally, AAM with detection on GPU, 0.7ms
			int minInd=bestFitTrainingSampleInd;
			globalTransformation=globalTransformation_optimal;
			for (int i=0;i<AAM_exp->shape_dim;i++)
			{
				AAM_exp->s_weight[i]=eigenVectors.at<double>(i,minInd);
			}


			float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
				globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
			float theta=-atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));

			AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
			AAM_exp->setInitialScale(scale);
			AAM_exp->setInitialTheta(theta);
		}


	}


	////////output parameters. all checked
	//if (AAM_exp->showSingleStep)
	//{
	//	cout<<"nearest Ind: "<<bestFitTrainingSampleInd<<endl;
	//	cout<<"visibleInd:\n";

	//	for (int i=0;i<finalInd.size();i++)
	//	{
	//		cout<<finalInd[i]<<" ";
	//	}
	//	cout<<endl;

	//	/*cout<<"\ndetected pts:\n";
	//	for (int i=0;i<fullIndNum;i++)
	//	{
	//		cout<<finalPos[i][0]<<" "<<finalPos[i][1]<<endl;
	//	}
	//	cout<<"\n shape parameters\n";
	//	for (int i=0;i<AAM_exp->shape_dim;i++)
	//	{
	//		cout<<AAM_exp->s_weight[i]<<" ";
	//	}
	//	cout<<endl;*/
	//}

	
	//else
	//{
	//	for (int i=0;i<AAM_exp->shape_dim;i++)
	//	{
	//		AAM_exp->s_weight[i]=0;
	//	}
	//}
		
	
	


	//1 3 6 10 12 14
	//for (int i=0;i<sizeof(specialInd)/sizeof(int);i++)
	//{
	//	if (maximumProb[specialInd[i]]<0.05)
	//	{
	//		finalInd.erase(finalInd.begin()+specialInd[i]);
	//	}
	//}
	




	//bigChange
	//if (state==1&&(!hasVelocity||(abs(veolocity[18])<2)))
	if (state==1&&!bigChange)
	{
	//	cout<<"should be one: "<<initialPara<<endl;
		AAM_exp->calculateData_onrun_AAM_combination(colorImg,finalInd,initialPara);
	}
	else// if (state==0)
	{
		//cout<<"single frame\n";
		AAM_exp->calculateData_onrun_AAM_combination(colorImg,finalInd,false);
	}

	//step .2: set up covariances matrix for each pixel
	//setupConv(colorImg.cols,colorImg.rows,20,finalPos,finalInd,host_preCalculatedConv);
	//float *absInd=new float[MAX_LABEL_NUMBER];


	


	//step .3: optimization on GPU
	//status=iterate_combination(colorImg.cols,colorImg.rows,0,0,currentShape,isAAMOnly&&!bigChange,showNN);

	lastlastTheta=lastTheta;
	status=iterate_combination(colorImg.cols,colorImg.rows,0,0,lastTheta,currentShape, currentShapePtsNum, isAAMOnly,showNN);

	////obtain the theta using eye angle
	//Point2f lEye=Point2f((currentShape[18]+currentShape[22])/2.0f,(currentShape[18+ptsNum]+currentShape[22+ptsNum])/2.0f);
	//Point2f rEye=Point2f((currentShape[26]+currentShape[30])/2.0f,(currentShape[26+ptsNum]+currentShape[30+ptsNum])/2.0f);
	//Point2f e_dist=rEye-lEye;
	//e_dist.x=e_dist.x/norm(e_dist);
	//e_dist.y=e_dist.y/norm(e_dist);

	//float eyeAngle=acosf(e_dist.x);
	//cout<<"angle from model: "<<lastTheta/3.1415926f*180.0f<<" eyeAngle: "<<eyeAngle/3.1415926f*180.0f<<endl;
	////obtain the theta using meanShape
	//vector<int> fullVisibleInd;
	//for (int i=0;i<fullIndNum;i++)
	//{
	//	fullVisibleInd.push_back(i);
	//}
	//for (int i=0;i<fullIndNum;i++)
	//{
	//	finalPos[i][0]=currentShape[combineIndList[i]];
	//	finalPos[i][1]=currentShape[combineIndList[i]+ptsNum];
	//}
	//
	//lastTheta=-atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	//cout<<"Theta using PCA model: "<<lastTheta/3.1415926f*180.0f<<endl;


	//getTransformationInfo(fullVisibleInd,finalPos,shapes,0);
	//lastTheta=-atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));

	//cout<<"Theta using mean shape: "<<lastTheta/3.1415926f*180.0f<<endl;
	//visualize current shape
	//m_pDrawColor->Draw(colorImgBackUP.ptr<BYTE>(), colorImgBackUP.rows * colorImgBackUP.cols * 4);

	//update v
	if (initialPara)
	{
		for (int i=0;i<fullIndNum;i++)
		{
			veolocity[i]=currentShape[combineIndList[i]]-lastPts[i];
			veolocity[i+fullIndNum]=currentShape[combineIndList[i]+ptsNum]-lastPts[i+fullIndNum];
		}
		thetaVeolosity=lastTheta-lastlastTheta;
		hasVelocity=true;

		//cout<<"we have veolocity!!\n";
	}

	for (int i=0;i<fullIndNum;i++)
	{
		/*lastPts[i]=currentShape[combineIndList[i]];
		lastPts[i+fullIndNum]=currentShape[combineIndList[i]+ptsNum];
		if (i==18)*/
		{
			lastPts[i]=finalPos[i][0];
			lastPts[i+fullIndNum]=finalPos[i][1];
		}
	
	}

	//pridict pts
	if (hasVelocity)
	{
		for (int i=0;i<fullIndNum*2;i++)
		{
			pridictedPts[i]=lastPts[i]+veolocity[i];
		}

	}



	////}
	//GTE("S");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"used time per iteration: "<<time<<"  /60= "<<time/60<<endl;

	return true;


}

bool AAM_Detection_Combination::track_combine(Mat &colorImg,Mat &depthImg,int sx,int ex,int sy,int ey,bool initialPara)
{

	//step 2: detection
	//GTB("S");
	//for (int ll=0;ll<200;ll++)
	//{



	int width=depthImg.cols;
	int height=depthImg.rows;

	

	//Step 1: transfer the data to gpu

	#pragma omp parallel for
	for (int i=0;i<depthImg.rows;i++)
	{
		for (int j=0;j<depthImg.cols;j++)
		{
			host_depthImage_global[i*depthImg.cols+j]=depthImg.at<float>(i,j);
			host_colorImage_global[i*depthImg.cols+j]=colorImg.at<uchar>(i,j);
		}
	}
	setData_onrun_shared(host_colorImage_global,host_depthImage_global,width, height);

	


	/*Mat tmpImg;
	colorImg.convertTo(tmpImg,CV_32FC1);
	host_colorImage=(float *)tmpImg.data;
	host_depthImage=(float *)depthImg.data;	
	setData_onrun_shared(host_colorImage,host_depthImage,width, height);*/

	/*for (int i=0;i<10;i++)
	{
		cout<<host_colorImage[i]<<" ";
	}
	cout<<endl;*/

	
		
	//GTB("A");
	bool enoughVisibleNum=predict_GPU_separated_combination(depthImg.cols,depthImg.rows,hostDetectionResult,finalPos,maximumProb,sx,ex,sy,ey,initialPara&&TemporalTracking,showProbMap);

	if (showProbMap)
	{
		//probMap[j*width+k+cIndex*width*height];
		//visualize the probMap
		Mat curProbMap=depthImg.clone()*0;
	//	cvtColor(curProbMap,curProbMap,CV_GRAY2BGR);
		int w_size=5;
		for (int i=0;i<fullIndNum;i++)
		{
			for (int j=finalPos[i][0]-w_size;j<finalPos[i][0]+w_size;j++)
			{
				for (int k=finalPos[i][1]-w_size;k<finalPos[i][1]+w_size;k++)
				{
					curProbMap.at<float>(k,j)=hostDetectionResult[k*width+j+i*width*height];
				}
				
			}

			//circle(curProbMap,Point(finalPos[i][0],finalPos[i][1]),3,Scalar(1));
		}
		namedWindow("probMap",CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
		imshow("probMap",curProbMap(Range(153,289),Range(640-348,640-205)));
		waitKey(1);
		//return false;
	}




	/*GTE("A");
	gCodeTimer.printTimeTree();
	double time = total_fps;
	cout<<"used time per iteration: "<<time<<endl;*/

	if (!enoughVisibleNum)
	{
		return false;
	}

	//check the probability map
	/*{
		checkProbMap(hostDetectionResult);
		return false;
	}*/
	
	/*if (state==2)
	{
		for (int i=0;i<ptsNum*2;i++)
		{
			currentShape[i]=0;
		}
		for (int i=0;i<fullIndNum;i++)
		{
			currentShape[combineIndList[i]]=finalPos[i][0];
			currentShape[combineIndList[i]+ptsNum]=finalPos[i][1];
		}
		return true;
	}*/
	
	bool bigChange=true;
	if (state==1)
	{
		int okayNum=0;
		for (int i=0;i<fullIndNum;i++)
		{
			if (abs(lastPts[i]-finalPos[i][0])<2&&abs(lastPts[i+fullIndNum]-finalPos[i][1])<2)
			{
				okayNum++;
			}

		}
		//cout<<okayNum<<endl;
		if (okayNum>12)
		{
			bigChange=false;
		}
		if (!(abs(lastPts[18]-finalPos[18][0])<2&&abs(lastPts[18+fullIndNum]-finalPos[18][1])<2))
			bigChange=true;
		if (!(abs(lastPts[15]-finalPos[15][0])<2&&abs(lastPts[15+fullIndNum]-finalPos[15][1])<2))
			bigChange=true;

		
		//else
		//	cout<<abs(lastPts[18]-finalPos[18][0])<<" "<<abs(lastPts[18+fullIndNum]-finalPos[18][1])<<endl;
		/*if (bigChange)
		{
			cout<<rand()<<" big change\n";
		}*/
	}
	
	//then, KNN. Currently, on CPU
	vector<int> finalInd;
	//ransac_noSampling(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb);
	tmpImg=colorImg.clone();
	//ransac_noSampling_parrllel(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb,&colorImg);

	vector<int> totalInd;
	for (int i=0;i<fullIndNum;i++)
	{
		totalInd.push_back(i);
		candidatePoints[i].clear();
		/*if (hasVelocity)
		{
			
			if (abs(pridictedPts[i]-finalPos[i][0])<10&&abs(pridictedPts[i+fullIndNum]-finalPos[i][1])<10)
			{
				candidatePoints[i].push_back(Point2f(pridictedPts[i],pridictedPts[i+fullIndNum]));
			}
			
		}*/
		/*if (state==1&&!bigChange)
		{
			candidatePoints[i].push_back(Point2f(lastPts[i],lastPts[i+fullIndNum]));
		}
		else*/
			candidatePoints[i].push_back(Point2f(finalPos[i][0],finalPos[i][1]));

		if (initialPara&&state!=2)
		{
			
			//if (abs(pridictedPts[i]-finalPos[i][0])<10&&abs(pridictedPts[i+fullIndNum]-finalPos[i][1])<10)
			/*if((combineIndList[i]<42||combineIndList[i]>63)&&(i!=1&&i!=3&&i!=6&&i!=10&&i!=12&&i!=14))*/
			//if((combineIndList[i]<42||combineIndList[i]>63))
			/*{
				candidatePoints[i].push_back(Point2f(lastPts[i],lastPts[i+fullIndNum]));
			}*/
			
		}
	}


	//	findSecondModes(finalPos,depthImg.cols,depthImg.rows,totalInd,hostDetectionResult,maximumProb,&colorImg,startX,endX,startY,endY);

	/*LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);*/
	//cout<<"mean Shift\n";
//	findSecondModes_Meanshift(finalPos,depthImg.cols,depthImg.rows,totalInd,hostDetectionResult,maximumProb,&colorImg,startX,endX,startY,endY);

	/*{
		for (int i=0;i<ptsNum*2;i++)
		{
			currentShape[i]=0;
		}
		for (int i=0;i<fullIndNum;i++)
		{
			currentShape[combineIndList[i]]=finalPos[i][0];
			currentShape[combineIndList[i]+ptsNum]=finalPos[i][1];
		}
	}
	return false;*/

	/*for (int i=0;i<fullIndNum;i++)
	{
		for (int j=0;j<candidatePoints[i].size();j++)
		{
			cout<<i<<" "<<candidatePoints[i][j].x<<" "<<candidatePoints[i][j].y<<endl;
		}
		
	}*/

	//return true;

	//LONGLONG   t1,t2; 
	//LONGLONG   persecond; 
	//double time;
	//
	//QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	//
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 

	//2ms
	
	bool isS;
	isS=geoHashing_Candidates_nearestInlier(finalPos,depthImg.cols,depthImg.rows,finalInd,maximumProb,"",NULL,NULL,candidatePoints);
	if (!isS)
	{
		//cout<<"no nn found!\n";
		return false;
	}

	//1 3 6 10 12 14
	//for (int i=0;i<sizeof(specialInd)/sizeof(int);i++)
	//{
	//	if (maximumProb[specialInd[i]]<0.05)
	//	{
	//		finalInd.erase(finalInd.begin()+specialInd[i]);
	//	}
	//}
	

	if (state==2)
	{
		for (int i=0;i<ptsNum*2;i++)
		{
			currentDetection[i]=0;
		}
		for (int i=0;i<finalInd.size();i++)
		{
			currentDetection[combineIndList[finalInd[i]]]=finalPos[finalInd[i]][0];
			currentDetection[combineIndList[finalInd[i]]+ptsNum]=finalPos[finalInd[i]][1];
				/*currentDetection[combineIndList[finalInd[i]]]=candidatePoints[finalInd[i]][0].x;
			currentDetection[combineIndList[finalInd[i]]+ptsNum]=candidatePoints[finalInd[i]][0].y;*/
		}
		/*for (int i=0;i<ptsNum*2;i++)
		{
			currentShape[i]=0;
		}
		for (int i=0;i<finalInd.size();i++)
		{
			currentShape[combineIndList[finalInd[i]]]=finalPos[finalInd[i]][0];
			currentShape[combineIndList[finalInd[i]]+ptsNum]=finalPos[finalInd[i]][1];
		}	
		return true;*/
	}

	/*{
		for (int i=0;i<ptsNum*2;i++)
		{
			currentShape[i]=0;
		}
		for (int i=0;i<fullIndNum;i++)
		{
			currentShape[combineIndList[i]]=finalPos[i][0];
			currentShape[combineIndList[i]+ptsNum]=finalPos[i][1];
		}
	}
	return false;*/


	//}
	//GTE("S");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"used time per iteration: "<<time<<"  /200= "<<time/200<<endl;


	//return;
	
	/*QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	time=(t2-t1)*1000/persecond; 
	cout<<"calculation on CPU time: "<<time<<"ms "<<endl;*/
	//return;
	//save the modes and KNN search result
	
	

	//finally, AAM with detection on GPU, 0.7ms
	int minInd=bestFitTrainingSampleInd;
	globalTransformation=globalTransformation_optimal;
	for (int i=0;i<AAM_exp->shape_dim;i++)
	{
		AAM_exp->s_weight[i]=eigenVectors.at<double>(i,minInd);
	}

	
	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=-atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));

	AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	AAM_exp->setInitialScale(scale);
	AAM_exp->setInitialTheta(theta);



	//if (state==1&&(!hasVelocity||(abs(veolocity[18])<2)))
	if (state==1&&!bigChange)
	{
		AAM_exp->calculateData_onrun_AAM_combination(colorImg,finalInd,initialPara);
	}
	else// if (state==0)
	{
		AAM_exp->calculateData_onrun_AAM_combination(colorImg,finalInd,false);
	}

	//step .2: set up covariances matrix for each pixel
	//setupConv(colorImg.cols,colorImg.rows,20,finalPos,finalInd,host_preCalculatedConv);
	//float *absInd=new float[MAX_LABEL_NUMBER];
	for (int i=0;i<finalInd.size();i++)
	{
		absInd_Global[i]=combineIndList[finalInd[i]];
	}
	setupConv_featurePts(colorImg.cols,colorImg.rows,20,finalPos,finalInd,absInd_Global);

	


	//step .3: optimization on GPU
	
	iterate_combination(colorImg.cols,colorImg.rows,0,0,lastTheta,currentShape, currentShapePtsNum, isAAMOnly&&!bigChange,showNN);

	//update v
	if (initialPara)
	{
		for (int i=0;i<fullIndNum;i++)
		{
			veolocity[i]=currentShape[combineIndList[i]]-lastPts[i];
			veolocity[i+fullIndNum]=currentShape[combineIndList[i]+ptsNum]-lastPts[i+fullIndNum];
		}
		hasVelocity=true;
	}

	for (int i=0;i<fullIndNum;i++)
	{
		/*lastPts[i]=currentShape[combineIndList[i]];
		lastPts[i+fullIndNum]=currentShape[combineIndList[i]+ptsNum];
		if (i==18)*/
		{
			lastPts[i]=finalPos[i][0];
			lastPts[i+fullIndNum]=finalPos[i][1];
		}
	
	}

	//pridict pts
	if (hasVelocity)
	{
		for (int i=0;i<fullIndNum*2;i++)
		{
			pridictedPts[i]=lastPts[i]+veolocity[i];
		}
	}



	////}
	//GTE("S");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"used time per iteration: "<<time<<"  /60= "<<time/60<<endl;

	return true;


}

void AAM_Detection_Combination::prepareModel(bool isApt)
{
	setupWeight(AAMWeight,RTWeight,priorWeight,localWeight);
	setupUsedIndex(combineIndList,fullIndNum);
	//read in the randomized trees

	///////////////////tree number test/////////////////////////
	//int treeNum=17;

	//rt=new RandTree(15,3,0,17,48,25);
	//rt->usingCUDA=false;
	//rt->trainStyle=0;rt->load_prob("D:\\Fuhao\\face dataset\\train_crossValidation\\depth\\trainedTree_17_15_48_22.txt");
	//rt->treeNum=treeNum;
	//

	//rt_colorRef=new RandTree(15,3,0,17,48,25);
	//rt_colorRef->usingCUDA=false;
	//rt_colorRef->trainStyle=1;
	////rt_colorRef->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThres.txt");
	//rt_colorRef->load_prob("D:\\Fuhao\\face dataset\\train_crossValidation\\color\\trainedTree_17_15_48_22.txt");
	//rt_colorRef->treeNum=treeNum;
	////////////////////////////////////////////////////////////

	/////////////////////////treeDepth test///////////////////////////////////
	//int treeNum=15;
	//int curDepth=9;

	//rt=new RandTree(curDepth,3,0,17,48,25);
	//rt->usingCUDA=false;
	//rt->trainStyle=0;rt->load_prob("D:\\Fuhao\\face dataset\\train_crossValidation\\depth_color_depthTest\\trainedTree_17_9_48_22_0.txt");
	////rt->treeNum=treeNum;


	//rt_colorRef=new RandTree(curDepth,3,0,17,48,25);
	//rt_colorRef->usingCUDA=false;
	//rt_colorRef->trainStyle=1;rt_colorRef->load_prob("D:\\Fuhao\\face dataset\\train_crossValidation\\depth_color_depthTest\\trainedTree_17_9_48_22_1.txt");
	//rt_colorRef->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThres.txt");
	//rt_colorRef->treeNum=treeNum;
	//////////////////////////////////////////////////////////////////////////

	///////////////////////windowsize test///////////////////////////////////
	//int treeNum=15;
	//int curDepth=6;
	//int windowsize=32;
	//rt=new RandTree(11,3,0,12,windowsize,25);
	//rt->usingCUDA=false;
	//rt->trainStyle=0;rt->load_prob("D:\\Fuhao\\face dataset\\train_crossValidation\\depth_color_windowsizeTest\\trainedTree_12_11_32_22_0.txt");
	////rt->treeNum=treeNum;


	//rt_colorRef=new RandTree(11,3,0,12,windowsize,25);
	//rt_colorRef->usingCUDA=false;
	//rt_colorRef->trainStyle=1;rt_colorRef->load_prob("D:\\Fuhao\\face dataset\\train_crossValidation\\depth_color_windowsizeTest\\trainedTree_12_11_32_22_1.txt");
	//rt_colorRef->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThres.txt");
	//rt_colorRef->treeNum=treeNum;
	////////////////////////////////////////////////////////////////////////

	

	rt=new RandTree(15,3,0,17,48,25);

	rt->usingCUDA=true;

	//rt->trainStyle=2;
	//rt->load("D:\\Fuhao\\RT Model\\trainedTree_17_15_48_25_color_depth.txt");

	//no threshold
	//rt->trainStyle=2;rt->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_depthColor.txt");
	//rt->trainStyle=1;rt->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_color.txt");
	
		//rt->trainStyle=0;rt->load_prob("D:\\Fuhao\\face dataset\\train_synDepthTest\\trainedTree_17_15_48_22_synDepth_Thres1.txt");
		//rt->trainStyle=0;rt->load_prob("D:\\Fuhao\\face dataset\\train_synDepthTest\\trainedTree_17_15_48_22_synDepth_noThres113.txt");
	//rt->trainStyle=0;rt->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_depth.txt");
	rt->trainStyle=0;rt->load_prob(depthRT_dir.c_str(),1);
	//cout<<rt->labelNum<<endl;
	//rt->trainStyle=0;rt->load_prob("D:\\Fuhao\\face dataset\\train_SYNandREAL\\trainedTree_17_15_48_22_depthThresSynReal.txt");

	//color with threshold
	//rt->trainStyle=1;rt->load_prob("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_color_thres.txt");
		//rt->trainStyle=1;rt->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThres.txt");

	//both with threshold
	///////////rt->trainStyle=2;rt->load_prob("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_both_thres.txt");//older version, the threshold of depth is not good
	////rt->trainStyle=2;rt->load_prob("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_bothThres_prob.txt");
	//rt->trainStyle=2;rt->load_prob("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_bothThres_faceSampling.txt");
	////both but only depth with threshold

	////no threshold, with left right prob
	////rt->trainStyle=2;rt->load_prob("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_depthColor_prob.txt");

	////new trained with threshold
	////rt->trainStyle=1;rt->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThres.txt");
	////rt->trainStyle=2;rt->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThresDepth.txt");

	//////////////NewADD for missing depth/////////////////////////
	rt_colorRef=new RandTree(15,3,0,17,48,25);
	rt_colorRef->usingCUDA=true;
	rt_colorRef->trainStyle=1;
	//rt_colorRef->load("D:\\Fuhao\\face dataset\\train_all_final\\trainedTree_17_15_48_22_colorThres.txt");
	rt_colorRef->load_prob(colorRT_dir.c_str(),0);
	////////////////////////////////////////////////////
	//set up data on GPU
	//return;

	//read in AAM
	//AAM_exp=new AAM_RealGlobal_GPU(.05,1,0);	//Detection, AAM and prior
	AAM_exp=new AAM_RealGlobal_GPU(RTWeight,AAMWeight,priorWeight,localWeight,isApt);	//Detection, AAM
	//AAM_exp=new AAM_RealGlobal_GPU(0,1,0);	//AAM only
	//AAM_exp=new AAM_RealGlobal_GPU(0,1,0.000001);	//Detection, AAM and prior
	//AAM_exp->getAllNeededData("D:\\Fuhao\\face dataset\\train_all_final\\trainedResault.txt");
	//AAM_exp->getAllNeededData("D:\\Fuhao\\face dataset\\train_larger database\\trainedResault.txt");
	//AAM_exp->getAllNeededData("D:\\Fuhao\\face dataset\\train_larger database\\trainedResault_90_90.txt");
	//AAM_exp->getAllNeededData("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\trainedResault_91_90.txt");
	AAM_exp->getAllNeededData(AAMModelPath.c_str());

	cout<<"shape and texture dim: "<<AAM_exp->shape_dim<<" "<<AAM_exp->texture_dim<<" pixNum: "<<AAM_exp->pix_num<<endl;
	//AAM_exp->getAllNeededData("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\trainedResault.txt");

	AAM_exp->isGlobaltransform=true;
	AAM_exp->outputtime=true;
	AAM_exp->showSingleStep=false;
	AAM_exp->setResizeSize(1);
	AAM_exp->startNum=0;
	//AAM_exp->setInitialTranslation( 85,110);
	//AAM_exp->setInitialScale(1.0f);
	AAM_exp->setCUDA(true);
	AAM_exp->setGPU(true);
	meanShapeCenter=AAM_exp->meanShape->getcenter();

	cout<<AAM_exp->pix_num<<endl;
	//set up the AAM data we use
	AAM_exp->prepareForTracking();

}

void AAM_Detection_Combination::buildHashTabel(string name)
{
	int cshapeNum,cfullPtsNum;
	int cptsNum=sizeof(combineIndList)/sizeof(int);
	//ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_feature.txt",ios::in);
//	ifstream in("D:\\Fuhao\\face dataset\\train_larger database\\allignedshape_feature_90_90.txt",ios::in);
	//ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_feature_91_90.txt",ios::in);

	ifstream in(name.c_str(),ios::in);

	in>>cshapeNum>>cfullPtsNum;

	float tmpVal;
	Mat cfullData=Mat::zeros(cshapeNum,cfullPtsNum*2,CV_64FC1);
	for (int i=0;i<cshapeNum;i++)
	{
		for (int j=0;j<cfullPtsNum*2;j++)
		{
			in>>tmpVal;
			cfullData.at<double>(i,j)=tmpVal;
		}
	}

	Mat data=Mat::zeros(cshapeNum,cptsNum*2,CV_64FC1);
	for (int i=0;i<cshapeNum;i++)
	{
		for (int j=0;j<cptsNum;j++)
		{
			data.at<double>(i,j)=cfullData.at<double>(i,combineIndList[j]);
			data.at<double>(i,j+cptsNum)=cfullData.at<double>(i,combineIndList[j]+cfullPtsNum);
		}
	}

	//	geoHashingMatrix *test=new geoHashingMatrix(data,0.8);
	geohashSearch=new GeoHashing(data,0.15);
	//geohashSearch->buildHashTabel(geohashSearch->basisNum,geohashSearch->basisTabel,data);
	geohashSearch->buildHashTabelVec(geohashSearch->basisNum,geohashSearch->basisTabel,data);
}

void AAM_Detection_Combination::buildHashTabel(Mat &shape)
{
	//int cshapeNum,cfullPtsNum;
	int cptsNum=sizeof(combineIndList)/sizeof(int);
	//ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_feature.txt",ios::in);
	//	ifstream in("D:\\Fuhao\\face dataset\\train_larger database\\allignedshape_feature_90_90.txt",ios::in);
	//ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\allignedshape_feature_91_90.txt",ios::in);

	//ifstream in(name.c_str(),ios::in);

	//in>>cshapeNum>>cfullPtsNum;

	int cshapeNum=shape.cols;
	int cfullPtsNum=shape.rows/2;

	Mat cfullData;
	cout << "transpose" << endl;
	transpose(shape,cfullData);
	cout << "done" << endl;

	/*float tmpVal;
	Mat cfullData=Mat::zeros(cshapeNum,cfullPtsNum*2,CV_64FC1);
	for (int i=0;i<cshapeNum;i++)
	{
		for (int j=0;j<cfullPtsNum*2;j++)
		{
			in>>tmpVal;
			cfullData.at<double>(i,j)=tmpVal;
		}
	}
*/
	Mat data=Mat::zeros(cshapeNum,cptsNum*2,CV_64FC1);
	for (int i=0;i<cshapeNum;i++)
	{
		for (int j=0;j<cptsNum;j++)
		{
			data.at<double>(i,j)=cfullData.at<double>(i,combineIndList[j]);
			data.at<double>(i,j+cptsNum)=cfullData.at<double>(i,combineIndList[j]+cfullPtsNum);
		}
	}

	//	geoHashingMatrix *test=new geoHashingMatrix(data,0.8);
	cout << "creating geo hashing ..." << endl;
	geohashSearch=new GeoHashing(data,0.8);
	cout << "done" << endl;
	//geohashSearch->buildHashTabel(geohashSearch->basisNum,geohashSearch->basisTabel,data);

	cout << "building hash table vector ..." << endl;
	geohashSearch->buildHashTabelVec(geohashSearch->basisNum,geohashSearch->basisTabel,data);
	cout << "done" << endl;
}



void AAM_Detection_Combination::geoHashing_Candidates(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,char *saveName,Mat *img,Mat *depthImg,vector<Point2f>*candidatePts)
{
	int cptsNum=sizeof(combineIndList)/sizeof(int);
	Mat inData=Mat::zeros(1,cptsNum*2,CV_64FC1);
	for (int i=0;i<cptsNum;i++)
	{
		inData.at<double>(0,i)=sampledPos[i][0];
		inData.at<double>(0,cptsNum+i)=sampledPos[i][1];
	}
	vector<int> KNNID;
	char name[500];
	sprintf(name, "AAM+detection/%s_%d_NN.jpg", AAM_exp->prefix,11361+AAM_exp->currentFrame);

	//output: the inlier index and corresponding best position
	geohashSearch->vote_countAll(inData,finalInd,KNNID,cptsNum*0.8,NNNum,candidatePoints,&colorImgBackUP,name);

	for (int i=0;i<cptsNum;i++)
	{
		sampledPos[i][0]=inData.at<double>(0,i);
		sampledPos[i][1]=inData.at<double>(0,i+cptsNum);
	}

	/////////dirty codes for connecting the original cpu code////////////
	bestFitTrainingSampleInd=KNNID[0];
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	Mat newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	globalTransformation_optimal=globalTransformation.clone();

	//////////////////////visulization/////////////////////////////////
	//Mat tmpIMg=(colorImgBackUP).clone();
	//for (int i=0;i<1;i++)
	//{
	//	Point c;
	//	for (int j=0;j<fullIndNum;j++)
	//	{
	//		c.x=newU.at<float>(j,0);
	//		c.y=newU.at<float>(j+fullIndNum,0);
	//		circle(tmpIMg,c,5,Scalar(0,255,0));
	//	}
	//}

	//for (int i=0;i<fullIndNum;i++)
	//	{
	//		
	//		for (int j=0;j<candidatePoints[i].size();j++)
	//		{
	//			//c.x=finalPos[i][0];
	//			//c.y=finalPos[i][1];
	//			//cout<<i<<" "<<candidatePoints[i][j].x<<" "<<candidatePoints[i][j].y<<endl;
	//			circle(tmpIMg,candidatePoints[i][j],2,Scalar(0,0,255));
	//		}
	//}

	//namedWindow("NN visualization");
	//imshow("NN visualization",tmpIMg);
	//waitKey();
	///////////////////////////////////////////////////////////////////////////

	//using all the index
		finalInd.clear();
		for (int i=0;i<fullIndNum;i++)
		{
			finalInd.push_back(i);
		}
	for (int ii=0;ii<finalInd.size();ii++)
	{
		int cindex=finalInd[ii];
		for (int j=0;j<candidatePoints[cindex].size();j++)
		{
			float currentDis=sqrt(((newU.at<float>(cindex,0)-candidatePoints[cindex][j].x)*
				(newU.at<float>(cindex,0)-candidatePoints[cindex][j].x)+
				(newU.at<float>(cindex+fullIndNum,0)-candidatePoints[cindex][j].y)*
				(newU.at<float>(cindex+fullIndNum,0)-candidatePoints[cindex][j].y)));
			float probAll=powf(e,-currentDis*currentDis/50);
			//probAll*=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
						//(2*sigma1*sigma1));
			AAM_exp->probForEachFeatureCandidates[ii][j]=probAll;

		/*	if (finalInd[ii]==2)
			{
				cout<<j<<" "<<currentDis<<" "<<probAll<<" "<<candidatePoints[cindex][j].x<<" "<<candidatePoints[cindex][j].y<<endl;
				
			}*/
			
		}
		AAM_exp->candidateNum[ii]=candidatePoints[cindex].size();
		//AAM_exp->probForEachFeature[ii]=1;
	}

	//use all the feature candidates
	for (int i=0;i<finalInd.size();i++)
	{
		AAM_exp->candidatePoints[i].clear();
		for (int j=0;j<candidatePoints[finalInd[i]].size();j++)
		{
			AAM_exp->candidatePoints[i].push_back(candidatePoints[finalInd[i]][j]);
		}
	}

	
	Mat KNNVec=Mat::zeros(eigenVectors.rows,NNNum,CV_64FC1);
	for (int i=0;i<NNNum;i++)
	{
		KNNVec.col(i)+=eigenVectors.col(KNNID[i]);
	}
	
	//if (localWeight>0)
	{
	//update the mean
	Mat mean_KNN=KNNVec.col(0)*0;
	for (int i=0;i<KNNVec.cols;i++)
	{
		mean_KNN+=KNNVec.col(i);
	}
	mean_KNN/=KNNVec.cols;
	for (int i=0;i<mean_KNN.rows;i++)
	{
		AAM_exp->priorMean[i]=mean_KNN.at<double>(i,0);
	}

	//update the conv
	for (int i=0;i<KNNVec.cols;i++)
	{
		KNNVec.col(i)-=mean_KNN;
	}
	Mat KNNVec_tran;
	transpose(KNNVec,KNNVec_tran);
	Mat convKNN=KNNVec*KNNVec_tran/KNNVec.cols;

	Mat conv_inv_KNN=convKNN.inv();

//	cout<<AAM_exp->priorSigma.cols<<" "<<AAM_exp->priorSigma.rows<<endl;
	for (int i=0;i<conv_inv_KNN.rows;i++)
	{
		for (int j=0;j<conv_inv_KNN.cols;j++)
		{
			AAM_exp->priorSigma.at<double>(i,j)=conv_inv_KNN.at<double>(i,j);
		}
	}

	if (AAM_exp->usingGPU)
	{
		float *p_mean=new float[mean_KNN.rows];
		int totalDim=(AAM_exp->shape_dim+AAM_exp->texture_dim+4);
		float *p_sigma=new float[totalDim*totalDim];

		for (int i=0;i<mean_KNN.rows;i++)
		{
			p_mean[i]=mean_KNN.at<double>(i,0);
		}

		for (int i=0;i<totalDim*totalDim;i++)
		{
			p_sigma[i]=0;
		}
		for (int i=0;i<conv_inv_KNN.rows;i++)
		{
			for (int j=0;j<conv_inv_KNN.cols;j++)
			{
				p_sigma[i*totalDim+j]=conv_inv_KNN.at<double>(i,j);
			}
		}
		setLocalPrior(p_mean,p_sigma,mean_KNN.rows,AAM_exp->texture_dim);

		delete []p_mean;
		delete []p_sigma;
	}

	if (localWeight>0)
	{


		//train the local PCA model
		int s_dim=eigenVectors.rows;
		//use KNNVec_tran to train
		CvMat *pData=cvCreateMat(KNNVec_tran.rows,KNNVec_tran.cols,CV_64FC1);
		for (int i=0;i<KNNVec_tran.rows;i++)
		{
			for (int j=0;j<KNNVec_tran.cols;j++)
			{
				//CV_MAT_ELEM(*pData,double,i,j)=shape[i]->ptsForMatlab[j];

				//here,we keep the shape in the same scale with the meanshape
				CV_MAT_ELEM(*pData,double,i,j)=KNNVec_tran.at<double>(i,j);
			}

		}
		if (AAM_exp->local_s_mean==NULL)
		{
			AAM_exp->local_s_mean = cvCreateMat(1, KNNVec_tran.cols, CV_64FC1);
			AAM_exp->m_local_s_mean=cvarrToMat(AAM_exp->local_s_mean);
		}

		if (AAM_exp->local_s_vec==NULL)
		{
			AAM_exp->local_s_vec=cvCreateMat( min(KNNVec_tran.rows,KNNVec_tran.cols), KNNVec_tran.cols, CV_64FC1); 
		}

		CvMat* s_value = cvCreateMat(1, min(KNNVec_tran.cols,KNNVec_tran.rows), CV_64FC1);
		//CvMat *s_PCAvec = cvCreateMat( min(KNNVec_tran.rows,KNNVec_tran.cols), KNNVec_tran.cols, CV_64FC1); 
		cvCalcPCA( pData, AAM_exp->local_s_mean, s_value, AAM_exp->local_s_vec, CV_PCA_DATA_AS_ROW );
		AAM_exp->m_local_mean=cvarrToMat(AAM_exp->local_s_mean);

		double sumEigVal=0;
		for (int i=0;i<s_value->cols;i++)
		{
			sumEigVal+=CV_MAT_ELEM(*s_value,double,0,i);
		}

		double sumCur=0;
		for (int i=0;i<s_value->cols;i++)
		{
			sumCur+=CV_MAT_ELEM(*s_value,double,0,i);
			if (sumCur/sumEigVal>=0.98)
			{
				AAM_exp->local_shape_dim=i+1;
				break;
			}
		}
		//cout<<"local dim: "<<AAM_exp->local_shape_dim<<endl;

		Mat curEigenVec=cvarrToMat(AAM_exp->local_s_vec);
		Mat usedEigenVec=curEigenVec.rowRange(Range(0,AAM_exp->local_shape_dim));
		Mat usedEigenVec_tran;
		transpose(usedEigenVec,usedEigenVec_tran);
		Mat localHessian=usedEigenVec_tran*usedEigenVec;
		localHessian=Mat::eye(localHessian.rows,localHessian.cols,CV_64FC1)-localHessian;

		Mat localHessian_tran;
		transpose(localHessian,localHessian_tran);
		localHessian=localHessian_tran*localHessian;
		for (int i=0;i<localHessian.rows;i++)
		{
			for (int j=0;j<localHessian.cols;j++)
			{
				AAM_exp->m_localHessian.at<double>(i,j)=localHessian.at<double>(i,j);
			}
		}
	}

	}
	//delete []distance;
	

	return;
}

bool AAM_Detection_Combination::geoHashing_Candidates_nearestInlier(float **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb,char *saveName,Mat *img,Mat *depthImg,vector<Point2f>*candidatePts)
{
	
	
	/*int cptsNum=sizeof(combineIndList)/sizeof(int);
	Mat inData=Mat::zeros(2,cptsNum,CV_64FC1);
	Mat inDataOldFormat=Mat::zeros(1,cptsNum*2,CV_64FC1);
	for (int i=0;i<cptsNum;i++)
	{
		inData.at<double>(0,i)=sampledPos[i][0];
		inData.at<double>(1,i)=sampledPos[i][1];
		inDataOldFormat.at<double>(0,i)=sampledPos[i][0];
		inDataOldFormat.at<double>(0,cptsNum+i)=sampledPos[i][1];
	}
	vector<int> KNNID;
	geohashSearch->vote_countAllVec(inData,inDataOldFormat,finalInd,KNNID,cptsNum*0.8,NNNum,candidatePoints);
*/

	//GTB("S");
	//for (int ll=0;ll<500;ll++)
	//{

	int cptsNum=sizeof(combineIndList)/sizeof(int);
	Mat inData=Mat::zeros(1,cptsNum*2,CV_64FC1);
	for (int i=0;i<cptsNum;i++)
	{
		inData.at<double>(0,i)=sampledPos[i][0];
		inData.at<double>(0,cptsNum+i)=sampledPos[i][1];
	}
	vector<int> KNNID;

	bool isS;
	isS=geohashSearch->vote_countAllVec_old(inData,finalInd,KNNID,cptsNum*0.8,NNNum,candidatePoints);
	if (!isS)
	{
		return false;
	}
	
	

	for (int i=0;i<cptsNum;i++)
	{
		sampledPos[i][0]=inData.at<double>(0,i);
		sampledPos[i][1]=inData.at<double>(0,i+cptsNum);
	}

	/////////dirty codes for connecting the original cpu code////////////
	bestFitTrainingSampleInd=KNNID[0];
	//cout<<"bestID: "<<bestFitTrainingSampleInd<<endl;
	//cout<<shapes.rows<<" "<<shapes.cols<<endl;
	getTransformationInfo(finalInd,sampledPos,shapes,bestFitTrainingSampleInd);
	Mat newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	globalTransformation_optimal=globalTransformation.clone();

	//////////////////////visulization/////////////////////////////////
	/*Mat tmpIMg=(colorImgBackUP).clone();
	

		Point c;
		for (int j=0;j<fullIndNum;j++)
		{
			c.x=sampledPos[j][0];
			c.y=sampledPos[j][1];
			circle(tmpIMg,c,5,Scalar(0,255,0));
		}

	for (int i=0;i<finalInd.size();i++)
	{
		c.x=sampledPos[finalInd[i]][0];
		c.y=sampledPos[finalInd[i]][1];
		circle(tmpIMg,c,5,Scalar(0,0,255));
	}
	namedWindow("to be used inliers");
	imshow("to be used inliers",tmpIMg);
	waitKey();*/

	//for (int i=0;i<fullIndNum;i++)
	//	{
	//		
	//		for (int j=0;j<candidatePoints[i].size();j++)
	//		{
	//			//c.x=finalPos[i][0];
	//			//c.y=finalPos[i][1];
	//			//cout<<i<<" "<<candidatePoints[i][j].x<<" "<<candidatePoints[i][j].y<<endl;
	//			circle(tmpIMg,candidatePoints[i][j],2,Scalar(0,0,255));
	//		}
	//}

	//namedWindow("NN visualization");
	//imshow("NN visualization",tmpIMg);
	//waitKey();
	///////////////////////////////////////////////////////////////////////////

	//using only the inlier index
	if (!AAM_exp->usingGPU)
	{
		for (int ii=0;ii<finalInd.size();ii++)
		{

			//AAM_exp->probForEachFeature[ii]=1;

			AAM_exp->candidatePoints[ii].clear();

			AAM_exp->candidatePoints[ii].push_back(Point(sampledPos[finalInd[ii]][0],sampledPos[finalInd[ii]][1]));

			AAM_exp->candidateNum[ii]=AAM_exp->candidatePoints[ii].size();
			for (int j=0;j<	AAM_exp->candidatePoints[ii].size();j++)
			{
				AAM_exp->probForEachFeatureCandidates[ii][j]=1;
			}

		}
	}
	

	/*LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);*/

	//cout<<"KNN ID\n";
	//for (int i=0;i<KNNID.size();i++)
	//{
	//	cout<<KNNID[i]<<" "<<endl;
	//}


	Mat KNNVec=Mat::zeros(eigenVectors.rows,NNNum,CV_64FC1);
	for (int i=0;i<NNNum;i++)
	{
		KNNVec.col(i)+=eigenVectors.col(KNNID[i]);
	}
	
	
	
		//update the mean
		Mat mean_KNN=KNNVec.col(0)*0;
		for (int i=0;i<KNNVec.cols;i++)
		{
			mean_KNN+=KNNVec.col(i);
		}
		mean_KNN/=KNNVec.cols;
		for (int i=0;i<mean_KNN.rows;i++)
		{
			AAM_exp->priorMean[i]=mean_KNN.at<double>(i,0);
		}

		//update the conv
		for (int i=0;i<KNNVec.cols;i++)
		{
			KNNVec.col(i)-=mean_KNN;
		}
		Mat KNNVec_tran;
		transpose(KNNVec,KNNVec_tran);
		Mat convKNN=KNNVec*KNNVec_tran/KNNVec.cols;

		Mat conv_inv_KNN=convKNN.inv();

	/*	QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
		double   time=(t2-t1)*1000/persecond; 
		cout<<"Local prior CPU: "<<time<<"ms "<<endl;*/

		//	cout<<AAM_exp->priorSigma.cols<<" "<<AAM_exp->priorSigma.rows<<endl;
		
		if (!AAM_exp->usingGPU)
		{
			for (int i=0;i<conv_inv_KNN.rows;i++)
			{
				for (int j=0;j<conv_inv_KNN.cols;j++)
				{
					AAM_exp->priorSigma.at<double>(i,j)=conv_inv_KNN.at<double>(i,j);
				}
			}
		}
	
		if (AAM_exp->usingGPU)
		{
			
			int totalDim=(AAM_exp->shape_dim+AAM_exp->texture_dim+4);
			

		/*	for (int i=0;i<mean_KNN.rows;i++)
			{
				
			}*/
			//#pragma omp parallel for

			/*if (AAM_exp->showSingleStep)
			{
				cout<<"cur mean shape:\n";
				for (int i=0;i<conv_inv_KNN.rows;i++)
					cout<<mean_KNN.at<double>(i,0)<<" ";
				cout<<endl;
			}*/

			for (int i=0;i<conv_inv_KNN.rows;i++)
			{
				p_mean_global[i]=mean_KNN.at<double>(i,0);

				for (int j=0;j<conv_inv_KNN.cols;j++)
				{
					p_sigma_global[i*totalDim+j]=conv_inv_KNN.at<double>(i,j);
				}
			}
			setLocalPrior(p_mean_global,p_sigma_global,mean_KNN.rows,AAM_exp->texture_dim);

		}

	return true;

	//}
	//GTE("S");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"used time per iteration: "<<time<<"  /500= "<<time/500<<endl;

}