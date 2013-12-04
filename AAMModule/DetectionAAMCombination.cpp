#include "DetectionAAMCombination.h"

int usedIndList[]={2,3  ,  10  ,  11   , 22 ,   23    , 0  ,   1  ,   8,     9  ,  24 ,   27   ,  12   , 13  ,  14   , 15  ,  16  ,  17 ,   18,
19 ,   20   , 21   , 25 ,   26};
int criticalList[]={2,3,10,11,22,23,20,0,1,8,9};
//vector<int> finalInd;

DetectionWithAAM::DetectionWithAAM(float _tPrecision)
{
	targetPrecison=_tPrecision;
	sampleNumberFromTrainingSet=ceil(log(1-targetPrecison)/log(5.0f/6.0f));
	sampleNumberFromProbMap=1;
	sampleNumberFromFeature=ceil(log(1-targetPrecison)/log(5.0f/9.0f));

	window_threshold_small=10;
	window_threshold_large=20;
	window_threshold_small*=window_threshold_small;
	window_threshold_large*=window_threshold_large;

	criticalIndNum=sizeof(criticalList)/sizeof(int);
	//criticalIndList.resize(criticalIndNum);
	//for (int i=0;i<criticalIndNum;i++)
	//{
	//	criticalIndList.push_back(criticalIndList[i]);
	//}
	
	w_critical=ceil((float)criticalIndNum*0.6);

	cout<<sampleNumberFromTrainingSet<<" "<<sampleNumberFromFeature<<endl;

	cout<<"loading randomized trees...\n";
	rt=new RandTree(15,3,0,17,48,25);

	rt->usingCUDA=true;

	rt->trainStyle=2;
	rt->load("D:\\Fuhao\\RT Model\\trainedTree_17_15_48_25_color_depth.txt");

	labelNum=rt->labelNum;
	//hostDetectionResult=rt->labelResult;
	hostDetectionResult=new float[MPN*(1+MAX_LABEL_NUMBER)];

	cout<<"loaded!\n";
	//goto NOAAM;

	cout<<"\nloading AAM...\n";
		//AAM_exp=new AAM_RealGlobal_GPU(.05,1,0);	//Detection, AAM and prior
	AAM_exp=new AAM_RealGlobal_GPU(.05,1,0.000001);	//Detection, AAM
	//AAM_exp=new AAM_RealGlobal_GPU(0,1,0);	//AAM only
	//AAM_exp=new AAM_RealGlobal_GPU(0,1,0.000001);	//Detection, AAM and prior
	AAM_exp->getAllNeededData("D:\\Fuhao\\AAM model\\trainedResault.txt");

	AAM_exp->isGlobaltransform=true;
	AAM_exp->outputtime=true;
	AAM_exp->showSingleStep=false;
	AAM_exp->setResizeSize(1);
	AAM_exp->startNum=0;
	//AAM_exp->setInitialTranslation( 85,110);
	//AAM_exp->setInitialScale(1.0f);
	AAM_exp->setCUDA(true);
	AAM_exp->setGPU(false);

	AAM_exp->prepareForTracking();
	meanShapeCenter=AAM_exp->meanShape->getcenter();
	ptsNum=AAM_exp->meanShape->ptsNum;

	shapeDim=AAM_exp->shape_dim;
	//load all shapes
	ifstream in("D:\\Fuhao\\AAM model\\allignedshape.txt",ios::in);
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

	Mat shapeCenter=shapes.clone();
	for (int i=1;i<=totalShapeNum;i++)
	{
		for (int j=0;j<ptsNum*2;j++)
		{
			shapeCenter.at<double>(j,i)=shapes.at<double>(j,0);
		}
		
	}
	eigenVectors=Mat::zeros(shapeDim,totalShapeNum+1,CV_64FC1);
	eigenVectors=AAM_exp->m_s_vec.rowRange(Range(0,shapeDim))*(shapes-shapeCenter);
	

	cout<<totalShapeNum<<" shapes...\n";
	cout<<"loaded!\n";

//NOAAM:
	host_colorImage=host_depthImage=NULL;

	cout<<"Initialization finished!\n";


	
	cout<<labelNum<<" "<<fullIndNum<<endl;

	fullDetectedPos.create(labelNum-1,1,CV_32FC1);

	fullTrainingPos=new Mat[totalShapeNum];
	int cind;
	for (int j=0;j<totalShapeNum;j++)
	{
		fullTrainingPos[j].create(fullIndNum*2,4,CV_32FC1);
		for (int i=0;i<fullIndNum;i++)
		{
			cind=usedIndList[i];
		
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

	globalTransformation.create(4,1,CV_32FC1);

	isFindModes=false;
}

void DetectionWithAAM::searchPics(string listName)
{
	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');

	int startNum=49-49;//477-49;//63-49;//381-49;//459-49;//7353-49;//428-49;
	AAM_exp->currentFrame=startNum;
	for (int ll=0;ll<_imgNum;ll++)
	{
	
		//readin the image name first
		in.getline(name,500,'\n');
		if (ll<startNum)//||ll>459-49)
		{
			continue;
		}
		string saveName=name;
		saveName=saveName.substr(0,ImgName.length()-4);
		string prob_name=saveName;
		saveName+=".txt";
	/*	if (ll>_imgNum-8)
		{
			continue;
		}*/
		
		Mat m_img=imread(name,0);

		Mat colorImg=imread(name);

		Mat depthImg=Mat::ones(colorImg.rows,colorImg.cols,CV_32FC1);


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
						if (depthImg.at<float>(i,j)!=0)
						{
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

		////face detection here
		int startX,endX,startY,endY;
		startX=287;
		startY=99;
		endX=402;
		endY=250;
		for (int i=0;i<depthImg.rows;i++)
		{
			for (int j=0;j<depthImg.cols;j++)
			{
				if (j<startX||j>endX||i<startY||i>endY)
				{
					depthImg.at<float>(i,j)=0;
				}
			}
		}

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

		DWORD HessianStart, HessianStop,d_totalstart;  
		HessianStart=GetTickCount();
		track_sampling_CPU(m_img,depthImg);
		HessianStop=GetTickCount();
		cout<<"\n************************************************\n";
		cout<<"frame  time: "<<(HessianStop-HessianStart)<<" ms"<<endl;
		cout<<"\n************************************************\n";
		AAM_exp->currentFrame++;
		}
		
}

void DetectionWithAAM::track(Mat &colorIMG,Mat &depthIMG)
{
	//0: send depth and image data to GPU
	if (host_colorImage==NULL)
	{
		host_colorImage=new float[MPN];
		host_depthImage=new float[MPN];
	}
	int i,j;
	for (i=0;i<depthIMG.rows;i++)
	{
		for (j=0;j<depthIMG.cols;j++)
		{
			host_depthImage[i*depthIMG.cols+j]=depthIMG.at<float>(i,j);
			host_colorImage[i*depthIMG.cols+j]=colorIMG.at<uchar>(i,j);
		}
	}

	setData_RT_onrun(host_colorImage,host_depthImage,depthIMG.cols,depthIMG.rows);

	//1 detection, need a clear version
	predict_GPU_withDepth_clean(depthIMG.cols,depthIMG.rows,hostDetectionResult,rt->trainStyle);
	//rt->predict_rt_depth_GPU(depthIMG,colorIMG,"");
	
	//visualization
	//int i,j;
	//checked
	//{		Mat probImg;
	//	cvtColor(colorIMG, probImg, CV_GRAY2RGB);
	//	int offset;
	//	int currentLabelIndex;
	//	double blendPara=0.7;
	//	float currentProb;
	//	int currentShowingLabel=10;
	//	string tmpName;
	//	int windowSize=rt->windowSize;
	//	for (currentShowingLabel=0;currentShowingLabel<rt->labelNum-1;currentShowingLabel++)
	//	{
	//		for (i=windowSize;i<depthIMG.cols-windowSize;i++)
	//		{
	//			//	cout<<i<<endl;
	//			for (j=windowSize;j<depthIMG.rows-windowSize;j++)
	//			{

	//				offset=j*depthIMG.cols+i;

	//				/*		if (labelResult[offset*(1+MAX_LABEL_NUMBER)]!=11)
	//				{
	//				continue;
	//				}
	//				if (labelResult[offset*(1+MAX_LABEL_NUMBER)+1+(int)labelResult[offset*(1+MAX_LABEL_NUMBER)]]<0.4)
	//				{
	//				continue;
	//				}
	//				currentLabelIndex=(float)labelResult[offset*(1+MAX_LABEL_NUMBER)]*(1000.0f/(float)(labelNum-1));*/

	//				currentProb=hostDetectionResult[offset*(1+MAX_LABEL_NUMBER)+1+currentShowingLabel];
	//			/*	if (currentProb<0.1)
	//				{
	//					continue;
	//				}*/
	//				currentLabelIndex=currentProb*1000.0f;
	//				//cout<<j<<" "<<i<<" "<<currentLabelIndex<<endl;
	//				if (currentLabelIndex==1000)
	//				{
	//					currentLabelIndex=999;
	//				}
	//				//circle(probImg,Point(i,j),3,Scalar(colorIndex[currentLabelIndex][2]*255,colorIndex[currentLabelIndex][1]*255,colorIndex[currentLabelIndex][0]*255));
	//				probImg.at<Vec3b>(j,i).val[0]=rt->colorIndex[currentLabelIndex][2]*255.0*blendPara+
	//					probImg.at<Vec3b>(j,i).val[0]*(1-blendPara);
	//				probImg.at<Vec3b>(j,i).val[1]=rt->colorIndex[currentLabelIndex][1]*255.0*blendPara+
	//					probImg.at<Vec3b>(j,i).val[1]*(1-blendPara);
	//				probImg.at<Vec3b>(j,i).val[2]=rt->colorIndex[currentLabelIndex][0]*255.0*blendPara+
	//					probImg.at<Vec3b>(j,i).val[2]*(1-blendPara);
	//				//	delete []feature;
	//			}
	//		}
	//		namedWindow("labeled results");
	//		imshow("labeled results",probImg);
	//		char c=waitKey();
	//	}
	//}

	//2 initialization
	float maxiMumProb[30];
	int maximumPos[30];
	int cind;
	int currentLabel;
	//find the maximum points and initialize
	for (int i=0;i<labelNum;i++)
	{
		maxiMumProb[i]=0;
	}
	for (int i=0;i<depthIMG.cols*depthIMG.rows;i++)
	{
		cind=i*(1+MAX_LABEL_NUMBER);
		currentLabel=hostDetectionResult[cind];
		if (currentLabel<labelNum-1)
		{

			if (hostDetectionResult[cind+currentLabel+1]>maxiMumProb[currentLabel])
			{
				maxiMumProb[currentLabel]=hostDetectionResult[cind+currentLabel+1];
				maximumPos[currentLabel]=i;
			}
		}
	}
	int finalPos[30][2];
	for (int i=0;i<labelNum-1;i++)
	{
		finalPos[i][0]=maximumPos[i]%depthIMG.cols;
		finalPos[i][1]=maximumPos[i]/depthIMG.cols;
	}

	
	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<labelNum-1;i++)
	//{
	//	c.x=maximumPos[i]%tmp.cols;
	//	c.y=maximumPos[i]/tmp.cols;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}
	
	//0 1 2 3 4 5 10 11 12 13 14 15
	//get the scale, rotation and translation
	//Note: ransac like process need to be added here
	//find the minimum pose in the database, or-----we use only 3 pose here: left, right and middle!
	//match the current pose to mean shape and find nearest neighbors in the training dataset
	//or: adding the constraints into the optimization
	vector<int> usedInd;
	usedInd.push_back(0);
	usedInd.push_back(1);
	usedInd.push_back(2);
	usedInd.push_back(3);
	usedInd.push_back(4);
	usedInd.push_back(5);
	usedInd.push_back(10);
	usedInd.push_back(11);
	usedInd.push_back(12);
	usedInd.push_back(13);
	usedInd.push_back(14);
	usedInd.push_back(15);

	
	int usedPtsNum=usedInd.size();
	Mat U=cv::Mat(usedPtsNum*2,1,CV_32FC1);
	Mat V=cv::Mat(usedPtsNum*2,4,CV_32FC1);
	Mat globalTransformation=cv::Mat(4,1,CV_32FC1);
	for (int i=0;i<usedPtsNum;i++)
	{
		cind=usedIndList[usedInd[i]];
		U.at<float>(i,0)=finalPos[usedInd[i]][0];
		U.at<float>(usedPtsNum+i,0)=finalPos[usedInd[i]][1];
		V.at<float>(i,0)=AAM_exp->meanShape->pts[cind][0]-meanShapeCenter[0];
		V.at<float>(i,1)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		V.at<float>(i,2)=1;
		V.at<float>(i,3)=0;
		V.at<float>(i+usedPtsNum,0)=AAM_exp->meanShape->pts[cind][1]-meanShapeCenter[1];
		V.at<float>(i+usedPtsNum,1)=-AAM_exp->meanShape->pts[cind][0]+meanShapeCenter[0];
		V.at<float>(i+usedPtsNum,2)=0;
		V.at<float>(i+usedPtsNum,3)=1;
	}
	solve(V,U,globalTransformation,DECOMP_SVD);

	//for (int i=0;i<4;i++)
	//{
	//	cout<<globalTransformation.at<float>(i,0)<<" ";
	//}
	//cout<<endl;

	//ofstream out("f:\\v.txt",ios::out);
	//for (int i=0;i<V.rows;i++)
	//{
	//	for (int j=0;j<V.cols;j++)
	//	{
	//		out<<V.at<float>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();

	//ofstream out1("f:\\u.txt",ios::out);
	//for (int i=0;i<U.rows;i++)
	//{
	//	for (int j=0;j<U.cols;j++)
	//	{
	//		out1<<U.at<float>(i,j)<<" ";
	//	}
	//	out1<<endl;
	//}
	//out1.close();

	for (int i=0;i<shapeDim;i++)
	{
		AAM_exp->parameters[i]=0;
	}

	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//Mat tmp_pos=V*globalTransformation;

	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<usedPtsNum;i++)
	//{
	//	c.x=tmp_pos.at<float>(i,0);
	//	c.y=tmp_pos.at<float>(i+usedPtsNum,0);
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}

	////1.40122 0.0269533 351.572 148.054
	//AAM_exp->setInitialTranslation( 351.572,148.054);
	//AAM_exp->setInitialScale(1.40122);
	//AAM_exp->setInitialTheta(0.0269533);
	//AAM_exp->iterate_clean(colorIMG);

	//cout<<scale<<" "<<theta<<" "<<globalTransformation.at<float>(2,0)<<" "<<globalTransformation.at<float>(3,0)<<endl;
	//3 AAM: also a clear version
	AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	AAM_exp->setInitialScale(scale);
	AAM_exp->setInitialTheta(theta);
	AAM_exp->iterate_clean(colorIMG);

}

void DetectionWithAAM::ransac_optimized(int **sampledPos,int width,int height,vector<int>&finalInd)
{
	srand(GetTickCount());
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	


	vector<int> traningSapleInd;
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	traningSapleInd.push_back(i);
	//}
	RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

	bool needCheckBestIndicesList;
	//go over every traing sample
	for (int i=0;i<sampleNumberFromTrainingSet;i++)
	{
		//cout<<i<<" "<<sampleNumberFromTrainingSet<<endl;
		//sample from probability map, currently we use global maxima only: sampledPos
		for (int j=0;j<sampleNumberFromProbMap;j++)
		{
			needCheckBestIndicesList=true;

			////if we have not tested best index list or it is updated, then use this to check
			//if (needCheckBestIndicesList)
			//{
			//	

			//	//RandSample_V1(fullIndNum,2,currentInd);	//sample the current used indices
			//}

			//sample from the feature locations

			int sampleTime=0;
			
			while(sampleTime<sampleNumberFromFeature)
			//for (int k=0;k<sampleNumberFromFeature;k++)
			{
				vector<int> currentInd;
				//when k=0, using current best index to check
				if (needCheckBestIndicesList&&finalInd.size()>0)
				{
					//cout<<"checking best ind\n";
					//if (needCheckBestIndicesList)
					{
						for(int jj=0;jj<finalInd.size();jj++)
						{
							currentInd.push_back(finalInd[jj]);
						}
						getTransformationInfo(currentInd,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters
						needCheckBestIndicesList=false;
					}					
				}
				//else, using the 2 based samples
				else
				{
					RandSample_V1(fullIndNum,2,currentInd);	//sample the current used indices
					

					sampleTime++;
					//if we have already checked the optimal list and current indices are included in the optimized index set, then ignore them
					if (finalInd.size()>0)
					{	
						int containedNum=0;
						for (int ll=0;ll<currentInd.size();ll++)
						{
							if (find(finalInd.begin(),finalInd.end(),currentInd[ll])!=finalInd.end())
							{
								containedNum++;
							}
						}
						if (containedNum==currentInd.size())
						{
						//	cout<<"passing the ind which are already included\n";
							continue;
						}
					}

					getTransformationInfo(currentInd,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters
				}
			
			



				Mat newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;



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

				int inlierNum=0;
				for (int l=0;l<criticalIndNum;l++)
				{
					if (find(tmp.begin(),tmp.end(),criticalList[l])!=tmp.end())
					{
						inlierNum++;
					}
				}

				//if there are enough inliers, then check its goodness
				if (inlierNum>=w_critical)
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
						prob+=hostDetectionResult[currentIndex*(1+MAX_LABEL_NUMBER)+tmp[jj]+1];
					}
					prob/=tmp.size();

					//cout<<"tmp size: "<<tmp.size()<<endl;

					
					if (prob>bestProb)
					{
						bestProb=prob;
						bestFitTrainingSampleInd=traningSapleInd[i];

						//if it can reach here, then we need to check the best index again
						//if (needCheckBestIndicesList)
						
							if(tmp.size()>finalInd.size())
								needCheckBestIndicesList=true;
							

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

	//cout<<"the best probility: "<<bestProb<<endl;

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



void DetectionWithAAM::ransac(int **sampledPos,int width,int height,vector<int>&finalInd)
{
	srand(GetTickCount());
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	


	vector<int> traningSapleInd;
	RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

	traningSapleInd.resize(totalShapeNum);
	for (int i=0;i<totalShapeNum;i++)
	{
		traningSapleInd[i]=i;
	}

	//go over every traing sample
	for (int i=0;i<sampleNumberFromTrainingSet;i++)
	{
		if (i!=16)
		{
			continue;
		}
		//sample from probability map, currently we use global maxima only: sampledPos
		for (int j=0;j<sampleNumberFromProbMap;j++)
		{
			//sample from the feature locations
			for (int k=0;k<sampleNumberFromFeature*2;k++)
			{
				vector<int> currentInd;
				RandSample_V1(fullIndNum,2,currentInd);	//sample the current used indices
				getTransformationInfo(currentInd,sampledPos,shapes,traningSapleInd[i]);	//get transformation parameters

				Mat newU=fullTrainingPos[traningSapleInd[i]]*globalTransformation;



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

				int inlierNum=0;
				for (int l=0;l<criticalIndNum;l++)
				{
					if (find(tmp.begin(),tmp.end(),criticalList[l])!=tmp.end())
					{
						inlierNum++;
					}
				}

				//if there are enough inliers, then check its goodness
				if (inlierNum>=w_critical)
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
						prob+=hostDetectionResult[currentIndex*(1+MAX_LABEL_NUMBER)+tmp[jj]+1];
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

void DetectionWithAAM::track_NoSampling(Mat &colorIMG,Mat &depthIMG)
{
	//0: send depth and image data to GPU
	if (host_colorImage==NULL)
	{
		host_colorImage=new float[MPN];
		host_depthImage=new float[MPN];
	}
	int i,j;
	for (i=0;i<depthIMG.rows;i++)
	{
		for (j=0;j<depthIMG.cols;j++)
		{
			host_depthImage[i*depthIMG.cols+j]=depthIMG.at<float>(i,j);
			host_colorImage[i*depthIMG.cols+j]=colorIMG.at<uchar>(i,j);
		}
	}

	setData_RT_onrun(host_colorImage,host_depthImage,depthIMG.cols,depthIMG.rows);

	//1 detection, need a clear version
	predict_GPU_withDepth_clean(depthIMG.cols,depthIMG.rows,hostDetectionResult,rt->trainStyle);

	cout<<"finished detection!\n";
	//2 initialization
	float *maxiMumProb=new float[30];
	int maximumPos[30];
	int cind;
	int currentLabel;
	//find the maximum points and initialize
	for (int i=0;i<labelNum;i++)
	{
		maxiMumProb[i]=0;
	}
	for (int i=0;i<depthIMG.cols*depthIMG.rows;i++)
	{
		cind=i*(1+MAX_LABEL_NUMBER);
		currentLabel=hostDetectionResult[cind];
		if (currentLabel<labelNum-1)
		{

			if (hostDetectionResult[cind+currentLabel+1]>maxiMumProb[currentLabel])
			{
				maxiMumProb[currentLabel]=hostDetectionResult[cind+currentLabel+1];
				maximumPos[currentLabel]=i;
			}
		}
	}


	int **finalPos;
	finalPos=new int*[30];
	for (int i=0;i<30;i++)
	{
		finalPos[i]=new int [2];
	}
	for (int i=0;i<labelNum-1;i++)
	{
		finalPos[i][0]=maximumPos[i]%depthIMG.cols;
		finalPos[i][1]=maximumPos[i]/depthIMG.cols;

		fullDetectedPos.at<float>(i,0)=finalPos[i][0];
		fullDetectedPos.at<float>(i+fullIndNum,0)=finalPos[i][1];
	}


	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<labelNum-1;i++)
	//{
	//	c.x=maximumPos[i]%tmp.cols;
	//	c.y=maximumPos[i]/tmp.cols;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}


	vector<int> finalInd;


	DWORD dwStart, dwStop;  
	DWORD d_totalstart;
	//sampleNumberFromTrainingSet=80;
	//sampleNumberFromFeature=100;
	dwStart=GetTickCount();
	ransac_noSampling(finalPos,depthIMG.cols,depthIMG.rows,finalInd,maxiMumProb);
	//ransac(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	//ransac_optimized(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	//ransac_distance(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	dwStop=GetTickCount();
	cout<<"sampling time: "<<(dwStop-dwStart)<<endl;



	//cout<<"sampled feature num: "<<finalInd.size()<<" which are"<<endl;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	cout<<finalInd[i]<<" ";
	//}
	//cout<<endl;

	//Mat tmp1=colorIMG.clone();
	//Point c1;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	c1.x=finalPos[finalInd[i]][0];
	//	c1.y=finalPos[finalInd[i]][1];
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp1,c1,5,255);

	//}

	//for (int i=0;i<labelNum-1;i++)
	//{
	//	c1.x=finalPos[i][0];
	//	c1.y=finalPos[i][1];
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp1,c1,2,255);

	//}
	//namedWindow("1");
	//imshow("1",tmp1);
	//waitKey();
	//return;

	Mat newU;
	float bestDistance=1000000;
	//float 
	//now check distance
	for (int i=0;i<totalShapeNum;i++)
	{
		getTransformationInfo(finalInd,finalPos,shapes,i);	//get transformation parameters
		newU=fullTrainingPos[i]*globalTransformation;
		float currentDis=0;
		for (int i=0;i<finalInd.size();i++)
		{
			currentDis+=(newU.at<float>(finalInd[i],0)-finalPos[finalInd[i]][0])*
				(newU.at<float>(finalInd[i],0)-finalPos[finalInd[i]][0])+
				(newU.at<float>(finalInd[i]+fullIndNum,0)-finalPos[finalInd[i]][1])*
				(newU.at<float>(finalInd[i]+fullIndNum,0)-finalPos[finalInd[i]][1]);	
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
	int minInd=bestFitTrainingSampleInd;
	globalTransformation=globalTransformation_optimal;

	/////////////////////////////////old bad sampling//////////////////////////////////////////////////////////////
	////sampling first
	//vector<int> traningSapleInd;
	//RandSample_V1(totalShapeNum,10,traningSapleInd);


	////0 1 2 3 4 5 10 11 12 13 14 15
	////get the scale, rotation and translation
	////Note: ransac like process need to be added here
	////find the minimum pose in the database, or-----we use only 3 pose here: left, right and middle!
	////match the current pose to mean shape and find nearest neighbors in the training dataset
	////or: adding the constraints into the optimization
	//vector<int> usedInd;
	//getVisibleInd(finalPos,usedInd,shapes,0);
	//int usedPtsNum=usedInd.size();


	/////////////////////////////////////////////
	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<usedPtsNum;i++)
	//{
	//	c.x=finalPos[usedInd[i]][0];
	//	c.y=finalPos[usedInd[i]][1];
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp,c,5,255);
	//	
	//}
	//namedWindow("1");
	//imshow("1",tmp);
	//waitKey();
	//return;
	//////////////////////////////////////////////

	////then find the nearest neighbors
	//float minimalDistance=100000000;
	//int minInd;
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	getTransformationInfo(usedInd,finalPos,shapes,i);
	//	Mat newU=fullTrainingPos[i]*globalTransformation;


	//	//Mat tmpImg=colorIMG.clone();
	//	//Point c;
	//	//for (int i=0;i<fullIndNum;i++)
	//	//{
	//	//	c.x=newU.at<float>(usedIndList[i],0);
	//	//	c.y=newU.at<float>(usedIndList[i]+fullIndNum,0);
	//	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	//	circle(tmpImg,c,5,255);
	//	//	
	//	//}
	//	//namedWindow("1");
	//	//imshow("1",tmpImg);
	//	//waitKey();

	//	vector<int> tmp;
	//	float distance=0;
	//	for (int j=0;j<usedPtsNum;j++)
	//	{
	//		cind=usedInd[j];
	//		distance+=(newU.at<float>(cind,0)-fullDetectedPos.at<float>(cind,0))*(newU.at<float>(cind,0)-fullDetectedPos.at<float>(cind,0))+
	//			(newU.at<float>(cind+fullIndNum,0)-fullDetectedPos.at<float>(cind+fullIndNum,0))*(newU.at<float>(cind+fullIndNum,0)-fullDetectedPos.at<float>(cind+fullIndNum,0));
	//	}
	//	if (distance<minimalDistance)
	//	{
	//		minimalDistance=distance;
	//		minInd=i;
	//	}
	//}

	//getTransformationInfo(usedInd,finalPos,shapes,minInd);

	//cout<<"visualization!\n";

	////Mat newU=fullTrainingPos[minInd]*globalTransformation;
	////Mat tmpImg=colorIMG.clone();
	////Point c;
	////for (int i=0;i<fullIndNum;i++)
	////{
	////	c.x=newU.at<float>(usedIndList[i],0);
	////	c.y=newU.at<float>(usedIndList[i]+fullIndNum,0);
	////	//c.x=V.at<float>(i,0)*1.0f/scale;
	////	//c.y=V.at<float>(i,1)*1.0f/scale;
	////	circle(tmpImg,c,5,255);
	////	
	////}
	////namedWindow("1");
	////imshow("1",tmpImg);
	////waitKey();
	//return;
	//////////////////////////////////////////////////////////////////////////////////

	for (int i=0;i<shapeDim;i++)
	{
		AAM_exp->parameters[i]=eigenVectors.at<double>(i,minInd);
	}

	//usedInd.push_back(0);
	//usedInd.push_back(1);
	//usedInd.push_back(2);
	//usedInd.push_back(3);
	//usedInd.push_back(4);
	//usedInd.push_back(5);
	//usedInd.push_back(10);
	//usedInd.push_back(11);
	//usedInd.push_back(12);
	//usedInd.push_back(13);
	//usedInd.push_back(14);
	//usedInd.push_back(15);



	//for (int i=0;i<4;i++)
	//{
	//	cout<<globalTransformation.at<float>(i,0)<<" ";
	//}
	//cout<<endl;

	//ofstream out("f:\\v.txt",ios::out);
	//for (int i=0;i<V.rows;i++)
	//{
	//	for (int j=0;j<V.cols;j++)
	//	{
	//		out<<V.at<float>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();

	//ofstream out1("f:\\u.txt",ios::out);
	//for (int i=0;i<U.rows;i++)
	//{
	//	for (int j=0;j<U.cols;j++)
	//	{
	//		out1<<U.at<float>(i,j)<<" ";
	//	}
	//	out1<<endl;
	//}
	//out1.close();



	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//Mat tmp_pos=V*globalTransformation;

	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<usedPtsNum;i++)
	//{
	//	c.x=tmp_pos.at<float>(i,0);
	//	c.y=tmp_pos.at<float>(i+usedPtsNum,0);
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}

	////1.40122 0.0269533 351.572 148.054
	//AAM_exp->setInitialTranslation( 351.572,148.054);
	//AAM_exp->setInitialScale(1.40122);
	//AAM_exp->setInitialTheta(0.0269533);
	//AAM_exp->iterate_clean(colorIMG);

	//cout<<scale<<" "<<theta<<" "<<globalTransformation.at<float>(2,0)<<" "<<globalTransformation.at<float>(3,0)<<endl;
	//3 AAM: also a clear version
	AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	AAM_exp->setInitialScale(scale);
	AAM_exp->setInitialTheta(theta);
	AAM_exp->iterate_clean(colorIMG);

}

void DetectionWithAAM::ransac_noSampling(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb)
{
	srand(time(NULL));
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	


	vector<int> traningSapleInd;
	RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

	//use all the training sample
	/*traningSapleInd.resize(totalShapeNum);
	for (int i=0;i<totalShapeNum;i++)
	{
		traningSapleInd[i]=i;
	}*/

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
						prob+=hostDetectionResult[currentIndex*(1+MAX_LABEL_NUMBER)+tmp[jj]+1]/maximumProb[tmp[jj]];
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

void DetectionWithAAM::ransac_noSampling_multiModes(int **sampledPos,int width,int height,vector<int>&finalInd,float *maximumProb)
{
		srand(time(NULL));
	float bestProb=0;
	
//	finalInd.resize(labelNum-1);

	


	vector<int> traningSapleInd;
	RandSample_V1(totalShapeNum,sampleNumberFromTrainingSet,traningSapleInd);

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
						prob+=hostDetectionResult[currentIndex*(1+MAX_LABEL_NUMBER)+tmp[jj]+1]/maximumProb[tmp[jj]];
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
	{
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


void DetectionWithAAM::track_sampling(Mat &colorIMG,Mat &depthIMG)
{
	//0: send depth and image data to GPU
	if (host_colorImage==NULL)
	{
		host_colorImage=new float[MPN];
		host_depthImage=new float[MPN];
	}
	int i,j;
	for (i=0;i<depthIMG.rows;i++)
	{
		for (j=0;j<depthIMG.cols;j++)
		{
			host_depthImage[i*depthIMG.cols+j]=depthIMG.at<float>(i,j);
			host_colorImage[i*depthIMG.cols+j]=colorIMG.at<uchar>(i,j);
		}
	}

	setData_RT_onrun(host_colorImage,host_depthImage,depthIMG.cols,depthIMG.rows);

	//1 detection, need a clear version
	predict_GPU_withDepth_clean(depthIMG.cols,depthIMG.rows,hostDetectionResult,rt->trainStyle);

	cout<<"finished detection!\n";
	//2 initialization
	float maxiMumProb[30];
	int maximumPos[30];
	int cind;
	int currentLabel;
	//find the maximum points and initialize
	for (int i=0;i<labelNum;i++)
	{
		maxiMumProb[i]=0;
	}
	for (int i=0;i<depthIMG.cols*depthIMG.rows;i++)
	{
		cind=i*(1+MAX_LABEL_NUMBER);
		currentLabel=hostDetectionResult[cind];
		if (currentLabel<labelNum-1)
		{

			if (hostDetectionResult[cind+currentLabel+1]>maxiMumProb[currentLabel])
			{
				maxiMumProb[currentLabel]=hostDetectionResult[cind+currentLabel+1];
				maximumPos[currentLabel]=i;
			}
		}
	}
	int **finalPos;
	finalPos=new int*[30];
	for (int i=0;i<30;i++)
	{
		finalPos[i]=new int [2];
	}
	for (int i=0;i<labelNum-1;i++)
	{
		finalPos[i][0]=maximumPos[i]%depthIMG.cols;
		finalPos[i][1]=maximumPos[i]/depthIMG.cols;

		fullDetectedPos.at<float>(i,0)=finalPos[i][0];
		fullDetectedPos.at<float>(i+fullIndNum,0)=finalPos[i][1];
	}

	
	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<labelNum-1;i++)
	//{
	//	c.x=maximumPos[i]%tmp.cols;
	//	c.y=maximumPos[i]/tmp.cols;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}
	
	
	vector<int> finalInd;


	DWORD dwStart, dwStop;  
	DWORD d_totalstart;
	//sampleNumberFromTrainingSet=80;
	//sampleNumberFromFeature=100;
	dwStart=GetTickCount();
	//ransac(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	ransac_optimized(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	dwStop=GetTickCount();
	cout<<"sampling time: "<<(dwStop-dwStart)<<endl;

	int minInd=bestFitTrainingSampleInd;
	globalTransformation=globalTransformation_optimal;

	//cout<<"sampled feature num: "<<finalInd.size()<<" which are"<<endl;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	cout<<finalInd[i]<<" ";
	//}
	//cout<<endl;

	Mat tmp1=colorIMG.clone();
	Point c1;
	for (int i=0;i<finalInd.size();i++)
	{
		c1.x=finalPos[finalInd[i]][0];
		c1.y=finalPos[finalInd[i]][1];
		//c.x=V.at<float>(i,0)*1.0f/scale;
		//c.y=V.at<float>(i,1)*1.0f/scale;
		circle(tmp1,c1,5,255);

	}

	for (int i=0;i<labelNum-1;i++)
	{
		c1.x=finalPos[i][0];
		c1.y=finalPos[i][1];
		//c.x=V.at<float>(i,0)*1.0f/scale;
		//c.y=V.at<float>(i,1)*1.0f/scale;
		circle(tmp1,c1,2,255);

	}
	namedWindow("1");
	imshow("1",tmp1);
	waitKey();
	return;


	/////////////////////////////////old bad sampling//////////////////////////////////////////////////////////////
	////sampling first
	//vector<int> traningSapleInd;
	//RandSample_V1(totalShapeNum,10,traningSapleInd);


	////0 1 2 3 4 5 10 11 12 13 14 15
	////get the scale, rotation and translation
	////Note: ransac like process need to be added here
	////find the minimum pose in the database, or-----we use only 3 pose here: left, right and middle!
	////match the current pose to mean shape and find nearest neighbors in the training dataset
	////or: adding the constraints into the optimization
	//vector<int> usedInd;
	//getVisibleInd(finalPos,usedInd,shapes,0);
	//int usedPtsNum=usedInd.size();


	/////////////////////////////////////////////
	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<usedPtsNum;i++)
	//{
	//	c.x=finalPos[usedInd[i]][0];
	//	c.y=finalPos[usedInd[i]][1];
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp,c,5,255);
	//	
	//}
	//namedWindow("1");
	//imshow("1",tmp);
	//waitKey();
	//return;
	//////////////////////////////////////////////

	////then find the nearest neighbors
	//float minimalDistance=100000000;
	//int minInd;
	//for (int i=0;i<totalShapeNum;i++)
	//{
	//	getTransformationInfo(usedInd,finalPos,shapes,i);
	//	Mat newU=fullTrainingPos[i]*globalTransformation;


	//	//Mat tmpImg=colorIMG.clone();
	//	//Point c;
	//	//for (int i=0;i<fullIndNum;i++)
	//	//{
	//	//	c.x=newU.at<float>(usedIndList[i],0);
	//	//	c.y=newU.at<float>(usedIndList[i]+fullIndNum,0);
	//	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	//	circle(tmpImg,c,5,255);
	//	//	
	//	//}
	//	//namedWindow("1");
	//	//imshow("1",tmpImg);
	//	//waitKey();

	//	vector<int> tmp;
	//	float distance=0;
	//	for (int j=0;j<usedPtsNum;j++)
	//	{
	//		cind=usedInd[j];
	//		distance+=(newU.at<float>(cind,0)-fullDetectedPos.at<float>(cind,0))*(newU.at<float>(cind,0)-fullDetectedPos.at<float>(cind,0))+
	//			(newU.at<float>(cind+fullIndNum,0)-fullDetectedPos.at<float>(cind+fullIndNum,0))*(newU.at<float>(cind+fullIndNum,0)-fullDetectedPos.at<float>(cind+fullIndNum,0));
	//	}
	//	if (distance<minimalDistance)
	//	{
	//		minimalDistance=distance;
	//		minInd=i;
	//	}
	//}

	//getTransformationInfo(usedInd,finalPos,shapes,minInd);

	//cout<<"visualization!\n";

	////Mat newU=fullTrainingPos[minInd]*globalTransformation;
	////Mat tmpImg=colorIMG.clone();
	////Point c;
	////for (int i=0;i<fullIndNum;i++)
	////{
	////	c.x=newU.at<float>(usedIndList[i],0);
	////	c.y=newU.at<float>(usedIndList[i]+fullIndNum,0);
	////	//c.x=V.at<float>(i,0)*1.0f/scale;
	////	//c.y=V.at<float>(i,1)*1.0f/scale;
	////	circle(tmpImg,c,5,255);
	////	
	////}
	////namedWindow("1");
	////imshow("1",tmpImg);
	////waitKey();
	//return;
	//////////////////////////////////////////////////////////////////////////////////

	for (int i=0;i<shapeDim;i++)
	{
		AAM_exp->parameters[i]=eigenVectors.at<double>(i,minInd);
	}

	//usedInd.push_back(0);
	//usedInd.push_back(1);
	//usedInd.push_back(2);
	//usedInd.push_back(3);
	//usedInd.push_back(4);
	//usedInd.push_back(5);
	//usedInd.push_back(10);
	//usedInd.push_back(11);
	//usedInd.push_back(12);
	//usedInd.push_back(13);
	//usedInd.push_back(14);
	//usedInd.push_back(15);

	
	
	//for (int i=0;i<4;i++)
	//{
	//	cout<<globalTransformation.at<float>(i,0)<<" ";
	//}
	//cout<<endl;

	//ofstream out("f:\\v.txt",ios::out);
	//for (int i=0;i<V.rows;i++)
	//{
	//	for (int j=0;j<V.cols;j++)
	//	{
	//		out<<V.at<float>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();

	//ofstream out1("f:\\u.txt",ios::out);
	//for (int i=0;i<U.rows;i++)
	//{
	//	for (int j=0;j<U.cols;j++)
	//	{
	//		out1<<U.at<float>(i,j)<<" ";
	//	}
	//	out1<<endl;
	//}
	//out1.close();



	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//Mat tmp_pos=V*globalTransformation;

	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<usedPtsNum;i++)
	//{
	//	c.x=tmp_pos.at<float>(i,0);
	//	c.y=tmp_pos.at<float>(i+usedPtsNum,0);
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}

	////1.40122 0.0269533 351.572 148.054
	//AAM_exp->setInitialTranslation( 351.572,148.054);
	//AAM_exp->setInitialScale(1.40122);
	//AAM_exp->setInitialTheta(0.0269533);
	//AAM_exp->iterate_clean(colorIMG);

	//cout<<scale<<" "<<theta<<" "<<globalTransformation.at<float>(2,0)<<" "<<globalTransformation.at<float>(3,0)<<endl;
	//3 AAM: also a clear version
	AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	AAM_exp->setInitialScale(scale);
	AAM_exp->setInitialTheta(theta);



	AAM_exp->iterate_clean(colorIMG);

}

void DetectionWithAAM::track_sampling_CPU(Mat &colorIMG,Mat &depthIMG)
{
	//0: send depth and image data to GPU
	if (host_colorImage==NULL)
	{
		host_colorImage=new float[MPN];
		host_depthImage=new float[MPN];
	}
	int i,j;
	for (i=0;i<depthIMG.rows;i++)
	{
		for (j=0;j<depthIMG.cols;j++)
		{
			host_depthImage[i*depthIMG.cols+j]=depthIMG.at<float>(i,j);
			host_colorImage[i*depthIMG.cols+j]=colorIMG.at<uchar>(i,j);
		}
	}

	setData_RT_onrun(host_colorImage,host_depthImage,depthIMG.cols,depthIMG.rows);

	//1 detection, need a clear version
	predict_GPU_withDepth_clean(depthIMG.cols,depthIMG.rows,hostDetectionResult,rt->trainStyle);

	cout<<"finished detection!\n";

	//then, find all the modes
	if (isFindModes)
	{
		rt->findProbModes(depthIMG.cols,depthIMG.rows,hostDetectionResult);
	}
	
	//2 initialization
	float maxiMumProb[30];
	int maximumPos[30];
	int cind;
	int currentLabel;
	//find the maximum points and initialize
	for (int i=0;i<labelNum;i++)
	{
		maxiMumProb[i]=0;
	}
	for (int i=0;i<depthIMG.cols*depthIMG.rows;i++)
	{
		cind=i*(1+MAX_LABEL_NUMBER);
		currentLabel=hostDetectionResult[cind];
		if (currentLabel<labelNum-1)
		{

			if (hostDetectionResult[cind+currentLabel+1]>maxiMumProb[currentLabel])
			{
				maxiMumProb[currentLabel]=hostDetectionResult[cind+currentLabel+1];
				maximumPos[currentLabel]=i;
			}
		}
	}
	int **finalPos;
	finalPos=new int*[30];
	for (int i=0;i<30;i++)
	{
		finalPos[i]=new int [2];
	}
	for (int i=0;i<labelNum-1;i++)
	{
		finalPos[i][0]=maximumPos[i]%depthIMG.cols;
		finalPos[i][1]=maximumPos[i]/depthIMG.cols;

		fullDetectedPos.at<float>(i,0)=finalPos[i][0];
		fullDetectedPos.at<float>(i+fullIndNum,0)=finalPos[i][1];
	}

	

	//Mat tmp=colorIMG.clone();
	//Point c;
	////for (int i=0;i<labelNum-1;i++)
	////{
	////	c.x=maximumPos[i]%tmp.cols;
	////	c.y=maximumPos[i]/tmp.cols;
	////	circle(tmp,c,5,255);
	////	
	////}

	//for (int i=0;i<1;i++)
	//{
	//	for (int j=0;j<rt->centerList[i].size();j++)
	//	{
	//		circle(tmp,rt->centerList[i][j],5,255);
	//	}
	//	

	//}
	//namedWindow("1");
	//imshow("1",tmp);
	//waitKey();


	vector<int> finalInd;


	DWORD dwStart, dwStop;  
	DWORD d_totalstart;
	//sampleNumberFromTrainingSet=80;
	//sampleNumberFromFeature=100;
	while(1)
	{

	
	dwStart=GetTickCount();
	//ransac(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	ransac_noSampling(finalPos,depthIMG.cols,depthIMG.rows,finalInd,maxiMumProb);
	//ransac_optimized(finalPos,depthIMG.cols,depthIMG.rows,finalInd);
	dwStop=GetTickCount();
	cout<<"sampling time: "<<(dwStop-dwStart)<<endl;

//	cout<<"checking modes with bestInd"<<bestFitTrainingSampleInd<<"\n";

	int minInd=bestFitTrainingSampleInd;
	globalTransformation=globalTransformation_optimal;
	
	if (isFindModes)
	{
		//then, decide which mode to use
		Mat newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
		float currentDis,currentBestDis;
		int bestInd;
		for (int ii=0;ii<fullIndNum;ii++)
		{
			//cout<<"checking label "<<ii<<endl;
			//cout<<"current mode num: "<<rt->centerList[ii].size()<<endl;
			//cout<<newU.at<double>(ii,0)<<" "<<newU.at<double>(ii+fullIndNum,0)<<endl;
			currentBestDis=10000000;
			for (int jj=0;jj<rt->centerList[ii].size();jj++)
			{
				
				currentDis=(newU.at<float>(ii,0)-rt->centerList[ii][jj].x)*(newU.at<float>(ii,0)-rt->centerList[ii][jj].x)+
					(newU.at<float>(ii+fullIndNum,0)-rt->centerList[ii][jj].y)*(newU.at<float>(ii+fullIndNum,0)-rt->centerList[ii][jj].y);

			//	cout<<newU.at<float>(ii,0)<<" "<<newU.at<float>(ii+fullIndNum,0)<<" "<<rt->centerList[ii][jj].x<<" "<<rt->centerList[ii][jj].y<<endl;
			//	cout<<"checking label "<<ii<<" current distance: "<<currentDis<<endl;
				if (currentDis<currentBestDis)
				{
					currentBestDis=currentDis;
					bestInd=jj;
				}
			}
			//cout<<ii<<" "<<"current best distance: "<<currentBestDis<<" bestInd: "<<bestInd<<" threshold: "<<window_threshold_large<<endl;
			if (currentBestDis<window_threshold_large)
			{
				if (find(finalInd.begin(),finalInd.end(),ii)==finalInd.end())
				{
					finalInd.push_back(ii);
					
				}
				finalPos[ii][0]=rt->centerList[ii][bestInd].x;
				finalPos[ii][1]=rt->centerList[ii][bestInd].y;
			}

		}
	}
	


	//cout<<"sampled feature num: "<<finalInd.size()<<" which are"<<endl;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	cout<<finalInd[i]<<" ";
	//}
	//cout<<endl;

	//Mat tmp1=colorIMG.clone();
	//cvtColor( tmp1, tmp1, CV_GRAY2RGB );
	//Scalar s,s1;
	//s.val[0]=s.val[1]=0;
	//s.val[2]=255;
	//Point c1;
	//for (int i=0;i<finalInd.size();i++)
	//{
	//	c1.x=finalPos[finalInd[i]][0];
	//	c1.y=finalPos[finalInd[i]][1];
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp1,c1,5,s);

	//}

	//Mat newU=fullTrainingPos[bestFitTrainingSampleInd]*globalTransformation;
	////for (int i=0;i<labelNum-1;i++)
	////{
	////	c1.x=finalPos[i][0];
	////	c1.y=finalPos[i][1];
	////	//c.x=V.at<float>(i,0)*1.0f/scale;
	////	//c.y=V.at<float>(i,1)*1.0f/scale;
	////	circle(tmp1,c1,2,255);

	////}
	//namedWindow("maximal");
	//imshow("maximal",tmp1);
	//waitKey();
	//return;


	//	for (int i=0;i<shape_dim;i++)
	//	{
	//		s_weight[i]=0;
	//	}

	for (int i=0;i<shapeDim;i++)
	{
		AAM_exp->s_weight[i]=eigenVectors.at<double>(i,minInd);
	}


	float scale=sqrtf(globalTransformation.at<float>(0,0)*globalTransformation.at<float>(0,0)+
		globalTransformation.at<float>(1,0)*globalTransformation.at<float>(1,0));
	float theta=atan2(globalTransformation.at<float>(1,0),globalTransformation.at<float>(0,0));

	//////////////////////////then setup the probability maps//////////////////////////////////////////////////////
	//if we use all the detection results, seems not good when pose change
	//rt->setupProbMaps(depthIMG.cols,depthIMG.rows,hostDetectionResult);
	//so, we try all the visible node now
	rt->setupProbMaps(depthIMG.cols,depthIMG.rows,hostDetectionResult,finalInd);

	

	//return true;
	//////////////////////////////////////////////////////////////////////////////////////////////////

	//Mat tmp_pos=V*globalTransformation;

	//Mat tmp=colorIMG.clone();
	//Point c;
	//for (int i=0;i<usedPtsNum;i++)
	//{
	//	c.x=tmp_pos.at<float>(i,0);
	//	c.y=tmp_pos.at<float>(i+usedPtsNum,0);
	//	//c.x=V.at<float>(i,0)*1.0f/scale;
	//	//c.y=V.at<float>(i,1)*1.0f/scale;
	//	circle(tmp,c,5,255);
	//	namedWindow("1");
	//	imshow("1",tmp);
	//	waitKey();
	//}

	////1.40122 0.0269533 351.572 148.054
	//AAM_exp->setInitialTranslation( 351.572,148.054);
	//AAM_exp->setInitialScale(1.40122);
	//AAM_exp->setInitialTheta(0.0269533);
	//AAM_exp->iterate_clean(colorIMG);

	//cout<<scale<<" "<<theta<<" "<<globalTransformation.at<float>(2,0)<<" "<<globalTransformation.at<float>(3,0)<<endl;
	//3 AAM: also a clear version
	AAM_exp->setInitialTranslation( globalTransformation.at<float>(2,0),globalTransformation.at<float>(3,0));
	AAM_exp->setInitialScale(scale);
	AAM_exp->setInitialTheta(theta);
	
	//set up the sampled feature locations
	for (int ll=0;ll<finalInd.size();ll++)
	{
		AAM_exp->shapeSample[ll][0]=finalPos[finalInd[ll]][0];
		AAM_exp->shapeSample[ll][1]=finalPos[finalInd[ll]][1];
	}

	if( AAM_exp->iterate_clean_CPU(colorIMG,rt))
		break;
	}

}

void DetectionWithAAM::getTransformationInfo(vector<int> &InputInd,int **finalPos,Mat &shapeList,int shapeInd)
{
	int usedPtsNum=InputInd.size();
	Mat U=cv::Mat(usedPtsNum*2,1,CV_32FC1);
	Mat V=cv::Mat(usedPtsNum*2,4,CV_32FC1);
	//Mat globalTransformationLocal=cv::Mat(4,1,CV_32FC1);
	int cind;
	for (int i=0;i<usedPtsNum;i++)
	{
		cind=usedIndList[InputInd[i]];
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
	solve(V,U,globalTransformation,DECOMP_SVD);

	//for (int i=0;i<globalTransformationLocal.rows;i++)
	//{
	//	globalTransformation.at<float>(i,0)=globalTransformationLocal.at<float>(i,0);
	//}
}

void DetectionWithAAM::getTransformationInfo_optimize(vector<int> &InputInd,int **finalPos,Mat &shapeList,int shapeInd)
{
	int usedPtsNum=InputInd.size();
	Mat U=cv::Mat(usedPtsNum*2,1,CV_32FC1);
	Mat V=cv::Mat(usedPtsNum*2,4,CV_32FC1);
	Mat globalTransformationLocal=cv::Mat(4,1,CV_32FC1);
	int cind;
	for (int i=0;i<usedPtsNum;i++)
	{
		cind=usedIndList[InputInd[i]];
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
	solve(V,U,globalTransformationLocal,DECOMP_SVD);

	Mat newU=fullTrainingPos[shapeInd]*globalTransformationLocal;

	vector<int> tmp;
	float distance;
	float threshold=20;
	for (int i=0;i<fullIndNum;i++)
	{
		distance=(newU.at<float>(i,0)-fullDetectedPos.at<float>(i,0))*(newU.at<float>(i,0)-fullDetectedPos.at<float>(i,0))+
			(newU.at<float>(i+fullIndNum,0)-fullDetectedPos.at<float>(i+fullIndNum,0))*(newU.at<float>(i+fullIndNum,0)-fullDetectedPos.at<float>(i+fullIndNum,0));
		if (distance<threshold)
		{
			tmp.push_back(i);
		}
	}
	if (tmp.size()>InputInd.size())
	{
		InputInd.clear();
		for (int i=0;i<tmp.size();i++)
		{
			InputInd.push_back(tmp[i]);
		}
	}
}

void DetectionWithAAM::getVisibleInd(int **finalPos,vector<int> &outputInd,Mat &shapeList,int shapeInd)
{
	vector<int> currentInd[50];
	int sampleTime=20;

	for (int i=0;i<sampleTime;i++)
	{
		RandSample_V1(fullIndNum,2,currentInd[i]);
		getTransformationInfo_optimize(currentInd[i],finalPos,shapeList,shapeInd);
	}
	
	int maxNum=0;
	int maxInd;
	for (int i=0;i<sampleTime;i++)
	{
		if (currentInd[i].size()>maxNum)
		{
			maxNum=currentInd[i].size();
			maxInd=i;
		}
	}

	outputInd.clear();
	for (int i=0;i<currentInd[maxInd].size();i++)
	{
		outputInd.push_back(currentInd[maxInd][i]);
	}
	
	
	//Mat newPos=
}