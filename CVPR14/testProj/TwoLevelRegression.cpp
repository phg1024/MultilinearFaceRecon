#include "TwoLevelRegression.h"
#include "GRandom.h"

#include "codetimer.h"

TwoLevelRegression::TwoLevelRegression(bool _isRot)
{
	showRes=false;
	isRot=_isRot;
}

void TwoLevelRegression::pridict(Mat &img, Shape &s,float givenS,float givenTx,float givenTy, bool showSingleStep)
{
	
	for(int i=0;i<T;i++)
	{
		//update translation
		s.syntheorize();
		s.estimateTrans_local(refShape);
		s.setScaleTranslatrion(givenS,givenTx,givenTy);
		if(showSingleStep)
		{
			char curName[50];
			sprintf(curName,"%d",i+1);
			namedWindow(curName);
			moveWindow(curName,20,20);
			s.visualizePts(curName);
			
		}
		for(int j=0;j<K;j++)
		{
			int fernID=i*K+j;
			ferns[fernID].pridict_directAdd(s);
		}		
	}

	
	if(showSingleStep)
		waitKey();
}

void TwoLevelRegression::pridict(Mat &img, Shape &s, bool showSingleStep)
{
	
	for(int i=0;i<T;i++)
	{
		//update translation
		s.syntheorize();
	//	cout<<"local "<<i<<" ";
		s.estimateTrans_local(refShape);
		if(showSingleStep)
		{
			char curName[50];
			sprintf(curName,"%d",i+1);
			namedWindow(curName);
			moveWindow(curName,20,20);
			s.visualizePts(curName);
			
		}
		//cout<<" 2nd regressor ";
		for(int j=0;j<K;j++)
		{
			int fernID=i*K+j;
			ferns[fernID].pridict_directAdd(s);
		}		
	}
	//cout<<endl;

	
	if(showSingleStep)
		waitKey();
}


void TwoLevelRegression::pridict(Mat &img,const int sampleNum, char *GTPts)
{
	;


}

float TwoLevelRegression::estimateScale(Shape &curShape, Rect curFaceRect)
{
	//step 1: obtain global rough scale first using x-axis eyebrow distance
	//Rect curRect=boundingRect(refShape.pts);
	float s_roough=(float)curFaceRect.width/(float)refFaceWidth;

	return s_roough;

	//then use this scale to obtain a new scale
	int eyeBrowInd1=17;
	int eyeBrowInd2=26;
	int minY=1000;
	for(int i=eyeBrowInd1;i<=eyeBrowInd2;i++)
		if(curShape.pts[i].y<minY)
			minY=curShape.pts[i].y;
	float curHeight=abs(curShape.pts[30].y-minY);

	float curFaceWidth=abs(curShape.pts[17].x-curShape.pts[26].x);;
	float s1=(float)curFaceRect.height*0.5/curHeight;
	float s2=s_roough;

	//cout<<curFaceRect.width*0.5f<<" "<<curHeight<<" ";
	cout<<s1<<" "<<s2<<endl;
//
//	return s1;
	return s1>s2?s2:s1;
}

void TwoLevelRegression::estimateST(Shape &curShape,Rect curFaceRect,float &s,float &tx,float &ty)
{
	//step 1: obtain global rough scale first using x-axis eyebrow distance
	//Rect curRect=boundingRect(refShape.pts);
	float s_rough=(float)curFaceRect.width/(float)refFaceWidth;
	s=s_rough;

	if(0)
	{
		Point2f nosePts=refShape.pts[30]*s;
		//float tx,ty;
		tx=curFaceRect.x+curFaceRect.width/2-nosePts.x;
		ty=curFaceRect.y+curFaceRect.width/2-nosePts.y;
	}
	else
	{
		Point2f shapeCenter;
		int ptsNUm=0;
		shapeCenter.x=shapeCenter.y=0;
		for(int i=17;i<=67;i++)
		{
			shapeCenter+=curShape.pts[i]*s;
			ptsNUm++;
		}
		shapeCenter.x/=ptsNUm;
		shapeCenter.y/=ptsNUm;

		tx=curFaceRect.x+curFaceRect.width/2-shapeCenter.x;
		ty=curFaceRect.y+curFaceRect.width/2-shapeCenter.y;
	}

		
}


Mat TwoLevelRegression::pridict_real(IplImage *img,const int sampleNum)
{
	//face detection first
	cout<<"detecting face\n";
	d.showFace=showRes;
	Rect faceRegion=d.findFace(img);

	if(faceRegion.x==-1)
	{
		Mat tmp;
		return tmp;
	}

	Mat res= predict_single(img,faceRegion,sampleNum);

	if(showRes)
	{
		Mat tmpOrg=cvarrToMat(img).clone();
		float ratio=1;
		if(tmpOrg.rows>700)
		{
			ratio=(float)tmpOrg.rows/700.0f;
			resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
		}
		for(int i=0;i<res.cols/2;i++)
			circle(tmpOrg,Point(res.at<float>(2*i)/ratio,res.at<float>(2*i+1)/ratio),2,Scalar(255),-1);



		imshow("finalRes",tmpOrg);
		waitKey();
	}

	return res;

	Mat curImg=d.getCurFace(faceRegion,img);

	Point2f curST=d.curST;
	float curScale=(float)curImg.cols/(float)refWidth;
	resize(curImg,curImg,Size(refWidth,refHeight));
	//waitKey();
	// return;

	 srand((unsigned)time(0));  

	int fullSize=inputShapes.size();
	vector<int> sampleInd;
	RandSample_V1(fullSize,sampleNum,sampleInd);

	//391 751 151 57 472
	//sampleInd[0]=391;sampleInd[1]=751;sampleInd[2]=151;sampleInd[3]=57;sampleInd[4]=472;
	//sampleInd[0]=807;

	vector<Shape> curShape(sampleNum);

	cout<<"aligning features\n";
	//if(GTPts!=NULL)
	{
		//try{
		GTB("1");
		#pragma omp parallel for
		for(int i=0;i<sampleNum;i++)
		{
			
			//set shape to current sample
			curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);

			
			
			curShape[i].orgImg=curImg.clone();

			//curShape[i].visualizePts("1");
			//waitKey();

			
			//curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);
			//cout<<i<<" ";
			pridict(curImg,curShape[i],false);
			//cout<<"post "<<i<<endl;
		}

		GTE("1");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"extraction time "<<time<<" ms"<<endl;
		/*}
		catch(Exception &e)
		{
			cout<<e.what();
		}*/

		//waitKey();
		//return Mat();

		//show curResult
		if(showRes&&0)
		{
			for(int i=0;i<sampleNum;i++)
			{
				char curName[50];
				sprintf(curName,"%d",i+1);
				namedWindow(curName);
				moveWindow(curName,400*i,20);
				curShape[i].visualizePts(curName);
			}
			//waitKey();
		}
		//finally, obtain the median for the final results
		Mat finalRes=Mat::zeros(sampleNum,curShape[0].n*2,CV_32FC1);
		for(int i=0;i<sampleNum;i++)
			finalRes.row(i)+=curShape[i].getFinalPosVector(curScale,curST);
		cv::sort(finalRes,finalRes,CV_SORT_EVERY_COLUMN );
		int usedRow=(sampleNum-1)/2;
		
		if(showRes)
		{
			Mat tmpOrg=cvarrToMat(img).clone();
			float ratio=1;
			if(tmpOrg.rows>700)
			{
				ratio=(float)tmpOrg.rows/700.0f;
				resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
			}
			for(int i=0;i<finalRes.cols/2;i++)
				circle(tmpOrg,Point(finalRes.at<float>(usedRow,2*i)/ratio,finalRes.at<float>(usedRow,2*i+1)/ratio),2,Scalar(255),-1);
			

			
			imshow("finalRes",tmpOrg);
			waitKey();
		}
		return finalRes.row(usedRow);
	}


}


bool TwoLevelRegression::pridict_real_full(IplImage *img,const int sampleNum)
{
	//face detection first
	cout<<"detecting face\n";
	d.showFace=showRes;
	vector<Rect> faceRegionList=d.findFaceFull(img);

	vector<Mat> facesMat= predict_real_givenRects(img,faceRegionList, sampleNum);

	if(facesMat.size()==0)
		return false;
	else
		return true;


	if(faceRegionList.size()==0)
	{
		//Mat tmp;
		return false;
	}

	vector<Mat> faceRes;
	for(int i=0;i<faceRegionList.size();i++)
	{
		Mat curRes=predict_single(img,faceRegionList[i],sampleNum);
		faceRes.push_back(curRes.clone());
	}

	

	if(showRes)
	{
		Mat tmpOrg=cvarrToMat(img).clone();
		float ratio=1;
		if(tmpOrg.rows>700)
		{
			ratio=(float)tmpOrg.rows/700.0f;
			resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
		}
		for(int f=0;f<faceRegionList.size();f++)
		{
			for(int i=0;i<faceRes[f].cols/2;i++)
				circle(tmpOrg,Point(faceRes[f].at<float>(2*i)/ratio,faceRes[f].at<float>(2*i+1)/ratio),2,Scalar(255),-1);
		}


		imshow("finalRes",tmpOrg);
		waitKey();
	}
	
	return true;

}


vector<Mat> TwoLevelRegression::predict_real_givenRects(IplImage *img,vector<Rect> &faceRegionList,const int sampleNum)
{
	//face detection first
	//cout<<"detecting face\n";
	//d.showFace=showRes;
	//vector<Rect> faceRegionList=d.findFaceFull(img);

	

	if(faceRegionList.size()==0)
	{
		vector<Mat> tmp;
		return tmp;
	}

	vector<Mat> faceRes;
	for(int i=0;i<faceRegionList.size();i++)
	{
		Mat curRes=predict_single(img,faceRegionList[i],sampleNum);
		faceRes.push_back(curRes.clone());
	}

	if(showRes)
	{
		Mat tmpOrg=cvarrToMat(img).clone();
		float ratio=1;
		if(tmpOrg.rows>700)
		{
			ratio=(float)tmpOrg.rows/700.0f;
			resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
		}
		for(int f=0;f<faceRegionList.size();f++)
		{
			for(int i=0;i<faceRes[f].cols/2;i++)
				circle(tmpOrg,Point(faceRes[f].at<float>(2*i)/ratio,faceRes[f].at<float>(2*i+1)/ratio),2,Scalar(255),-1);
		}


		imshow("finalRes",tmpOrg);
		waitKey();
	}
	
	return faceRes;
}

vector<Mat> TwoLevelRegression::predict_real_givenRects_L2(IplImage *img,vector<Rect> &faceRegionList,TwoLevelRegression &model_lv2,const int sampleNum)
{
	//face detection first
	//cout<<"detecting face\n";
	//d.showFace=showRes;
	//vector<Rect> faceRegionList=d.findFaceFull(img);

	

	if(faceRegionList.size()==0)
	{
		vector<Mat> tmp;
		return tmp;
	}

	vector<Mat> faceRes;
	for(int i=0;i<faceRegionList.size();i++)
	{
		Mat curRes=predict_single_lv2(img,faceRegionList[i],sampleNum,model_lv2);
		faceRes.push_back(curRes.clone());
	}

	if(showRes)
	{
		Mat tmpOrg=cvarrToMat(img).clone();
		float ratio=1;
		if(tmpOrg.rows>700)
		{
			ratio=(float)tmpOrg.rows/700.0f;
			resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
		}
		for(int f=0;f<faceRegionList.size();f++)
		{
			for(int i=0;i<faceRes[f].cols/2;i++)
				circle(tmpOrg,Point(faceRes[f].at<float>(2*i)/ratio,faceRes[f].at<float>(2*i+1)/ratio),2,Scalar(255),-1);
		}


		imshow("finalRes",tmpOrg);
		waitKey();
	}
	
	return faceRes;
}


Mat TwoLevelRegression::predict_single(IplImage *img, Rect faceRegion,int sampleNum)
{
	Mat curImg=d.getCurFace(faceRegion,img);

	Point2f curST=d.curST;
	float curScale=(float)curImg.cols/(float)refWidth;
	resize(curImg,curImg,Size(refWidth,refHeight));
	//waitKey();
	// return;

	 srand((unsigned)time(0));  

	int fullSize=inputShapes.size();
	vector<int> sampleInd;
	RandSample_V1(fullSize,sampleNum,sampleInd);

	//391 751 151 57 472
	//sampleInd[0]=391;sampleInd[1]=751;sampleInd[2]=151;sampleInd[3]=57;sampleInd[4]=472;
	//sampleInd[0]=807;

	vector<Shape> curShape(sampleNum);

	bool showStepRes=false;
	if(showRes&&sampleNum==1)
		showStepRes=true;
	//cout<<"aligning features\n";
	//if(GTPts!=NULL)
	{
		//try{
		GTB("1");
		#pragma omp parallel for
		for(int i=0;i<sampleNum;i++)
		{
			
			//set shape to current sample
			curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);

			
			
			curShape[i].orgImg=curImg.clone();

			//curShape[i].visualizePts("1");
			//waitKey();

			
			//curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);
			//cout<<i<<" ";
			pridict(curImg,curShape[i],showStepRes);
			//cout<<"post "<<i<<endl;
		}

		GTE("1");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"extraction time "<<time<<" ms"<<endl;
		/*}
		catch(Exception &e)
		{
			cout<<e.what();
		}*/

		//waitKey();
		//return Mat();

		//show curResult
		if(showRes&&0)
		{
			for(int i=0;i<sampleNum;i++)
			{
				char curName[50];
				sprintf(curName,"%d",i+1);
				namedWindow(curName);
				moveWindow(curName,400*i,20);
				curShape[i].visualizePts(curName);
			}
			//waitKey();
		}
		//finally, obtain the median for the final results
		Mat finalRes=Mat::zeros(sampleNum,curShape[0].n*2,CV_32FC1);
		for(int i=0;i<sampleNum;i++)
			finalRes.row(i)+=curShape[i].getFinalPosVector(curScale,curST);
		cv::sort(finalRes,finalRes,CV_SORT_EVERY_COLUMN );
		int usedRow=(sampleNum-1)/2;
		
		//check 
		if(0)
		{
			Shape s(finalRes.cols/2);
			s.ptsVec+=finalRes.row(usedRow);

			for(int i=0;i<s.ptsVec.cols/2;i++)
			{
				s.ptsVec.at<float>(2*i)-=curST.x;
				s.ptsVec.at<float>(2*i+1)-=curST.y;
			}
			s.ptsVec/=curScale;
			s.syntheorize();
			s.setImg(curImg);
			//do a final round optimization
			s.visualizePts("ptsCur");
			pridict(curImg,s,showStepRes);

			Mat tmpRes=s.getFinalPosVector(curScale,curST);
			for(int i=0;i<s.ptsVec.cols;i++)
				finalRes.at<float>(usedRow,i)=tmpRes.at<float>(i);
		}
		return finalRes.row(usedRow);
	}

}

Mat TwoLevelRegression::predict_single_lv2(IplImage *img, Rect faceRegion,int sampleNum, TwoLevelRegression &model)
{
	Mat curImg=d.getCurFace(faceRegion,img);

	Point2f curST=d.curST;
	float curScale=(float)curImg.cols/(float)refWidth;
	resize(curImg,curImg,Size(refWidth,refHeight));
	//waitKey();
	// return;

	 srand((unsigned)time(0));  

	int fullSize=inputShapes.size();
	vector<int> sampleInd;
	RandSample_V1(fullSize,sampleNum,sampleInd);

	//391 751 151 57 472
	//sampleInd[0]=391;sampleInd[1]=751;sampleInd[2]=151;sampleInd[3]=57;sampleInd[4]=472;
	//sampleInd[0]=807;

	vector<Shape> curShape(sampleNum);

	bool showStepRes=false;
	if(showRes&&sampleNum==1)
		showStepRes=true;
	//cout<<"aligning features\n";
	//if(GTPts!=NULL)
	{
		//try{
		GTB("1");
		#pragma omp parallel for
		for(int i=0;i<sampleNum;i++)
		{
			
			//set shape to current sample
			curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);

			
			
			curShape[i].orgImg=curImg.clone();

			//curShape[i].visualizePts("1");
			//waitKey();

			
			//curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);
			//cout<<i<<" ";
			pridict(curImg,curShape[i],showStepRes);
			//cout<<"post "<<i<<endl;
		}

		GTE("1");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"extraction time "<<time<<" ms"<<endl;
		/*}
		catch(Exception &e)
		{
			cout<<e.what();
		}*/

		//waitKey();
		//return Mat();

		//show curResult
		if(showRes&&0)
		{
			for(int i=0;i<sampleNum;i++)
			{
				char curName[50];
				sprintf(curName,"%d",i+1);
				namedWindow(curName);
				moveWindow(curName,400*i,20);
				curShape[i].visualizePts(curName);
			}
			//waitKey();
		}
		//finally, obtain the median for the final results
		Mat finalRes=Mat::zeros(sampleNum,curShape[0].n*2,CV_32FC1);
		for(int i=0;i<sampleNum;i++)
			finalRes.row(i)+=curShape[i].ptsVec;
		cv::sort(finalRes,finalRes,CV_SORT_EVERY_COLUMN );
		int usedRow=(sampleNum-1)/2;

		if(showRes)
		{
			Mat tmpOrg=curImg.clone();
			float ratio=1;
			/*if(tmpOrg.rows>700)
			{
				ratio=(float)tmpOrg.rows/700.0f;
				resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
			}*/
			//for(int f=0;f<faceRegionList.size();f++)
			{
				for(int i=0;i<finalRes.cols/2;i++)
					circle(tmpOrg,Point(finalRes.at<float>(usedRow,2*i)/ratio,finalRes.at<float>(usedRow,2*i+1)/ratio),2,Scalar(255),-1);
			}


			imshow("finalRes_L1",tmpOrg);
			//waitKey();
		}

		Shape s(finalRes.cols/2);
		s.ptsVec+=finalRes.row(usedRow);
		s.syntheorize();
		Mat curRes=model.predict_single_directImgShape(curImg,s);
		
		s.ptsVec*=0;
		s.ptsVec+=curRes;
		Mat finalMat=s.getFinalPosVector(curScale,curST);
		//check 
		if(0)
		{
			Shape s(finalRes.cols/2);
			s.ptsVec+=finalRes.row(usedRow);

			for(int i=0;i<s.ptsVec.cols/2;i++)
			{
				s.ptsVec.at<float>(2*i)-=curST.x;
				s.ptsVec.at<float>(2*i+1)-=curST.y;
			}
			s.ptsVec/=curScale;
			s.syntheorize();
			s.setImg(curImg);
			//do a final round optimization
			s.visualizePts("ptsCur");
			pridict(curImg,s,showStepRes);

			Mat tmpRes=s.getFinalPosVector(curScale,curST);
			for(int i=0;i<s.ptsVec.cols;i++)
				finalRes.at<float>(usedRow,i)=tmpRes.at<float>(i);
		}
		return finalMat;
	}

}


Mat TwoLevelRegression::predict_single_direct(IplImage *img, int sampleNum)
{
	Mat curImg=cvarrToMat(img).clone();

	//imshow("curImg",curImg);
	//waitKey();
	// return;

	 srand((unsigned)time(0));  

	int fullSize=inputShapes.size();
	vector<int> sampleInd;
	RandSample_V1(fullSize,sampleNum,sampleInd);

	//391 751 151 57 472
	//sampleInd[0]=391;sampleInd[1]=751;sampleInd[2]=151;sampleInd[3]=57;sampleInd[4]=472;
	//sampleInd[0]=807;

	vector<Shape> curShape(sampleNum);

	bool showStepRes=false;
	if(showRes&&sampleNum==1)
		showStepRes=true;
	//cout<<"aligning features\n";
	//if(GTPts!=NULL)
	{
		//try{
		GTB("1");
		#pragma omp parallel for
		for(int i=0;i<sampleNum;i++)
		{
			
			//set shape to current sample
			curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);

			
			
			curShape[i].orgImg=curImg.clone();

			//curShape[i].visualizePts("1");
			//waitKey();

			
			//curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);
			//cout<<i<<" ";
			pridict(curImg,curShape[i],showStepRes);
			//cout<<"post "<<i<<endl;
		}

		GTE("1");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"extraction time "<<time<<" ms"<<endl;
		/*}
		catch(Exception &e)
		{
			cout<<e.what();
		}*/

		//waitKey();
		//return Mat();

		//show curResult
		if(showRes&&0)
		{
			for(int i=0;i<sampleNum;i++)
			{
				char curName[50];
				sprintf(curName,"%d",i+1);
				namedWindow(curName);
				moveWindow(curName,400*i,20);
				curShape[i].visualizePts(curName);
			}
			waitKey();
		}
		//finally, obtain the median for the final results
		Mat finalRes=Mat::zeros(sampleNum,curShape[0].n*2,CV_32FC1);
		for(int i=0;i<sampleNum;i++)
			finalRes.row(i)+=curShape[i].ptsVec;
		cv::sort(finalRes,finalRes,CV_SORT_EVERY_COLUMN );
		int usedRow=(sampleNum-1)/2;
		
		
		return finalRes.row(usedRow);
	}

}



Mat TwoLevelRegression::predict_single_directImgShape(Mat &img, Shape &s)
{
	Mat curImg=img;

	//imshow("curImg",curImg);
	//waitKey();
	// return;

	 srand((unsigned)time(0));  

	int fullSize=inputShapes.size();
	
	//391 751 151 57 472
	//sampleInd[0]=391;sampleInd[1]=751;sampleInd[2]=151;sampleInd[3]=57;sampleInd[4]=472;
	//sampleInd[0]=807;

	int sampleNum=1;
	vector<Shape> curShape(sampleNum);

	bool showStepRes=false;

	//cout<<"aligning features\n";
	//if(GTPts!=NULL)
	{
		//try{
		GTB("1");
		#pragma omp parallel for
		for(int i=0;i<sampleNum;i++)
		{
			
			//set shape to current sample
			curShape[i].setShapeOnly(s);

			
			
			curShape[i].orgImg=curImg.clone();

			//curShape[i].visualizePts("1");
			//waitKey();

			
			//curShape[i].setShapeOnly(inputShapes[sampleInd[i]]);
			//cout<<i<<" ";
			pridict(curImg,curShape[i],true);
			//cout<<"post "<<i<<endl;
		}

		GTE("1");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"extraction time "<<time<<" ms"<<endl;
		/*}
		catch(Exception &e)
		{
			cout<<e.what();
		}*/

		//waitKey();
		//return Mat();

		//show curResult
		if(showRes&&0)
		{
			for(int i=0;i<sampleNum;i++)
			{
				char curName[50];
				sprintf(curName,"%d",i+1);
				namedWindow(curName);
				moveWindow(curName,400*i,20);
				curShape[i].visualizePts(curName);
			}
			waitKey();
		}
		//finally, obtain the median for the final results
		Mat finalRes=Mat::zeros(sampleNum,curShape[0].n*2,CV_32FC1);
		for(int i=0;i<sampleNum;i++)
			finalRes.row(i)+=curShape[i].ptsVec;
		cv::sort(finalRes,finalRes,CV_SORT_EVERY_COLUMN );
		int usedRow=(sampleNum-1)/2;
		
		
		return finalRes.row(usedRow);
	}

}



Mat TwoLevelRegression::pridict_evaluate(IplImage *img,const int sampleNum, char *GTPts)
{
	Mat gtPts;
	Shape gtShape;
	gtShape.load(GTPts,false);
	gtPts=gtShape.ptsVec;


	//Mat finalRes=Mat::zeros(sampleNum,gtShape.n*2,CV_32FC1);

	Mat resMat=pridict_real(img,sampleNum);

	Mat finalResMat=Mat::zeros(1,gtShape.n*2*2,CV_32FC1);

	if(resMat.empty())
		return finalResMat;

	//check it is raughly correct
	if(abs(resMat.at<float>(2*30)-gtPts.at<float>(2*30))>15&&abs(resMat.at<float>(2*30+1)-gtPts.at<float>(2*30+1))>15)
		return finalResMat;
	for(int i=0;i<gtShape.n*2;i++)
		{
			finalResMat.at<float>(i)=resMat.at<float>(i);
			finalResMat.at<float>(i+gtShape.n*2)=gtPts.at<float>(i);
		}
	return finalResMat;
	
}


void TwoLevelRegression::pridict_GT(IplImage *img, char *GTPts)
{
	Mat gtPts;
	Shape gtShape;
	gtShape.load(GTPts,true);
	gtPts=gtShape.ptsVec;


	//use the true shape for intialization
	pridict(cvarrToMat(img),gtShape,true);
}

void TwoLevelRegression::visualizeModel(char *refShapeName)
{
	Shape curShape;
	curShape.load(refShapeName);
	for(int i=0;i<T;i++)
	{
		char curName[50];
		sprintf(curName,"%d_offset",i+1);
		ferns[i*K].visualize(curName,curShape);
	}
}


void TwoLevelRegression::getDS(vector<ShapePair> &curShape)
{
	int N=curShape.size();
	dsVector=Mat::zeros(N,curShape[0].n*2,CV_32FC1);
	for(int i=0;i<N;i++)
	{
		dsVector.row(i)+=(curShape[i].gtShape-curShape[i]).ptsVec;	
		

		//check
		if(0)
		{
			refShape.visualize(refShape.orgImg,curShape[i].pts,"s0");
			refShape.visualize(refShape.orgImg,curShape[i].gtShape.pts,"sg");


			for(int j=0;j<10;j++)
			{
				cout<<"pts:\n";
				cout<<curShape[i].pts[j]<<" "<<curShape[i].gtShape.pts[j]<<endl;
				cout<<curShape[i].ptsVec.at<float>(2*j)<<" "<<curShape[i].ptsVec.at<float>(2*j+1)<<" "<<
					curShape[i].gtShape.ptsVec.at<float>(2*j)<<" "<<curShape[i].gtShape.ptsVec.at<float>(2*j+1)<<endl;

				cout<<"ptsDif:\n";
				cout<<dsVector.at<float>(i,2*j)<<" "<<dsVector.at<float>(i,2*j+1)<<endl;
				cout<<curShape[i].pts[j]-curShape[i].gtShape.pts[j]<<endl;
			}
			waitKey();
		}

	}
}

void TwoLevelRegression::prepareTrainingData(vector<Shape> &shapes_input, Shape &refS,int augNum)
{
	refShape.setShape(refS);

	//randomized input shape first
	/*for(int i=0;i<shapes_input.size();i++)
	{
		shapes_input[i].estimateTrans(refShape);
		shapes_input[i].addLocalST(refS,refWidth,refHeight);

	}*/

	//check
	if(0)
	{
		for(int i=0;i<shapes_input.size();i++)
		{
			shapes_input[i].visualizePts("ptsCheck");
			shapes_input[i].visualize(refS.orgImg,shapes_input[i].pts,"ptsCheck1");
			waitKey();
		}
	}

	int N=shapes_input.size();


	shapes.resize(N*augNum);

	int curShapeID=0;

	for(int i=0;i<N;i++)
	{
		vector<int> curInd;

		if(N<augNum)
			augNum=N;
		RandSample_V1(N,augNum,curInd);

		curInd[0]=i;

		for(int j=0;j<curInd.size();j++)
		{
			/*shapes[curShapeID].setShapeOnly(shapes_input[curInd[j]]);
			shapes[curShapeID].estimateTrans(refShape);
			shapes[curShapeID].setGTShape(shapes_input[i],refS);
			shapes[curShapeID].setImg(shapes_input[i].orgImg);
			shapes[curShapeID].setScaleTranslatrion(shapes[curShapeID].gtShape.s,shapes[curShapeID].gtShape.tx,
				shapes[curShapeID].gtShape.ty);*/
			shapes[curShapeID].setImg(shapes_input[i].orgImg);
			shapes[curShapeID].setShapeOnly(shapes_input[curInd[j]]);
			shapes[curShapeID].setGTShape(shapes_input[i],refS);

			//cout<<shapes[curShapeID].gtShape.pts[36]<<" ";
		

			curShapeID++;
		}
	}
}


void TwoLevelRegression::prepareTrainingData_refFace(vector<Shape> &shapes_input, Shape &refS)
{
	refShape.setShape(refS);

	//randomized input shape first
	/*for(int i=0;i<shapes_input.size();i++)
	{
		shapes_input[i].estimateTrans(refShape);
		shapes_input[i].addLocalST(refS,refWidth,refHeight);

	}*/

	//check
	if(0)
	{
		for(int i=0;i<shapes_input.size();i++)
		{
			shapes_input[i].visualizePts("ptsCheck");
			shapes_input[i].visualize(refS.orgImg,shapes_input[i].pts,"ptsCheck1");
			waitKey();
		}
	}

	int N=shapes_input.size();


	shapes.resize(N*1);

	int curShapeID=0;

	for(int i=0;i<N;i++)
	{
		shapes[curShapeID].setImg(shapes_input[i].orgImg);
			shapes[curShapeID].setShapeOnly(refS);
			shapes[curShapeID].setGTShape(shapes_input[i],refS);
			curShapeID++;
	}
}


void TwoLevelRegression::processRefShape(Shape &curShape)
{
	/*int ptsNum=curShape.n;

	int minX,minY,maxX,maxY;
	minX=minY=10000;maxX=maxY=0;
	for(int i=0;i<ptsNum;i++)
	{
		int curX,curY;
		curX=curShape.pts[i].x;
		curY=curShape.pts[i].y;
		if(curX<minX)
			minX=curX;
		if(curX>maxX)
			maxX=curX;
		if(curY<minY)
			minY=curY;
		if(curY>maxY)
			maxY=curY;
	}*/

	Rect bRect=boundingRect(curShape.pts);

	

	//then enlarge the rect a little bit
	int enlargedW=(int)(bRect.width*0.2)*2;
	int enlargedY=(int)(bRect.height*0.2)*2;

	refWidth=bRect.width+enlargedW/2;
	refHeight=bRect.height+enlargedY/2;
	tl.x=bRect.x-enlargedW/4;
	tl.y=bRect.y-enlargedY/4;


	bRect.width+=enlargedW;
	bRect.height+=enlargedY;

	bRect.x-=enlargedW/2;
	bRect.y-=enlargedY/2;

	//then update the pts
	int ptsNum=curShape.n;
	for(int i=0;i<ptsNum;i++)
	{
		curShape.ptsVec.at<float>(2*i)-=bRect.x;
		curShape.ptsVec.at<float>(2*i+1)-=bRect.y;
	}
	curShape.syntheorize();

	tl.x-=bRect.x;
	tl.y-=bRect.y;


	curShape.orgImg=curShape.orgImg(bRect);

	//check
	if(0)
	{
		Mat tmp=curShape.orgImg.clone();
		for(int i=0;i<ptsNum;i++)
			circle(tmp,curShape.pts[i],1,Scalar(255));
		imshow("refShape",tmp);
		waitKey();
	}
}

void TwoLevelRegression::prepareTrainingData(char *nameList)
{
	vector<string> nameStrList;
	char curName[100];
	ifstream in(nameList,ios::in);
	int num;
	in>>num;
	in.getline(curName,99);
	
	for(int i=0;i<num;i++)
	{
		
		in.getline(curName,99);
		nameStrList.push_back(curName);
	}
	cout<<endl;

	//then normalize the face
	vector<Shape> allShapes;
	vector<bool> goodLabel;
	allShapes.resize(num);
	goodLabel.resize(num);
	
	//#pragma omp parallel for 
	for(int i=0;i<num;i++)
	{
		if(i%20==0||1)
			cout<<i<<" ";
		Shape curShape=normalizeTrainingData((char *)nameStrList[i].c_str());
		if(curShape.s==-1)
			goodLabel[i]=false;
		else
			goodLabel[i]=true;
		allShapes[i].setShape(curShape);
	}

	inputShapes.clear();
	for(int i=0;i<num;i++)
	{
		if(goodLabel[i])
			inputShapes.push_back(allShapes[i]);
	}
}

Shape TwoLevelRegression::normalizeTrainingData( char *name,bool isRef)
{
	Shape curShape;
	curShape.load(name);

	Rect gtRect=boundingRect(curShape.pts);

	d.showFace=false;
	Rect faceRect=d.findFaceGT(curShape.ImgPtr,gtRect);

	if(faceRect.x==-1)
	{
		curShape.s=-1;
		return curShape;
	}

	Mat newFace=d.getCurFace(faceRect,curShape.ImgPtr);

	float toRefScale=1;
	if(!isRef)
	{
		toRefScale=(float)refWidth/(float)newFace.cols;
		resize(newFace,newFace,Size(refWidth,refHeight));
	}


	//revise the pts here
	for(int i=0;i<curShape.n;i++)
	{
		curShape.ptsVec.at<float>(2*i)-=faceRect.x;
		curShape.ptsVec.at<float>(2*i+1)-=faceRect.y;
	}
	curShape.ptsVec*=toRefScale;
	curShape.syntheorize();
	curShape.setImg(newFace);

	if(showRes)
	{
		cout<<newFace.size()<<" "<<refWidth<<endl;
		curShape.visualize(newFace,curShape.pts,"curRes");
		waitKey();
	}

	return curShape;
}

void TwoLevelRegression::train(char *nameList,char *paraSetting)
{
	int TT,KK,FF,PP,augNum;
	char refShapeName[100];
	{
		ifstream in(paraSetting,ios::in);
		in>>TT>>KK>>FF>>PP>>augNum;
		in.getline(refShapeName,99);
		in.getline(refShapeName,99);
		in.close();

		//refShape.load(refShapeName);
		//process the refShape to only use the face region
		//processRefShape(refShape);

		refShape=normalizeTrainingData(refShapeName,true);
		refWidth=refShape.orgImg.cols;
		refHeight=refShape.orgImg.rows;
	}

	refShapeStr=refShapeName;

	cout<<"loading images\n";

	string processedDir=nameList;
	processedDir=processedDir.substr(0,processedDir.find_last_of("\\")+1)+"processed\\";
	string processedNameList=processedDir;
	processedNameList+="ptsList.txt";
	ifstream in_processed(processedNameList.c_str(),ios::in);
	if(in_processed)
	{
		char curName[100];
		int num;
		in_processed>>num;
		in_processed.getline(curName,99);
		inputShapes.resize(num);
		for(int i=0;i<num;i++)
		{
			if(i%20==0)
				cout<<i<<" ";
			in_processed.getline(curName,99);
			inputShapes[i].load(curName);
			//inputShapes[i].visualizePts("curShape");
			//waitKey();
		}
	}
	else
	{
		in_processed.close();
		prepareTrainingData(nameList);

		//save the image and shapes, as well as the ptsList
		for(int i=0;i<inputShapes.size();i++)
		{
			char curImgName[100];
			sprintf(curImgName,"%s%d.png",processedDir.c_str(),i);
			imwrite(curImgName,inputShapes[i].orgImg);

			char curPtsName[100];
			sprintf(curPtsName,"%s%d.pts",processedDir.c_str(),i);
			ofstream out(curPtsName,ios::out);
			out<<"version: 1\n";
			out<<"n_points:  "<<inputShapes[i].n<<endl;
			out<<"{\n";
			for(int j=0;j<inputShapes[i].n;j++)
				out<<inputShapes[i].pts[j].x<<" "<<inputShapes[i].pts[j].y<<endl;
			out<<"}\n";
			out.close();
		}

		//output the ptsList
		ofstream out(processedNameList.c_str(),ios::out);
		out<<inputShapes.size()<<endl;
		for(int i=0;i<inputShapes.size();i++)
		{
				char curPtsName[100];
				sprintf(curPtsName,"%s%d.pts",processedDir.c_str(),i);
				out<<curPtsName<<endl;
		}
		out.close();
		exit(1);
	}
	//char curName[100];
	//ifstream in(nameList,ios::in);
	//int num;
	//in>>num;
	//in.getline(curName,99);
	//inputShapes.resize(num);
	//for(int i=0;i<num;i++)
	//{
	//	if(i%20==0)
	//		cout<<i<<" ";
	//	in.getline(curName,99);
	//	inputShapes[i].load(curName);
	//}
	//cout<<endl;

	//saving all aligned shapes
	/*{
		cout<<"saving all shapes\n";
		string allShapeName=nameList;
		allShapeName=allShapeName.substr(0,allShapeName.find_last_of('\\'));
		allShapeName+="allShape.bin";
		cout<<allShapeName<<endl;

		ofstream out(allShapeName,ios::binary);
		out.write((char *)&num,sizeof(int));
		for(int i=0;i<num;i++)
		{
			Shape cur;
			cur.setShape(inputShapes[i]);
			cur.estimateTrans(refShape);
			cur.save(out);
		}
		out.close();
	}*/
	//return;

	cout<<"training\n";
	train(TT,KK,FF,PP,augNum);
}

void TwoLevelRegression::train(int _T,int _K,int F, int P, int augNum)
{
	cout<<"preparing training data\n";
	//prepareTrainingData(inputShapes,refShape,augNum);
	prepareTrainingData_refFace(inputShapes,refShape);
	//check
	if(0)
	{
		for(int i=0;i<shapes.size();i++)
		{
			shapes[i].visualizePts("curPPts");
			cout<<shapes[i].s<<" ";
			waitKey();
		}
	}


	T=_T;
	K=_K;

	ferns.resize(T*K);
	int N=shapes.size();
	//level 1
	for(int t=0;t<T;t++)
	{
		cout<<"1st level "<<t<<endl;
		//update the global translations to meanshape
		//cout<<"aligning images\n";
		for(int i=0;i<N;i++)
		{
			if(i%1000==0)
				cout<<i<<" ";;
			shapes[i].estimateTrans_local(refShape);
			//shapes[i].setScaleTranslatrion(shapes[i].gtShape.s,shapes[i].gtShape.tx,
			//	shapes[i].gtShape.ty);

			//shapes[i].visualizePts("curPPts");
			//waitKey();
		}
		//then calculate ds again
		getDS(shapes);

		//generate P feature locations
		//cout<<"getting P features\n";
		generateFeatureLocation(P,refShape);

		

		//do level two regressor training
		cout<<"2nd level ";
		for(int k=0;k<K;k++)
		{
			cout<<k<<" ";
			vector<Point> selectedIndexPariInd(F);
			vector<float> thresholdList(F);
			//select F best index-pair

			//while(1)
			{
			//	cout<<"finding f features\n";
				cv::theRNG().state = cv::getTickCount();
				Mat dirFull=Mat::zeros(shapes[0].n*2,F,CV_32FC1);
				randu(dirFull,Scalar::all(-1), Scalar::all(1));
				#pragma omp parallel for  
				for(int f=0;f<F;f++)
				{
					//randomized a direction
					//srand(cv::getTickCount()); 
					
					Mat curDir=dirFull.col(f);
					
					curDir/=norm(curDir);
					//cout<<curDir.t()<<endl;
					//project dS using this direction
					Mat projectedDs=(dsVector*curDir);
					projectedDs-=mean(projectedDs).val[0];	//centerlize

					//find the highest correleation
					Mat corResult=myCorr(projectedDs,featureVector,correlationMat);

					//cout<<corResult.t()<<endl;
					double minVal; double maxVal; Point minLoc; Point maxLoc;
					minMaxLoc( corResult, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
					selectedIndexPariInd[f].x=maxLoc.y/P;
					selectedIndexPariInd[f].y=maxLoc.y%P;
				}

				//check
				if(0)
				{
					Mat tmp=refShape.orgImg.clone();
					for(int i=0;i<selectedIndexPariInd.size();i++)
					{
						int curInd1=selectedIndexPariInd[i].x;
						int curInd2=selectedIndexPariInd[i].y;

						cout<<curInd1<<" "<<curInd2<<endl;

						Point featurePts1=refShape.pts[featureLocations[curInd1].z];
						Point featurePts2=refShape.pts[featureLocations[curInd2].z];

						float curTx1=featureLocations[curInd1].x;float curTy1=featureLocations[curInd1].y;
						float curTx2=featureLocations[curInd2].x;float curTy2=featureLocations[curInd2].y;

						line(tmp,Point(curTx1+featurePts1.x,curTy1+featurePts1.y),Point(curTx2+featurePts2.x,
							curTy2+featurePts2.y),Scalar(255));

					}
					imshow("selected pair",tmp);
					waitKey();
				}
			}

			//randomize F threshold here
			Mat curFeatureDifMat=Mat::zeros(featureVector.rows,F,CV_32FC1);
			for(int f=0;f<F;f++)
			{
				int curInd1=selectedIndexPariInd[f].x;
				int curInd2=selectedIndexPariInd[f].y;
				//obtain the range first
				Mat curFeatureDif=featureVector.col(curInd1)-featureVector.col(curInd2)+(featureVectorMean[curInd1]-featureVectorMean[curInd2]);
				curFeatureDifMat.col(f)+=curFeatureDif;
				//check
				if(0)
				{
					Mat sorted;
					cv::sort(curFeatureDif,sorted,CV_SORT_EVERY_COLUMN);
					cout<<sorted.t()<<endl;

				}

				int selectedThresInd=RandDouble_c01o()*featureVector.cols;
				thresholdList[f]=curFeatureDif.at<float>(selectedThresInd);
				
				/*double minVal; double maxVal; Point minLoc; Point maxLoc;
				minMaxLoc( curFeatureDif, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );
				thresholdList[f]=(RandDouble_c01c()*0.6+0.2)*(maxVal-minVal)+minVal;*/	
			}
			//cout<<endl<<selectedIndexPariInd<<endl;

			int curFurnID=t*K+k;
			ferns[curFurnID].train(featureLocations,selectedIndexPariInd,thresholdList,shapes,curFeatureDifMat);
			//locate each image then using the fern
			

			//learn the fern: calculate and store the delta shape
			//also, update the shape for next level calculation
			/*cout<<"size of bins\n";
			for(int i=0;i<ferns[curFurnID].binPool.size();i++)
				cout<<ferns[curFurnID].binPool[i].size()<<" ";
			cout<<endl;*/
			//go over each bin
			for(int i=0;i<ferns[curFurnID].binPool.size();i++)
			{
				for(int j=0;j<ferns[curFurnID].binPool[i].size();j++)
				{
					int curID=ferns[curFurnID].binPool[i][j];

					//shapes[curID].visualizePts("B4");
					shapes[curID]+=ferns[curFurnID].dsList[i];
					shapes[curID].syntheorize();
				/*	cout<<ferns[curFurnID].dsList[i].pts<<endl;
					shapes[curID].visualizePts("after");
					waitKey();*/
				}
			}


		}
		cout<<endl;
	}
}

void TwoLevelRegression::saveFerns_bin(char *name)
{
	ofstream out(name,ios::binary);
	//out<<T<<" "<<K<<endl;
	out.write((char *)&T,sizeof(int)*1);
	out.write((char *)&K,sizeof(int)*1);

	//cout<<"saving ferns\n";
	for(int i=0;i<T;i++)
	{
		for(int j=0;j<K;j++)
		{
			/*if(j%30==0)
				cout<<j<<" ";*/
			ferns[i*K+j].saveBin(out);
		}
		cout<<endl;
	}

	//output the refShape info
	//out<<refShapeStr<<endl;
	refShape.save(out);

	int num=inputShapes.size();
		out.write((char *)&num,sizeof(int));
		for(int i=0;i<num;i++)
		{
			Shape cur;
			cur.setShape(inputShapes[i]);
			//cur.estimateTrans(refShape);
			cur.save(out);
		}

		//refWidth and refHeight
		//refWidth=refHeight=249;
		out.write((char *)&refWidth,sizeof(int)*1);
		out.write((char *)&refHeight,sizeof(int)*1);

	out.close();
}

void TwoLevelRegression::saveFerns(char *name)
{
	ofstream out(name,ios::out);
	out<<T<<" "<<K<<endl;
	for(int i=0;i<T;i++)
	{
		for(int j=0;j<K;j++)
		{
			ferns[i*K+j].save(out);
		}
	}

	//output the refShape info
	out<<refShapeStr<<endl;
	out.close();
}


void TwoLevelRegression::loadFerns(char *name)
{
	ifstream in(name,ios::in);
	in>>T>>K;

	cout<<"loading ferns\n";
	ferns.resize(T*K);
	for(int i=0;i<T;i++)
	{
		cout<<i<<endl;
		for(int j=0;j<K;j++)
		{
			if(j%20==0)
				cout<<j<<" ";
			ferns[i*K+j].load(in);
		}
		cout<<endl;
	}

	cout<<"reading refShape\n";
	char Name[100];
	in.getline(Name,99);
	in.getline(Name,99);
	refShapeStr=Name;


	refShape.load(Name);
	processRefShape(refShape);
	//cout<<refShapeStr<<endl;
	in.close();
}

void TwoLevelRegression::loadFerns_bin(char *name)
{
	ifstream in(name,ios::binary);
	//in>>T>>K;
	in.read((char *)&T,sizeof(int));
	in.read((char *)&K,sizeof(int));

	cout<<"loading ferns\n";
	ferns.resize(T*K);
	for(int i=0;i<T;i++)
	{
		//cout<<i<<endl;
		for(int j=0;j<K;j++)
		{
			/*if(j%20==0)
				cout<<j<<" ";*/
			ferns[i*K+j].loadBin(in);
		}
		//cout<<endl;
	}

	//output ds here for testing
	
	//for(int i=0;i<T;i++)
	//{
	//	int curID=i*K;
	//	char name[100];
	//	sprintf(name,"D:\\Fuhao\\face dataset new\\faceRegression\\evlRes\\%d_Fern.txt",i);
	//	ofstream out(name,ios::out);
	//	for(int j=0;j<ferns[curID].dsList.size();j++)
	//	{
	//		for(int k=0;k<ferns[curID].dsList[j].n;k++)
	//			out<<ferns[curID].dsList[j].pts[k].x<<" "<<ferns[curID].dsList[j].pts[k].y<<" ";
	//		out<<endl;
	//	}

	//	out.close();

	//}
	/*cout<<"reading refShape\n";
	char Name[100];
	in.getline(Name,99);
	in.getline(Name,99);
	refShapeStr=Name;*/
	
	refShape.load(in);
	//Rect refRect=boundingRect(refShape.pts);
	//refFaceWidth=refRect.width;

	refFaceWidth=abs(refShape.pts[17].x-refShape.pts[26].x)*1.03f;

	//load in all training shapes
	//we should have all training shapes stored in the end of the file. read from somewhere else instead
	{
		//ifstream in("D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\allShape.bin",ios::binary);
		int fullShapeNum;
		in.read((char *) &fullShapeNum,sizeof(int));
		inputShapes.resize(fullShapeNum);
		for(int i=0;i<fullShapeNum;i++)
			inputShapes[i].load(in);
		//in.close();
	}

	in.read((char *)&refWidth,sizeof(int)*1);
	in.read((char *)&refHeight,sizeof(int)*1);

	in.close();


	//reload all input shapes here
	if(0)
	{
		inputShapes.clear();
		ifstream in("D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\processed\\ptsList.txt",ios::in);

		char curName[100];
		int num;
		in>>num;
		in.getline(curName,99);
		inputShapes.resize(num);
		for(int i=0;i<num;i++)
		{
			if(i%20==0)
				cout<<i<<" ";
			in.getline(curName,99);
			inputShapes[i].load(curName);
			//inputShapes[i].visualizePts("curShape");
			//waitKey();
		}
		in.close();

		saveFerns_bin(name);

		cout<<"finished saving\n";
	}
	//test the reference image here
	
	//IplImage *refImg=cvCloneImage(refImg);
	

}


//need to test
Mat TwoLevelRegression::myCorr(Mat &projectedDs,Mat &featureVector,
		Mat &correlationMat)
{
	int P=featureVector.cols;
	Mat corDis=Mat::zeros(P,1,CV_32FC1);
	for(int i=0;i<P;i++)
		corDis.at<float>(i)=featureVector.col(i).dot(projectedDs);

	float dsDot=projectedDs.dot(projectedDs);
	//select the index-pair with highest correlation
	Mat corResult=Mat::zeros(P*P,1,CV_32FC1);
	for(int i=0;i<P;i++)
	{
		for(int j=0;j<P;j++)
		{
			corResult.at<float>(i*P+j)=(corDis.at<float>(i)-corDis.at<float>(j))/
				sqrtf((correlationMat.at<float>(i,i)+correlationMat.at<float>(j,j)-
				2*correlationMat.at<float>(i,j))*dsDot);
			//cout<<corResult.at<float>(i*P+j)<<" ";
		}
		//cout<<endl;
	}
	return corResult;
}

void TwoLevelRegression::generateFeatureLocation(int P, Shape &refShape)
{
	int totalInd=refWidth*refHeight;
	vector<int> selectedInd;
	//RandSample_V1(totalInd,P,selectedInd);
	featureLocations.resize(P);

	RNG rng( 0xFFFFFFFF );
	selectedInd.resize(P);
	for(int i=0;i<P;i++)
		selectedInd[i]=-1;
	while(1)
	{
	cv::theRNG().state = cv::getTickCount();
	float sigma=0.2;
	for(int i=0;i<P;i++)
	{
		//cout<<rng.gaussian(sigma)<<" ";
		while(1)
		{
			int x=(rng.gaussian(sigma)+0.5)*refWidth;
			int y=(rng.gaussian(sigma)+0.45)*refHeight;
			x=x>=0?x:0;
			x=x<refWidth?x:refWidth-1;
			y=y>=0?y:0;y=y<refHeight?y:refHeight;
			int realInd=x*refHeight+y;

			bool found=false;
			for(int j=0;j<i;j++)
			{
				if(selectedInd[j]==realInd)
				{
					found=true;break;
				}
			}
			if(!found)
			{
				selectedInd[i]=realInd;
				break;
			}
		}
		
	}
	for(int i=0;i<P;i++)
	{
		int x=selectedInd[i]/refHeight;
		int y=selectedInd[i]%refHeight;

		x+=tl.x;y+=tl.y;

		//find the nearest feature point index
		int nearestId=0;
		double nearestDis=refWidth*refWidth+refHeight*refHeight+100;
		for(int j=0;j<refShape.n;j++)
		{
			double curDis=(x-refShape.pts[j].x)*(x-refShape.pts[j].x)+
				(y-refShape.pts[j].y)*(y-refShape.pts[j].y);

			if(curDis<nearestDis)
			{
				nearestDis=curDis;
				nearestId=j;
			}
		}
		featureLocations[i].x=x-refShape.pts[nearestId].x;
		featureLocations[i].y=y-refShape.pts[nearestId].y;
		featureLocations[i].z=nearestId;
	}
	
	//check
	if(0)
	{
		Mat tmp=refShape.orgImg.clone();
		for(int i=0;i<P;i++)
		{
			circle(tmp,Point((refShape.pts[featureLocations[i].z].x+featureLocations[i].x),
				refShape.pts[featureLocations[i].z].y+featureLocations[i].y),1,Scalar(255));
		}
		imshow("P samples",tmp);
		waitKey();
	}
	}
	//then obtain the feature vectors

	cout<<"obtaining feature vectors\n";
	int N=shapes.size();
	featureVector=Mat::zeros(N,P,CV_32FC1);
	for(int i=0;i<N;i++)
	{	
		featureVector.row(i)+=shapes[i].getCurFeature(featureLocations);
	}

	//centerlize the feature vector and store their mean values
	cout<<"centerlizing vectors\n";
	featureVectorMean.resize(P);
	for(int i=0;i<P;i++)
	{
		featureVectorMean[i]=mean(featureVector.col(i)).val[0];
		featureVector.col(i)-=featureVectorMean[i];
	}

	cout<<"correlation precalcluations "<<featureVector.cols<<endl;
	correlationMat=cv::Mat::zeros(P,P,CV_32FC1);
	//selfCorrelationMat=Mat::zeros(P,1,CV_32FC1);
	#pragma omp parallel for  
	for(int i=0;i<P;i++)
	{
		//cout<<i<<" ";
		//selfCorrelationMat.at<float>(i,0)=featureVector.col(i).dot(featureVector.col(i));
		for(int j=0;j<P;j++)
		{
			correlationMat.at<float>(i,j)=featureVector.col(i).dot(featureVector.col(j));
		}
	}

	//calculate the P*P correlation matrix

	////finally, obtain the P^2 index pari values
	//cout<<"obtianing P^2 index pair\n";
	//featurePairVal=Mat::zeros(P*P,N,CV_32FC1);
	//featurePairMean.resize(P*P);
	////featurePairVal_centerlized=Mat::zeros(P*P,N,CV_32FC1);
	//featurePIndex.resize(P*P);

	//#pragma omp parallel for  
	//for(int i=0;i<P;i++)
	//{
	//	for(int j=0;j<P;j++)
	//	{
	//		int curCol=i*P+j;
	//		featurePIndex[curCol]=Point(j,i);
	//		featurePairVal.row(curCol)+=(featureVector.col(j)-featureVector.col(i)).t();
	//		featurePairMean[curCol]=mean(featurePairVal.row(curCol)).val[0];
	//		featurePairVal.row(curCol)-=featurePairMean[curCol];
	//	}
	//}
}