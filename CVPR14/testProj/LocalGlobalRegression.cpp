#include "LocalGlobalRegression.h"
#include "GRandom.h"
#include "codeTimer.h"

bool myfunction (int i,int j) { return (i<j); }

LocalGlobalRegression::LocalGlobalRegression()
{
	candidateNum=10; //10

	//candidateNum=1;
	candidateRadius.resize(candidateNum);
	for(int i=0;i<candidateRadius.size();i++)
		candidateRadius[i]=0.05+i*0.05;//+0.35;

	/*candidateNum=1;
	candidateRadius.clear();
	candidateRadius.push_back(0.5);*/

	isRot=true;

	//eyeIndL=2;
	//eyeIndR=3;
}

Mat LocalGlobalRegression::predict_Real_CVPR14(IplImage *img)
{
	
	Rect faceRegion=d.findFace(img);

	if(faceRegion.x==-1)
	{
		Mat tmp;
		return tmp;
	}

	
	
	Mat curImg=d.getCurFace(faceRegion,img);
	Point2f curST=d.curST;
	float curScale=(float)curImg.cols/(float)refWidth;
	resize(curImg,curImg,Size(refWidth,refHeight));

	//GTB("1");
	Shape initialShape(refShape.n);
	initialShape.setShapeOnly(refShape);
	predict_CVPR14(curImg,initialShape);
	/*GTE("1");
	gCodeTimer.printTimeTree();
	double time = total_fps;
	cout<<"prediction time "<<time<<" ms"<<endl;*/

	Mat res=initialShape.getFinalPosVector(curScale,curST);
	//check here
	if(1)
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
	

}


Mat LocalGlobalRegression::predict_single(IplImage *img, Rect facialRect)
{
	
	Rect faceRegion=facialRect;

	if(faceRegion.x==-1)
	{
		Mat tmp;
		return tmp;
	}

	
	
	Mat curImg=d.getCurFace(faceRegion,img);
	Point2f curST=d.curST;
	float curScale=(float)curImg.cols/(float)refWidth;
	resize(curImg,curImg,Size(refWidth,refHeight));

	GTB("1");
	Shape initialShape(refShape.n);
	initialShape.setShapeOnly(refShape);
	predict_CVPR14(curImg,initialShape);
	GTE("1");
	gCodeTimer.printTimeTree();
	double time = total_fps;
	cout<<"prediction time "<<time<<" ms"<<endl;

	Mat res=initialShape.getFinalPosVector(curScale,curST);
	//check here
	if(0)
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
	

}


vector<Mat> LocalGlobalRegression::predict_real_givenRects(IplImage *img,vector<Rect> &faceRegionList)
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
		Mat curRes=predict_single(img,faceRegionList[i]);
		faceRes.push_back(curRes.clone());
	}

	if(0)
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



void LocalGlobalRegression::predict_CVPR14(Mat &img, Shape &s)
{
	if(0)
	{
		Mat tmpOrg=img.clone();
		float ratio=1;
		if(tmpOrg.rows>700)
		{
			ratio=(float)tmpOrg.rows/700.0f;
			resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
		}
		Mat res=s.ptsVec;
		for(int i=0;i<res.cols/2;i++)
			circle(tmpOrg,Point(res.at<float>(2*i)/ratio,res.at<float>(2*i+1)/ratio),2,Scalar(255),-1);



		imshow("Initial",tmpOrg);
		waitKey();
	}
	
	for(int i=0;i<T;i++)
	{
		s.estimateTrans_local(refShape);
		for(int j=0;j<K;j++)
		{
			//local preiction
			forests[i*K+j].predict(img,s);
		}		

		if(0)
		{
			Mat tmpOrg=img.clone();
			float ratio=1;
			if(tmpOrg.rows>700)
			{
				ratio=(float)tmpOrg.rows/700.0f;
				resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
			}
			Mat res=s.ptsVec;
			for(int j=0;j<res.cols/2;j++)
				circle(tmpOrg,Point(res.at<float>(2*j)/ratio,res.at<float>(2*j+1)/ratio),2,Scalar(255),-1);


			char curName[20];
			sprintf(curName,"Res_%d",i+1);
			imshow(curName,tmpOrg);
			//waitKey();
		}
	}

	
		if(0)
	{
		Mat tmpOrg=img.clone();
		float ratio=1;
		if(tmpOrg.rows>700)
		{
			ratio=(float)tmpOrg.rows/700.0f;
			resize(tmpOrg,tmpOrg,Size(tmpOrg.cols/ratio,tmpOrg.rows/ratio));
		}
		Mat res=s.ptsVec;
		for(int i=0;i<res.cols/2;i++)
			circle(tmpOrg,Point(res.at<float>(2*i)/ratio,res.at<float>(2*i+1)/ratio),2,Scalar(255),-1);



		imshow("finalRes",tmpOrg);
		waitKey();
	}

}

void LocalGlobalRegression::train_CVPR14(char *nameList,char *paraSetting)
{
	char refShapeName[100];
	char validationSetListName[100];
	{
		ifstream in(paraSetting,ios::in);
		in>>T>>N>>D>>K>>augNum;
		in.getline(refShapeName,99);
		in.getline(refShapeName,99);
		in.getline(validationSetListName,99);
		in.close();

		//refShape.load(refShapeName);
		//process the refShape to only use the face region
		//processRefShape(refShape);

		refShape=normalizeTrainingData(refShapeName,true);
		refWidth=refShape.orgImg.cols;
		refHeight=refShape.orgImg.rows;

		ShapePtsNum=refShape.n;

		//then scale all radius to normal face size
		//Point2f eyeDif=refShape.pts[36]-refShape.pts[45];
		//eyeDis=sqrtf(eyeDif.x*eyeDif.x+eyeDif.y*eyeDif.y);
		
		//hardcoded here for testing
		eyeDis=87.07f;

		//for(int i=0;i<candidateNum;i++)
			//candidateRadius[i]*=eyeDis;

		//load validation set
		if(1)
		{
			ifstream in_validation(validationSetListName,ios::in);
			char curName[100];
			int num;
			in_validation>>num;
			in_validation.getline(curName,99);
			validateShapes.resize(num);
			for(int i=0;i<num;i++)
			{
				if(i%20==0)
					cout<<i<<" ";
				in_validation.getline(curName,99);
				validateShapes[i].load(curName);
				//inputShapes[i].visualizePts("curShape");
				//waitKey();
			}
			in_validation.close();

			//check
			if(0)
			{
				for(int i=0;i<num;i++)
				{
					validateShapes[i].visualizePts("probImg");
						waitKey();
				}
			}

			//set the gt shape 
			for(int i=0;i<num;i++)
			{
				validateShapes[i].setGTShape(validateShapes[i],refShape);
				validateShapes[i].setShapeOnly(refShape);
			}

			//check
			if(0)
			{
				for(int i=0;i<num;i++)
				{
					cout<<validateShapes[i].dS().ptsVec<<endl;
					for(int j=0;j<validateShapes[i].n;j++)
					{
						cout<<validateShapes[i].pts[j]<<" "<<validateShapes[i].gtShape.pts[j]<<endl;
						validateShapes[i].visualizePts("probImg");
						waitKey();
					}
					/*Shape tmp=validateShapes[i].dS();
					for(int j=0;j<tmp.n;j++)
					{
						if(tmp.pts[j].x>1000||tmp.pts[j].y>1000)
						{
							validateShapes[i].visualizePts("probImg");
							waitKey();
						}
					}*/
				}
			}
		}
	


	}
	

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

	cout<<"training\n";
	train_CVPR14();
}

void LocalGlobalRegression::train_CVPR14()
{
	//cout<<" curPtsNum: "<<ShapePtsNum<<endl;
	//prepare shapes for training
	cout<<"preparing training data\n";
	prepareTrainingData(inputShapes,refShape,augNum);
	//prepareTrainingData_refFace(inputShapes,refShape);

	//set tree numbers for each forest according to ptsNum
	int treeNumEachForest=N/ShapePtsNum;
	cout<<"number of trees for each forest: "<<treeNumEachForest<<endl;
	//treeNumEachForest=treeNumEachForest>5?5:treeNumEachForest;
	
	int sampleFullNum=shapes.size();
	int validationSize=sampleFullNum*0.1;
	
	forests.resize(T*K);
	for(int i=0;i<T;i++)
	{
		//update local R and S
		for(int j=0;j<shapes.size();j++)
		{
			shapes[j].estimateTrans_local(refShape);
		}
		for(int j=0;j<validateShapes.size();j++)
		{
			validateShapes[j].estimateTrans_local(refShape);
		}

		//do cross validation

		float curRadius;
		if(i==0)
			curRadius=0.5*eyeDis;
		else if(i==1)
			curRadius=0.4*eyeDis;
		else if(i==2)
			curRadius=0.15*eyeDis;
		else if(i==3)
			curRadius=0.1*eyeDis;
		else if(i==4)
			curRadius=0.1*eyeDis;

		
		if(i==0)
		{
			candidateRadius.clear();
			candidateRadius.push_back(0.3);candidateRadius.push_back(0.4);candidateRadius.push_back(0.35);
			//candidateRadius.push_back(0.45);candidateRadius.push_back(0.5);candidateRadius.push_back(0.35);
		}
		else if(i==1)
		{
			candidateRadius.clear();
			candidateRadius.push_back(0.25); candidateRadius.push_back(0.3);
			candidateRadius.push_back(0.35);//candidateRadius.push_back(0.4);candidateRadius.push_back(0.45);
		}
		else if(i==2)
		{
			candidateRadius.clear();
			candidateRadius.push_back(0.15);candidateRadius.push_back(0.2);candidateRadius.push_back(0.25);
		}
		else if(i==4)
		{
			candidateRadius.clear();
			candidateRadius.push_back(0.05);candidateRadius.push_back(0.1);candidateRadius.push_back(0.15);
		}
		else
		{
			candidateRadius.clear();
			candidateRadius.push_back(0.05);candidateRadius.push_back(0.1);candidateRadius.push_back(0.15);
		}
		candidateNum=candidateRadius.size();

		int bestCandidateInd=-1;
		vector<vector<RF_WholeFace>> forests_candidates(candidateNum);	
		if(1)
		{
			//obtain the training and validation set first
			//vector<int> validInd, trainInd,fullInd;
			//for(int j=0;j<sampleFullNum;j++)
			//	fullInd.push_back(j);
			//RandSample_V1(sampleFullNum,validationSize,validInd);
			//std::sort(validInd.begin(),validInd.end(),myfunction);
			//trainInd.resize(sampleFullNum-validationSize);

			//std::set_difference(fullInd.begin(),fullInd.end(),validInd.begin(),validInd.end(),trainInd.begin());
			////generate the training and testing set
			//vector<ShapePair> trainingSet, testSet;
			//for(int j=0;j<validationSize;j++)
			//	testSet.push_back(shapes[validInd[j]]);
			//for(int j=0;j<trainInd.size();j++)
			//	trainingSet.push_back(shapes[trainInd[j]]);
			float startAdjustVal=0.0f;
			/*if(i==0)
				startAdjustVal=0.2;
			else if(i==1)
				startAdjustVal=0.1;
			else if(i==5)
				curRadius=-0.0025;
			else
				curRadius=0;*/
		
			//then test all candidate radius
			
			//#pragma omp parallel for
			for(int c=0;c<candidateNum;c++)
			{
				vector<ShapePair> curShapes=shapes;
				forests_candidates[c].resize(K);
				for(int j=0;j<K;j++)
				{
					cout<<"training candidate forests "<<c<<endl;
					forests_candidates[c][j].train_unit_validation(curShapes,&refShape,treeNumEachForest,D,(startAdjustVal+candidateRadius[c])*eyeDis, &validateShapes);

					//for testing only
					//updateShapes(curShapes,forests_candidates[c][j]);
				}
			}

			//calculate errors and select the one with minimum error
			cout<<"checking prediction errors\n";
			vector<float> errors(candidateNum);
			for(int c=0;c<candidateNum;c++)
			{
				//cout<<"checking error "<<c<<endl;
				errors[c]=evlError(validateShapes,forests_candidates[c]);
			}


			int minInd=0;
			float minErr=errors[0];
			for(int j=1;j<candidateNum;j++)
			{
				if(minErr>errors[j])
				{
					minErr=errors[j];minInd=j;
				}
			}

			bestCandidateInd=minInd;

			for(int c=0;c<candidateNum;c++)
			{
				cout<<" [ "<<startAdjustVal+candidateRadius[c]<<" , "<<errors[c]<<" ] ";
			}
			cout<<"best radius: "<<startAdjustVal+candidateRadius[minInd]<<endl;

			curRadius=(startAdjustVal+candidateRadius[minInd])*eyeDis;
			//check
			if(0)
			{
				Mat tmp=refShape.orgImg.clone();
				for(int k=0;k<refShape.n;k++)
					circle(tmp,refShape.pts[k],curRadius,Scalar(255),1);
				imshow("curBestRadius",tmp);
				waitKey();
			}
				//train using the full set
			
		}

		
		
		cout<<"training using the best radius "<<curRadius/eyeDis<<endl;
		if(bestCandidateInd==-1)
		{
			for(int j=0;j<K;j++)
			{
				forests[i*K+j].train_unit_validation(shapes,&refShape,treeNumEachForest,D,curRadius,&validateShapes);

				cout<<"--------tree size (should be "<<shapes[0].n<<" ): "<<forests[i*K+j].forests.size()<<endl;
			}
		}
		else
		{
			char curName[50];
			sprintf(curName,"forest_stage_%d",i+1);
			ofstream out(curName,ios::binary);
			for(int j=0;j<K;j++)
			{
				forests_candidates[bestCandidateInd][j].save(out);
			}
			out.close();

			ifstream in(curName,ios::binary);
			for(int j=0;j<K;j++)
			{
				forests[i*K+j].load(in);
			}
		}
	
		


		cout<<"evluating errors on validation set\n";
		float error1=evlError(validateShapes,forests,i*K);
		cout<<"Stage "<<i+1<<" average error: "<<error1<<endl;


		//for(int j=0;j<K;j++)
		//{
		//	//check
		//	if(0)
		//	{
		//		Mat tmpImg=testShapes[i].orgImg.clone();
		//		for(int i=0;i<curShape.n;i++)
		//			circle(tmpImg,curShape.pts[i],1,Scalar(255));
		//		imshow("curImg",tmpImg);

		//		cout<<forests_candidates.size()<<endl;
		//		waitKey();
		//	}
		//	//forests_candidates[j].showSingleStep=true;
		//	forests_candidates[j].predict(testShapes[i].orgImg, curShape);
		//}

		for(int j=0;j<shapes.size();j++)
		{
			for(int k=0;k<K;k++)
			{
				forests[i*K+k].predict(shapes[j].orgImg,shapes[j]);
			}
		}

		//also, update the validation Shapes
		for(int j=0;j<validateShapes.size();j++)
		{
			for(int k=0;k<K;k++)
			{
				forests[i*K+k].predict(validateShapes[j].orgImg,validateShapes[j]);
			}
		}

	}	
}

void LocalGlobalRegression::updateShapes(vector<ShapePair> &curShapes,RF_WholeFace &forest)
{
	for(int i=0;i<curShapes.size();i++)
	{
		//curShapes[i].visualizePts("oldShape");
		//cout<<curShapes[i].dS().ptsVec<<endl;
		//waitKey();
		forest.predict(curShapes[i].orgImg,curShapes[i]);

		//curShapes[i].visualizePts("newShape");
		//waitKey();
	}
}

float LocalGlobalRegression::evlError(vector<ShapePair> &testShapes,vector<RF_WholeFace> &forests_candidates, int startInd)
{
	float errorRes=0;
	int shapeNum=testShapes.size();

	int effectiveNum=0;
	for(int i=0;i<shapeNum;i++)
	{
		Shape curShape;
		curShape.setShapeOnly(testShapes[i]);
		curShape.orgImg=testShapes[i].orgImg;
		curShape.RS_local=testShapes[i].RS_local;

		if(0)
		{
			testShapes[i].visualizePts("Pts");
			waitKey();
		}

		//cout<<"evling shape "<<i<<endl;
		for(int j=startInd;j<startInd+K;j++)
		{
			//check
			if(0)
			{
				Mat tmpImg=testShapes[i].orgImg.clone();
				for(int i=0;i<curShape.n;i++)
					circle(tmpImg,curShape.pts[i],1,Scalar(255));
				imshow("curImg",tmpImg);

				cout<<forests_candidates.size()<<endl;
				waitKey();
			}
			//forests_candidates[j].showSingleStep=true;
			forests_candidates[j].predict(testShapes[i].orgImg, curShape);
		}

		//calculate the error
		//cout<<"calculating error "<<i<<endl;
		float curError=getError(curShape,testShapes[i].gtShape);
		//if(sqrtf(curError/(testShapes[0].n*2))>15&&0)
		//	continue;
		//else
		{
			errorRes+=curError;
			effectiveNum++;
		}
		
		//cout<<"curRes"<<" "<<errorRes<<" "<<sqrtf(errorRes/(testShapes[0].n*2*(i+1)))<<endl;
	}
	

	errorRes/=testShapes.size()*testShapes[0].n*2;

	
	return sqrtf(errorRes);
}

float LocalGlobalRegression::getError(Shape &curShape,Shape &gtShape)
{
	float res=0;

	//cout<<curShape.ptsVec<<endl;
	//cout<<gtShape.ptsVec<<endl;
	//float curEyeDis;
	//Point2f eyeDif=gtShape.pts[eyeIndL]-gtShape.pts[eyeIndR]; //2,3
	//curEyeDis=(eyeDif.x*eyeDif.x+eyeDif.y*eyeDif.y);

	Mat dif=curShape.ptsVec-gtShape.ptsVec;

	////cout<<dif<<endl;
	//Mat ptsVec1=curShape.ptsVec;
	//Mat pteVec2=gtShape.ptsVec;
	//for(int i=0;i<curShape.n;i++)
	//	res+=sqrtf((ptsVec1.at<float>(2*i)-pteVec2.at<float>(2*i))*(ptsVec1.at<float>(2*i)-pteVec2.at<float>(2*i))
	//	+(ptsVec1.at<float>(2*i+1)-pteVec2.at<float>(2*i+1))*(ptsVec1.at<float>(2*i+1)-pteVec2.at<float>(2*i+1)))/curEyeDis;

	res=dif.dot(dif);
	//check
	if(0&&sqrtf(res/(curShape.n*2))>15)
	{
		curShape.visualizePts("curShape");
		cout<<res<<" "<<sqrtf(res/(curShape.n*2))<<endl;
		//cout<<"error: "<<res/curShape.n<<" eyeDis: "<<curEyeDis<<endl;
		waitKey();
	}
	//cout<<sqrtf(res/(curShape.n*2))<<" ";
	//res=sum(dif).val[0];
	return res;
}



void LocalGlobalRegression::getUpdate(Mat &W,RF_WholeFace &forest, Shape &s)
{
	//update s to s+Ds here

}


void LocalGlobalRegression::save_CVPR14(char *name)
{
	ofstream out(name,ios::binary);
	//T,K
	out.write((char *)&T,sizeof(int));
	out.write((char *)&K,sizeof(int));
	//refShape
	refShape.save(out);
	out.write((char *)&refWidth,sizeof(int));
	out.write((char *)&refHeight,sizeof(int));

	//forests for each stage
	int treeNum=forests.size();
	out.write((char *)&treeNum,sizeof(int));
	for(int i=0;i<forests.size();i++)
		forests[i].save(out);

	out.close();
}

void LocalGlobalRegression::load_CVPR14(char *name)
{
	ifstream in(name,ios::binary);
	
	in.read((char *)&T,sizeof(int));
	in.read((char *)&K,sizeof(int));

	//refShape
	refShape.load(in);
	in.read((char *)&refWidth,sizeof(int));
	in.read((char *)&refHeight,sizeof(int));
	ShapePtsNum=refShape.n;
	eyeDis=87.07f;

	int treeNum;
	in.read((char *)&treeNum,sizeof(int));
	forests.resize(treeNum);

	for(int i=0;i<treeNum;i++)
	{
		cout<<"tree "<<i<<endl;
		forests[i].load(in);
	}

	////load in refShape here temparily
	//refShape=normalizeTrainingData("D:\\Fuhao\\face dataset new\\faceRegression\\refShape\\image_0808.pts",true);
	//refWidth=refShape.orgImg.cols;
	//refHeight=refShape.orgImg.rows;

	//T=5;K=1;

	in.close();

}

void LocalGlobalRegression::visualizeModel()
{
	for(int i=0;i<forests.size();i++)
	{
		char curName[10];
		sprintf(curName,"%d_forests",i+1);
		forests[i].visualize(curName);
	}
	waitKey();
}

Mat LocalGlobalRegression::pridict_evaluate(IplImage *img, char *GTPts)
{
	Mat gtPts;
	Shape gtShape;
	gtShape.load(GTPts,false);
	gtPts=gtShape.ptsVec;


	//Mat finalRes=Mat::zeros(sampleNum,gtShape.n*2,CV_32FC1);

	Mat resMat=predict_Real_CVPR14(img);

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