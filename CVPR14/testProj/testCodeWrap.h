#include "TwoLevelRegression.h"
#include "codetimer.h"
#include "Evaluation.h"
#include "FaceFounder.h"
#include "test.h"
#include "RealtimeTracking.h"

#include "TwoLevelRegression_LV2.h"

void train(char *name,bool isRot)
{
	char *nameList="D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\ptsList.txt";
	//char *nameList="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\ptsList.txt";
	char *paraSettingFile="D:\\Fuhao\\FacialFeatureDetection_Regression\\regressionSetting.txt";
	TwoLevelRegression trainer(isRot);
	trainer.train(nameList,paraSettingFile);

	trainer.saveFerns_bin(name);
}

void pridict(char *modelName)
{
	TwoLevelRegression trainer;
	trainer.loadFerns_bin(modelName);

	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset\\ptsList.txt";
	ifstream in(testSetName,ios::in);
	int num;
	in>>num;
	char curName[500];
	in.getline(curName,499);

	vector<string> nameStrList;
	for(int i=0;i<num;i++)
	{
		in.getline(curName,499);
		nameStrList.push_back(curName);
	}
	Mat nullImg;
	for(int i=0;i<nameStrList.size();i++)
	{
		trainer.pridict(nullImg,5,(char *)nameStrList[i].c_str());
	}
}

void evaluation(int sampleNum=5)
{
	char testNameList[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset1\\ptsList.txt";

	vector<string> modelNameList;
	vector<string> evlResList;

	/*modelNameList.push_back("model_Uni_Rot_large.bin");
	modelNameList.push_back("model_Uni_Rot.bin");

	evlResList.push_back("D:\\Fuhao\\face dataset new\\faceRegression\\evlRes\\model_Uni_Rot_large_large11.txt");
	evlResList.push_back("D:\\Fuhao\\face dataset new\\faceRegression\\evlRes\\model_Uni_Rot_lfpw_large11.txt");*/

	modelNameList.push_back("D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\model_Uni_Rot_large_gaussian.bin");
	char saveName[500];
	sprintf(saveName,"D:\\Fuhao\\face dataset new\\faceRegression\\evlRes\\model_Uni_Rot_large(gaussian)_large%d.txt",sampleNum);
	evlResList.push_back(saveName);

	
	for(int i=0;i<evlResList.size();i++)
	{
		Evaluation evlCorrect;	
		evlCorrect.doevaluation((char *)modelNameList[i].c_str(),testNameList,
			(char *)evlResList[i].c_str(),sampleNum);
	}
}

void evaluation_stepCheck()
{
	char testNameList[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset1\\ptsList.txt";

	char *modelName="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\model_Uni_Rot_large_gaussian.bin";

	Evaluation evlCorrect;	
	evlCorrect.checkIteration(modelName,testNameList);
}

void evaluation_GTCheck()
{
	char testNameList[]="D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\processed\\ptsList.txt";

	char *modelName="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\model_Uni_Rot_large_gaussian.bin";

	Evaluation evlCorrect;	
	evlCorrect.checkConvergeGT(modelName,testNameList);
}

void evaluation_ModelCheck()
{
	char *refName="D:\\Fuhao\\face dataset new\\faceRegression\\refShape\\image_0808.pts";

	char *modelName="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\model_Uni_Rot_large_gaussian.bin";

	Evaluation evlCorrect;	
	evlCorrect.checkModel(modelName,refName);
}

void faceDetect()
{
	//Shape refShape;
	//refShape.load("D:\\Fuhao\\face dataset new\\faceRegression\\refShape\\image_0808.pts");

	//TwoLevelRegression tmp;
	//tmp.processRefShape(refShape);

	//imshow("imgCheck",cvarrToMat(refShape.ImgPtr));
	//waitKey();

	IplImage *img=cvLoadImage("D:\\Fuhao\\face dataset new\\faceRegression\\testset\\image_0005.png");
	FaceDetector d;
	Rect fRect=d.findFace(img);

	cout<<fRect.width<<" "<<fRect.height<<endl;
}


void pridict_real(char *modelName,int sampleNum=5)
{
	TwoLevelRegression trainer;
	trainer.showRes=true;
	trainer.loadFerns_bin(modelName);
	//return;

	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset1\\imgList.txt";
	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testInternet\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\test_realCapture\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\DataBase\\ibug\\imgList.txt";
	ifstream in(testSetName,ios::in);
	int num;
	in>>num;
	char curName[500];
	in.getline(curName,499);

	vector<string> nameStrList;
	for(int i=0;i<num;i++)
	{
		in.getline(curName,499);
		nameStrList.push_back(curName);
	
	}
	
	for(int i=0;i<nameStrList.size();i++)
	{
		cout<<nameStrList[i].c_str()<<endl;
		IplImage *img=cvLoadImage(nameStrList[i].c_str(),0);
		IplImage *destination=img;
		/*if(img->height>600)
		{
			float ratio=600.0f/(float)img->height;
			     destination = cvCreateImage
( cvSize((int)(img->width*ratio) , (int)(img->height*ratio) ),
                                     img->depth, img->nChannels );
				cvResize(img, destination); 
		}*/
		trainer.pridict_real_full(destination,sampleNum);
	}
}


void pridict_real_twoLevel(char *modelName,char *modelNameL2,int sampleNum=5)
{
	TwoLevelRegression_LV2 trainer;
	trainer.showRes=true;
	trainer.loadFull(modelName,modelNameL2);
	//return;
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset1\\imgList.txt";
	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testInternet\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\test_realCapture\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\DataBase\\ibug\\imgList.txt";
	ifstream in(testSetName,ios::in);
	int num;
	in>>num;
	char curName[500];
	in.getline(curName,499);

	vector<string> nameStrList;
	for(int i=0;i<num;i++)
	{
		in.getline(curName,499);
		nameStrList.push_back(curName);
	
	}
	
	for(int i=0;i<nameStrList.size();i++)
	{
		cout<<nameStrList[i].c_str()<<endl;
		IplImage *img=cvLoadImage(nameStrList[i].c_str(),0);
		IplImage *destination=img;
		/*if(img->height>600)
		{
			float ratio=600.0f/(float)img->height;
			     destination = cvCreateImage
( cvSize((int)(img->width*ratio) , (int)(img->height*ratio) ),
                                     img->depth, img->nChannels );
				cvResize(img, destination); 
		}*/
		trainer.predict_real_lv2(destination,sampleNum);
	}
}


void predict_direct(char *modelName,int sampleNum=5)
{
	TwoLevelRegression trainer;
	trainer.showRes=true;
	trainer.loadFerns_bin(modelName);
	//return;

	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset1\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testInternet\\imgList.txt";
	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\processed\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\test_realCapture\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\DataBase\\ibug\\imgList.txt";
	ifstream in(testSetName,ios::in);
	int num;
	in>>num;

	//num=10;

	char curName[500];
	in.getline(curName,499);

	vector<string> nameStrList;
	for(int i=0;i<num;i++)
	{
		in.getline(curName,499);
		nameStrList.push_back(curName);
	
	}
	
	Mat finalRes=Mat::zeros(num,trainer.refShape.n*2,CV_32FC1);
	for(int i=0;i<nameStrList.size();i++)
	{
		cout<<i<<" "<<nameStrList[i].c_str()<<endl;
		IplImage *img=cvLoadImage(nameStrList[i].c_str(),0);
		IplImage *destination=img;
		/*if(img->height>600)
		{
			float ratio=600.0f/(float)img->height;
			     destination = cvCreateImage
( cvSize((int)(img->width*ratio) , (int)(img->height*ratio) ),
                                     img->depth, img->nChannels );
				cvResize(img, destination); 
		}*/
		Rect faceRegion;
		faceRegion.x=faceRegion.y=0;
		faceRegion.width=img->width;
		faceRegion.height=img->height;
		finalRes.row(i)+=trainer.predict_single_direct(destination,sampleNum);

		//cout<<finalRes.row(i)<<endl;
		//check result here
		if(0)
		{
			Mat tmp=cvarrToMat(destination).clone();
			for(int j=0;j<finalRes.cols/2;j++)
			{
				circle(tmp,Point(finalRes.at<float>(i,2*j),finalRes.at<float>(i,2*j+1)),
					3,Scalar(255));
			}
			imshow("curRes",tmp);
			waitKey();
		}
	}

	string outputName=testSetName;
	outputName=outputName.substr(0,outputName.find_last_of('\\')+1);
	outputName+="1stRes.bin";
	cout<<outputName<<endl;
	ofstream out(outputName,ios::binary);
	for(int i=0;i<num;i++)
	{
		Shape s(finalRes.cols/2);
		s.ptsVec+=finalRes.row(i);
		s.syntheorize();
		

		//check result here
		if(0)
		{
			Mat tmp=imread(nameStrList[i]);
			for(int j=0;j<finalRes.cols/2;j++)
			{
				circle(tmp,Point(finalRes.at<float>(i,2*j),finalRes.at<float>(i,2*j+1)),
					3,Scalar(255));
				circle(tmp,s.pts[j],
					3,Scalar(0,255,0));
			}
			imshow("curRes",tmp);
			waitKey();
		}

		s.save(out);
	}
	out.close();
}


void testST()
{
	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset\\imgList.txt";
	
	Test test;
	test.getScaleTranslation("model_bilinear.bin",testSetName);
}

void prepareImages(char *nameList)
{
	//char *nameList="D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\ptsList.txt";
	TwoLevelRegression trainer;
	trainer.prepareTrainingData((char *)nameList);
}

void realtimeFace(char *modelName,int sampleNum)
{
	TwoLevelRegression trainer(true);
	trainer.loadFerns_bin(modelName);
	FaceTracker t(&trainer);
	t.setSampleNum(sampleNum);
	t.start();
}

