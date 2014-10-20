#pragma once

#include "TwoLevelRegression.h"
#include "codetimer.h"
#include "Evaluation.h"
#include "Evaluation_cvpr14.h"
#include "FaceFounder.h"
#include "test.h"
#include "RealtimeTracking.h"

#include "TwoLevelRegression_LV2.h"
#include "LocalGlobalRegression.h"

#include <atlbase.h>
#include <atlconv.h>
#include "W_train.h"

void testLearnW(char *inputName)
{
	//load data
	ifstream in(inputName,ios::binary);
	int sampleNum;
	in.read((char *)&sampleNum,sizeof(int));
	vector<Shape> ds(sampleNum);
	vector<LBFFeature> lbfFeatureList(sampleNum);

	for(int i=0;i<sampleNum;i++)
		ds[i].load(in);
	for(int i=0;i<sampleNum;i++)
	{
		int onesNum;
		in.read((char *)&(lbfFeatureList[i].totalNum),sizeof(int));
		in.read((char *)&onesNum,sizeof(int));
		lbfFeatureList[i].onesInd.resize(onesNum);
		for(int j=0;j<onesNum;j++)
		{
			in.read((char *)&(lbfFeatureList[i].onesInd[j]),sizeof(int));
		}
	}

	//then test W here
	vector<float> c_list;
	c_list.push_back(0.001f);
	

	/*vector<Mat> W_list_new1(c_list.size());
	W_train w_trainer1(1);
	w_trainer1.train_multipleC(ds,lbfFeatureList,W_list_new1,c_list);*/

	vector<Mat> W_list_new(c_list.size());
	W_train w_trainer;
	w_trainer.train_multipleC(ds,lbfFeatureList,W_list_new,c_list);

	/*for(int i=0;i<W_list_new[0].rows;i++)
	{
		cout<<W_list_new[0].row(i)<<endl;

		Mat tmp(10,10,CV_8UC3);
		imshow("pause",tmp);
		waitKey();
	}*/

	//return;
	Mat W;
	RF_WholeFace tmp;
	W=Mat::zeros(ds[0].n*2,lbfFeatureList[0].totalNum+1,CV_32FC1);
	tmp.learnW_unit(ds,lbfFeatureList,W,0.001f);

	for(int i=0;i<W.rows;i++)
	{
		cout<<W(Range(i,i+1),Range(0,10))<<endl;
		cout<<W_list_new[0](Range(i,i+1),Range(0,10))<<endl;
		Mat dif=abs(W_list_new[0].row(i)-W.row(i));
		double min, max;
		cv::minMaxLoc(dif, &min, &max);
		cout<<"max dif: "<<max<<endl;

		{
		/*	Mat dif=abs(W_list_new1[0].row(i)-W.row(i));
			double min, max;
			cv::minMaxLoc(dif, &min, &max);
			cout<<"max dif: "<<max<<endl;*/
		}

		Mat tmp(10,10,CV_8UC3);
		imshow("pause",tmp);
		waitKey();
	}



}

vector<vector<int>> readData(char *inputName)
{
	vector<vector<int>>res;
	ifstream in("W_Train\\dataX_0.dat",ios::in);
	while(in)
	{
		char name[10000];
		in.getline(name,9999);
		vector<int> onesInd;
		string nameStr=name;
		for(int i=0;i<nameStr.length();i++)
		{
			if(nameStr[i]==':')
			{
				for(int j=i-1;j--;j>=0)
				{
					if(nameStr[j]==' ')
					{
						string curIndStr=nameStr.substr(j+1,i-j);
						int curInd= atoi(curIndStr.c_str())-1; 
						if(nameStr[i+1]=='1')
							onesInd.push_back(curInd);
						break;
					}
				}
			}
		}
		res.push_back(onesInd);
	}

	return res;
}

void loadW(char *name)
{
	RF_WholeFace tmp;

	Mat W;
	tmp.getW(name,W);

	cout<<W.size()<<endl;

	vector<vector<int>> onesInd=readData("W_Train\\dataX_10.dat");

	vector<float> res(onesInd.size());
	for(int i=0;i<onesInd.size();i++)
	{
		res[i]=0;
		for(int j=0;j<onesInd[i].size();j++)
		{
			res[i]+=W.at<float>(onesInd[i][j]);
		}
		res[i]+=W.at<float>(W.rows-1);
	}
	//cout<<W<<endl;

	for(int i=0;i<20;i++)
		cout<<res[i]<<" ";
	cout<<endl;
}

void runEXE(char *str)
{
	LPCTSTR lpApplicationName=CA2W(str);
   // additional information
   STARTUPINFO si;     
   PROCESS_INFORMATION pi;

   // set the size of the structures
   ZeroMemory( &si, sizeof(si) );
   si.cb = sizeof(si);
   ZeroMemory( &pi, sizeof(pi) );

  // start the program up
  CreateProcess( NULL,   // the path
    CA2W(str),        // Command line // -s 12 -p 0 -B 1 D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\hear_scale D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\heart_scal_model_new
    NULL,           // Process handle not inheritable
    NULL,           // Thread handle not inheritable
    FALSE,          // Set handle inheritance to FALSE
    0,              // No creation flags
    NULL,           // Use parent's environment block
    NULL,           // Use parent's starting directory 
    &si,            // Pointer to STARTUPINFO structure
    &pi );           // Pointer to PROCESS_INFORMATION structure
    
   // Wait until child process exits.
   WaitForSingleObject( pi.hProcess, INFINITE );

    // Close process and thread handles. 
    CloseHandle( pi.hProcess );
    CloseHandle( pi.hThread );

}
void treeSLTest() //passed
{
	//build a test trees
	RandomForests tree;
	tree.nodes.resize(2);
	

	tree.nodes[0].left=new TreeNode;
	tree.nodes[0].right=new TreeNode;

	tree.nodes[0].index_pair.threshold=2;
	tree.nodes[0].index_pair.u=Point2f(2,4);

	tree.nodes[0].left->leafInfo=new LeafInfo;
	tree.nodes[0].left->leafInfo->sampleNum=1;
	tree.nodes[0].left->leafInfo->ptsDif=Point2f(-1,2);
	tree.nodes[0].left->leafInfo->LBF=Mat::zeros(1,10,CV_32FC1)+2;

	tree.nodes[0].right->leafInfo=new LeafInfo;
	tree.nodes[0].right->leafInfo->sampleNum=2;
	tree.nodes[0].right->leafInfo->ptsDif=Point2f(1,-2);
	tree.nodes[0].right->leafInfo->LBF=Mat::zeros(1,10,CV_32FC1)+3;


	tree.nodes[1].left=new TreeNode;
	tree.nodes[1].right=new TreeNode;

	tree.nodes[1].index_pair.threshold=3;
	tree.nodes[1].index_pair.u=Point2f(2,65);

	tree.nodes[1].right->leafInfo=new LeafInfo;
	tree.nodes[1].right->leafInfo->sampleNum=22;
	tree.nodes[1].right->leafInfo->ptsDif=Point2f(11,-22);
	tree.nodes[1].right->leafInfo->LBF=Mat::zeros(1,10,CV_32FC1)+6;

	TreeNode *curNode=tree.nodes[1].left;
	curNode->left=new TreeNode;
	curNode->index_pair.threshold=65;
	curNode->index_pair.u=Point2f(22,65);
	curNode->index_pair.v=Point2f(-2,650);

	curNode=curNode->left;
	curNode->leafInfo=new LeafInfo;
	curNode->leafInfo->sampleNum=21;
	curNode->leafInfo->ptsDif=Point2f(11,-2);
	curNode->leafInfo->LBF=Mat::zeros(1,10,CV_32FC1)-13;

	tree.printTrees();

	cout<<"----------------------------------------------\n";

	//get LBF and check
	tree.transformFormat();
	
	vector<LeafInfo *> leafInfo;
	leafInfo.push_back(tree.TreeVectors[0][2]->leafInfo);
	leafInfo.push_back(tree.TreeVectors[1][1]->leafInfo);

	LBFFeature finalFeature=tree.getLBF(leafInfo);
	Mat curLBF=Mat::zeros(1,finalFeature.totalNum,CV_32FC1);
	for(int i=0;i<finalFeature.onesInd.size();i++)
	{
		curLBF.at<float>(finalFeature.onesInd[i])=1;
	}
	cout<<curLBF<<endl;


	//ofstream out("testTree.bin");
	//tree.save(out);
	//out.close();

	//ifstream in("testTree.bin");
	//RandomForests newTree;
	//newTree.load(in);
	//newTree.printTrees();
}



void pridict(char *modelName)
{

	cout<<"loading modles\n";
	LocalGlobalRegression trainer;
	trainer.load_CVPR14(modelName);


	cout<<"loading testing data\n";
	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testInternet\\imgList.txt";
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
		IplImage *img=cvLoadImage(nameStrList[i].c_str(),0);
		trainer.predict_Real_CVPR14(img);
	}
}


void pridict_direct14(char *modelName)
{

	cout<<"loading modles\n";
	LocalGlobalRegression trainer;
	trainer.load_CVPR14(modelName);
	
	if(1)
	{
		trainer.visualizeModel();
		return;
	}


	cout<<"loading testing data\n";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset\\imgList.txt";
	char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\processed\\imgList.txt";
	//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testInternet\\imgList.txt";
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
		IplImage *img=cvLoadImage(nameStrList[i].c_str(),0);
		Shape curShape(trainer.refShape.n);
		curShape.setShapeOnly(trainer.refShape);
		trainer.predict_CVPR14(cvarrToMat(img),curShape);
	}
}

void train(char *name)
{
	//char *nameList="D:\\Fuhao\\face dataset new\\faceRegression\\trainset\\ptsList.txt";
	//char *nameList="D:\\Fuhao\\face dataset new\\faceRegression\\testset\\ptsList.txt";
	char *nameList="D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\ptsList.txt";
	char *paraSettingFile="D:\\Fuhao\\FacialFeatureDetection_Regression\\regressionSetting_CVPR14_Accurate.txt";
	LocalGlobalRegression trainer;
	trainer.train_CVPR14(nameList,paraSettingFile);
	
	trainer.save_CVPR14(name);

	/*LocalGlobalRegression trainer1;
	trainer1.load_CVPR14(name);
	float error2=trainer1.evlError(trainer.shapes,trainer1.forests);

	cout<<error1<<" "<<error2<<" "<<(error1==error2)<<endl;

	return;*/

	//do prediction directly
	{
		cout<<"loading testing data\n";
		//char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset\\ptsList.txt";
		char testSetName[]="D:\\Fuhao\\face dataset new\\faceRegression\\testInternet\\imgList.txt";
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
			IplImage *img=cvLoadImage(nameStrList[i].c_str(),0);
			trainer.predict_Real_CVPR14(img);
		}
	}
}

void evaluation_cvpr14()
{
	char testNameList[]="D:\\Fuhao\\face dataset new\\faceRegression\\testset1\\ptsList.txt";

	vector<string> modelNameList;
	vector<string> evlResList;

	//modelNameList.push_back("model_LocalGlobal_RandLarge_WRCV_fast.bin");
	//modelNameList.push_back("model_LocalGlobal_RandLarge_WRCV_moreComplete_fast.bin");
	modelNameList.push_back("model_LocalGlobal_RandLarge_WRCV_Advanced_fast.bin");
	//evlResList.push_back("D:\\Fuhao\\face dataset new\\faceRegression\\evlRes_cvpr14\\model_cv_3Candidates.txt");
	evlResList.push_back("D:\\Fuhao\\face dataset new\\faceRegression\\evlRes_cvpr14\\model_cv_4Candidates_advanced.txt");

	//modelNameList.push_back("D:\\Fuhao\\face dataset new\\faceRegression\\trainsetLarge\\model_Uni_Rot_large_gaussian.bin");
	//char saveName[500];
	//sprintf(saveName,"D:\\Fuhao\\face dataset new\\faceRegression\\evlRes\\model_Uni_Rot_large(gaussian)_large%d.txt",sampleNum);
	//evlResList.push_back(saveName);

	
	for(int i=0;i<evlResList.size();i++)
	{
		Evaluation_CVPR14 evlCorrect;	
		evlCorrect.doevaluation((char *)modelNameList[i].c_str(),testNameList,
			(char *)evlResList[i].c_str());
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

void realtimeFace(char *modelName)
{
	LocalGlobalRegression trainer;
	trainer.load_CVPR14(modelName);
	FaceTracker t(&trainer);
	t.start();
}

