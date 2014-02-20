#ifndef RANDTREE_H
#define RANDTREE_H

#define NOMINMAX
#include <Windows.h>

#include "opencv2/ml/ml.hpp"
#include "opencv2/core/core_c.h"
#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include <stdio.h>
#include <map>
#include <iostream>
#include "shape.h"
#include <vector>
#include "GRandom.h"
#include <math.h>
#include "sharedDefination.h"
#include <vector>
using namespace std;
using namespace cv;

#define LEFT_NULL 99999
#define RIGHT_NULL 11111
#define LEFT_NODE 999990
#define RIGHT_NODE 111110
#define TREE_START 888888
#define LEAF_NODE_LEFT 123456
#define LEAF_NODE_RIGHT 654321
#define ROOT_NODE 666666
#define END_TREE 798798

#define GRIDDIM 10
struct Sample
{
	int label;
	CvPoint pos;
	Mat imgMat;
	Mat gradientMap;
	double **LocalGrids;
};




struct sampleImage
{
	sampleImage()
	{
		;
	}
	sampleImage(Mat depthImage,Mat colorImage,double **point,int num,Shape *shape=NULL)
	{
		img=depthImage.clone();
		gradientMap=colorImage.clone();
		pts=new double *[num];
		for (int i=0;i<num;i++)
		{
			pts[i]=new double [2];
			pts[i][0]=point[i][0];
			pts[i][1]=point[i][1];
		}
	}

	sampleImage(Mat dst,double **point,int num,Shape *shape=NULL)
	{
		img=dst.clone();
		pts=new double *[num];
		for (int i=0;i<num;i++)
		{
			pts[i]=new double [2];
			pts[i][0]=point[i][0];
			pts[i][1]=point[i][1];
		}
		return;
		
		CvMat *x_kernel=cvCreateMat(3,3,CV_64FC1);
		CvMat *y_kernel=cvCreateMat(3,3,CV_64FC1);
		//CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=-1;
		//CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=1;
		//CV_MAT_ELEM(*x_kernel,double,1,0)=-2;CV_MAT_ELEM(*x_kernel,double,1,2)=2;
		//CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;

		CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=0;
		CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=0;
		CV_MAT_ELEM(*x_kernel,double,1,0)=-0.5;CV_MAT_ELEM(*x_kernel,double,1,2)=0.5;
		CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;
		cvTranspose(x_kernel,y_kernel);

		Mat g_x,g_y;
		g_x=cvCreateMat(dst.rows,dst.cols,CV_64FC1);
		g_y=cvCreateMat(dst.rows,dst.cols,CV_64FC1);

		Mat d_img=img.clone();
		d_img.convertTo(d_img,CV_64FC1);

		filter2D(d_img,g_x,g_x.depth(),cvarrToMat(x_kernel));  
		filter2D(d_img,g_y,g_x.depth(),cvarrToMat(y_kernel)); 
		//cvFilter2D(img,g_y,y_kernel,cvPoint(-1,-1));  
		Mat gradientMap_double=cvCreateMat(dst.rows,dst.cols,CV_64FC1);
		pow(g_x,2.0f,g_x);
		pow(g_y,2.0f,g_y);

		//cout<<g_x.depth()<<" "<<gradientMap_double.depth()<<endl;
		sqrt(g_x+g_y,gradientMap_double);

		double meanV=mean(gradientMap_double).val[0];
		for (int i=0;i<gradientMap_double.rows;i++)
		{
			for (int j=0;j<gradientMap_double.cols;j++)
			{
				if (gradientMap_double.at<double>(i,j)<=meanV)
				{
					gradientMap_double.at<double>(i,j)=0;
				}
				else
				{
					gradientMap_double.at<double>(i,j)=1;
				}
			}
		}

		//eliminating the isolated values
		Mat labels=gradientMap_double.clone()*0;
		vector<Point> pointList;
		vector<Point> pointSaveList;
		int tx,ty;
		//int totalNum;
		for (int i=0;i<gradientMap_double.rows;i++)
		{
			for (int j=0;j<gradientMap_double.cols;j++)
			{
				pointList.clear();
				pointSaveList.clear();
				if (gradientMap_double.at<double>(i,j)==1&&labels.at<double>(i,j)==0)
				{
				//	totalNum=1;
			/*		namedWindow("1");
					imshow("1",gradientMap_double);
					waitKey();*/

					pointList.push_back(Point(i,j));
					pointSaveList.push_back(Point(i,j));
					labels.at<double>(i,j)=1;
					while (pointList.size()!=0)
					{
						Point tmpP=pointList.at(pointList.size()-1);
						//cout<<tmpP.x<<" "<<tmpP.y<<endl;
						pointList.pop_back();
							
						
						
						tx=tmpP.x-1;
						ty=tmpP.y-1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x;
						ty=tmpP.y-1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x+1;
						ty=tmpP.y-1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x-1;
						ty=tmpP.y;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x+1;
						ty=tmpP.y;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x-1;
						ty=tmpP.y+1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x;
						ty=tmpP.y+1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x+1;
						ty=tmpP.y+1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}
					}
					if (pointSaveList.size()<100)
					{
						for (int k=0;k<pointSaveList.size();k++)
						{
							Point tmp=pointSaveList.at(k);
							gradientMap_double.at<double>(tmp.x,tmp.y)=0;
						}
					}

					/*namedWindow("1");
					imshow("1",gradientMap_double);
					waitKey();*/
				}

			
			}
		}

		gradientMap=img.clone();
		for (int m=0;m<gradientMap.rows;m++)
		{
			for (int n=0;n<gradientMap.cols;n++)
			{
				gradientMap.at<uchar>(m,n)=gradientMap_double.at<double>(m,n)*255;
			}
		}

		/*namedWindow("1");
		imshow("1",gradientMap);
		waitKey();*/
		
	
	}

	/*~sampleImage()
	{
		;
	}*/
	void distroy(int num)
	{
		img.release();
		//cvFree((void**)&img);
		/*char* imgData = (char*)img.data;

		delete [] imgData;*/
		for (int i=0;i<num;i++)
		{
			delete []pts[i];
		}
		delete []pts;
		//~sampleImage();
	}
	Mat img;
	Mat gradientMap;
	double **pts;
};

class RandTree
{
public:
	RandTree();
	//~RandTree();
	RandTree(int _max_depth,int _min_sample_count,double _regression_accuracy,int _max_num_of_trees_in_the_forest,int,int);
	void SetWindowSize(int w_size);

	int response_idx;

	void train(string name);
	 CvRTrees rtrees;
	 CvMLData data;

	  CvTrainTestSplit spl;

	  void getData(string name);
	  void getSample(string name);
	  void getSample(int num);
	  Shape **shape;
	  Mat fullData;
	  Mat response;
	  CvMat *cvfulldata,*cvResponse;
	 // int sampleNum;
	  int sampleDim;
	  int shapeNum;

	  float* getDiscriptor(IplImage *,double *center,Mat &list);
	  Mat indexList;
	  bool haveIndex;
	
	  void predict(IplImage *);
	  Mat & setValue(CvSeq*);

	 // string modelName;
	  //Mat & setValue_Pts(CvSeq*);

	  vector<Sample> samples; //all the training samples
	 // int originalSampleNum;

	  int max_depth;
	  int min_sample_count;
	  int max_num_of_trees_in_the_forest;
	  double regression_accuracy;
	  int windowSize;
	  void sampleWindowIndex(int sampledNum,vector<CvPoint> &SampledPosition,vector<CvPoint> &SampledPosition1);
	  void split(Node *node);
	  void split_WithDepth(Node *node);
	  void RandTree::split_color(Node *node);
	  void caculateProb(Node *node);//caculate p for each leaf
	  void train();
	  int treeNum;
	  int rootSampleNum;
	  Node **roots;
	  bool fullRandom;
	  Mat img;
	  bool is_satisfied;
	  int sampleNum;
	  int labelNum;//the number of classes

	  int getLabel(Node *root,Mat mat,CvPoint &);//pridict label from single tree
	  void getProb(Node *root,Mat mat,CvPoint &,LeafNode &);//return a leafnode
	  void getProb(Node *root,Mat mat,Mat gradient,CvPoint &,LeafNode &);//return a leafnode
	  void getProb(Node *root);//recursively get all the probs

	  void outPutLeaves(Node *root);
	  void predict_rt(IplImage *img);
	  void predict_rt(string ImgName);
	  void predict_fulltest(string ImgName);
	  
	  void setThreshold(double);
	  void total_pridict(Mat mat,CvPoint &pos,LabelResult &);
	  void pridict_prob(Mat mat,CvPoint &pos,LabelResult &);
	
	  void pridict_prob(Mat mat,Mat gradient,CvPoint &pos,LabelResult &);


	  string path;
	  void save();//save the tree
	  void save(int);//save the tree
	  void save(string name);//save the tree
	  ofstream out;
	  ifstream in_file;
	  void saveTree(Node *,ofstream &);
	  void outputTree(Node *);//the cout version of saving a tree
	  void load(string name);
	
	  void loadTree(Node *last,ifstream &in);
	  double threshold;
	
	  int totalSampleNumTest;

	  void showTree(Node *root,Mat mat,CvPoint &pos);
	  void showTree_recur(Node *root,Mat mat,CvPoint &pos);

	  void showTree(Node *root,Mat mat,Mat gradient,CvPoint &pos);
	  void showTree_recur(Node *root,Mat mat,Mat gradient,CvPoint &pos);


	  int showMaxLayer;
	  Mat finalImage;
	  int *layerNum;
	  	int ababdantsize;

		//the sampled images from giving training images
	vector<sampleImage> **sampledImage;

	void output_probimg(Mat mat,string fileName,bool isshow=false);

	double colorIndex[1000][3];

	//predict an image list
	float predictPoint(int x,int y,Mat &colorImg,Mat &depthImg,int index=0,float refProb=1);
	void predict_imgList(string listName);
	void predict_DepthImgList(string listName);
	void predict_imgList_fast(string listName,bool wirteFile=false);
	void predict_DepthList_fast(string listName,bool writeFile=false);
	void predict_DepthList_fast_withLabelCenter(string listName,bool writeFile=false);

	void predict_img_transform(string imgName,double angle=0, double scale=1,bool isshow=false, bool isflip=false);
	//void clearPts();

	void imageFeaturePts(string imgname);

	bool usingCUDA;
	
	void predict_imgList_GPU(string listName);
	void predict_imgList_GPU_depth(string listName);

	void predict_rt_GPU(IplImage *img,string);
	void predict_rt_depth_GPU(Mat &depthImage,Mat &colorImg,string);
	void predict_rt_color_GPU(Mat &depthImage,Mat &colorImg,string);
	void predict_rt_depth_color_GPU(Mat &depthImage,Mat &colorImg,string);

	float *host_inputImage;

	float *host_colorImage;
	float *host_depthImage;

	float *labelResult;

	vector<Sample> *addedSamples;

	bool isevaluating;
	int trainStyle;
	void split_gradient(Node *node);
	void sampleWindowIndex_gradient(int sampledNum,vector<int> &gradientCandidate,vector<CvPoint> &SampledPosition1,vector<CvPoint> &SampledPosition2);
	void split_output(Node *node,string namebase);

	void RGB2YIQ(Mat &src,Mat &dst);

	void split_mixture(Node *node);

	void getGradientMap(Mat &src,Mat &dst);

	string *nameList;
	void getNameList(string name);
	int TotalShapeNum;

	void distroyAllData();
	int sampleNumEveryTime;

	///////////the combination on CPU/////////////////
	//set up the prob map from detection results
	Mat *probabilityMap;
	Mat *probabilityMap_full;
	vector<Point>*centerList;
	void setupProbMaps(int width,int height,float *result);
	void setupProbMaps(int width,int height,float *result,vector<int>&visibleInd);
	void findProbModes(int width, int height,float *result);
	int interestPtsInd[MAX_LABEL_NUMBER];
	int numOfLabels;


	//deal with missing data
	float predictPoint_missingData(int x,int y,Mat &colorImg,Mat &depthImg,int index=0,float refProb=1);
	float predictPoint_missingData(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &tmp);
	void pridict_prob_missingData(Mat mat,CvPoint &pos,LabelResult &);
	void pridict_prob_missingData(Mat mat,Mat gradient,CvPoint &pos,LabelResult &);

	void getProb_missingData(Node *root,Mat mat,CvPoint &,LeafNode &);//return a leafnode
	void getProb_missingData(Node *root,Mat mat,Mat gradient,CvPoint &,vector<LeafNode> &);//return a leafnode
	void predict_IMG(Mat &colorIMg,Mat &depthImg,Mat *result,int startX=0,int endX=10000,int startY=0,int endY=10000);

	//combination of depth and color image
	void predict_IMG_Both(Mat &colorIMg,Mat &depthImg,Mat *result,RandTree *rt_color,RandTree *rt_depth,int startX=0,int endX=10000,int startY=0,int endY=10000);
	void predictPoint(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &);
	//double standardDepth;
	//int trainStyle;

	  void load_prob(string name,int type=0);
	   void loadTree_prob(Node *last,ifstream &in);
	   void predict_IMG_LRprob(Mat &colorIMg,Mat &depthImg,Mat *result,int startX=0,int endX=10000,int startY=0,int endY=10000);
	   float predictPoint_LRprob(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &tmp);
	   void pridict_prob_LRprob(Mat mat,Mat gradient,CvPoint &pos,LabelResult &);
	   void getProb_LRprob(Node *root,Mat mat,Mat gradient,CvPoint &,vector<LeafNode> &);//return a leafnode
	   //float predictPoint_missingData(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &tmp);

	   //visualize process with depth and color
	   void showTree_both(Node *root,Mat &mat,Mat &gradient,CvPoint &pos);
	   void showTree_recur_both(Node *root,Mat &mat,Mat &gradient,CvPoint &pos);

	   void showTree_both_pathOnly(Node *root,Mat &mat,Mat &gradient,CvPoint &pos);
};

#endif