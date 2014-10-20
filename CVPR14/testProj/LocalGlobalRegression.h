#pragma once
#include "RF_WholeFace.h"
#include "TwoLevelRegression.h"

class LocalGlobalRegression: public TwoLevelRegression
{
public:
	LocalGlobalRegression();
	//--------------train
	void train_CVPR14(char *nameList,char *paraSetting);
	void train_CVPR14();

	//--------------regression
	Mat predict_Real_CVPR14(IplImage *img);
	Mat predict_single(IplImage *img, Rect facialRect);

	void predict_CVPR14(Mat &img, Shape &s);
	void getUpdate(Mat &W,RF_WholeFace &forest, Shape &s);

	//SL
	void save_CVPR14(char *name);
	void load_CVPR14(char *name);

	vector<ShapePair> *testShapes;
	float evlError(vector<ShapePair> &,vector<RF_WholeFace> &,int startInd=0);
	float getError(Shape &curShape,Shape &gtShape);

	void updateShapes(vector<ShapePair> &shapes,RF_WholeFace &forest);

	vector<ShapePair> validateShapes;

	vector<RF_WholeFace> forests; //should be whole face then

	void visualizeModel();

	//evaluation
	Mat pridict_evaluate(IplImage *img, char *GTPts=NULL);

	//multiface detection
	vector<Mat> predict_real_givenRects(IplImage *img,vector<Rect> &faceRegionList);
	
private:
	vector<Mat> WList;	//global weight
	
	int T,N,D;//level num, 2nd level tree total Num and depth of trees
	int K;//2nd level num

	int augNum,ShapePtsNum;

	Mat fullLBF;

	//Shape refShape;
	vector<double>candidateRadius; //0.05:0.05:0.5
	int candidateNum;

	float eyeDis;
};