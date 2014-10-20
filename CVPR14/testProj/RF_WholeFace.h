#pragma once
#include "RandomForest.h"
#include "Shape.h"
class RF_WholeFace
{
public:
	RF_WholeFace();
	//void train(vector<ShapePair> &shapes, int treeNum, int depth);
	void train_unit_validation(vector<ShapePair> &shapes, Shape *refShape,int treeNum, int depth, float radius,vector<ShapePair> *);
	void train_unit(vector<ShapePair> &shapes, Shape *refShape,int treeNum, int depth, float radius);
	void learnW(vector<ShapePair> &shapes,vector<LBFFeature> &trainLBF, Mat &curW, float C);
	void learnW_unit(vector<Shape> &ds,vector<LBFFeature> &trainLBF, Mat &curW, float C);
	void learnW_crossValidation(vector<ShapePair> &shapes,vector<ShapePair> *validationShapes, vector<RandomForests> &forests, Mat &curW);
	float optimalC;
	vector<float> c_list;
	float getError(LBFFeature &lbfFeatureValidationList, Mat &, ShapePair &validationShapes);

	Mat trainSingle(vector<Shape> &dSList, int ind, vector<LBFFeature> &, float C, bool isX);

	void trainW(char *);
	void getW(char *, Mat &);

	vector<RandomForests> forests;	//with size PtsNum
	void predict(Mat &img, Shape &s);

	Mat fastAdd(vector<int> &onesInd,int l, int r);
	LBFFeature predict_local(Mat &img, Shape &s);
	Mat LBF;

	//float evlError(vector<ShapePair> &,vector<RandomForests> &, Mat &curW);
	//vector<ShapePair> *testShapes;

	Mat W;// W can be either local or global

	//for testing only
	bool showSingleStep;

	

	void save(ofstream &);
	void load(ifstream &);

	void visualize(char *name);
private:
	//vector<double>candidateRadius; //0.05:0.05:0.5
	//int candidateNum;
};