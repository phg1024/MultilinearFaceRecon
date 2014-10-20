#pragma once

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "RandomForest.h"
#include "linear.h"
using namespace std;

class W_train
{
public:
	W_train(int startInd=0);

	
	void train_singleC(vector<Shape> &ds,vector<LBFFeature> &trainLBF, Mat &curW, float C);

	void train_multipleC(vector<Shape> &shapes,vector<LBFFeature> &trainLBF, vector<Mat> &WList, vector<float> &C_List);

	void train_unit(problem &prob, Mat &W, parameter &param);

	void train_batch(vector<problem> &probs, Mat &W, float C);
	int startInd;

	bool showSingleStep;
private:
	void obtainProblems(vector<problem> &probs, vector<Shape> &ds, vector<LBFFeature> &trainLBF);
	void obtainProblem(problem &prob, vector<Shape> &ds, int ptsInd, vector<LBFFeature> &trainLBF, problem *ref=NULL);
	feature_node *x_space;
	double bias;
	void setPara(parameter &param, double c);
};