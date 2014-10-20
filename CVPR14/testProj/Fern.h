#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cv.h"      
#include "highgui.h" 
#include <iostream>
#include "shape.h"
using namespace cv;
using namespace std;

struct FernNode
{
	int ptsInd1,ptsInd2;
	float threshold;
	float dx1,dy1,dx2,dy2;

	void save(ofstream &);
	void load(ifstream &);
	//Shape *ds;
};

class Fern
{
public:
	Fern();
	~Fern();
	void train(vector<Point3f> &featureOffsets,vector<Point> &indexPairInd,
		vector<float> &threshold,vector<ShapePair> &shapes,Mat &curFeature);



	//compair at a particular fern node
	int getBinVal(Shape &, FernNode &fernNode);

	Shape pridict(Shape &s);

	void pridict_directAdd(Shape &s);

	void buildFern(int F,int curInd,vector<Point3f> &featureOffsets,
		vector<Point> &indexPairInd,vector<float> &threshold);

	vector<FernNode> fernNodes;

	vector<Shape> dsList;

	void convertDSMat();
	Mat *dsMat;

	inline int findBins(Shape &shape,vector<FernNode> &);
	int findBinsDirect(Mat &features,vector<FernNode> &);

	vector<vector<int>> binPool;//store the shape for each bin

	void save(ofstream &);
	void load(ifstream &);

	void saveBin(ofstream &);
	void loadBin(ifstream &);

	//Mat tmpMat;

	//validation
	void visualize(char *name,Shape &s);
};