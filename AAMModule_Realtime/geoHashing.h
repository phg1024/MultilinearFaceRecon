#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"

#define NOMINMAX
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <map>
#include "HashTable.h"
#include "LSAlignment.h"

using namespace std;
using namespace cv;

struct Property
{
	Property(int _exampleID,int _basisID)
	{
		ExampleID=_exampleID;
		basisID=_basisID;
	}
	int ExampleID;
	int basisID;
};

struct basisPair
{
	int id1,id2;
};



//
//struct hashEntry
//{
//	hashEntry(int _basisID,int _exampleID)
//	{
//		basisID=_basisID;
//		propertyList=new vector<Property>;
//		propertyList->push_back(Property(_exampleID,keyBasis));
//	}
//	int basisID;
//	vector<Property> *propertyList;
//};

class GeoHashing
{
public:
	GeoHashing(Mat &shapes,float _dsize);

	float dSize;

	HashTable *table;

	int ptsNum;

	int shapeNum;

	int basisNum;
	basisPair *basisTabel;

	//SparseMat keyIndexTable;
	//basis*example
	Mat keyIndexTable;
	int smallestX;
	int smallestY;

	vector<vector<Property>> *dataTabel; //the same id as table indicates

	
	//memory and time efficient
	vector<vector<vector<int>*>*> dataTabelVec;
	void buildSingleTabelVec(Mat &data,int basisID,float discretNum);
	void buildHashTabelVec(int pairNum,basisPair* pairList,Mat &data);
	void vote_countAllVec(Mat &data,Mat &dataOldFormat,vector<int>&exampleCandidate,vector<int> &,int thresNum,int nnnum=30,vector<Point2f>*candidatePts=NULL,Mat *img=NULL,char *name=NULL);
	bool vote_countAllVec_old(Mat &data,vector<int>&exampleCandidate,vector<int> &,int thresNum,int nnnum=30,vector<Point2f>*candidatePts=NULL,Mat *img=NULL,char *name=NULL);
	bool reSort_vec(Mat &dectedFeatures,int *basisInd,vector<int> &inlierList,int nnNum=30,Mat *img=NULL,char *name=NULL,vector<Point2f>*candidatePts=NULL);


	float discretWindow;
	void buildSingleTabel(Mat &data,int basisID,float discretNum);
	void buildHashTabel(int pairNum,basisPair* pairList,Mat &data);

	
	float discretValue(float input,float disceteNum);
	void buildBasisTabel(int featureNum,basisPair* pairList);

	void showTabel();


	void vote(Mat &data,vector<int>&exampleCandidate,int thresNum);

	void initialVote(Mat &data,vector<int>&usedCandidateID,vector<Point>*candidatePts,Mat *img=NULL);
	void vote_countAll(Mat &data,vector<int>&exampleCandidate,vector<int> &,int thresNum,int nnnum=30,vector<Point2f>*candidatePts=NULL,Mat *img=NULL,char *name=NULL);



	void saveTable(char *name);
	void loadTable(char *name);

	Mat originalTrainingData;

	//float tmpEntry[1000][2];

	string inputImgName;

	LSSquare alignTools;
	void reSort(Mat &dectedFeatures,int *basisInd,vector<int> &inlierList,int nnNum=30,Mat *img=NULL,char *name=NULL,vector<Point2f>*candidatePts=NULL);


	void vote_weighted(Mat &data,vector<int>&exampleCandidate,int thresNum);
	float getWeight(float u,float v,float basisDis,float s,float x,float y,float sigma);


	//for speed up, put parameter definitions here
	Mat rotation;
	Mat *voteNum;
	/*Mat voteAll;*/
};