#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"

#include <Windows.h>
#include <iostream>
#include <fstream>
#include <map>
#include "HashTable.h"
using namespace std;
using namespace cv;

//struct Property
//{
//	Property(int _exampleID,int _basisID)
//	{
//		ExampleID=_exampleID;
//		basisID=_basisID;
//	}
//	int ExampleID;
//	int basisID;
//};

struct basisPairMatrix
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

class geoHashingMatrix
{
public:
	geoHashingMatrix(Mat &shapes,float _dsize);

	float dSize;

	HashTable *table;

	int ptsNum;

	int shapeNum;

	Mat keyIndexTable;
	int smallestX;
	int smallestY;

	int basisNum;
	basisPairMatrix *basisTabel;

	//int **: basis*example
	vector<Mat> *dataTabel; //the same id as table indicates

	
	float discretWindow;
	void buildSingleTabel(Mat &data,int basisID,float discretNum);
	void buildHashTabel(int pairNum,basisPairMatrix* pairList,Mat &data);

	
	float discretValue(float input,float disceteNum);
	void buildBasisTabel(int featureNum,basisPairMatrix* pairList);

	void showTabel();


	void vote(Mat &data,vector<int>&exampleCandidate,int thresNum);
	void vote_countAll(Mat &data,vector<int>&exampleCandidate,int thresNum);

	void saveTable(char *name);
	void loadTable(char *name);

	Mat originalTrainingData;

	//float tmpEntry[1000][2];
};