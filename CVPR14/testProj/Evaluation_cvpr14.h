#pragma once

#include "LocalGlobalRegression.h"

class Evaluation_CVPR14
{
public:
	void doevaluation(char *modelName,char *testNameList,char *saveName);
	void evaluate(char *testNameList,LocalGlobalRegression *trainer);
	Mat evalRes;
	void saveRes(char *);
	void analysisError(Mat &);

	//check result of each iteration
	void checkIteration(char *modelName,char *testNameList);

	//check result started from GT shape
	void checkConvergeGT(char *modelName,char *testNameList);

	void obtainNameList(char *testNameList,vector<string> &nameStrList);

	void checkModel(char *modelName, char *refShapeName);
};