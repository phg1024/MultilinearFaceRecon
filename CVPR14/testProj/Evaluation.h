#pragma once

#include "TwoLevelRegression.h"

namespace ESRAligner {
  class Evaluation
  {
  public:
    void doevaluation(char *modelName, char *testNameList, char *saveName, int sampleNum);
    void evaluate(char *testNameList, TwoLevelRegression *trainer, int sampleNum);
    Mat evalRes;
    void saveRes(char *);
    void analysisError(Mat &);

    //check result of each iteration
    void checkIteration(char *modelName, char *testNameList);

    //check result started from GT shape
    void checkConvergeGT(char *modelName, char *testNameList);

    void obtainNameList(char *testNameList, vector<string> &nameStrList);

    void checkModel(char *modelName, char *refShapeName);
  };

}