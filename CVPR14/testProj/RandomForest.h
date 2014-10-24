#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"      
#include "opencv/highgui.h" 
#include <iostream>
#include "Shape.h"
using namespace cv;
using namespace std;

namespace ESRAligner {

  struct LBFFeature
  {
    LBFFeature()
    {
      totalNum = 0;
    }
    int totalNum;
    vector<int> onesInd;
  };

  struct LeafInfo
  {
    int sampleNum;
    Point2f ptsDif;

    Mat LBF;
    int fullNum;
    int oneInd;

    Mat getLBF();
    //int leafInd;
  };

  struct FeaturePair
  {
    FeaturePair()
    {
      u = v = Point2f(0, 0);
      threshold = 0;
    }
    Point2f u, v;
    float threshold;


    /*void operator=(const FeaturePair &indexPair)
    {
    u=indexPair.u;
    v=indexPair.v;
    threshold=indexPair.threshold;
    }*/
  };

  struct TreeNode
  {
    TreeNode()
    {
      left = right = NULL;
      leafInfo = NULL;
      leftInd = rightInd = -1;
      //isVisited=false;
    }

    void save(ofstream &out);
    void load(ifstream &in);

    TreeNode *left, *right;

    FeaturePair index_pair;

    LeafInfo *leafInfo;

    int curInd;
    int leftInd, rightInd;
    //bool isVisited;
  };



  class RandomForests
  {
  public:
    RandomForests(int _sampleNum = 500);
    //note: shapes need to be aligned to reference shape, so that we have R and s
    void train(vector<ShapePair> &shapes, int ind, float radius, int treeNum, int depth);

    LBFFeature predict(Mat &img, Shape &s, int ptsInd);
    LBFFeature getLBF(vector<LeafInfo *> &leafInfo);
    int leafNum;
    int sampleNum;



    //form 1: link
    vector<TreeNode> nodes;

    //form 2: vectors
    vector<vector<TreeNode *>> TreeVectors;
    void transformFormat();

    //predict
    LeafInfo *getLeaf(Mat &, Shape &s, int ptsInd, TreeNode *);

    void setRefShape(Shape *);

    bool showSingleStep;

    void save(ofstream &);
    void load(ifstream &);


    void printTrees();
    void printSingleTree(TreeNode *);

    Mat visualize();
    void drawRes(TreeNode *node, Mat &res, int treeInd);
    void drawRes_node(TreeNode *node, Mat &res, Point2f tl, Point2f c);
  private:
    void train_eachTree(vector<ShapePair> &shapes, vector<int> sampleInd, int ptsInd, float radius, int curDepth, TreeNode* node, Point2f lastMu = Point2f(-1000, -1000));

    //generate index paris with optimal radius, using reference shape here
    void genPoints(vector<FeaturePair> &curPoints, float radius, int GivenSampleNum);
    Point2f generatePts(float radius);

    int tree_depth;

    int findBestSplit(vector<float> &);

    //split
    void getInd(vector<ShapePair> &shapes, int ptsInd, vector<int> sampleInd, FeaturePair &indexPair,
      vector<int> &leftNodeIndex, vector<int> &rightNodeIndex);

    bool indexPairTest(Shape &s, int ptsInd, FeaturePair &indexPair);

    bool indexPairTest(Mat &img, Shape &s, int ptsInd, TreeNode *node);


    //split: calculate variance
    float getVar(vector<ShapePair> &shapes, int ind, vector<int> &sampleInd, FeaturePair& indexPair);
    float getMeanShape(vector<ShapePair> &shapes, vector<int> &ind, int ptsInd);

    Shape *refShape;
    int treeNum;

    int totalLeafNum;
    vector<int> LeafNumEachForest;

    vector<vector<float>> difVal_global;
    vector<FeaturePair> IndexPairs_global;
    float getVars(vector<ShapePair> &shapes, int ptsInd, vector<int> &sampleInd, FeaturePair &curIndexPair, int indexPairInd, Point2f&leftMu, Point2f &rightMu, Point2f &parentMu);
    void getInd(vector<int> sampleInd, int indexPairInd, FeaturePair &indexPair,
      vector<int> &leftNodeIndex, vector<int> &rightNodeIndex);
    float getMeanShape_mu(vector<ShapePair> &shapes, vector<int> &ind, int ptsInd, Point2f&mu);
  };

}