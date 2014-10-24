#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"      
#include "opencv/highgui.h" 
#include <iostream>
#include "Fern.h"
#include "shape.h"
using namespace cv;
using namespace std;


#include "FaceFounder.h"

namespace ESRAligner {

  class TwoLevelRegression
  {
  public:

    TwoLevelRegression(bool isRot = true);
    //highest level: nameList and parameter setting
    void train(char *nameList, char *paraSetting);

    //----------training-----------------------------
    //normalize all images
    void prepareTrainingData(char *nameList);
    Shape normalizeTrainingData(char *name, bool isRef = false);
    //--------------------------------------------

    //------------testing------------------------
    void pridict(Mat &, Shape &, bool showSingleStep = false);
    //---------------------------------------------
    void processRefShape(Shape&);

    int T, K;
    //vector<Level1Regressor> L1Regressors;
    vector<Fern> ferns;
    //Fern fern;
    void pridict(Mat &, Shape &, float givenS, float givenTx, float givenTy, bool showSingleStep = false);

    //T: num of L1 regressors
    //K: num of L2 regressors
    //F: num of features used in each Fern
    //P: num of candidate feature pair
    void train(int T, int K, int F, int P, int augNum);

    //augment the shapes to form shape pairs
    void prepareTrainingData(vector<Shape> &, Shape &, int augNum);

    //use ref face
    void prepareTrainingData_refFace(vector<Shape> &, Shape &);

    //generate P feature locations, and form the P^2 feature pair, 
    //also the convarance
    void generateFeatureLocation(int, Shape &refShape);

    vector<Mat> trainingImgs;
    vector<ShapePair> shapes;	//the augmented training data
    vector<Shape> inputShapes;//the pure training data

    Shape refShape;
    float refFaceWidth;

    int refWidth, refHeight;
    Point tl;

    vector<cv::Point3f> featureLocations;

    //save which two indices out of P indices are used
    vector<cv::Point> featurePIndex;
    Mat correlationMat;
    Mat selfCorrelationMat;

    Mat featureVector;
    vector<float> featureVectorMean;

    Mat featurePairVal;
    vector<float> featurePairMean;
    Mat dsVector;

    void getDS(vector<ShapePair> &curShape);

    //parameter1: template, parameter 2: feature vectors
    Mat myCorr(Mat &projectedDs, Mat &featureVector,
      Mat &correlationMat);

    string refShapeStr;
    void saveFerns(char *name);
    void loadFerns(char *name);

    void saveFerns_bin(char *name);
    void loadFerns_bin(char *name);
    //augment the shape

    void pridict(Mat &img, const int sampleNum = 5, char *GTPts = NULL);

    //-----------evaluation and validation-----------------
    Mat pridict_evaluate(IplImage *img, const int sampleNum = 5, char *GTPts = NULL);
    void pridict_GT(IplImage *img, char *GTPts);
    void visualizeModel(char *refShapeName);


    //estimate scale and translation using face detection
    FaceDetector d;
    Mat pridict_real(IplImage *img, const int sampleNum = 5);

    bool pridict_real_full(IplImage *img, const int sampleNum = 5);
    vector<Mat> predict_real_givenRects(IplImage *img, vector<Rect> &faceRegionList, const int sampleNum = 5);
    vector<Mat> predict_real_givenRects_L2(IplImage *img, vector<Rect> &faceRegionList, TwoLevelRegression &model, const int sampleNum = 5);
    Mat predict_single(IplImage *img, Rect facialRect, int sampleNum);

    //two level models
    Mat predict_single_lv2(IplImage *img, Rect facialRect, int sampleNum, TwoLevelRegression &model);

    //direct use current image for detection
    Mat predict_single_direct(IplImage *img, int sampleNum);
    //dorect ise current image and shape for regression
    Mat predict_single_directImgShape(Mat &img, Shape &s);

    bool showRes;
    float estimateScale(Shape &curShape, Rect curFaceRect);

    void estimateST(Shape &curShape, Rect curFaceRect, float &s, float &tx, float &ty);

    bool isRot;
  };

}