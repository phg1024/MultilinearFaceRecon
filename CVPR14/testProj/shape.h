#pragma once
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv/cv.h"      
#include "opencv/highgui.h" 
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;

namespace ESRAligner {

#define eyeIndL 2
#define eyeIndR 3
  class Shape
  {
  public:
    Shape();
    Shape(int n);
    Shape(Shape *);

    void setPtsNum(int);

    void setShape(Shape &);

    void setImg(Mat &);
    void setShapeOnly(Shape &);
    void setScaleTranslatrion(float _s, float _tx, float _ty);
    int n;//size of points
    //vector<Point3f> pts;

    vector<Point2f> pts;
    Mat ptsVec;

    Shape operator+(const Shape &);
    Shape operator-(const Shape &);
    void operator+=(const Shape &);
    Shape operator/(const float&);
    //for inverse translation
    void estimateTrans(Shape &ref);

    //the local adjustment, only need to calculate s (and R?)
    void estimateTrans_local(Shape &ref);
    float s_local;
    Mat RS_local;
    //new add0926: add random local scale and transformation
    void addLocalST(Shape &ref, int refWidth, int refHeight);


    Mat R;
    float tx, ty;
    float s;

    //Mat pts in reference space to the image space using R, tx, ty and s
    Mat getCurFeature(vector<Point3f> &pts);
    Mat getFeature(Mat &, vector<Point2f> &);

    float getCurFeature(int ptsInd, Point2f offset);
    float getCurFeature_GivenImg(Mat &img, int ptsInd, Point2f offset);
    float getSubPixel(Mat &img, Point2f pts);

    IplImage *ImgPtr;
    Mat orgImg;

    void syntheorize();

    void load(char*, bool readImg = true);

    void generateGlobalFeature(vector<Point2f> &pts_out);

    Mat getFinalPosVector(float curScale, Point2f curST);

    //for testing
    void visualize(Mat &, vector<Point2f> &, char *);
    void visualizePts(char *);

    void save(ofstream &out);
    void load(ifstream &in);
  };

  class ShapePair :public Shape
  {
  public:
    Shape gtShape;

    void setGTShape(Shape &input, Shape &ref);
    //void estimateTrans
    Shape dS();
  };

}