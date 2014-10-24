#pragma once

#include <opencv/cv.h>
#include <opencv/highgui.h>

// Our library
#include "LibFace.h"
#include "Face.h"

using namespace std;

//Use namespace libface in the library.
using namespace libface;
using namespace cv;

namespace ESRAligner {

  class FaceDetector
  {
  public:
    FaceDetector(bool isShowFace = false);

    bool showFace;
    float scale;
    LibFace libFace;
    Rect findFace(IplImage *img);
    vector<Rect> findFaceFull(IplImage *img);
    //vector<Point2f> curSTList;

    Rect findFaceGT(IplImage *img, Rect &gtRect);

    Mat getCurFace(Rect &, IplImage *img);
    Point2f curST;

    void findFaceFull(IplImage *img, vector<Rect> &);

    Rect getRect(Face &result);

    Rect getEnlargedRect(Face &result);
  };

}