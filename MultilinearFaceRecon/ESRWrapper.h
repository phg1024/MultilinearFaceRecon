#pragma once

#include "phgutils.h"
#include "../CVPR14/testProj/testCode_14.h"
#include "Utils/Timer.h"

class ESRWrapper
{
public:
  ESRWrapper(void);
  ~ESRWrapper(void);

  void reset();
  float getTrackingError() const {
    return 0;
  }

  // tracking interface
  const vector<float>& track(const unsigned char* cimg, const unsigned char* dimg, int w, int h);

  float getPrepTime() const {
    return tPrep.elapsed();
  }

  void printTimeStats();

protected:
  void setup();

private:
  // input data
  Mat colorImage, colorIMG_Gray, depthImg;

  // flag: whether initial fitting or not
  bool initial;
  bool isRecording;
  // bounding box of the face region
  int startX, endX, startY, endY;

  ESRAligner::FaceDetector detector;
  ESRAligner::LocalGlobalRegression engine;

  // engine related parameters
  int startNum;	// not sure what it is
  int curStatus;

  float lastShape[200], currentShape[200];
  vector<float> ptsList;

  // returned points
  vector<float> f;
  vector<float> eptf;	// dummy

  PhGUtils::Timer tPrep, tDetect, tPredict;
};

