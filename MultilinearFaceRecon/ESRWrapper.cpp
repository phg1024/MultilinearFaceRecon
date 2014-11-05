#include "ESRWrapper.h"

ESRWrapper::ESRWrapper()
{
  setup();
}


ESRWrapper::~ESRWrapper()
{
}

void ESRWrapper::reset() {

}

void ESRWrapper::resize(int w, int h) {
  colorImage.create(h, w, CV_8UC4);
  depthImg = cv::Mat::zeros(h, w, CV_32FC1);
}

void ESRWrapper::setup() {
  char *modelName = "../Data/model_LocalGlobal_RandLarge_WRCV_Advanced_fast.bin";
  engine.load_CVPR14(modelName);

  colorImage.create(480, 640, CV_8UC4);
  depthImg = cv::Mat::zeros(480, 640, CV_32FC1);

  f.reserve(256);
}

void ESRWrapper::printTimeStats() {
  cout << "Preparation time " << tPrep.gap() * 1000.0 << " ms." << endl;
  cout << "Detection time " << tDetect.gap() * 1000.0 << " ms." << endl;
  cout << "Prediction time " << tPredict.gap() * 1000.0 << " ms." << endl;
}

const vector<float>& ESRWrapper::track(const unsigned char* cimg,
  const unsigned char* dimg, int w, int h) {
  cout << "tracking a " << w << "x" << h << " image." << endl;
  /// realtime tracking related
  tPrep.tic();
  // copy the image over here
  memcpy(colorImage.ptr<BYTE>(), cimg, sizeof(unsigned char)*w*h * 4);

  // convert to gray image
  cvtColor(colorImage, colorIMG_Gray, CV_BGR2GRAY);

  IplImage pframe(colorImage);
  IplImage pgray(colorIMG_Gray);
  tPrep.toc();

  tDetect.tic();
  vector<Rect> rects = detector.findFaceFull(&pframe);
  tDetect.toc();
  if (rects.empty()) return eptf;

  /*
  cout << "bounding box:" << rects.front().tl().x 
       << ", " << rects.front().tl().y << "; "
       << rects.front().br().x << ", " 
       << rects.front().br().y << endl;
  */
  tPredict.tic();
  vector<cv::Mat> facesMat = engine.predict_real_givenRects(&pgray, rects);
  tPredict.toc();
  if (facesMat.empty()) return eptf;

  /// only track the first face
  f.resize(facesMat.front().cols);
  int npts = facesMat.front().cols / 2;
  for (int i = 0; i < npts; ++i) {
    f[i] = facesMat.front().at<float>(i * 2);
    f[i + npts] = facesMat.front().at<float>(i * 2 + 1);
    //cout << f[i] << '\t';
  }
  //cout << endl;

  return f;
}