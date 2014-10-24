#include "RealtimeTracking.h"

namespace ESRAligner {

  FaceTracker::FaceTracker(TwoLevelRegression *_reg, bool _isCVPR14)
  {
    reg = _reg;
    sampleNum = 15;

    isCVPR14 = _isCVPR14;
  }


  FaceTracker::FaceTracker(LocalGlobalRegression *_reg)
  {
    reg_cvpr14 = _reg;
    isCVPR14 = true;
  }

  void FaceTracker::setSampleNum(int _sampleNum)
  {
    sampleNum = _sampleNum;

  }

  void FaceTracker::start()
  {
    //start the camera
    CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);

    if (!capture) // check if we succeeded

      return;

    //Mat frame;

    for (;;)
    {


      IplImage* frame = cvQueryFrame(capture);

      vector<Rect> rects = d.findFaceFull(frame);

      Mat curFrame = cvarrToMat(frame);

      if (1)
      {
        IplImage* gray = cvCreateImage(cvGetSize(frame), IPL_DEPTH_8U, 1);
        cvCvtColor(frame, gray, CV_RGB2GRAY);

        vector<Mat> facesMat;

        if (isCVPR14)
          facesMat = reg_cvpr14->predict_real_givenRects(gray, rects);
        else
          facesMat = reg->predict_real_givenRects(gray, rects, sampleNum);



        for (int f = 0; f < facesMat.size(); f++)
        {
          for (int i = 0; i < facesMat[f].cols / 2; i++)
            circle(curFrame, Point(facesMat[f].at<float>(2 * i), facesMat[f].at<float>(2 * i + 1)), 2, Scalar(255), -1);
        }
      }
      else
      {
        for (int i = 0; i < rects.size(); i++)
          rectangle(curFrame, rects[i], Scalar(255, 0, 0));
      }

      imshow("Res", curFrame);
      //waitKey();
      int l = waitKey(3);
      if (l == 27)
        break;
    }
  }

}