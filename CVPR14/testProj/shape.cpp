#include "shape.h"
#include "GRandom.h"

namespace ESRAligner {

  Shape::Shape()
  {
    n = 68;	//need to check
    ptsVec = Mat::zeros(1, n * 2, CV_32FC1);

    syntheorize();
  }

  Shape::Shape(int _n)
  {
    n = _n;
    ptsVec = Mat::zeros(1, n * 2, CV_32FC1);

    syntheorize();

  }

  void Shape::save(ofstream &out)
  {
    out.write((char*)&n, sizeof(int)* 1);
    out.write((char *)ptsVec.data, sizeof(float)*n * 2);
  }

  void Shape::load(ifstream &in)
  {
    in.read((char*)&n, sizeof(int)* 1);
    setPtsNum(n);

    in.read((char *)ptsVec.data, sizeof(float)*n * 2);
    syntheorize();
  }

  void Shape::setPtsNum(int _n)
  {
    n = _n;
    ptsVec = Mat::zeros(1, n * 2, CV_32FC1);

    syntheorize();

  }

  Shape::Shape(Shape *s)
  {
    n = s->n;
    ptsVec = s->ptsVec.clone();
    syntheorize();

    orgImg = s->orgImg;
  }

  void Shape::setShape(Shape &curS)
  {
    n = curS.n;
    ptsVec = curS.ptsVec.clone();
    syntheorize();
    orgImg = curS.orgImg;
  }

  void Shape::setShapeOnly(Shape &curS)
  {
    n = curS.n;
    ptsVec = curS.ptsVec.clone();
    syntheorize();
    //orgImg=curS.orgImg;
  }

  void Shape::setImg(Mat &img)
  {
    orgImg = img;
  }

  void Shape::syntheorize()
  {
    if (pts.size() != n)
      pts.resize(n);
    for (int i = 0; i < n; i++)
      pts[i] = Point2f(ptsVec.at<float>(0, 2 * i), ptsVec.at<float>(0, 2 * i + 1));
  }

  Shape Shape::operator+(const Shape &ds)
  {
    Shape s(this);
    s.ptsVec += ds.ptsVec;
    s.syntheorize();
    return s;
  }

  Shape Shape::operator/(const float &d)
  {
    Shape s(this);
    s.ptsVec /= d;
    s.syntheorize();
    return s;
  }

  Shape Shape::operator-(const Shape &ds)
  {
    Shape s(this);
    s.ptsVec -= ds.ptsVec;
    s.syntheorize();
    return s;
  }

  void Shape::operator+=(const Shape &ds)
  {
    //Shape s(this);
    this->ptsVec += ds.ptsVec;
    //this->syntheorize();

    //return *this;
  }

  void Shape::load(char *name, bool readImg)
  {
    char strInfo[100];
    ifstream in(name);
    in.getline(strInfo, 99);//version
    in.getline(strInfo, 99);//ptsInfo
    string numStr = strInfo;
    numStr = numStr.substr(numStr.find(':') + 1, numStr.length() - numStr.find(':'));
    n = stoi(numStr);
    //in>>n;
    in.getline(strInfo, 99);//{
    ptsVec = Mat::zeros(1, n * 2, CV_32FC1);
    for (int i = 0; i < n; i++)
    {
      in >> ptsVec.at<float>(2 * i) >> ptsVec.at<float>(2 * i + 1);
      //cout<<ptsVec.at<float>(2*i)<<" "<<ptsVec.at<float>(2*i+1)<<endl;
    }
    in.close();
    ptsVec -= 1;
    syntheorize();

    if (0)
    {
      //use only 17 21 22 26 30 36 39 43 45 48 51 54 57 62 66
      //int usedInd[]={30,17,21,22,26,36,39,43,45,48,51,54,57,62,66};
      int usedInd[] = { 30, 17, 36, 45, 21, 22, 26, 48, 54, 39, 42 };
      int usedPtsNum = sizeof(usedInd) / sizeof(int);
      if (1)
      {

        Mat ptsVecNew = Mat::zeros(1, usedPtsNum * 2, CV_32FC1);
        for (int i = 0; i < usedPtsNum; i++)
        {
          ptsVecNew.at<float>(2 * i) = ptsVec.at<float>(2 * usedInd[i]);
          ptsVecNew.at<float>(2 * i + 1) = ptsVec.at<float>(2 * usedInd[i] + 1);
        }
        ptsVec = ptsVecNew;
        n = usedPtsNum;
        //pts.clear();
        syntheorize();
      }
    }

    //also load the image
    if (readImg)
    {
      string imgName = name;
      imgName = imgName.substr(0, imgName.length() - 3) + "png";

      ifstream intest(imgName.c_str(), ios::in);
      if (!intest)
      {
        imgName = name;
        imgName = imgName.substr(0, imgName.length() - 3) + "jpg";
      }
      intest.close();

      ImgPtr = cvLoadImage(imgName.c_str(), 0);
      orgImg = cvarrToMat(ImgPtr);
    }
  }

  //we can assume the refShape has already been normalized
  void Shape::estimateTrans(Shape &ref)
  {
    //check
    /*if(1)
    {
    visualize(orgImg,pts,"orgPts");
    }*/


    //estimate tx,ty and s
    Mat ptsMat = Mat(pts);
    Scalar meanPts = cv::mean(ptsMat);
    //float xMean=meanPts.val[0];
    //float yMean=meanPts.val[1];

    Mat ptsMat_ref = Mat(ref.pts).clone();
    Scalar meanPts_ref = cv::mean(ptsMat_ref);
    //float xMean_ref=meanPts_ref.val[0];
    //float yMean_ref=meanPts_ref.val[1];



    //centerlize
    ptsMat -= meanPts;
    ptsMat_ref -= meanPts_ref;

    /*if(1)
    {
    visualize(orgImg,pts,"orgPts1");
    }*/

    //calculate S
    s = sqrtf(ptsMat.dot(ptsMat) / ptsMat_ref.dot(ptsMat_ref));

    //for(int i=0;i<n;i++)
    //ptsMat.at<Point2f>(i,0)-=Point2f(tx,ty);
    ptsMat /= s;
    ptsMat += meanPts_ref;

    tx = (meanPts - meanPts_ref*s).val[0];
    ty = (meanPts - meanPts_ref*s).val[1];

    //syntheorize ptsVec
    for (int i = 0; i < n; i++)
    {
      ptsVec.at<float>(2 * i) = pts[i].x;
      ptsVec.at<float>(2 * i + 1) = pts[i].y;
    }

    /*if(1)
    {
    visualize(orgImg,pts,"orgPts2");
    }*/
    //check
    if (0)
    {
      visualize(ref.orgImg, pts, "registeredPts");
      waitKey();
    }

  }

  void Shape::estimateTrans_local(Shape &ref)
  {

    //use full rigid transform here
    {
      Mat dstMat = Mat(pts);
      Mat srcMat = Mat(ref.pts);

      Scalar meanDst = cv::mean(dstMat);
      Scalar meanSrc = cv::mean(srcMat);

      Mat dstRealMat = Mat::zeros(n, 2, CV_32FC1);
      Mat srcRealMat = Mat::zeros(n, 2, CV_32FC1);

      for (int i = 0; i < n; i++)
      {
        dstRealMat.at<float>(i, 0) = pts[i].x - meanDst.val[0];
        dstRealMat.at<float>(i, 1) = pts[i].y - meanDst.val[1];
        srcRealMat.at<float>(i, 0) = ref.pts[i].x - meanSrc.val[0];
        srcRealMat.at<float>(i, 1) = ref.pts[i].y - meanSrc.val[1];
      }

      solve(srcRealMat.t()*srcRealMat, srcRealMat.t()*dstRealMat, RS_local);


      if (0)
      {
        Mat newPts = srcRealMat*srcRealMat;

        vector<Point2f> ptsNew;
        for (int i = 0; i < n; i++)
        {
          ptsNew.push_back(Point2f(newPts.at<float>(i, 0) + meanDst.val[0],
            newPts.at<float>(i, 1) + meanDst.val[1]));
        }
        Mat tmp = ref.orgImg.clone();
        for (int i = 0; i < ref.n; i++)
        {
          circle(tmp, pts[i], 5, Scalar(255));
          circle(tmp, ptsNew[i], 3, Scalar(255));
        }
        imshow("registeredPts", tmp);
        waitKey();
      }

      return;
    }
    //cout<<"before: \n"<<Mat(pts)<<endl;
    //estimate tx,ty and s
    Mat ptsMat = Mat(pts).clone();
    Scalar meanPts = cv::mean(ptsMat);
    //float xMean=meanPts.val[0];
    //float yMean=meanPts.val[1];

    Mat ptsMat_ref = Mat(ref.pts).clone();
    Scalar meanPts_ref = cv::mean(ptsMat_ref);
    //float xMean_ref=meanPts_ref.val[0];
    //float yMean_ref=meanPts_ref.val[1];



    //centerlize
    ptsMat -= meanPts;
    ptsMat_ref -= meanPts_ref;

    /*if(1)
    {
    visualize(orgImg,pts,"orgPts1");
    }*/

    //calculate S
    s_local = sqrtf(ptsMat.dot(ptsMat) / ptsMat_ref.dot(ptsMat_ref));

    //cout<<"after: \n"<<Mat(pts)<<endl;
    /*if(1)
    {
    visualize(orgImg,pts,"orgPts2");
    }*/
    //check
    if (0)
    {
      visualize(ref.orgImg, pts, "registeredPts");
      waitKey();
    }

  }

  void Shape::addLocalST(Shape &ref, int refWidth, int refHeight)
  {
    Point2f orgNoseTip = pts[30] * s + Point2f(tx, ty);
    //obtain the certer first
    Point2f center;
    center.x = center.y = 0;
    for (int i = 0; i < n; i++)
      center += pts[i];
    center.x /= n; center.y /= n;

    Mat centeredPts = ptsVec.clone();
    for (int i = 0; i < n; i++)
    {
      centeredPts.at<float>(2 * i) -= center.x;
      centeredPts.at<float>(2 * i + 1) -= center.y;
    }

    //random scale between 0.8-1.2
    float s_scale = 0.9 + RandDouble_c01c()*0.2;
    centeredPts *= s_scale;

    //add random T
    float randomRange = 0.2;
    float tx_local = (RandDouble_c01c() * 2 * randomRange - randomRange)*refWidth;
    float ty_local = (RandDouble_c01c() * 2 * randomRange - randomRange)*refHeight;

    //tx_local=ty_local=0;
    for (int i = 0; i < n; i++)
    {
      centeredPts.at<float>(2 * i) += tx_local + center.x;
      centeredPts.at<float>(2 * i + 1) += ty_local + center.y;
    }

    //update scale and T then
    centeredPts.copyTo(ptsVec);
    syntheorize();

    s /= s_scale;

    Point2f NewNoseTip = pts[30] * s;
    Point2f difNew = orgNoseTip - NewNoseTip;
    tx = difNew.x;
    ty = difNew.y;

  }

  void Shape::visualizePts(char *name)
  {
    //vector<Point2f> curRealPts;
    //generateGlobalFeature(curRealPts);
    //cout<<pts.size()<<endl;
    Mat tmp = orgImg.clone();
    for (int i = 0; i < pts.size(); i++)
      circle(tmp, pts[i], 2, Scalar(255), -1);

    //Rect rect=boundingRect(curRealPts);
    //tmp=tmp(rect);

    imshow(name, tmp);
  }

  void Shape::visualize(Mat &img, vector<Point2f> &curPts, char *name)
  {

    Mat tmp = img.clone();
    for (int i = 0; i < curPts.size(); i++)
      circle(tmp, curPts[i], 2, Scalar(255), -1);

    //find the bounding box
    /*Rect rect=boundingRect(curPts);
    tmp=tmp(rect);*/
    imshow(name, tmp);
    //waitKey();


  }

  void Shape::setScaleTranslatrion(float _s, float _tx, float _ty)
  {
    s = _s;
    tx = _tx;
    ty = _ty;
  }

  Mat Shape::getFinalPosVector(float curScale, Point2f curST)
  {
    Mat res = ptsVec*curScale;
    for (int i = 0; i < n; i++)
    {
      res.at<float>(2 * i) += curST.x;
      res.at<float>(2 * i + 1) += curST.y;
    }
    return res;
  }

  void Shape::generateGlobalFeature(vector<Point2f> &pts_out)
  {
    pts_out.resize(pts.size());
    for (int i = 0; i < pts.size(); i++)
      pts_out[i] = pts[i] * s + Point2f(tx, ty);
  }

  Mat Shape::getCurFeature(vector<Point3f> &pts_in)
  {

    //cout<<pts_in[25]<<" "<<pts[pts_in[25].z]<<endl;
    vector<Point2f> inputPts(pts_in.size());
    for (int i = 0; i < inputPts.size(); i++)
    {
      int usedInd = pts_in[i].z;


      /*float ctx=pts_in[i].x*s_local;
      float cty=pts_in[i].y*s_local;*/

      float tx_pure = pts_in[i].x;
      float ty_pure = pts_in[i].y;
      float ctx = tx_pure*RS_local.at<float>(0, 0) + ty_pure*RS_local.at<float>(1, 0);
      float cty = tx_pure*RS_local.at<float>(0, 1) + ty_pure*RS_local.at<float>(1, 1);

      inputPts[i].x = pts[usedInd].x + ctx;
      inputPts[i].y = pts[usedInd].y + cty;
    }


    return getFeature(orgImg, inputPts);
  }


  float Shape::getCurFeature(int ptsInd, Point2f pts_in)
  {
    float tx_pure = pts_in.x;
    float ty_pure = pts_in.y;
    float ctx = tx_pure*RS_local.at<float>(0, 0) + ty_pure*RS_local.at<float>(1, 0);
    float cty = tx_pure*RS_local.at<float>(0, 1) + ty_pure*RS_local.at<float>(1, 1);

    Point2f inputPts;
    inputPts.x = pts[ptsInd].x + ctx;
    inputPts.y = pts[ptsInd].y + cty;

    return getSubPixel(orgImg, inputPts);
  }

  float Shape::getCurFeature_GivenImg(Mat &img, int ptsInd, Point2f pts_in)
  {
    float tx_pure = pts_in.x;
    float ty_pure = pts_in.y;
    float ctx = tx_pure*RS_local.at<float>(0, 0) + ty_pure*RS_local.at<float>(1, 0);
    float cty = tx_pure*RS_local.at<float>(0, 1) + ty_pure*RS_local.at<float>(1, 1);

    Point2f inputPts;
    inputPts.x = pts[ptsInd].x + ctx;
    inputPts.y = pts[ptsInd].y + cty;

    return getSubPixel(img, inputPts);
  }

  Mat Shape::getFeature(Mat &img, vector<Point2f> &pts)
  {
    Mat res = Mat::zeros(1, pts.size(), CV_32FC1);
    for (int i = 0; i < pts.size(); i++)
    {
      if (pts[i].x<1 || pts[i].y<1 || pts[i].x>img.cols - 2 || pts[i].y>img.rows - 2)
        res.at<float>(0, i) = 0;
      else
        //do we need to interpolate??
        res.at<float>(0, i) = getSubPixel(img, pts[i]);
    }
    //cout<<res.at<float>(25)<<" "<<pts[25]<<endl;
    return res;
  }

  float Shape::getSubPixel(Mat &img, Point2f current_pos)
  {
    if (current_pos.x<0 || current_pos.x>img.cols - 2 || current_pos.y<0 || current_pos.y>img.rows - 2)
      return 0;
    //bilinear interpolation
    float dx = current_pos.x - (int)current_pos.x;
    float dy = current_pos.y - (int)current_pos.y;

    float weight_tl = (1.0 - dx) * (1.0 - dy);
    float weight_tr = (dx)* (1.0 - dy);
    float weight_bl = (1.0 - dx) * (dy);
    float weight_br = (dx)* (dy);

    Point tl = Point(current_pos.x, current_pos.y);
    Point tr = Point(tl.x + 1, tl.y);
    Point bl = Point(tl.x, tl.y + 1);
    Point br = Point(tl.x + 1, tl.y + 1);

    float finalVal = img.at<uchar>(tl)*weight_tl + img.at<uchar>(tr)*weight_tr +
      img.at<uchar>(bl)*weight_bl + img.at<uchar>(br)*weight_br;

    return finalVal;
  }


  void ShapePair::setGTShape(Shape &input, Shape &ref)
  {
    gtShape.setShapeOnly(input);
    //gtShape.setScaleTranslatrion(input.s,input.tx,input.ty);
    return;
    gtShape.setShapeOnly(input);
    gtShape.estimateTrans(ref);	//align the groundtruth shape with reference

    //then disturb the scale, tx and ty
    float s_scale = 0.95 + RandDouble_c01c()*0.1;
    gtShape.s *= s_scale;


    if (0)
    {
      float tx_t = (-10 + RandDouble_c01c()*20.0f) / gtShape.s;
      float ty_t = (-10 + RandDouble_c01c()*20.0f) / gtShape.s;


      gtShape.tx += tx_t;
      gtShape.ty += ty_t;
    }
    else
    {
      Point2f curPtsCenter = input.pts[30];
      Point2f curCenter = ref.pts[30] * gtShape.s;
      Point2f TDif = curPtsCenter - curCenter;
      gtShape.tx = TDif.x;
      gtShape.ty = TDif.y;
    }
  }

  Shape ShapePair::dS()
  {
    return gtShape - (*this);
  }

}