#include "Fern.h"

namespace ESRAligner {
  void FernNode::save(ofstream &out)
  {
    float val[7];
    val[0] = ptsInd1;
    val[1] = ptsInd2;
    val[2] = threshold;
    val[3] = dx1;
    val[4] = dy1;
    val[5] = dx2;
    val[6] = dy2;

    out.write((char *)val, sizeof(float)* 7);

  }

  void FernNode::load(ifstream &in)
  {
    float val[7];


    in.read((char *)val, sizeof(float)* 7);

    ptsInd1 = val[0];
    ptsInd2 = val[1];
    threshold = val[2];
    dx1 = val[3];
    dy1 = val[4];
    dx2 = val[5];
    dy2 = val[6];

  }

  Fern::Fern()
  {
    //tmpMat=Mat::zeros(1,68*2,CV_32FC1);
  }

  Fern::~Fern()
  {
    //delete []dsMat;
  }

  void Fern::convertDSMat()
  {
    int dsLenght = dsList.size();
    dsMat = new Mat[dsLenght];
    for (int i = 0; i < dsList.size(); i++)
    {
      dsMat[i] = dsList[i].ptsVec;
    }
  }

  void Fern::train(vector<Point3f> &featureOffsets, vector<Point> &indexPairInd,
    vector<float> &threshold, vector<ShapePair> &shapes, Mat &curDifMat)
  {
    int ptsNum = shapes[0].n;
    int F = indexPairInd.size();
    buildFern(F, 0, featureOffsets, indexPairInd, threshold);

    int binSize = pow(2, F);
    dsList.resize(binSize);
    binPool.resize(binSize);
    //find bin for each shape
    int N = shapes.size();

    //check
    if (0)
    {
      for (int i = 0; i < N; i++)
      {
        Mat tmpIMg = shapes[i].orgImg.clone();
        for (int j = 0; j < fernNodes.size(); j++)
        {
          int id1 = fernNodes[j].ptsInd1;
          int id2 = fernNodes[j].ptsInd2;

          vector<Point2f> ptsPos;
          ptsPos.resize(2);



          ptsPos[0] = shapes[i].pts[id1] + Point2f(fernNodes[j].dx1, fernNodes[j].dy1)*shapes[i].s_local;
          ptsPos[1] = shapes[i].pts[id2] + Point2f(fernNodes[j].dx2, fernNodes[j].dy2)*shapes[i].s_local;

          Mat orgImg = shapes[i].orgImg;

          //cout<<shapes[i].pts[id1]<<" "<<fernNodes[j].dx1<<" "<<fernNodes[j].dy1<<endl;
          //cout<<ptsPos[0]<<" "<<shapes[i].s<<" "<<shapes[i].tx<<" "<<shapes[i].ty<<endl;
          /*	float s=shapes[i].s;
            Point2f curT=Point2f(shapes[i].tx,shapes[i].ty);
            ptsPos[0]*=s;
            ptsPos[1]*=s;

            ptsPos[0]+=curT;
            ptsPos[1]+=curT;*/

          line(tmpIMg, ptsPos[0], ptsPos[1], Scalar(255));



          float dif = shapes[i].getSubPixel(orgImg, ptsPos[0]) - shapes[i].getSubPixel(orgImg, ptsPos[1]);
          /*	cout<<fernNodes[j].dx1<<" "<<fernNodes[j].dy1<<" "<<
              shapes[i].pts[id1]<<" "<<ptsPos[0]<<" "<<s<<" "<<curT.x<<" "<<curT.y<<"--";*/
          cout << dif << "   ";
        }
        cout << endl;
        cout << curDifMat.row(i) << endl;
        imshow("curDif", tmpIMg);

        shapes[i].visualizePts("curShape");
        waitKey();
      }
    }


    for (int i = 0; i < N; i++)
    {
      //shapes[i].visualizePts("curRegisteredImg");
      //int binID=findBins(shapes[i],fernNodes);
      int binID = findBinsDirect(curDifMat.row(i), fernNodes);
      binPool[binID].push_back(i);
    }

    //calculate the optimal delta
    float beta = 1000;
    for (int i = 0; i<binSize; i++)
    {
      Shape dS(ptsNum);
      int O = binPool[i].size();
      if (O>0)
      {
        for (int j = 0; j < O; j++)
        {
          dS += shapes[binPool[i][j]].dS();
        }
        dS = dS / (O + beta);
      }
      dsList[i] = dS;
    }
  }

  void Fern::save(ofstream &out)
  {
    out << fernNodes.size() << endl;
    for (int i = 0; i < fernNodes.size(); i++)
    {
      out << fernNodes[i].ptsInd1 << " " << fernNodes[i].ptsInd2 << " " <<
        fernNodes[i].threshold << " " << fernNodes[i].dx1 << " " <<
        fernNodes[i].dy1 << " " << fernNodes[i].dx2 << " " << fernNodes[i].dy2 << endl;
    }

    out << dsList.size() << " " << dsList[0].n << endl;
    for (int i = 0; i < dsList.size(); i++)
    {
      for (int j = 0; j < dsList[i].pts.size(); j++)
      {
        out << dsList[i].pts[j].x << " " << dsList[i].pts[j].y << " ";
      }
      out << endl;
    }
  }


  void Fern::saveBin(ofstream &out)
  {
    //out<<fernNodes.size()<<endl;
    int fernNodeSize = fernNodes.size();
    out.write((char *)&fernNodeSize, sizeof(int)* 1);
    for (int i = 0; i < fernNodes.size(); i++)
    {
      fernNodes[i].save(out);
    }

    //out<<dsList.size()<<" "<<dsList[0].n<<endl;
    int dsSize = dsList.size();
    out.write((char *)&dsSize, sizeof(int)* 1);
    out.write((char *)&(dsList[0].n), sizeof(int)* 1);
    for (int i = 0; i < dsList.size(); i++)
    {
      dsList[i].save(out);
      /*for(int j=0;j<dsList[i].pts.size();j++)
      {
      out<<dsList[i].pts[j].x<<" "<<dsList[i].pts[j].y<<" ";
      }
      out<<endl;*/
    }
  }


  void Fern::load(ifstream &in)
  {
    int fernodeSize;
    in >> fernodeSize;
    fernNodes.resize(fernodeSize);

    for (int i = 0; i < fernodeSize; i++)
      in >> fernNodes[i].ptsInd1 >> fernNodes[i].ptsInd2 >> fernNodes[i].threshold >> fernNodes[i].dx1 >>
      fernNodes[i].dy1 >> fernNodes[i].dx2 >> fernNodes[i].dy2;

    int dsSize, ptsNum;
    in >> dsSize >> ptsNum;
    dsList.resize(dsSize);
    for (int i = 0; i < dsSize; i++)
    {
      dsList[i].setPtsNum(ptsNum);
      for (int j = 0; j < ptsNum; j++)
      {
        in >> dsList[i].ptsVec.at<float>(2 * j) >> dsList[i].ptsVec.at<float>(2 * j + 1);
      }
      dsList[i].syntheorize();
    }
  }

  void Fern::loadBin(ifstream &in)
  {
    int fernodeSize;
    in.read((char *)(&fernodeSize), sizeof(int)* 1);
    fernNodes.resize(fernodeSize);

    for (int i = 0; i < fernodeSize; i++)
      fernNodes[i].load(in);

    int dsSize, ptsNum;
    in.read((char *)&dsSize, sizeof(int)* 1);
    in.read((char *)&ptsNum, sizeof(int)* 1);
    dsList.resize(dsSize);
    for (int i = 0; i < dsSize; i++)
    {
      dsList[i].load(in);
    }

    //convertDSMat();
  }


  int Fern::getBinVal(Shape &shape, FernNode &fernNode)
  {
    int id1 = fernNode.ptsInd1;
    int id2 = fernNode.ptsInd2;

    vector<Point2f> ptsPos;
    ptsPos.resize(2);

    //float s_local=shape.s_local;
    //ptsPos[0]=shape.pts[id1]+Point2f(fernNode.dx1,fernNode.dy1)*s_local;
    //ptsPos[1]=shape.pts[id2]+Point2f(fernNode.dx2,fernNode.dy2)*s_local;

    Mat RS_local = shape.RS_local;
    float ctx1 = fernNode.dx1*RS_local.at<float>(0, 0) + fernNode.dy1*RS_local.at<float>(1, 0);
    float cty1 = fernNode.dx1*RS_local.at<float>(0, 1) + fernNode.dy1*RS_local.at<float>(1, 1);
    float ctx2 = fernNode.dx2*RS_local.at<float>(0, 0) + fernNode.dy2*RS_local.at<float>(1, 0);
    float cty2 = fernNode.dx2*RS_local.at<float>(0, 1) + fernNode.dy2*RS_local.at<float>(1, 1);

    ptsPos[0] = shape.pts[id1] + Point2f(ctx1, cty1);
    ptsPos[1] = shape.pts[id2] + Point2f(ctx2, cty2);

    /*float s=shape.s;
    Point2f curT=Point2f(shape.tx,shape.ty);
    ptsPos[0]*=s;
    ptsPos[1]*=s;

    ptsPos[0]+=curT;
    ptsPos[1]+=curT;*/

    /*Mat resVec=shape.s*Mat(ptsPos,false);
    resVec.row(0)+=shape.tx;
    resVec.row(1)+=shape.ty;

    vector<Point2f> newPts;
    resVec.copyTo(Mat(newPts,false));*/

    //check
    if (0)
    {
      shape.visualizePts("curShape");
      Mat tmp = shape.orgImg.clone();
      circle(tmp, ptsPos[0], 3, Scalar(255), -1);
      circle(tmp, ptsPos[1], 3, Scalar(255), -1);
      imshow("curCpomp", tmp);
      waitKey();
    }

    Mat orgImg = shape.orgImg;

    /*float val1,val2;
    val1=val2=0;
    if(ptsPos[0].x>=0&&ptsPos[0].x<orgImg.cols&&ptsPos[0].y>=0&&ptsPos[0].y<orgImg.rows)
    val1=orgImg.at<uchar>(ptsPos[0]);
    if(ptsPos[1].x>=0&&ptsPos[1].x<orgImg.cols&&ptsPos[1].y>=0&&ptsPos[1].y<orgImg.rows)
    val2=orgImg.at<uchar>(ptsPos[1]);
    float dif=val1-val2;*/


    float dif = shape.getSubPixel(orgImg, ptsPos[0]) - shape.getSubPixel(orgImg, ptsPos[1]);

    return dif > fernNode.threshold;
  }


  Shape Fern::pridict(Shape &s)
  {
    //int binID=findBins(s,fernNodes);


    int binID = 0;
    for (int i = 0; i < fernNodes.size(); i++)
    {
      int comVal = getBinVal(s, fernNodes[i]);
      binID <<= 1;
      binID |= comVal;
    }

    return dsList[binID];
  }

  void Fern::pridict_directAdd(Shape &s)
  {
    //int binID=findBins(s,fernNodes);

    int binID = 0;
    for (int i = 0; i < fernNodes.size(); i++)
    {
      int comVal = getBinVal(s, fernNodes[i]);
      binID <<= 1;
      binID |= comVal;
    }
    s += dsList[binID];

    /*for(int i=0;i<s.n*2;i++)
      s.ptsVec.at<float>(i)+=dsList[binID].ptsVec.at<float>(i);*/
    //s.ptsVec+=dsMat[binID];
  }

  void Fern::buildFern(int F, int curInd, vector<Point3f> &featureOffsets,
    vector<Point> &indexPairInd, vector<float> &threshold)
  {
    fernNodes.resize(F);
    for (int i = 0; i < F; i++)
    {
      int curInd1 = indexPairInd[i].x;
      int curInd2 = indexPairInd[i].y;
      fernNodes[i].ptsInd1 = featureOffsets[curInd1].z;
      fernNodes[i].dx1 = featureOffsets[curInd1].x;
      fernNodes[i].dy1 = featureOffsets[curInd1].y;

      fernNodes[i].ptsInd2 = featureOffsets[curInd2].z;
      fernNodes[i].dx2 = featureOffsets[curInd2].x;
      fernNodes[i].dy2 = featureOffsets[curInd2].y;

      fernNodes[i].threshold = threshold[i];
    }
  }

  int Fern::findBins(Shape &shape, vector<FernNode> &fernNode)
  {
    int code = 0;
    for (int i = 0; i < fernNode.size(); i++)
    {
      int comVal = getBinVal(shape, fernNode[i]);
      code <<= 1;
      code |= comVal;
    }
    return code;
  }

  int Fern::findBinsDirect(Mat &features, vector<FernNode> &fernNode)
  {
    int code = 0;
    for (int i = 0; i<fernNode.size(); i++)
    {
      int comVal = features.at<float>(i)>fernNode[i].threshold;
      code <<= 1;
      code |= comVal;
    }
    return code;
  }

  void Fern::visualize(char *name, Shape &s)
  {
    Mat img = s.orgImg.clone();



    for (int i = 0; i < fernNodes.size(); i++)
    {
      Point2f pt1, pt2;
      pt1 = s.pts[fernNodes[i].ptsInd1];
      pt1.x += fernNodes[i].dx1;
      pt1.y += fernNodes[i].dy1;

      pt2 = s.pts[fernNodes[i].ptsInd2];
      pt2.x += fernNodes[i].dx2;
      pt2.y += fernNodes[i].dy2;

      line(img, pt1, pt2, Scalar(255), 2);
    }

    imshow(name, img);

  }

}