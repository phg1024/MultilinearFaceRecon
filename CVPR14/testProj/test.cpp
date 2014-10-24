#include "test.h"

namespace ESRAligner {

  void Test::getScaleTranslation(char *modelName, char *fileListName)
  {
    char testNameList[] = "D:\\Fuhao\\face dataset new\\faceRegression\\testset\\ptsList.txt";

    TwoLevelRegression trainer;
    trainer.loadFerns_bin(modelName);

    //obtain the nameList
    char testSetName[] = "D:\\Fuhao\\face dataset new\\faceRegression\\testset\\imgList.txt";
    ifstream in(testSetName, ios::in);
    int num;
    in >> num;
    char curName[100];
    in.getline(curName, 99);

    vector<string> nameStrList;
    for (int i = 0; i < num; i++)
    {
      in.getline(curName, 99);
      nameStrList.push_back(curName);
    }
    vector<Shape> shapes(nameStrList.size());
    vector<bool> isSampleGood(nameStrList.size());


    //obtain the estimated s,t and gt s,t
    vector<Point3f> ST_estimated(nameStrList.size());
    vector<Point3f> ST_gt(nameStrList.size());
    FaceDetector d;

    for (int i = 0; i < nameStrList.size(); i++)
    {
      if (i % 20 == 0)
        cout << i << " ";
      IplImage *img = cvLoadImage(nameStrList[i].c_str(), 0);
      vector<Rect> faceRects;
      d.findFaceFull(img, faceRects);

      //load in the shape 
      string curPtsName = nameStrList[i];
      curPtsName = curPtsName.substr(0, curPtsName.length() - 3) + "pts";
      shapes[i].load((char *)curPtsName.c_str());

      //select the correct face
      int faceInd = isAFace(faceRects, shapes[i]);
      if (faceInd == -1)
      {
        isSampleGood[i] = false;
        continue;
      }

      Rect curFaceRect = faceRects[faceInd];
      isSampleGood[i] = true;
      trainer.estimateST(shapes[i], curFaceRect, ST_estimated[i].x, ST_estimated[i].y, ST_estimated[i].z);
      shapes[i].estimateTrans(trainer.refShape);
      ST_gt[i].x = shapes[i].s; ST_gt[i].y = shapes[i].tx; ST_gt[i].z = shapes[i].ty;


    }
    cout << endl;


    //save for analysis
    ofstream out("D:\\Fuhao\\face dataset new\\faceRegression\\evlRes\\st.txt", ios::out);
    for (int i = 0; i < isSampleGood.size(); i++)
    {
      if (isSampleGood[i])
      {
        out << ST_estimated[i].x << " " << ST_estimated[i].y << " " << ST_estimated[i].z << " " <<
          ST_gt[i].x << " " << ST_gt[i].y << " " << ST_gt[i].z << endl;
      }
    }
    out.close();

  }

  int Test::isAFace(vector<Rect> &faceRects, Shape &s)
  {
    Rect curRect = boundingRect(s.pts);
    int res = -1;
    for (int i = 0; i < faceRects.size(); i++)
    {
      if (isCollide(faceRects[i], curRect))
      {
        return i;
      }
    }

    return res;
  }

  bool Test::isCollide(Rect &a, Rect &b)
  {
    return (abs(a.x - b.x) * 2 < (a.width + b.width)) &&
      (abs(a.y - b.y) * 2 < (a.height + b.height));
  }

}