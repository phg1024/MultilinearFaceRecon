#include "evaluation.h"

namespace ESRAligner {

  void Evaluation::doevaluation(char *modelName, char *testNameList, char *saveName, int sampleNum)
  {
    TwoLevelRegression trainer;
    trainer.loadFerns_bin(modelName);

    cout << "do evaluation\n";
    evaluate(testNameList, &trainer, sampleNum);

    cout << "saving res\n";
    saveRes(saveName);

  }

  void Evaluation::obtainNameList(char *testNameList, vector<string> &nameStrList)
  {
    ifstream in(testNameList, ios::in);
    int num;
    in >> num;
    char curName[100];
    in.getline(curName, 99);

    //vector<string> nameStrList;
    nameStrList.clear();
    for (int i = 0; i < num; i++)
    {
      in.getline(curName, 99);
      nameStrList.push_back(curName);
    }
  }

  void Evaluation::evaluate(char *testNameList, TwoLevelRegression *trainer, int sampleNum)
  {
    Mat nullImg;

    //char testSetName[]=testNameList;//"D:\\Fuhao\\face dataset new\\faceRegression\\testset\\ptsList.txt";


    int ptsNum = trainer->refShape.n;

    vector<string> nameStrList;
    obtainNameList(testNameList, nameStrList);

    evalRes = Mat::zeros(nameStrList.size(), ptsNum * 2 * 2, CV_32FC1);
    for (int i = 0; i < nameStrList.size(); i++)
    {
      //if(i<27)
      //continue;
      //cout<<i<<" ";
      string imgName = nameStrList[i];
      imgName = imgName.substr(0, imgName.length() - 3) + "png";

      ifstream inTmp(imgName.c_str(), ios::in);
      if (!inTmp)
      {
        inTmp.close();
        imgName = nameStrList[i];
        imgName = imgName.substr(0, imgName.length() - 3) + "jpg";
      }
      IplImage *img = cvLoadImage(imgName.c_str(), 0);

      /*if(i==27)
        trainer->showRes=true;
        else
        trainer->showRes=false;*/
      evalRes.row(i) += trainer->pridict_evaluate(img, sampleNum, (char *)nameStrList[i].c_str());
      cvReleaseImage(&img);
    }

    //analysisError(evalRes);
  }

  void Evaluation::saveRes(char *name)
  {
    ofstream out(name, ios::out);
    for (int i = 0; i < evalRes.rows; i++)
    {
      for (int j = 0; j < evalRes.cols; j++)
      {
        out << evalRes.at<float>(i, j) << " ";
      }
      out << endl;
    }
    out.close();
  }

  void Evaluation::analysisError(Mat &evalRes)
  {

  }

  void Evaluation::checkIteration(char *modelName, char *testNameList)
  {
    TwoLevelRegression trainer;
    trainer.loadFerns_bin(modelName);
    trainer.showRes = true;

    vector<string> nameStrList;
    obtainNameList(testNameList, nameStrList);
    for (int i = 0; i < nameStrList.size(); i++)
    {
      //if(i<27)
      //continue;
      //cout<<i<<" ";
      string imgName = nameStrList[i];
      imgName = imgName.substr(0, imgName.length() - 3) + "png";

      ifstream inTmp(imgName.c_str(), ios::in);
      if (!inTmp)
      {
        inTmp.close();
        imgName = nameStrList[i];
        imgName = imgName.substr(0, imgName.length() - 3) + "jpg";
      }
      IplImage *img = cvLoadImage(imgName.c_str(), 0);
      trainer.pridict_real(img, 1);
    }

  }

  void Evaluation::checkConvergeGT(char *modelName, char *testNameList)
  {
    TwoLevelRegression trainer;
    trainer.loadFerns_bin(modelName);
    trainer.showRes = true;

    vector<string> nameStrList;
    obtainNameList(testNameList, nameStrList);


    for (int i = 0; i < nameStrList.size(); i++)
    {
      //if(i<27)
      //continue;
      //cout<<i<<" ";
      string imgName = nameStrList[i];
      imgName = imgName.substr(0, imgName.length() - 3) + "png";

      ifstream inTmp(imgName.c_str(), ios::in);
      if (!inTmp)
      {
        inTmp.close();
        imgName = nameStrList[i];
        imgName = imgName.substr(0, imgName.length() - 3) + "jpg";
      }
      IplImage *img = cvLoadImage(imgName.c_str(), 0);
      trainer.pridict_GT(img, (char *)nameStrList[i].c_str());
    }

  }


  void Evaluation::checkModel(char *modelName, char *refName)
  {
    TwoLevelRegression trainer;
    trainer.loadFerns_bin(modelName);

    trainer.visualizeModel(refName);

    waitKey();
  }

}