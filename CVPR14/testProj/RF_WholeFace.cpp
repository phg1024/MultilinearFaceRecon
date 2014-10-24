#include "RF_WholeFace.h"

#include <atlbase.h>
#include <atlconv.h>
#include "CodeTimer.h"

#include "W_train.h"

namespace ESRAligner {

  void RF_WholeFace::trainW(char *str)
  {
    LPCTSTR lpApplicationName = CA2W(str);
    // additional information
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    // set the size of the structures
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // start the program up
    CreateProcess(NULL,   // the path
      CA2W(str),        // Command line // -s 12 -p 0 -B 1 D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\hear_scale D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\heart_scal_model_new
      NULL,           // Process handle not inheritable
      NULL,           // Thread handle not inheritable
      FALSE,          // Set handle inheritance to FALSE
      0,              // No creation flags
      NULL,           // Use parent's environment block
      NULL,           // Use parent's starting directory 
      &si,            // Pointer to STARTUPINFO structure
      &pi);           // Pointer to PROCESS_INFORMATION structure

    // Wait until child process exits.
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Close process and thread handles. 
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

  }

  RF_WholeFace::RF_WholeFace()
  {
    showSingleStep = false;

    //float startVal=powf(2,0);
    /*c_list.push_back(0.002);
    c_list.push_back(0.004);
    c_list.push_back(0.006);
    c_list.push_back(0.008);
    c_list.push_back(0.01);*/
    //c_list.push_back(1);
    //c_list.push_back(0.0000025);
    c_list.push_back(0.000005);
    c_list.push_back(0.00001);
    c_list.push_back(0.000025);
    c_list.push_back(0.00005);
    c_list.push_back(0.0001);
    c_list.push_back(0.000125);
    /*c_list.push_back(0.00025);
    c_list.push_back(0.0005);
    c_list.push_back(0.001);
    c_list.push_back(0.002);
    c_list.push_back(0.004);

    c_list.push_back(0.008);
    c_list.push_back(0.016);
    c_list.push_back(0.032);*/
    /*c_list.push_back(0.064);
    c_list.push_back(0.128);
    c_list.push_back(0.256);
    c_list.push_back(0.512);
    c_list.push_back(0.1);*/
    /*candidateNum=10;
    candidateRadius.resize(candidateNum);
    for(int i=0;i<candidateRadius.size();i++)
    candidateRadius[i]=0.05+i*0.05;*/
  }

  //void RF_WholeFace::train(vector<ShapePair> &shapes, int treeNum, int depth)
  //{
  //	//forests.resize(treeNum);
  //	//
  //
  //	//vector<vector<RandomForests>> forests_candidates(candidateNum);
  //	//vector<Mat> W_list;
  //	//for(int i=0;i<candidateNum;i++)
  //	//{
  //	//	train_unit(shapes,treeNum,depth,candidateRadius[i],forests_candidates[i]);
  //	//	learnW(shapes,forests_candidates[i],W_list[i]);
  //	//}
  //
  //	//vector<float> errors(candidateNum);
  //	//for(int i=0;i<candidateNum;i++)
  //	//	errors[i]=evlError(*testShapes,forests_candidates[i],W_list[i]);
  //
  //	//int minInd=0;
  //	//float minErr=errors[0];
  //	//for(int i=1;i<candidateNum;i++)
  //	//{
  //	//	if(minErr>errors[i])
  //	//	{
  //	//		minErr=errors[i];minInd=i;
  //	//	}
  //	//}
  //
  //	//forests=forests_candidates[minInd];
  //	//W=W_list[minInd].clone();
  //
  //}

  void RF_WholeFace::learnW_crossValidation(vector<ShapePair> &shapes, vector<ShapePair> *validationShapes, vector<RandomForests> &forests, Mat &curW)
  {
    //lbf feature for training first
    //cout<<"obtaining LBF for training set\n";
    vector<LBFFeature> lbfFeatureList;
    lbfFeatureList.resize(shapes.size());
    for (int i = 0; i < lbfFeatureList.size(); i++)
    {
      lbfFeatureList[i] = predict_local(shapes[i].orgImg, shapes[i]);
    }

    //lbf feature for validation set
    //cout<<"obtaining LBF for validation set\n";
    vector<LBFFeature> lbfFeatureValidationList;
    lbfFeatureValidationList.resize(validationShapes->size());
    for (int i = 0; i < lbfFeatureValidationList.size(); i++)
    {
      //check
      if (0)
      {
        validationShapes->at(i).visualizePts("curValImg");
        waitKey();
      }
      //cout<<validationShapes->at(i).ptsVec<<endl;
      lbfFeatureValidationList[i] = predict_local(validationShapes->at(i).orgImg, validationShapes->at(i));
    }

    //then do cross-validation on validation shapes
    //cout<<"doing cross validation on W\n";
    // vector<vector<RandomForests>> curForestCandidates(c_list.size());
    vector<Mat> W_list(c_list.size());

    if (0)
    {
#pragma omp parallel for  
      for (int i = 0; i < c_list.size(); i++)
      {
        //cout<<"[ "<<i<<" "<<c_list.size()<<"]"<<" ";
        Mat valW;
        learnW(shapes, lbfFeatureList, valW, c_list[i]);
        W_list[i] = valW.clone();
      }
    }
    else
    {
      vector<Shape> dSList;
      for (int i = 0; i < lbfFeatureList.size(); i++)
      {
        Shape curDS = shapes[i].dS();
        dSList.push_back(curDS);
      }
      W_train w_trainer;
      w_trainer.train_multipleC(dSList, lbfFeatureList, W_list, c_list);
    }

    //check
    if (0)
    {
      vector<Shape> dSList;
      for (int i = 0; i < lbfFeatureList.size(); i++)
      {
        Shape curDS = shapes[i].dS();
        dSList.push_back(curDS);
      }

      int sampleNum = dSList.size();
      ofstream out("testW.txt", ios::binary);
      out.write((char *)&sampleNum, sizeof(int));
      for (int i = 0; i < sampleNum; i++)
        dSList[i].save(out);
      for (int i = 0; i < sampleNum; i++)
      {
        out.write((char *)&(lbfFeatureList[i].totalNum), sizeof(int));
        int oneNum = lbfFeatureList[i].onesInd.size();
        out.write((char *)&oneNum, sizeof(int));
        for (int j = 0; j < oneNum; j++)
        {
          out.write((char *)&(lbfFeatureList[i].onesInd[j]), sizeof(int));
        }
      }
      out.close();

      /*vector<Mat> W_list_new(c_list.size());
      W_train w_trainer;
      w_trainer.train_multipleC(dSList,lbfFeatureList,W_list_new,c_list);*/

      for (int i = 0; i < 1; i++)
      {
        cout << lbfFeatureList[i].totalNum << endl;
        imshow("stop", validationShapes->at(i).orgImg);
        waitKey();
      }

    }
    //cout<<endl;

    //test to see which has the smallist error
    vector<float> errors(c_list.size());
    for (int i = 0; i < errors.size(); i++)
    {
      errors[i] = 0;
      for (int j = 0; j < lbfFeatureValidationList.size(); j++)
      {
        errors[i] += getError(lbfFeatureValidationList[j], W_list[i], validationShapes->at(j));
        //errors[i]+=error_cur;
      }
    }

    int minInd = 0;
    float minError = errors[0];
    for (int i = 0; i<errors.size(); i++)
    {
      cout << "[ " << c_list[i] << " " << sqrtf(errors[i] / (shapes[0].n * 2 * validationShapes->size())) * 10 << "] ";
      if (minError>errors[i])
      {
        minError = errors[i];
        minInd = i;
      }
    }
    //cout<<endl;

    cout << "setting values " << c_list[minInd] << " index: " << minInd << " minError: " << sqrtf(errors[minInd] / (shapes[0].n * 2 * validationShapes->size())) * 10 << endl;
    curW = W_list[minInd].clone();
    optimalC = c_list[minInd];
  }

  float RF_WholeFace::getError(LBFFeature &lbfFeatureValidationList, Mat &curW, ShapePair &validationShapes)
  {
    Mat finalRes = Mat::zeros(curW.rows, 1, CV_32FC1) + curW.col(curW.cols - 1);
    for (int i = 0; i < lbfFeatureValidationList.onesInd.size(); i++)
      finalRes += curW.col(lbfFeatureValidationList.onesInd[i]);

    //cout<<finalRes<<endl;
    Mat ptsDif = validationShapes.dS().ptsVec - finalRes.t();

    //check here
    if (0)
    {
      validationShapes.visualizePts("curShape");
      cout << finalRes.t() << endl;
      cout << validationShapes.dS().ptsVec << endl;
      cout << ptsDif.dot(ptsDif) << endl;
      waitKey();
    }
    //float curEyeDis;
    //Point2f eyeDif=validationShapes.gtShape.pts[eyeIndL]-validationShapes.gtShape.pts[eyeIndR]; //2,3
    //curEyeDis=(eyeDif.x*eyeDif.x+eyeDif.y*eyeDif.y);

    float res = ptsDif.dot(ptsDif) / 100; //||(s-s_g)/eyeDis||^2

    return res;

  }



  void RF_WholeFace::learnW(vector<ShapePair> &shapes, vector<LBFFeature> &lbfFeatureList, Mat &curW, float C)
  {
    //arrange W locally first
    //cout<<"obtaining LBF\n";
    /*vector<LBFFeature> lbfFeatureList;
    lbfFeatureList.resize(shapes.size());
    for(int i=0;i<lbfFeatureList.size();i++)
    {
    lbfFeatureList[i]=predict_local(shapes[i].orgImg, shapes[i]);
    }*/

    //output features and delta for training
    curW = Mat::zeros(shapes[0].n * 2, lbfFeatureList[0].totalNum + 1, CV_32FC1);
    //cout<<"output LBF\n";
    vector<Shape> dSList;
    for (int i = 0; i < lbfFeatureList.size(); i++)
    {
      Shape curDS = shapes[i].dS();
      dSList.push_back(curDS);

      //check
      if (0)
      {
        Shape dS = shapes[i].dS();
        bool oKay = true;
        for (int j = 0; j<dS.n; j++)
        {
          if (dS.pts[j].x>500 || dS.pts[j].y > 500)
          {
            oKay = false;
            break;
          }
        }
        if (!oKay)
        {
          Mat img = shapes[i].orgImg.clone();
          for (int j = 0; j < shapes[i].n; j++)
          {
            circle(img, shapes[i].pts[j], 2, Scalar(255));
            circle(img, shapes[i].gtShape.pts[j], 2, Scalar(255), -1);
          }
          imshow("imgProblem", img);
          waitKey();
        }
      }
    }

    learnW_unit(dSList, lbfFeatureList, curW, C);

    return;

    //not used
    for (int i = 0; i < shapes[0].n; i++)
    {
      //cout<<"training W for "<<i<<" X"<<endl;
      Mat W = trainSingle(dSList, i, lbfFeatureList, C, true);
      curW.row(2 * i) += W.t();

      //cout<<"training W for "<<i<<" Y"<<endl;
      Mat W1 = trainSingle(dSList, i, lbfFeatureList, C, false);
      curW.row(2 * i + 1) += W1.t();

    }
  }

  void RF_WholeFace::learnW_unit(vector<Shape> &dSList, vector<LBFFeature> &lbfFeatureList, Mat &curW, float C)
  {
    //arrange W locally first
    //cout<<"obtaining LBF\n";
    /*vector<LBFFeature> lbfFeatureList;
    lbfFeatureList.resize(shapes.size());
    for(int i=0;i<lbfFeatureList.size();i++)
    {
    lbfFeatureList[i]=predict_local(shapes[i].orgImg, shapes[i]);
    }*/

    //output features and delta for training

    for (int i = 0; i < dSList[0].n; i++)
    {
      cout << "training W for " << i << " X" << endl;
      Mat W = trainSingle(dSList, i, lbfFeatureList, C, true);
      curW.row(2 * i) += W.t();

      cout << "training W for " << i << " Y" << endl;
      Mat W1 = trainSingle(dSList, i, lbfFeatureList, C, false);
      curW.row(2 * i + 1) += W1.t();

    }
  }


  Mat RF_WholeFace::trainSingle(vector<Shape> &dSList, int ind, vector<LBFFeature> &lbfFeatureList, float C, bool isX)
  {
    char curDataName[100];
    if (isX)
      sprintf(curDataName, "C:\\Users\\Fuhao\\Documents\\GitHub\\CVPR14_trainingFast\\testProj\\W_Train\\dataX_%d_%f.dat", ind, C);
    else
      sprintf(curDataName, "C:\\Users\\Fuhao\\Documents\\GitHub\\CVPR14_trainingFast\\testProj\\W_Train\\dataY_%d_%f.dat", ind, C);
    cout << curDataName << endl;
    ofstream out(curDataName, ios::out);
    for (int i = 0; i < dSList.size(); i++)
    {
      //Shape curDS=dSList[i];
      //for(int j=0;j<dSList[i].n;j++)
      if (isX)
        out << dSList[i].pts[ind].x << " ";
      else
        out << dSList[i].pts[ind].y << " ";


      for (int j = 0; j < lbfFeatureList[i].onesInd.size(); j++)
      {
        out << lbfFeatureList[i].onesInd[j] + 1 << ":" << 1 << " ";
      }
      if (i == 0 && lbfFeatureList[i].onesInd.at(lbfFeatureList[i].onesInd.size() - 1) != lbfFeatureList[i].totalNum - 1)
        out << lbfFeatureList[i].totalNum << ":" << 0 << " ";
      out << endl;
    }
    out.close();

    //cout<<"training LBF\n";
    char curModelName[100];
    if (isX)
      sprintf(curModelName, "C:\\Users\\Fuhao\\Documents\\GitHub\\CVPR14_trainingFast\\testProj\\W_Train\\modelX_%d_%f.dat", ind, C);
    else
      sprintf(curModelName, "C:\\Users\\Fuhao\\Documents\\GitHub\\CVPR14_trainingFast\\testProj\\W_Train\\modelY_%d_%f.dat", ind, C);
    string WPath = curModelName;

    char fullPathName[500];
    //sprintf(fullPathName,"D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\liblinear-1.94\\windows\\train.exe -s 12 -p 0 -B 1 -q -c %f ",C);
    sprintf(fullPathName, "D:\\Fuhao\\FacialFeatureDetection_Regression\\code\\libLinear_mine\\x64\\Release\\libLinear_mine.exe -s 12 -p 0 -B 1 -q -c %f ", C);
    string fullPath = fullPathName;
    fullPath = fullPath + curDataName + " " + WPath;
    trainW((char *)fullPath.c_str());

    //cout<<"obtaining W\n";
    //load W
    Mat W;
    getW((char *)WPath.c_str(), W);
    return W;
  }

  void RF_WholeFace::getW(char *path, Mat &curW)
  {
    ifstream in(path, ios::in);
    if (!in)
    {
      cout << "no model file\n";
      return;
    }

    char curInfo[100];


    while (1)
    {
      in.getline(curInfo, 99);
      string tmp = curInfo;
      if (tmp.length() == 1 && tmp[0] == 'w')
        break;
    }

    in.getline(curInfo, 99);
    string tmp = curInfo;
    int colNum = 0;
    for (int i = 0; i < tmp.length(); i++)
    {
      if (tmp[i] == ' ')
        colNum++;
    }

    in.seekg(0, ios::beg);

    while (1)
    {
      in.getline(curInfo, 99);
      string tmp = curInfo;
      if (tmp.length() == 1 && tmp[0] == 'w')
        break;
    }


    vector<double> W_mat;
    while (in)
    {
      float tmp;
      in >> tmp;
      W_mat.push_back(tmp);
    }
    W_mat.pop_back();

    curW = Mat::zeros(W_mat.size() / colNum, colNum, CV_32FC1);
    for (int i = 0; i < curW.rows; i++)
    {
      for (int j = 0; j < curW.cols; j++)
      {
        curW.at<float>(i, j) = W_mat[i*curW.cols + j];
      }
    }

  }

  //float RF_WholeFace::evlError(vector<ShapePair> &testShapes,vector<RandomForests> &forests_candidates, Mat &curW)
  //{
  //	float errorRes=0;
  //	return errorRes;
  //}

  void RF_WholeFace::train_unit(vector<ShapePair> &shapes, Shape *refShape, int treeNum, int depth, float radius)
  {
    assert(0); //ensure this will not excute
    int ptsNum = shapes[0].n;
    forests.resize(ptsNum);
    //for(int i=0;i<treeNum;i++)
    {
      cout << ptsNum << " learning forest for point ";
#pragma omp parallel for
      for (int j = 0; j < ptsNum; j++)
      {
        cout << j << " ";
        forests[j].setRefShape(refShape);
        forests[j].train(shapes, j, radius, treeNum, depth);
      }
      cout << endl;
    }

    //cout<<"learning W\n";
    //learnW(shapes,forests,W,optimalC);
  }

  void RF_WholeFace::train_unit_validation(vector<ShapePair> &shapes, Shape *refShape, int treeNum, int depth, float radius, vector<ShapePair> *validationShapes)
  {
    int ptsNum = shapes[0].n;
    forests.resize(ptsNum);
    //for(int i=0;i<treeNum;i++)
    {
      //cout<<ptsNum<<" learning forest for point ";
      int unitStepNum = 100;
      int stepNum = ptsNum / unitStepNum + 1;
      GTB("1");
      for (int i = 0;; i++)
      {
        cout << "[ " << i << " " << stepNum << " ]" << " ";
        int startInd = i*unitStepNum;
        int endInd = (i + 1)*unitStepNum;
        if (startInd >= ptsNum)
          break;

        endInd = endInd < ptsNum ? endInd : ptsNum;
#pragma omp parallel for
        for (int j = startInd; j < endInd; j++)
        {
          //cout<<j<<" ";
          forests[j].setRefShape(refShape);
          forests[j].train(shapes, j, radius, treeNum, depth);
        }
      }
      GTE("1");
      gCodeTimer.printTimeTree();
      double time = total_fps;
      cout << "forest learning time " << time << " ms" << endl;
    }

    cout << "learning W ";
    //GTB("2");
    learnW_crossValidation(shapes, validationShapes, forests, W);
    /*GTE("2");
    gCodeTimer.printTimeTree();
    double time1 = total_fps;
    cout<<"forest learning time "<<time1<<" ms"<<endl;*/

  }

  Mat RF_WholeFace::fastAdd(vector<int> &onesInd, int l, int r)
  {

    if (l == r)
      return W.col(onesInd[l]);
    else if (r == l + 1)
      return W.col(onesInd[l]) + W.col(onesInd[r]);
    else
    {
      int mid = (l + r) / 2;
      if (mid > l&&mid < r)
      {
        return fastAdd(onesInd, l, mid) + fastAdd(onesInd, mid + 1, r);
      }
    }

  }

  //need to be checked
  void RF_WholeFace::predict(Mat &img, Shape &s)
  {
    //cout<<"in pts prediction "<<s.n<<" "<<forests.size()<<endl;

    //GTB("LBF");
    LBFFeature finalLBF = predict_local(img, s);

    /*GTE("LBF");
      gCodeTimer.printTimeTree();
      double time = total_fps;
      cout<<"LBF time "<<time<<" ms"<<endl;*/
    //cout<<"applying W here\n";
    //apply W here
    /*Mat finalRes=Mat::zeros(W.rows,1,CV_32FC1)+W.col(W.cols-1);
    for(int i=0;i<finalLBF.onesInd.size();i++)
    finalRes+=W.col(finalLBF.onesInd[i]);*/

    //GTB("SUM");
    //time consuming, 2.xms
    Mat finalRes = fastAdd(finalLBF.onesInd, 0, finalLBF.onesInd.size() - 1);
    finalRes += W.col(W.cols - 1);

    //cout<<W.cols<<" "<<finalLBF.onesInd.size()<<endl;

    //check
    //if(1)
    //{
    //	//cout<<finalRes.t()<<endl;
    //	//cout<<finalRes1.t()<<endl;
    //	cout<<(finalRes-finalRes1).t()<<endl;
    //	imshow("pause",img);
    //	waitKey();
    //}

    //cout<<finalRes<<endl;

    s.ptsVec += finalRes.t();

    /*GTE("SUM");
      gCodeTimer.printTimeTree();
      double time1 = total_fps;
      cout<<"SUM time "<<time1<<" ms"<<endl;*/

    //	//check
    //if(1)
    //{
    //	s.visualize(img,s.pts,"before");
    //}

    s.syntheorize();

    ////check
    //if(1)
    //{
    //	s.visualize(img,s.pts,"after");
    //	waitKey();
    //}

    //Mat LBF=Mat::zeros(1,forests.leafNum*s.n,CV_32FC1);
    //return 
  }

  LBFFeature RF_WholeFace::predict_local(Mat &img, Shape &s)
  {

    LBFFeature finalFeature;
    vector<LBFFeature> localLBFResult(s.n);

    //#pragma omp parallel for 	
    //time consuming, 0.95ms
    for (int i = 0; i < s.n; i++)
      localLBFResult[i] = forests[i].predict(img, s, i);

    int curFullNum = 0;
    for (int i = 0; i < s.n; i++)
    {
      //if(showSingleStep)
      //cout<<i<<" "<<forests.size()<<endl;
      //forests[i].showSingleStep=showSingleStep;
      //LBFFeature curLBFFeature=forests[i].predict(img,s,i);
      for (int j = 0; j < localLBFResult[i].onesInd.size(); j++)
        finalFeature.onesInd.push_back(localLBFResult[i].onesInd[j] + curFullNum);
      curFullNum += localLBFResult[i].totalNum;
      finalFeature.totalNum += localLBFResult[i].totalNum;
    }
    return finalFeature;
  }





  void RF_WholeFace::save(ofstream &out)
  {
    cout << "saving " << forests.size() << " trees\n";
    int tNum = forests.size();
    out.write((char *)&tNum, sizeof(int));
    for (int i = 0; i < tNum; i++)
      forests[i].save(out);

    //save W
    int rows = W.rows;
    int cols = W.cols;
    out.write((char *)&rows, sizeof(int));
    out.write((char *)&cols, sizeof(int));
    out.write((char *)W.data, sizeof(float)*W.rows*W.cols);
  }

  void RF_WholeFace::load(ifstream &in)
  {
    int tNum;
    in.read((char *)&tNum, sizeof(int));

    forests.resize(tNum);
    for (int i = 0; i < tNum; i++)
    {
      //cout<<"forest "<<i<<endl;
      forests[i].load(in);
    }

    int rows;
    int cols;
    in.read((char *)&rows, sizeof(int));
    in.read((char *)&cols, sizeof(int));
    W = Mat::zeros(rows, cols, CV_32FC1);
    in.read((char *)W.data, sizeof(float)*W.rows*W.cols);
  }

  void RF_WholeFace::visualize(char *name)
  {
    vector<Mat> imgVis(forests.size());
    for (int i = 0; i < forests.size(); i++)
    {
      imgVis[i] = forests[i].visualize();
    }
    Mat finalRes = Mat::zeros(imgVis[0].rows*forests.size(), imgVis[0].cols, imgVis[0].type());
    for (int i = 0; i < imgVis.size(); i++)
    {
      finalRes(Range(i*imgVis[0].rows, (i + 1)*imgVis[0].rows), Range(0, imgVis[0].cols)) += imgVis[i];
    }

    imshow(name, finalRes);
  }

}