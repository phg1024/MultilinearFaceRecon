#include "randtree.h"
#include   <imagehlp.h> 
#pragma   comment(lib, "imagehlp.lib ")

//3 4 11 12 23 24 1 2 9 10 25 28 5 6 7 8 13 14 15 16 17 18 19 20 21 22 26 27
//int interestInd[]={2,3  ,  10  ,  11   , 22 ,   23    , 0  ,   1  ,   8,     9  ,  24 ,   27   ,  4 ,    5  ,   6  ,   7  ,  12   , 13  ,  14   , 15  ,  16  ,  17 ,   18,
//  19 ,   20   , 21   , 25 ,   26};

//int interestInd[]={2,3  ,  10  ,  11   , 22 ,   23    , 0  ,   1  ,   8,     9  ,  24 ,   27   ,  12   , 13  ,  14   , 15  ,  16  ,  17 ,   18,
//19 ,   20   , 21   , 25 ,   26};

int interestInd[]={0,4,8,12,16,18,20,22,24,26,28,30,42,45,48,51,56,61,76,35,38};

RandTree::RandTree()
{
	response_idx=0;
	sampleDim=0;
	haveIndex=false;
	usingCUDA=false;
	host_inputImage=new float[MPN];

	host_colorImage=new float[MPN];
	host_depthImage=new float[MPN];

	labelResult=new float[MPN*(1+MAX_LABEL_NUMBER)];

	for (int i=0;i<MPN*(1+MAX_LABEL_NUMBER);i++)
	{
		labelResult[i]=0;
	}

	addedSamples=new vector<Sample>;
	isevaluating=false;

	trainStyle=0;
	sampleNumEveryTime=2000;
	probabilityMap=NULL;
	probabilityMap_full=NULL;
	centerList=NULL;
	//windowSize=32;
}

RandTree::RandTree(int _max_depth,int _min_sample_count,double _regression_accuracy,int _max_num_of_trees_in_the_forest,int wind_size,int classNum)
{
	max_depth=_max_depth;
	min_sample_count=_min_sample_count;
	regression_accuracy=_regression_accuracy;
	max_num_of_trees_in_the_forest=_max_num_of_trees_in_the_forest;
	treeNum=0;//the number of trees
	rootSampleNum=10;//sample num for root
	fullRandom=false;
	is_satisfied=false;
	usingCUDA=false;
	labelNum=classNum;
	windowSize=wind_size;

	ifstream in("D:\\Fuhao\\colorIndex.txt",ios::in);
	for (int i=0;i<1000;i++)
	{
		in>>colorIndex[i][0]>>colorIndex[i][1]>>colorIndex[i][2];
	}
	host_inputImage=new float[MPN];
	labelResult=new float[MPN*(1+MAX_LABEL_NUMBER)];
	cout<<"labelResult CPU set"<<MPN*(1+MAX_LABEL_NUMBER)<<endl;
	addedSamples=new vector<Sample>;
	isevaluating=false;
	trainStyle=0;
	sampleNumEveryTime=2000;

	host_colorImage=new float[MPN];
	host_depthImage=new float[MPN];

	probabilityMap=NULL;
	probabilityMap_full=NULL;
	centerList=NULL;
}


void RandTree::setupProbMaps(int width,int height,float *result)
{
	numOfLabels=labelNum-1;
	
	if (probabilityMap==NULL)
	{
		probabilityMap=new Mat[numOfLabels];
		for (int i=0;i<numOfLabels;i++)
		{
			probabilityMap[i]=Mat::zeros(height,width,CV_64FC1);
		}
	}

	int cwidth,cheight;
	int cind;
	for (int i=0;i<numOfLabels;i++)
	{
		//cout<<i<<endl;
		for (int j=0;j<height*width;j++)
		{
			cwidth=j%width;
			cheight=j/width;
			cind=j*(1+MAX_LABEL_NUMBER);
		/*	if (j%100==0)
			{
				cout<<cwidth<<" "<<cheight<<endl;
			}*/
			//probabilityMap[i].at<double>(cheight,cwidth)=0;
			
			probabilityMap[i].at<double>(cheight,cwidth)=result[cind+i+1];
		}
	}

	for (int i=0;i<numOfLabels;i++)
	{
		interestPtsInd[i]=interestInd[i];
	}

	//for (int i=0;i<numOfLabels;i++)
	//{
	//	namedWindow("1");
	//	imshow("1",probabilityMap[i]);
	//	waitKey();
	//}
	

	//for (int i=0;i<numOfLabels;i++)
	//{
	//	namedWindow("1");
	//	imshow("1",probabilityMap[i]);
	//	waitKey();
	//}
	
}

void RandTree::setupProbMaps(int width,int height,float *result,vector<int> &visibleInd)
{
	numOfLabels=visibleInd.size();
	
	if (probabilityMap==NULL)
	{
		probabilityMap=new Mat[labelNum-1];
		for (int i=0;i<labelNum-1;i++)
		{
			probabilityMap[i]=Mat::zeros(height,width,CV_64FC1);
		}
	}

	int cwidth,cheight;
	int cind;
	for (int i=0;i<numOfLabels;i++)
	{
		//cout<<i<<endl;
		for (int j=0;j<height*width;j++)
		{
			cwidth=j%width;
			cheight=j/width;
			//cind=j*(1+MAX_LABEL_NUMBER);
		/*	if (j%100==0)
			{
				cout<<cwidth<<" "<<cheight<<endl;
			}*/
			//probabilityMap[i].at<double>(cheight,cwidth)=0;
			
			//probabilityMap[i].at<double>(cheight,cwidth)=result[cind+visibleInd[i]+1];
			probabilityMap[i].at<double>(cheight,cwidth)=result[j+visibleInd[i]*width*height];
		}
	}

	for (int i=0;i<numOfLabels;i++)
	{
		interestPtsInd[i]=interestInd[visibleInd[i]];
	}

	//for (int i=0;i<numOfLabels;i++)
	//{
	//	namedWindow("1");
	//	imshow("1",probabilityMap[i]);
	//	waitKey();
	//}
	
}

void RandTree::findProbModes(int width, int height,float *result)
{
	//set up the probability map first
	if (probabilityMap_full==NULL)
	{
		probabilityMap_full=new Mat[labelNum-1];
		for (int i=0;i<labelNum-1;i++)
		{
			probabilityMap_full[i]=Mat::zeros(height,width,CV_64FC1);
		}
		centerList=new vector<Point>[labelNum-1];
	}

	int cwidth,cheight;
	int cind;
	for (int i=0;i<labelNum-1;i++)
	{
		//cout<<i<<endl;
		for (int j=0;j<height*width;j++)
		{
			cwidth=j%width;
			cheight=j/width;
			cind=j*(1+MAX_LABEL_NUMBER);
		/*	if (j%100==0)
			{
				cout<<cwidth<<" "<<cheight<<endl;
			}*/
			//probabilityMap[i].at<double>(cheight,cwidth)=0;
			
			probabilityMap_full[i].at<double>(cheight,cwidth)=result[cind+i+1];
		}
	}

	//then get modes for all the features
	Mat currentProbMap;
	int numberCount[30];
	int threshold=10;
	double centerPts[30][2];
	for (int i=0;i<labelNum-1;i++)
	{
		currentProbMap=probabilityMap_full[i];
		Mat label=currentProbMap.clone()*0;
		label-=2;
		int maxIdx[3]; 
		double testMaxval=500;
		minMaxIdx(currentProbMap, 0,&testMaxval, 0, maxIdx);

		//set the seeds
		vector<Point> seeds;
		centerList[i].clear();
		//Vector<Point> center;
		for (int ii=0;ii<currentProbMap.rows;ii++)
		{
			for (int j=0;j<currentProbMap.cols;j++)
			{
				if (currentProbMap.at<double>(ii,j)>=0.6*testMaxval)
				{
					seeds.push_back(Point(j,ii));
					/*label.at<double>(ii,j)=-1;*/
					for (int m=j-2;m<=j+2;m++)
					{
						for (int n=ii-2;n<=ii+2;n++)
						{
							//seeds.push_back(Point(m,n));
							label.at<double>(n,m)=-1;
						}
					}
					

				}
			}
		}

	/*	Mat centerVis1=label.clone()*0;
		for (int k=0;k<seeds.size();k++)
		{
			circle(centerVis1,seeds[k],3,255);
		}

		namedWindow("seeds");
		imshow("seeds",centerVis1);
		waitKey();*/

		Point currentPoint;
		vector<Point> CurrentPts;
		for (int ii=0;ii<seeds.size();ii++)
		{
			numberCount[ii]=0;
			currentPoint.x=seeds[ii].x;
			currentPoint.y=seeds[ii].y;
			CurrentPts.clear();
			if (label.at<double>(currentPoint.y,currentPoint.x)==-1)	//not visited
			{
				CurrentPts.push_back(currentPoint);
				label.at<double>(currentPoint.y,currentPoint.x)=ii+1;
				while (CurrentPts.size()!=0)
				{
					currentPoint.x=CurrentPts[CurrentPts.size()-1].x;
					currentPoint.y=CurrentPts[CurrentPts.size()-1].y;
					CurrentPts.pop_back();
					
					numberCount[ii]++;
					//break;

					if (label.at<double>(currentPoint.y,currentPoint.x+1)==-1)
					{
						label.at<double>(currentPoint.y,currentPoint.x+1)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x+1,currentPoint.y));
					}

					if (label.at<double>(currentPoint.y+1,currentPoint.x+1)==-1)
					{
						label.at<double>(currentPoint.y+1,currentPoint.x+1)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x+1,currentPoint.y+1));
					}

					if (label.at<double>(currentPoint.y+1,currentPoint.x)==-1)
					{
						label.at<double>(currentPoint.y+1,currentPoint.x)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x,currentPoint.y+1));
					}

					if (label.at<double>(currentPoint.y+1,currentPoint.x-1)==-1)
					{
						label.at<double>(currentPoint.y+1,currentPoint.x-1)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x-1,currentPoint.y+1));
					}


					if (label.at<double>(currentPoint.y,currentPoint.x-1)==-1)
					{
						label.at<double>(currentPoint.y,currentPoint.x-1)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x-1,currentPoint.y));
					}

					if (label.at<double>(currentPoint.y-1,currentPoint.x-1)==-1)
					{
						label.at<double>(currentPoint.y-1,currentPoint.x-1)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x-1,currentPoint.y-1));
					}

					if (label.at<double>(currentPoint.y-1,currentPoint.x)==-1)
					{
						label.at<double>(currentPoint.y-1,currentPoint.x)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x,currentPoint.y-1));
					}

					if (label.at<double>(currentPoint.y-1,currentPoint.x+1)==-1)
					{
						label.at<double>(currentPoint.y-1,currentPoint.x+1)=ii+1;
						CurrentPts.push_back(Point(currentPoint.x+1,currentPoint.y-1));
					}
				}

		/*		namedWindow("currentLabelMap");
				imshow("currentLabelMap",label*30);
				waitKey();*/
			
			}

			
			////then, exclude all the negetive and isolate value
			//for (int i=0;i<label.rows;i++)
			//{
			//	for (int j=0;j<label.cols;j++)
			//	{
			//		if (label.at<double>(i,j)<0)
			//		{
			//		}
			//	}
			//}
			
		/*	namedWindow("currentLabelMap");
			imshow("currentLabelMap",label*30);
			waitKey();*/
		
		}

		//then, find the mode for all the large enough region
		int currentLabel;
		float maxValue[30]={0};
		for (int k=0;k<30;k++)
		{
			centerPts[k][0]=centerPts[k][1]=0;
			maxValue[k]=0;
		}

		
		
		for (int k=0;k<label.rows;k++)
		{
			for (int l=0;l<label.cols;l++)
			{
				currentLabel=label.at<double>(k,l)-1;
				if (currentLabel>=0&&numberCount[currentLabel]>threshold)
				{
					if (currentProbMap.at<double>(k,l)>maxValue[currentLabel])
					{
						maxValue[currentLabel]=currentProbMap.at<double>(k,l);
						centerPts[currentLabel][0]=l;
						centerPts[currentLabel][1]=k;
					}
					
					//num++;
				}
				else
				{
					label.at<double>(k,l)=0;
				}
			}
		}

		//cout<<num<<" "<<endl;

		for (int k=0;k<seeds.size();k++)
		{
			if (numberCount[k]>threshold)
			{
				centerList[i].push_back(Point(centerPts[k][0],centerPts[k][1]));
			}
		}

		
		//cout<<"label "<<i<<" with "<<centerList[i].size()<<" modes"<<endl;
		//Mat centerVis=label.clone()*0;
		//for (int k=0;k<centerList[i].size();k++)
		//{
		//	circle(centerVis,centerList[i][k],3,255);
		//}
		//namedWindow("Prob_Cluster");
		//imshow("Prob_Cluster",label);
		////waitKey();

		//namedWindow("currentCandidateMode");
		//imshow("currentCandidateMode",centerVis);
		//waitKey();
	}
}

void RandTree::train(string name)
{
	int ind=name.find_last_of('\\');
	if (ind<0)
	{
		ind=name.find_last_of('/');
	}
	string dirName=name.substr(0,ind+1);
	string dataname=dirName+"RT_tree_Resault.txt";
	ifstream in(dataname.c_str(),ios::in);

//	if (!in)
	{
		getData(name);
	}

	


	//get sample num
	int label;
	float tmp;
	int i,j,k;
	int sampleNum=0;
	while (in)
	{
		in>>label;
		for (i=0;i<sampleDim;i++)
		{
			in>>tmp;
		}
		sampleNum++;
	}
	sampleNum--;

	//sampleNum=;

	cvfulldata=cvCreateMat(sampleNum,sampleDim,CV_32F);
	cvResponse=cvCreateMat(1,sampleNum,CV_32F);
	fullData=cvarrToMat(cvfulldata);
	response=cvarrToMat(cvResponse);
	in.clear();
	in.seekg(0);

	for (i=0;i<sampleNum;i++)
	{
		in>>label;
	/*	if(label>1)
			label=1;
		else
			label=0;*/
		response.at<float>(0,i)=label;
		for(j=0;j<sampleDim;j++)
		{
			in>>tmp;
			fullData.at<float>(i,j)=tmp;
		}
	}
	//for (i=0;i<sampleNum;i++)
	//{
	//	cout<<response.at<float>(0,i)<<" ";
	//}

	//定义R.T.训练用参数，CvDTreeParams的扩展子类，但并不用到CvDTreeParams（单一决策树）所需的所有参数。比如说，R.T.通常不需要剪枝，因此剪枝参数就不被用到。
	//	max_depth  单棵树所可能达到的最大深度
	//	min_sample_count  树节点持续分裂的最小样本数量，也就是说，小于这个数节点就不持续分裂，变成叶子了
	//	regression_accuracy  回归树的终止条件，如果所有节点的精度都达到要求就停止
	//	use_surrogates  是否使用代理分裂。通常都是false，在有缺损数据或计算变量重要性的场合为true，比如，变量是色彩，而图片中有一部分区域因为光照是全黑的
	//	max_categories  将所有可能取值聚类到有限类，以保证计算速度。树会以次优分裂（suboptimal split）的形式生长。只对2种取值以上的树有意义
	//	priors  优先级设置，设定某些你尤其关心的类或值，使训练过程更关注它们的分类或回归精度。通常不设置
	//	calc_var_importance  设置是否需要获取变量的重要值，一般设置true
	//	nactive_vars  树的每个节点随机选择变量的数量，根据这些变量寻找最佳分裂。如果设置0值，则自动取变量总和的平方根
	//	max_num_of_trees_in_the_forest  R.T.中可能存在的树的最大数量
	//	forest_accuracy  准确率（作为终止条件）
	//	termcrit_type  终止条件设置
	//	-- CV_TERMCRIT_ITER  以树的数目为终止条件，max_num_of_trees_in_the_forest生效
	//	-- CV_TERMCRIT_EPS  以准确率为终止条件，forest_accuracy生效
	//	-- CV_TERMCRIT_ITER | CV_TERMCRIT_EPS  两者同时作为终止条件
	cout<<"training\n";
	rtrees.train(cvfulldata,CV_ROW_SAMPLE ,cvResponse,0,0,0,0,
		CvRTParams( 15, 2, 0, false, 16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER ));
	cout<<"Train Error: "<<rtrees.get_train_error()<<endl;
	string modelName=dirName+"Trained RTtree.txt";
	rtrees.save(modelName.c_str());
	return;
	//rtrees.train( &data, CvRTParams( 10, 2, 0, false, 16, 0, true, 0, 100, 0, CV_TERMCRIT_ITER ));
}

float* RandTree::getDiscriptor(IplImage *img,double *center,Mat &indlist)
{
	int i,j;
	float *feature=new float[indlist.rows];
	Mat m_img=cvarrToMat(img);
	/*for (i=0;i<indlist.rows;i++)
	{
		feature[i]=m_img.at<uchar>(center[1]+indlist.at<float>(i,1),center[0]+indlist.at<float>(i,0))-
			m_img.at<uchar>(center[1]+indlist.at<float>(i,3),center[0]+indlist.at<float>(i,2));
	}*/
	for (i=0;i<indlist.rows;i++)
	{
		feature[i]=m_img.at<uchar>(center[1]+indlist.at<float>(i,1),center[0]+indlist.at<float>(i,0))>
			m_img.at<uchar>(center[1]+indlist.at<float>(i,3),center[0]+indlist.at<float>(i,2));
	}

	//Mat t_img=cvarrToMat(img).clone();
	//Point c;
	//for (i=0;i<indlist.rows;i++)
	//{
	//	c.x=center[0]+indlist.at<float>(i,2);
	//	c.y=center[1]+indlist.at<float>(i,3);
	//	circle(t_img,c,5,255);
	//	c.x=center[0]+indlist.at<float>(i,0);
	//	c.y=center[1]+indlist.at<float>(i,1);
	//	circle(t_img,c,5,255);
	//}
	//namedWindow("local Region");
	//imshow("local Region",t_img);
	//waitKey(1);
	return feature;
}

void RandTree::getData(string name)
{
	//get dirname
	cout<<"loading data\n";
	int ind=name.find_last_of('\\');
	if (ind<0)
	{
		ind=name.find_last_of('/');
	}
	string dirName=name.substr(0,ind+1);

	fstream in(name.c_str(),ios::in);
	in>>shapeNum;
	//shapeNum=5;
	shape=new Shape*[shapeNum];
	char cname[500];
	//	in.getline(cname,499);
	string cfilename;
	for (int i=0;i<shapeNum;i++)
	{
		//	name="";
		shape[i]=new Shape;
		in.getline(cname,499);
		cfilename=cname;
		if(cfilename.length()<3)
		{
			i--;
			continue;
		}
		shape[i]->getVertex(cfilename);
	}
	
	//then, get the discriptors

	//comparsion
	//get the indexList
	ifstream in1("F:\\imgdata\\index.txt",ios::in);
	int temp;
	sampleDim=0; 
	int i,j,k,l;
	while(in1)
	{
		for (i=0;i<4;i++)
		{
			in1>>temp;
		}
		sampleDim++;
	}
	sampleDim--;
	indexList.create(sampleDim,4,CV_32F);
	in1.clear();
	in1.seekg(0);
	for(i=0;i<sampleDim;i++)
	{
		for (j=0;j<4;j++)
		{
			in1>>temp;
			indexList.at<float>(i,j)=temp;
		}
	}
	haveIndex=true;

	//get the feature
	//output file
	string saveName=dirName+"RT_tree_Resault.txt";
	ofstream out_com(saveName.c_str(),ios::out);
	int indList[]={0,8,16,24,42,48};
	float *feature;
	double center[2];
	int tmpLabel;
		vector <float> featureAll;
	int totalInd=0;
	int windowSize=5;
	int step=(windowSize-1)/2;
	int m,n;
	for (i=0;i<shapeNum;i++)
	{
		//cout<<i<<endl;
		featureAll.clear();
		//totalInd=0;
		//set the label
		Mat img(cvGetSize(shape[i]->hostImage).height,cvGetSize(shape[i]->hostImage).width,CV_32F);
		img=100000;
		for(j = 0;j < sizeof(indList)/sizeof(int); j++ )
		{
			for (m=shape[i]->pts[indList[j]][1]-step;m<=shape[i]->pts[indList[j]][1]+step;m++)
			{
				for (n=shape[i]->pts[indList[j]][0]-step;n<=shape[i]->pts[indList[j]][0]+step;n++)
				{
					img.at<float>(m,n)=j;
				}
			}
			
		}
	
		//output the facial region
		//cout<<"getting and outputing feature "<<i<<endl;
		//for (j=shape[i]->minx;j<shape[i]->maxx+1;j++)
		//{
		//	//cout<<j<<endl;
		//	for (k=shape[i]->miny;k<shape[i]->maxy+1;k++)
		//	{
		//		center[0]=j;center[1]=k;
		//		feature=getDiscriptor(shape[i]->hostImage,center,indexList);

		//		out_com<<img.at<float>(center[1],center[0])<<" ";
		//		for (l=0;l<sampleDim;l++)
		//		{
		//			out_com<<feature[l]<<" ";
		//		}
		//
		//		delete []feature;
		//	}
		//}

		//output feature points only
		cout<<"getting and outputing feature "<<i<<endl;
		for(j = 0;j < sizeof(indList)/sizeof(int); j++ )
		{
			for (m=shape[i]->pts[indList[j]][1]-step;m<=shape[i]->pts[indList[j]][1]+step;m++)
			{
				for (n=shape[i]->pts[indList[j]][0]-step;n<=shape[i]->pts[indList[j]][0]+step;n++)
				{
					center[0]=n;center[1]=m;
					feature=getDiscriptor(shape[i]->hostImage,center,indexList);
					out_com<<j<<" ";
					for (k=0;k<sampleDim;k++)
					{
						out_com<<feature[k]<<" ";
					}
					out_com<<endl;
				}
			}
		
		}
		for (j=0;j<shape[i]->ptsNum;j++)
		{
			if (j!=0&&j!=8&&j!=16&&j!=24&&j!=42&&j!=48)
			{
				//cout<<j<<endl;
				feature=getDiscriptor(shape[i]->hostImage,shape[i]->pts[j],indexList);
				//out_com<<tmpLabel<<" ";
				out_com<<img.at<float>(shape[i]->pts[j][1],shape[i]->pts[j][0])<<" ";
				for (k=0;k<sampleDim;k++)
				{
					out_com<<feature[k]<<" ";
				}
				out_com<<endl;
				//tmpLabel++;
			}
		}
		//////////////////////////////////////////////////////////

	/*	cout<<"Outputing feature"<<i<<endl;

		int totalNum=featureAll.size()/(1+sampleDim);
		int curInd;
		for (j=0;j<totalNum;j++)
		{
			curInd=j*(1+sampleDim);
			
			for (k=curInd+0;k<curInd+1+sampleDim;k++)
			{
				out_com<<featureAll[k]<<" ";
			}
			out_com<<endl;
		}*/
		//float tmp;
		//for (j=shape[i]->minx;j<shape[i]->maxx+1;j++)
		//{
		//	//cout<<j<<endl;
		//	for (k=shape[i]->miny;k<shape[i]->maxy+1;k++)
		//	{
		//		center[0]=j;center[1]=k;
		//		out_com<<img.at<float>(center[1],center[0])<<" ";
		//		for (l=0;l<sampleDim;l++)
		//		{
		//			out_com<<featureAll[totalInd]<<" ";
		//			totalInd++;
		//		}
		//		out_com<<endl;
		//	}
		//}
	


		//Mat img1=img.clone();
		//img1=0;
		//for (j=shape[i]->minx;j<shape[i]->maxx+1;j++)
		//{
		//	for (k=shape[i]->miny;k<shape[i]->maxy+1;k++)
		//	{
		//		center[0]=j;center[1]=k;
		//		img1.at<float>(center[1],center[0])=255;
		//	}
		//}
		//namedWindow("label");
		//imshow("label",img1);
		//waitKey();


		//continue;

		//for(j = 0;j < sizeof(indList)/sizeof(int); j++ )
		//{
		//	feature=getDiscriptor(shape[i]->hostImage,shape[i]->pts[indList[j]],indexList);
		//	out_com<<j<<" ";
		//	for (k=0;k<sampleDim;k++)
		//	{
		//		out_com<<feature[k]<<" ";
		//	}
		//	out_com<<endl;
		//}

		//////other pts
		//tmpLabel=sizeof(indList)/sizeof(int);
		//for (j=0;j<shape[i]->ptsNum;j++)
		//{
		//	if (j!=0&&j!=8&&j!=16&&j!=24&&j!=42&&j!=48)
		//	{
		//		//cout<<j<<endl;
		//		feature=getDiscriptor(shape[i]->hostImage,shape[i]->pts[j],indexList);
		//		//out_com<<tmpLabel<<" ";
		//		out_com<<100<<" ";
		//		for (k=0;k<sampleDim;k++)
		//		{
		//			out_com<<feature[k]<<" ";
		//		}
		//		out_com<<endl;
		//		tmpLabel++;
		//	}
		//}

	}
	out_com.close();


	//surf
	//int i,j;
	//CvSURFParams params = cvSURFParams(500, 1);
	// CvSeq* objectKeypoints=0, *objectDescriptors = 0;
	// CvMemStorage* storage4Pts = cvCreateMemStorage(0);
	//
	// //output file
	// string saveName=dirName+"RT_tree_Resault.txt";
	// ofstream out(saveName.c_str(),ios::out);


	// cout<<"extracting feature discriptors to train\n";
	//int length=-1;
	//int featPtsNum=6;
	//CvMemStorage* storage = cvCreateMemStorage(0);
	//// copy descriptors
	//CvSeqReader obj_reader;
	//float* obj_ptr;
	//int indList[]={0,8,16,24,42,48};
	////int featurePtsNum=6;
	//for (int i=0;i<shapeNum;i++)
	//{
	//	//cout<<i<<endl;
	//	objectKeypoints=cvCreateSeq( 0, /* sequence of points */
	//		sizeof(CvSeq), /* header size - no extra fields */
	//		sizeof(CvSURFPoint), /* element size */
	//		storage4Pts /* the container storage */ );
	///*	objectDescriptors = cvCreateSeq( objectKeypoints->flags, sizeof(CvSeq),
	//		sizeof(CvSURFPoint), storage );*/

	//	for(j = 0;j < sizeof(indList)/sizeof(int); j++ )
	//	{
	//		CvSURFPoint pt;
	//		pt.pt.x=shape[i]->pts[indList[j]][0];
	//		pt.pt.y=shape[i]->pts[indList[j]][1];
	//		pt.size=10;
	//		cvSeqPush( objectKeypoints, &pt);
	//	}
	//	////other pts
	//	for (j=0;j<shape[i]->ptsNum;j++)
	//	{
	//		if (j!=0&&j!=8&&j!=16&&j!=24&&j!=42&&j!=48)
	//		{
	//			CvSURFPoint pt;
	//			pt.pt.x=shape[i]->pts[j][0];
	//			pt.pt.y=shape[i]->pts[j][1];
	//			pt.size=5;
	//			cvSeqPush( objectKeypoints, &pt);
	//		}
	//	}
	//	//here,we need to do the multi-scale extraction
	//	cvExtractSURF( shape[i]->hostImage, 0, &objectKeypoints, &objectDescriptors, storage, params,1 );
	//	//cvExtractSURF( shape[i]->hostImage, 0, &objectKeypoints, &objectDescriptors, storage, params);
	//	//cvExtractSURF( shape[i]->hostImage, 0, &objectKeypoints, &objectDescriptors, storage, params,1);
	//	if (length==-1)
	//	{
	//		length=(int)(objectDescriptors->elem_size/sizeof(float));
	//		fullData.create(objectDescriptors->total,length,CV_32F);
	//	}

	//	 obj_ptr = fullData.ptr<float>(0);
	//	  cvStartReadSeq( objectDescriptors, &obj_reader );
	//	for(j = 0; j < objectDescriptors->total; j++ )
	//	{
	//		const float* descriptor = (const float*)obj_reader.ptr;
	//		CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
	//		memcpy(obj_ptr, descriptor, length*sizeof(float));
	//		obj_ptr += length;
	//	}
	//	for(j=0;j<fullData.rows;j++)
	//	{
	//		if (j<featPtsNum)
	//		{
	//			out<<j<<" ";
	//		}
	//		else
	//			out<<j<<" ";
	//		
	//		for (int k=0;k<fullData.cols;k++)
	//		{
	//			out<<fullData.at<float>(j,k)<<" ";
	//		}
	//		out<<endl;
	//	}
	////	cvRelease(&objectKeypoints);
	////	cvRelease(&objectDescriptors);
	//}
	//out.close();

	//sampleDim=length;
}

void RandTree::predict(IplImage *img)
{
	rtrees.load("F:\\imgdata\\Video 2 Train\\Trained RTtree.txt");
	//cout<<"Train Error: "<<rtrees.get_train_error()<<endl;
	Mat AllLabel(img->height,img->width,CV_32F);
	int i,j,k,l;
	Mat m_img=cvarrToMat(img);
	float *feature;
	ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel.txt",ios::in);
	if (1)
	{
		if (!haveIndex)
		{
			ifstream in1("F:\\imgdata\\index.txt",ios::in);
			int temp;
			sampleDim=0; 
			int i,j,k;
			while(in1)
			{
				for (i=0;i<4;i++)
				{
					in1>>temp;
				}
				sampleDim++;
			}
			sampleDim--;
			indexList.create(sampleDim,4,CV_32F);
			in1.clear();
			in1.seekg(0);
			for(i=0;i<sampleDim;i++)
			{
				for (j=0;j<4;j++)
				{
					in1>>temp;
					indexList.at<float>(i,j)=temp;
				}
			}
		}
		//CvSeq* objectKeypoints=0, *objectDescriptors = 0;
		//CvMemStorage* storage4Pts = cvCreateMemStorage(0);
		//CvMemStorage* storage = cvCreateMemStorage(0);
		//objectKeypoints=cvCreateSeq( 0, /* sequence of points */
		//	sizeof(CvSeq), /* header size - no extra fields */
		//	sizeof(CvSURFPoint), /* element size */
		//	storage4Pts /* the container storage */ );
		//CvSURFParams params = cvSURFParams(500, 1);
		//vector<CvSURFPoint> pts;
		//int i,j;
		//for (i=5;i<img->width-5;i++)
		//{
		//	for(j=5;j<img->height-5;j++)
		//	{
		//		CvSURFPoint pt;
		//		pt.pt.x=i;
		//		pt.pt.y=j;
		//		pt.size=10;
		//		cvSeqPush( objectKeypoints, &pt);
		//		pts.push_back(pt);
		//	}
		//}
		//cout<<"extracting features\n";
		//cvExtractSURF( img, 0, &objectKeypoints, &objectDescriptors, storage, params,1 );



		////predict for every point
		////assign value to mat file
		//CvSeqReader obj_reader;
		//int length=(int)(objectDescriptors->elem_size/sizeof(float));
		//CvMat *data=cvCreateMat(objectDescriptors->total,length,CV_32F);
		//Mat m_data=cvarrToMat(data);
		//float* obj_ptr;
		//obj_ptr = m_data.ptr<float>(0);
		//cvStartReadSeq( objectDescriptors, &obj_reader );
		//for(j = 0; j < objectDescriptors->total; j++ )
		//{
		//	const float* descriptor = (const float*)obj_reader.ptr;
		//	CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
		//	memcpy(obj_ptr, descriptor, length*sizeof(float));
		//	obj_ptr += length;
		//}
		//get the feature set
		int size=windowSize;
		cout<<"predicting labels\n";
		double center[2];
		ofstream out("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel.txt",ios::out);
		for (i=size;i<m_img.cols-size;i++)
		{
		//	cout<<i<<endl;
			for (j=size;j<m_img.rows-size;j++)
			{
				center[0]=i;
				center[1]=j;
				feature=getDiscriptor(img,center,indexList);
				Mat cdata(1,sampleDim,CV_32F,feature);
				out<<i<<" "<<j<<" "<<rtrees.predict(cdata)<<endl;
				delete []feature;
			}
		}
		out.close();
		
		//for(i=0;i<objectDescriptors->total;i++)
		//{
		//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
		//}
		//out.close();
	}
	//else
	{
		//display
		ifstream in("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel.txt",ios::in);
		AllLabel=0;
		int y,x;
		int label;
		int i;
		while(in)
		{
			in>>y>>x>>label;
			AllLabel.at<float>(x,y)=label;
		}

		//show all in one image
		//Mat currentLabel(img->height,img->width,CV_32F);
		////show directly
		//Mat ImageWithLabel=cvarrToMat(img).clone();
		//int currentID=2;
	//	for (currentID=0;currentID<6;currentID++)
	//	{
	//		for (i=0;i<currentLabel.rows;i++)
	//		{
	//			for (j=0;j<currentLabel.cols;j++)
	//			{
	//				currentLabel.at<float>(i,j)=(AllLabel.at<float>(i,j)==currentID);
	//			}
	//		}

	//		
	//		Point c;

	//		for (i=0;i<currentLabel.rows;i++)
	//		{
	//			for (j=0;j<currentLabel.cols;j++)
	//			{
	//				if(AllLabel.at<float>(i,j)==currentID)
	//				{
	//					c.x=j;
	//					c.y=i;
	//					circle(ImageWithLabel,c,2,255);
	//				}
	//			}
	//		}
	//	}
	//	char name[50];
	////	sprintf(name, "feature ", ID+1);
	//	namedWindow("feature");
	//	imshow("feature",ImageWithLabel);
	//	waitKey();
			

			//show single image

		Mat currentLabel(img->height,img->width,CV_32F);
		int currentID=2;
		for (currentID=0;currentID<6;currentID++)
		{
			for (i=0;i<currentLabel.rows;i++)
			{
				for (j=0;j<currentLabel.cols;j++)
				{
					currentLabel.at<float>(i,j)=(AllLabel.at<float>(i,j)==currentID);
				}
			}

		//	//show directly
		//	Mat ImageWithLabel=cvarrToMat(img).clone();
		//	Point c;

		//	for (i=0;i<currentLabel.rows;i++)
		//	{
		//		for (j=0;j<currentLabel.cols;j++)
		//		{
		//			if(AllLabel.at<float>(i,j)==currentID)
		//			{
		//				c.x=j;
		//				c.y=i;
		//				circle(ImageWithLabel,c,2,255);
		//			}
		//		}
		//	}
		//	char name[50];
		//	sprintf(name, "feature %d", currentID+1);
		//	namedWindow(name);
		//	imshow(name,ImageWithLabel);
		//	//////////////////////////////////////////////////


			//local window search
			int localWindowSize=10;
			double threshold=0.5;
			int center[2];
			int sum;
			vector<CvPoint> filterdLabel;
			for (i=localWindowSize;i<img->width-localWindowSize;i++)
			{
				for (j=localWindowSize;j<img->height-localWindowSize;j++)
				{
					center[0]=j;center[1]=i;
					sum=0;
					for (k=center[0]-localWindowSize/2;k<center[0]+localWindowSize/2;k++)
					{
						for (l=center[1]-localWindowSize/2;l<center[1]+localWindowSize/2;l++)
						{
							if (currentLabel.at<float>(k,l)==1)
							{
								sum++;
							}
						}
					}
					if (sum>localWindowSize*localWindowSize*threshold)
					{
						filterdLabel.push_back(cvPoint(i,j));
					}
				}
			}
			cout<<"remained pts number: "<<filterdLabel.size()<<endl;
			Mat ImageWithLabel=cvarrToMat(img).clone();
			Point c;
			for (i=0;i<filterdLabel.size();i++)
			{
				c.x=filterdLabel.at(i).x;
				c.y=filterdLabel.at(i).y;
				circle(ImageWithLabel,c,1,255);
			}
			char name[50];
			sprintf(name, "feature %d", currentID+1);
			namedWindow(name);
			imshow(name,ImageWithLabel);
			waitKey();
		}
	}
	


	

	//int index;
	//for (i=5;i<img->width-5;i++)
	//{
	//	for(j=5;j<img->height-5;j++)
	//	{
	//		index=i*(img->width-10)+j;
	//		out<<i<<" "<<j<<" "<<rtrees.predict(m_data.row(i))<<endl;
	//	}
	//}
	//out.close();

}

Mat& RandTree::setValue(CvSeq*objectDescriptors)
{
	int i,j;
	CvSeqReader obj_reader;
	int length=(int)(objectDescriptors->elem_size/sizeof(CvSURFPoint));
	Mat m_data;
	{
		m_data.create(objectDescriptors->total,length,CV_32F);
	}
	float* obj_ptr;
	obj_ptr = m_data.ptr<float>(0);
	cvStartReadSeq( objectDescriptors, &obj_reader );
	for(j = 0; j < objectDescriptors->total; j++ )
	{
		const float* descriptor = (const float*)obj_reader.ptr;
		CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
		memcpy(obj_ptr, descriptor, length*sizeof(float));
		obj_ptr += length;
	}
	return m_data;
}

//Mat& RandTree::setValue_Pts(CvSeq*objectDescriptors)
//{
//	int i,j;
//	CvSeqReader obj_reader;
//	int length=(int)(objectDescriptors->elem_size/sizeof(float));
//	Mat m_data;
//	{
//		m_data.create(objectDescriptors->total,length,CV_32F);
//	}
//	float* obj_ptr;
//	obj_ptr = m_data.ptr<float>(0);
//	cvStartReadSeq( objectDescriptors, &obj_reader );
//	for(j = 0; j < objectDescriptors->total; j++ )
//	{
//		const float* descriptor = (const float*)obj_reader.ptr;
//		CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
//		memcpy(obj_ptr, descriptor, length*sizeof(float));
//		obj_ptr += length;
//	}
//	return m_data;
//}


void RandTree::SetWindowSize(int w_size)
{
	windowSize=w_size;
}

void RandTree::pridict_prob(Mat mat,CvPoint &pos,LabelResult& result)
{
	//LabelResult result;
	if (treeNum<1)
	{
		return ;
	}
	double *label_prob_all=new double[labelNum];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_all[i]=0;
	}
	int i,j;
	LeafNode leaf;
	for (i=0;i<treeNum;i++)
	{
		getProb(roots[i],mat,pos,leaf);
		if (leaf.leaf_node!=NULL)
		{
			for (j=0;j<labelNum;j++)
			{
				label_prob_all[j]+=leaf.leaf_node->num_of_each_class[j];
			}

		}
	}

	//find the most frequent label
	int maxInd=0;
	double maxNum=-1;
	for (i=0;i<labelNum;i++)
	{
		if (label_prob_all[i]>maxNum)
		{
			maxNum=label_prob_all[i];
			maxInd=i;
		}
	}

	//if (label_prob_all[maxInd]>threshold)
	{
		result.label=maxInd;
		result.prob=label_prob_all[maxInd]/(double)treeNum;
		//result.prob_all=new double[labelNum];
		for (i=0;i<labelNum;i++)
		{
			result.prob_all[i]=label_prob_all[i]/(double)treeNum;
		}
		//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	}
	delete []label_prob_all;

}

void RandTree::pridict_prob(Mat mat,Mat gradient,CvPoint &pos,LabelResult& result)
{
	//LabelResult result;
	if (treeNum<1)
	{
		return ;
	}
	double *label_prob_all=new double[labelNum];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_all[i]=0;
	}
	int i,j;
	LeafNode leaf;
	for (i=0;i<treeNum;i++)
	{
		getProb(roots[i],mat,gradient,pos,leaf);
		if (leaf.leaf_node!=NULL)
		{
			for (j=0;j<labelNum;j++)
			{
				label_prob_all[j]+=leaf.leaf_node->num_of_each_class[j];
			}

		}
	}

	//find the most frequent label
	int maxInd=0;
	double maxNum=-1;
	for (i=0;i<labelNum;i++)
	{
		if (label_prob_all[i]>maxNum)
		{
			maxNum=label_prob_all[i];
			maxInd=i;
		}
	}

	//if (label_prob_all[maxInd]>threshold)
	{
		result.label=maxInd;
		result.prob=label_prob_all[maxInd]/(double)treeNum;
		//result.prob_all=new double[labelNum];
		for (i=0;i<labelNum;i++)
		{
			result.prob_all[i]=label_prob_all[i]/(double)treeNum;
		}
		//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	}
	delete []label_prob_all;

}

void RandTree::total_pridict(Mat mat,CvPoint &pos,LabelResult &result)
{

	if (treeNum<1)
	{
		return;
	}
	int *label_all=new int[labelNum];
	int i,j;
	for (i=0;i<labelNum;i++)
	{
		label_all[i]=0;
	}
	int label_res;
	for (i=0;i<treeNum;i++)
	{
		label_res=getLabel(roots[i],mat,pos);
		//	cout<<"tree"<<i<<" pridiction: "<<label_res<<endl;
		if (label_res>=0)
		{
			label_all[label_res]++;
		}

	}
	//find the most labeled label
	int maxInd=0;
	int maxNum=-1;
	for (i=0;i<labelNum;i++)
	{
		if (label_all[i]>maxNum)
		{
			maxNum=label_all[i];
			maxInd=i;
		}
	}

	double x=(double)label_all[maxInd]/(double)treeNum;
	if (x>threshold)
	{
		result.label=maxInd;
		result.prob=x;
		//result.prob_all=new double[labelNum];
		for (i=0;i<labelNum;i++)
		{
			;
		}
	}
	//return result;
}

void RandTree::train()
{
	srand(time(NULL));
	//prepare the root
	roots=new Node *[max_num_of_trees_in_the_forest];
	for (int i=0;i<max_num_of_trees_in_the_forest;i++)
	{
		roots[i]=new Node;
		roots[i]->nLevel=0;
		//roots[i]->sampleInd=new vector<int>;
	}
	treeNum=0;//no tree at the first

	//int curTreeNum=0;
	int i,j;
	//int *labels=new label[labelNum];
	double lastError;
	double CurError;
	CurError=100000;
	lastError=CurError-1;

	vector<int> choosenPicId;

	int testImgNum=10;
	choosenPicId.resize(testImgNum);
	//int interestInd[]={0,8, 16 ,24, 42, 48, 4 ,12, 20, 28, 45, 51};
	//int interestInd[]={5,6, 10,11,2,3,4,7,9,12,17,18};
	//int interestInd[]={0};
	for (int curTreeNum=0;curTreeNum<max_num_of_trees_in_the_forest;curTreeNum++)
	{
		//cout<<curTreeNum<<endl;
		if (curTreeNum>0&&sampleNumEveryTime!=-1)
		{
			distroyAllData();
		}
		if(sampleNumEveryTime!=-1||(sampleNumEveryTime==-1&&curTreeNum==0))
			getSample(sampleNumEveryTime);
		for (int j=0;j<samples.size();j++)
		{
			roots[curTreeNum]->sampleInd->push_back(j);
		}
					/////////////////////////////////////////////////////////
			//test the full images and add new training samples to the vector

			if (CurError<=0)//make it for sure that the training data has right response
			//if(0)
			{
				LabelResult tmp;
				Mat gradientMap;
				cout<<"perform evaluating\n";
				//step 1: randomly pick 10 images and do full test
				//RandIntVector_cABc(0,shapeNum,choosenPicId);
				RandSample(shapeNum,testImgNum,choosenPicId);
				for (int et=0;et<choosenPicId.size();et++)
				{
					choosenPicId.at(et)--;
					cout<<choosenPicId.at(et)<<" ";
				}
				cout<<endl;

				int stepx,stepy;
				int cx,cy;
				int steplength=10;
				Mat m_img;
				int currentLabel;
				//0 8 16 24 42 48 4 12 20 28 45 51
				for (int et=0;et<choosenPicId.size();et++)
				{
					if (trainStyle==0)
					{
						m_img=sampledImage[choosenPicId.at(et)]->at(0).img;
					}
			/*		else if (trainStyle==1)
					{
						m_img=sampledImage[choosenPicId.at(et)]->at(0).gradientMap;
					}*/
					else if (trainStyle==1||trainStyle==2)
					{
						m_img=sampledImage[choosenPicId.at(et)]->at(0).img;
						gradientMap=sampledImage[choosenPicId.at(et)]->at(0).gradientMap;
					}

				/*	namedWindow("1");
					imshow("1",m_img);
					waitKey();*/
						for (i=windowSize;i<m_img.cols-windowSize;i+=steplength)
						{
							//	cout<<i<<endl;
							for (j=windowSize;j<m_img.rows-windowSize;j+=steplength)
							{
								stepx=RandInt_cABc(1, steplength);
								stepy=RandInt_cABc(1, steplength);
								cx=i+stepx;
								cy=j+stepy;

								//always check the depth
								if (m_img.at<float>(cy,cx)==0)
								{
									continue;
								}
								currentLabel=sizeof(interestInd)/sizeof(int);
								for (int k=0;k<sizeof(interestInd)/sizeof(int);k++)
								{
									if (abs(shape[et]->pts[interestInd[k]][0]-cx)<1&&abs(shape[et]->pts[interestInd[k]][1]-cy)<1)
									{
										currentLabel=k;

										/*namedWindow("1");
										imshow("1",m_img);
										circle(m_img,Point(i,j),1,Scalar(255));
										waitKey();*/
									}
								}

							/*	currentLabel=labelNum-1;
								for (int k=0;k<sizeof(interestInd)/sizeof(int);k++)
								{
									if (abs(shape[et]->pts[interestInd[k]][0]-i)<1&&abs(shape[et]->pts[interestInd[k]][1]-j)<1)
									{
										currentLabel=0;
									}
								}*/
								if (trainStyle==0)
								{
									pridict_prob(m_img,cvPoint(cx,cy),tmp);
								}
								else if (trainStyle==1||trainStyle==2)
								{
									pridict_prob(m_img,gradientMap,cvPoint(cx,cy),tmp);
								}

								
								if (tmp.label!=currentLabel)
								{
									Sample tmpSample;
									//tmpSample.imgMat=m_img;
									if (trainStyle==0)
									{
										tmpSample.imgMat=sampledImage[choosenPicId.at(et)]->at(0).img;
									}
									else if (trainStyle==1)
									{
										tmpSample.imgMat=sampledImage[choosenPicId.at(et)]->at(0).gradientMap;
									}
									else if (trainStyle==2)
									{
										tmpSample.imgMat=sampledImage[choosenPicId.at(et)]->at(0).img;
									}
									tmpSample.gradientMap=sampledImage[choosenPicId.at(et)]->at(0).gradientMap;
									tmpSample.label=currentLabel;
									tmpSample.pos.x=cx;
									tmpSample.pos.y=cy;
									//addedSamples->push_back(tmpSample);
									samples.push_back(tmpSample);
								}
							}
						}

					}

					/*for (int i=curTreeNum+1;i<max_num_of_trees_in_the_forest;i++)
					{
						for (int j=0;j<addedSamples->size();j++)
						{
							roots[i]->sampleInd->push_back(-j);
						}
					}*/

					for (int i=curTreeNum+1;i<max_num_of_trees_in_the_forest;i++)
					{
						roots[i]->sampleInd->clear();
						for (int j=0;j<samples.size();j++)
						{
							roots[i]->sampleInd->push_back(j);
						}
					}
					//isevaluating=true;
				}

		while(1)
		{
			if (trainStyle==0)
			{
				split_WithDepth(roots[curTreeNum]);
			}
			else if (trainStyle==1)
			{
				//split(roots[curTreeNum]);
				split_color(roots[curTreeNum]);
			}
			else if (trainStyle==2)
			{
				//split_gradient(roots[curTreeNum]);
				split_mixture(roots[curTreeNum]);
			}
			

			//////////////////////for tracking the process//////////////////////////////
			//split_output(roots[curTreeNum],"F:\\imgdata\\Rock sec20\\training process\\mouth corner\\");

			//Mat mat=imread("F:\\imgdata\\Rock sec20\\train_evaluation\\cvtColor_undist_sync_webCam_00000.png");
			//CvPoint pos;
			//pos.x=shape[0]->pts[42][0];
			//pos.y=shape[0]->pts[42][1];

			////for(int i=0;i<40;i++)
			////{
			////	namedWindow("1");
			////	circle(mat,Point(rt.shape[0]->pts[i][0],rt.shape[0]->pts[i][1]),2,Scalar(255));
			////	imshow("1",mat);
			////	waitKey();
			////}

			//showTree(roots[0],mat,pos);
			////////////////////////////////////////////////////////////

			

			/*	totalSampleNumTest=0;
			outPutLeaves(roots[curTreeNum]);
			cout<<"totalNum: "<<totalSampleNumTest<<endl;*/
			//caculate the posterior for each class at each leaf, this is done in split
			//caculateProb(roots[curTreeNum]);
			treeNum=curTreeNum+1;


			//test the accruacy
			cout<<"testing with "<<treeNum<<"trees\n";
			int predictedLabel;
			int totalCorrectNum=0;
			LabelResult tmp;
			for (i=0;i<samples.size();i++)
			{	
				//if (total_pridict(roots[j],samples[i].imgMat)==samples[i].label,samples[i].pos)
				if (trainStyle==0)
				{
					pridict_prob(samples[i].imgMat,samples[i].pos,tmp);
				}
				else if (trainStyle==1||trainStyle==2)
				{
					pridict_prob(samples[i].imgMat,samples[i].gradientMap,samples[i].pos,tmp);
				}

				
				predictedLabel=tmp.label;
				if(predictedLabel!=samples[i].label)
				{
					totalCorrectNum++;
					//cout<<predictedLabel<<" "<<samples[i].label<<endl;
				}



				//display the image and show the label
				/*	cout<<predictedLabel<<" "<<samples[i].label<<endl;
				Mat tmp=samples[i].imgMat.clone();
				Point x;
				x.x=samples[i].pos.x;x.y=samples[i].pos.y;
				circle(tmp,x,3,255);
				namedWindow("1");
				imshow("1",tmp);
				waitKey();*/

			}

			//see if any output has label 1
			cout<<endl<<curTreeNum<<" "<<totalCorrectNum<<" "<< sampleNum<<" "<<samples.size()<<endl<<endl;
			//	double tmp=(double)totalCorrectNum/(double) sampleNum;
			//	cout<<tmp<<endl;
			CurError=(double)totalCorrectNum/(double) sampleNum;


	
				/////////////////////////////////////////////////////////



				//if (lastError>=CurError)
				{
					break;
				}
		}
	
		if (CurError<=regression_accuracy)
		{
			;
			//is_satisfied=true;
		}
		if (is_satisfied)
		{
			break;
		}
		lastError=CurError;

		save(curTreeNum+1);
	}
}

void RandTree::caculateProb(Node *root)
{
	LeafNode leaf;
	for (int i=0;i<sampleNum;i++) //get all the numbers we need
	{
		getProb(root,samples[i].imgMat,samples[i].pos,leaf);
		if (leaf.leaf_node==NULL)
		{
			continue;
		}
		if (leaf.leaf_node->num_of_each_class==NULL)
		{
			leaf.leaf_node->num_of_each_class=new float[labelNum];
			for (int j=0;j<labelNum;j++)
			{
				leaf.leaf_node->num_of_each_class[j]=0;
			}
		}
		leaf.leaf_node->num_all++;
		leaf.leaf_node->num_of_each_class[leaf.label]++;
	}	

	//then caculate the probs
	getProb(root);
}

void RandTree::getProb(Node *root)
{
	if (root==NULL)
	{
		return;
	}
	//leaf
	if (root->l_child==NULL&&root->r_child==NULL)
	{
		if (root->num_all==0)
		{
			if (root->num_of_each_class==NULL)
			{
				root->num_of_each_class=new float[labelNum];
			}
			for (int i=0;i<labelNum;i++)
			{
				root->num_of_each_class[i]=0;
			}
		}
		else
		{
			for (int i=0;i<labelNum;i++)
			{
				root->num_of_each_class[i]/=(float)root->num_all;
				if (root->num_of_each_class[0]>0)
				{
					cout<<1<<endl;
				}
			}
		}
	}
	else
	{
		getProb(root->l_child);
		getProb(root->r_child);
	}
}

void RandTree::outPutLeaves(Node *root)
{
	if (root==NULL)
	{
		return;
	}
	//leaf
	
	if (root->l_child==NULL&&root->r_child==NULL)
	{
		if(root->num_of_each_class[0]>0)
		{
			cout<<"level: "<<root->nLevel<<" ";
			for (int i=0;i<labelNum;i++)
			{
				cout<<root->num_of_each_class[i]<<" ";
			}
			cout<<endl;
		}
		
	//	totalSampleNumTest+=root->num_of_each_class[0]*root->num_all;
	}
	else
	{
		outPutLeaves(root->l_child);
		outPutLeaves(root->r_child);
	}
}

void RandTree::sampleWindowIndex(int sampledNum,vector<CvPoint> &SampledPosition1,vector<CvPoint> &SampledPosition2)
{
	SampledPosition1.resize(sampledNum);
	SampledPosition2.resize(sampledNum);
	int num_of_pixels_in_window=windowSize*windowSize;
	vector<int> pos1,pos2;
	//srand(time(NULL));
	//RandSample(num_of_pixels_in_window,sampledNum,pos1);
	int tmp;
	pos1.resize(sampledNum);
	for (int i=0;i<sampledNum;i++)
	{
		tmp = RandInt_cABc(1, num_of_pixels_in_window);
		pos1[i]=tmp;
	}
	//srand(time(NULL));
	//RandSample(num_of_pixels_in_window,sampledNum,pos2);
	pos2.resize(sampledNum);
	for (int i=0;i<sampledNum;i++)
	{
		tmp = RandInt_cABc(1, num_of_pixels_in_window);
		while(tmp==pos1[i])
		{
			tmp = RandInt_cABc(1, num_of_pixels_in_window);
		}
		pos2[i]=tmp;
	}
	int mid=(windowSize+1)/2;
	for (int i=0;i<sampledNum;i++)
	{
		SampledPosition1[i].x=(pos1[i]-1)/windowSize-mid;
		SampledPosition1[i].y=(pos1[i]-1)%windowSize-mid;
		SampledPosition2[i].x=(pos2[i]-1)/windowSize-mid;
		SampledPosition2[i].y=(pos2[i]-1)%windowSize-mid;
	}
	;//RandSample()
}

void RandTree::sampleWindowIndex_gradient(int sampledNum,vector<int> &gradientCandidate,vector<CvPoint> &SampledPosition1,vector<CvPoint> &SampledPosition2)
{
	SampledPosition1.resize(sampledNum);
	SampledPosition2.resize(sampledNum);
	int num_of_pixels_in_window=windowSize*windowSize;
	vector<int> pos1,pos2;
	//srand(time(NULL));
	//RandSample(num_of_pixels_in_window,sampledNum,pos1);
	int tmp;
	pos1.resize(sampledNum);
	for (int i=0;i<sampledNum;i++)
	{
		tmp = RandInt_cABc(1, gradientCandidate.size());
		pos1[i]=gradientCandidate.at(tmp-1);
	}
	//srand(time(NULL));
	//RandSample(num_of_pixels_in_window,sampledNum,pos2);
	pos2.resize(sampledNum);
	for (int i=0;i<sampledNum;i++)
	{
		tmp = RandInt_cABc(1, num_of_pixels_in_window);
		while(tmp==pos1[i])
		{
			tmp = RandInt_cABc(1, num_of_pixels_in_window);
		}
		pos2[i]=tmp;
	}
	int mid=(windowSize+1)/2;
	for (int i=0;i<sampledNum;i++)
	{
		SampledPosition1[i].x=(pos1[i]-1)/windowSize-mid;
		SampledPosition1[i].y=(pos1[i]-1)%windowSize-mid;
		SampledPosition2[i].x=(pos2[i]-1)/windowSize-mid;
		SampledPosition2[i].y=(pos2[i]-1)%windowSize-mid;
	}
	;//RandSample()
}

void RandTree::split_gradient(Node *node)
{
	//cout<<node->nLevel<<" "<<node->sampleInd->size()<<"   ";
	//if (node->nLevel==5&&node->sampleInd->size()==122)
	//{
	//	cout<<"see what happens...\n";
	//}
	if (node->nLevel>=max_depth)//never exceed the max depth
	{
		node->num_all=node->sampleInd->size();
		int i,j;
		if (node->num_of_each_class==NULL)
		{
			node->num_of_each_class=new float[labelNum];
			for (i=0;i<labelNum;i++)
			{
				node->num_of_each_class[i]=0;
			}
		}
		//get the total num
		for (i=0;i<node->sampleInd->size();i++)
		{		
			node->num_of_each_class[samples[node->sampleInd->at(i)].label]++;
		}
		//get the prob
		for (i=0;i<labelNum;i++)
		{
			node->num_of_each_class[i]/=(float)node->num_all;
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<node->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;


		return;
	}
	//vector<int> l_sampleList;	//current filtered sample in left child
	//vector<int> r_sampleList;	//current filtered sample in right child
	//test the accruacy, if enough, return
	

	
	vector<CvPoint> SampledPosition1,SampledPosition2;
	int curSampleNum;
	if (node->nLevel==0)//root
	{
		curSampleNum=rootSampleNum;
	}
	else
	{
		curSampleNum=100*node->nLevel;//100d
	}
	//cout<<"sampling data on level "<<node->nLevel<<endl;
	
	//cout<<"sampling done!\n";

	int splitInd;
	int currentSampleInd;
	vector<int> SampledPositionL,SampledPositionR; //the samples goes to left and right
	int leftNum,rightNum;
	
	{
		//cout<<"finding optimal split test\n";
		int i,j,threshold,l;
		double * proportion1;double *proportion2;
		proportion1=new double [labelNum];
		proportion2=new double [labelNum];
		double maxGain=-100000000000000;
		int maxGainInd=-1;
		int maxThreshold=-1;
		double gain;
		
		Mat tmpGradient;
		Mat activeIndex=Mat::zeros(windowSize,windowSize,CV_8UC1);
		vector<int> sailentAreaIndex;
		while(maxGainInd==-1) //just to make sure there will be a test that both left and right are non-empty
		{
			//sample the first pixel set by using large graditude region only
			sailentAreaIndex.clear();
			for (i=0;i<node->sampleInd->size();i++)
			{
				currentSampleInd=node->sampleInd->at(i);
				//add currentInd into the index list if it is salient enough

				tmpGradient=samples[currentSampleInd].gradientMap(Range(samples[currentSampleInd].pos.y-windowSize/2,samples[currentSampleInd].pos.y+windowSize/2),
					Range(samples[currentSampleInd].pos.x-windowSize/2,samples[currentSampleInd].pos.x+windowSize/2));

				activeIndex+=tmpGradient;

			/*	for (int m=0;m<tmpGradient.rows;m++)
				{
					for (int n=0;n<tmpGradient.cols;n++)
					{
						if (tmpGradient.at<uchar>(m,n)!=0)
						{
							if (find(sailentAreaIndex.begin(),sailentAreaIndex.end(),m*tmpGradient.rows+n)==sailentAreaIndex.end())
							{
								sailentAreaIndex.push_back(m*tmpGradient.rows+n+1);
							}
							
						}
					}
				}*/
				/*namedWindow("1");
				imshow("1",activeIndex);
				waitKey();*/
			}
			for (int m=0;m<tmpGradient.rows;m++)
			{
				for (int n=0;n<tmpGradient.cols;n++)
				{
					if (activeIndex.at<uchar>(m,n)!=0)
					{
							sailentAreaIndex.push_back(m*tmpGradient.rows+n+1);
					}
				}
			}
			sampleWindowIndex_gradient(curSampleNum,sailentAreaIndex,SampledPosition1,SampledPosition2);
			//sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d
			
			//Threshold test
			for (i=0;i<curSampleNum;i++)
			{
				splitInd=i;
				//for (threshold=0;threshold<254;threshold+=10)
				for(l=0;l<5;l++)
				{
					threshold=RandInt_cABc(0,254);

				////////////no threshold//////////////////
	/*			for(l=0;l<1;l++)
				{
					threshold=0;*/
					SampledPositionL.clear();
					SampledPositionR.clear();
					for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
					{
						currentSampleInd=node->sampleInd->at(j);

						if (currentSampleInd<0)
						{
							if (addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition1[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
								SampledPosition1[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)>
								addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition2[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
								SampledPosition2[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}
						}
						else
						{
							if(isevaluating)
							cout<<SampledPosition1[splitInd].y<<" "<<samples[currentSampleInd].pos.y<<" "<<SampledPosition1[splitInd].x<<" "<<
									samples[currentSampleInd].pos.x<<endl;
							if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
								SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
								samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
								SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}
						}
						//cout<<i<<endl;
				
					}
					if (SampledPositionL.size()==0||SampledPositionR.size()==0)
					{
						continue;
					}
					//caculate the entrop 
					for (j=0;j<labelNum;j++)
					{
						proportion1[j]=proportion2[j]=0;
					}
					for (j=0;j<SampledPositionL.size();j++)
					{
						if (SampledPositionL.at(j)>=0)
						{
							proportion1[samples[SampledPositionL.at(j)].label]++;
						}
						else
						{
							proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
						}
						
					}
					for (j=0;j<SampledPositionR.size();j++)
					{
						if (SampledPositionR.at(j)>=0)
						{
							proportion2[samples[SampledPositionR.at(j)].label]++;
						}
						else
						{
							proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
						}
						
					}

					//..get the probability
					if (SampledPositionL.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]/=(double)SampledPositionL.size();
						}
					}
					if (SampledPositionR.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion2[j]/=(double)SampledPositionR.size();
						}
					}

					//..get the entropy
					double entropy[2];
					entropy[0]=entropy[1]=0;
					for (j=0;j<labelNum;j++)
					{
						if (proportion1[j]!=0)
						{
							entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
						}
						if (proportion2[j]!=0)
						{
							entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
						}		
					}

					//get the gain
					gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
					if (gain>maxGain)
					{
						maxGain=gain;
						maxGainInd=i;
						maxThreshold=threshold;
					}
				}
			}
		}
		
		
		//finally get the left and right sample
		SampledPositionL.clear();
		SampledPositionR.clear();
		splitInd=maxGainInd;
		node->pos1[0]=SampledPosition1[splitInd].x;
		node->pos1[1]=SampledPosition1[splitInd].y;
		node->pos2[0]=SampledPosition2[splitInd].x;
		node->pos2[1]=SampledPosition2[splitInd].y;
		node->threshold=maxThreshold;
		for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
		{
			currentSampleInd=node->sampleInd->at(j);
			//cout<<i<<endl;
			if (currentSampleInd<0)
			{
				if (addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition1[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
					SampledPosition1[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)>
					addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition2[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
					SampledPosition2[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
			else
			{
				if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
					SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
					samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
					SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
	
		}
		
		delete []proportion1;
		delete []proportion2;


	}
	leftNum=SampledPositionL.size();
	rightNum=SampledPositionR.size();

	//if no need to stop, split and call them recursively
	if (leftNum>min_sample_count)
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		split_gradient(leftChild);
	}
	else //treat it as a leaf
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		
		leftChild->num_all=leftChild->sampleInd->size();
		int i,j;
		if (leftChild->num_all==0)
		{
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					leftChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<leftChild->sampleInd->size();i++)
			{		
				if (leftChild->sampleInd->at(i)>=0)
				{
					leftChild->num_of_each_class[samples[leftChild->sampleInd->at(i)].label]++;
				}
				else
				{
					leftChild->num_of_each_class[addedSamples->at(-leftChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]/=(float)leftChild->num_all;
			}
		}
	

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<leftChild->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;

		
	}
	if (rightNum>min_sample_count)
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		split_gradient(rightChild);
	}
	else //treat it as a leaf
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		
		rightChild->num_all=rightChild->sampleInd->size();
		int i,j;
		if (rightChild->num_all==0)
		{
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					rightChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<rightChild->sampleInd->size();i++)
			{	
				if (rightChild->sampleInd->at(i)>=0)
				{
					rightChild->num_of_each_class[samples[rightChild->sampleInd->at(i)].label]++;
				}
				else
				{
					rightChild->num_of_each_class[addedSamples->at(-rightChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]/=(float)rightChild->num_all;
			}

		}

	/*	for (i=0;i<labelNum;i++)
		{
			cout<<rightChild->num_of_each_class[i]<<" ";
		}
		cout<<endl;*/
	}

}

void RandTree::split_output(Node *node,string namebase)
{
	//cout<<node->nLevel<<" "<<node->sampleInd->size()<<"   ";
	//if (node->nLevel==5&&node->sampleInd->size()==122)
	//{
	//	cout<<"see what happens...\n";
	//}

	//out_p<<node->nLevel<<" "<<node->sampleInd->size()<<" "<<123<<endl;
	//for (int i=0;i<node->sampleInd->size();i++)
	//{
	//	out_p<<node->sampleInd->at(i)<<" "<<samplesnode->sampleInd->at(i);
	//}
	//out_p<<endl;

	char name_withInd[50];
	//LPCWSTR str = namebase.c_str();
	//if(!PathFileExists(namebase.c_str()))
	//	CreateDirectory(namebase.c_str(),NULL);

	if (node->nLevel>=max_depth)//never exceed the max depth
	{
		node->num_all=node->sampleInd->size();
		int i,j;
		if (node->num_of_each_class==NULL)
		{
			node->num_of_each_class=new float[labelNum];
			for (i=0;i<labelNum;i++)
			{
				node->num_of_each_class[i]=0;
			}
		}
		//get the total num
		for (i=0;i<node->sampleInd->size();i++)
		{		
			node->num_of_each_class[samples[node->sampleInd->at(i)].label]++;
		}
		//get the prob
		for (i=0;i<labelNum;i++)
		{
			node->num_of_each_class[i]/=(float)node->num_all;
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<node->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;


		return;
	}
	//vector<int> l_sampleList;	//current filtered sample in left child
	//vector<int> r_sampleList;	//current filtered sample in right child
	//test the accruacy, if enough, return
	

	
	vector<CvPoint> SampledPosition1,SampledPosition2;
	int curSampleNum;
	if (node->nLevel==0)//root
	{
		curSampleNum=rootSampleNum;
	}
	else
	{
		curSampleNum=100*node->nLevel;//100d
	}
	//cout<<"sampling data on level "<<node->nLevel<<endl;
	
	//cout<<"sampling done!\n";

	int splitInd;
	int currentSampleInd;
	vector<int> SampledPositionL,SampledPositionR; //the samples goes to left and right
	int leftNum,rightNum;
	if (fullRandom)
	{
		sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d
		int l=SampledPosition1.size();
		//select a question

		while (SampledPositionL.size()==0||SampledPositionR.size()==0)
		{
			SampledPositionL.clear();
			SampledPositionR.clear();
			splitInd=RandInt_cABc(0,l-1);

			node->pos1[0]=SampledPosition1[splitInd].x;
			node->pos1[1]=SampledPosition1[splitInd].y;
			node->pos2[0]=SampledPosition2[splitInd].x;
			node->pos2[1]=SampledPosition2[splitInd].y;
			//check if the left num is more than min value




			for (int i=0;i<node->sampleInd->size();i++) //if larger, go to left. Otherwise right
			{
				currentSampleInd=node->sampleInd->at(i);
				//cout<<i<<endl;
				if (samples[currentSampleInd].imgMat.at<uchar>(node->pos1[1]+samples[currentSampleInd].pos.y,
					node->pos1[0]+samples[currentSampleInd].pos.x)>
					samples[currentSampleInd].imgMat.at<uchar>(node->pos2[1]+samples[currentSampleInd].pos.y,
					node->pos2[0]+samples[currentSampleInd].pos.x))
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
		}

	}
	else //maximum entropy
	{
		//cout<<"finding optimal split test\n";
		int i,j,threshold,l;
		double * proportion1;double *proportion2;
		proportion1=new double [labelNum];
		proportion2=new double [labelNum];
		double maxGain=-100000000000000;
		int maxGainInd=-1;
		int maxThreshold=-1;
		double gain;
		
		
		while(maxGainInd==-1) //just to make sure there will be a test that both left and right are non-empty
		{
			sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d

			////binary test
			//for (i=0;i<curSampleNum;i++)
			//{
			//	SampledPositionL.clear();
			//	SampledPositionR.clear();
			//	splitInd=i;
			//	for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
			//	{
			//		currentSampleInd=node->sampleInd->at(j);
			//		//cout<<i<<endl;
			//		if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
			//			SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
			//			samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
			//			SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x))
			//			//if(1)
			//		{
			//			SampledPositionL.push_back(currentSampleInd);
			//		}
			//		else
			//		{ 
			//			SampledPositionR.push_back(currentSampleInd);
			//		}
			//	}
			//	if (SampledPositionL.size()==0||SampledPositionR.size()==0)
			//	{
			//		continue;
			//	}
			//	//caculate the entrop 
			//	for (j=0;j<labelNum;j++)
			//	{
			//		proportion1[j]=proportion2[j]=0;
			//	}
			//	for (j=0;j<SampledPositionL.size();j++)
			//	{
			//		proportion1[samples[SampledPositionL.at(j)].label]++;
			//	}
			//	for (j=0;j<SampledPositionR.size();j++)
			//	{
			//		proportion2[samples[SampledPositionR.at(j)].label]++;
			//	}

			//	//..get the probability
			//	if (SampledPositionL.size()>0)
			//	{
			//		for (j=0;j<labelNum;j++)
			//		{
			//			proportion1[j]/=(double)SampledPositionL.size();
			//		}
			//	}
			//	if (SampledPositionR.size()>0)
			//	{
			//		for (j=0;j<labelNum;j++)
			//		{
			//			proportion2[j]/=(double)SampledPositionR.size();
			//		}
			//	}

			//	//..get the entropy
			//	double entropy[2];
			//	entropy[0]=entropy[1]=0;
			//	for (j=0;j<labelNum;j++)
			//	{
			//		if (proportion1[j]!=0)
			//		{
			//			entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
			//		}
			//		if (proportion2[j]!=0)
			//		{
			//			entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
			//		}		
			//	}
	
			//	//get the gain
			//	gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
			//	if (gain>maxGain)
			//	{
			//		maxGain=gain;
			//		maxGainInd=i;
			//	}
			////	break;
			//	//if (node->nLevel==5&&node->sampleInd->size()==122)
			///*	{
			//		cout<<gain<<" "<<maxGain<<" "<<maxGainInd<<endl;
			//	}*/
			//}

			//Threshold test
			for (i=0;i<curSampleNum;i++)
			{
				splitInd=i;
				//for (threshold=0;threshold<254;threshold+=10)
				for(l=0;l<5;l++)
				{
					threshold=RandInt_cABc(0,254);

					if (trainStyle==1)
					{
						threshold=0;
						if (l>0)
						{
							break;
						}
					}

				////////////no threshold//////////////////
	/*			for(l=0;l<1;l++)
				{
					threshold=0;*/
					SampledPositionL.clear();
					SampledPositionR.clear();
					for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
					{
						currentSampleInd=node->sampleInd->at(j);

						if (currentSampleInd<0)
						{
							if (addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition1[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
								SampledPosition1[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)>
								addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition2[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
								SampledPosition2[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}
						}
						else
						{
							if(isevaluating)
							cout<<SampledPosition1[splitInd].y<<" "<<samples[currentSampleInd].pos.y<<" "<<SampledPosition1[splitInd].x<<" "<<
									samples[currentSampleInd].pos.x<<endl;
							if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
								SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
								samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
								SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}
						}
						//cout<<i<<endl;
				
					}
					if (SampledPositionL.size()==0||SampledPositionR.size()==0)
					{
						continue;
					}
					//caculate the entrop 
					for (j=0;j<labelNum;j++)
					{
						proportion1[j]=proportion2[j]=0;
					}
					for (j=0;j<SampledPositionL.size();j++)
					{
						if (SampledPositionL.at(j)>=0)
						{
							proportion1[samples[SampledPositionL.at(j)].label]++;
						}
						else
						{
							proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
						}
						
					}
					for (j=0;j<SampledPositionR.size();j++)
					{
						if (SampledPositionR.at(j)>=0)
						{
							proportion2[samples[SampledPositionR.at(j)].label]++;
						}
						else
						{
							proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
						}
						
					}

					//..get the probability
					if (SampledPositionL.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]/=(double)SampledPositionL.size();
						}
					}
					if (SampledPositionR.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion2[j]/=(double)SampledPositionR.size();
						}
					}

					//..get the entropy
					double entropy[2];
					entropy[0]=entropy[1]=0;
					for (j=0;j<labelNum;j++)
					{
						if (proportion1[j]!=0)
						{
							entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
						}
						if (proportion2[j]!=0)
						{
							entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
						}		
					}

					//get the gain
					gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
					if (gain>maxGain)
					{
						maxGain=gain;
						maxGainInd=i;
						maxThreshold=threshold;
					}
				}
			}

			//RGB test



		}
		
		
		//finally get the left and right sample
		SampledPositionL.clear();
		SampledPositionR.clear();
		splitInd=maxGainInd;
		node->pos1[0]=SampledPosition1[splitInd].x;
		node->pos1[1]=SampledPosition1[splitInd].y;
		node->pos2[0]=SampledPosition2[splitInd].x;
		node->pos2[1]=SampledPosition2[splitInd].y;
		node->threshold=maxThreshold;




		for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
		{
			currentSampleInd=node->sampleInd->at(j);
			//cout<<i<<endl;
			if (currentSampleInd<0)
			{
				if (addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition1[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
					SampledPosition1[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)>
					addedSamples->at(-currentSampleInd).imgMat.at<uchar>(SampledPosition2[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
					SampledPosition2[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
			else
			{
				if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
					SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
					samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
					SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
	
		}

		char levelName[10];
		sprintf(levelName, "%d\\", node->nLevel+1);

		MakeSureDirectoryPathExists((namebase+"left_"+levelName).c_str());
		Mat currentImg;
		for (int i=0;i<SampledPositionL.size();i++)
		{
			currentImg=samples[SampledPositionL.at(i)].imgMat(Range(samples[SampledPositionL.at(i)].pos.y-windowSize/2,
				samples[SampledPositionL.at(i)].pos.y+windowSize/2),Range(samples[SampledPositionL.at(i)].pos.x-windowSize/2,
				samples[SampledPositionL.at(i)].pos.x+windowSize/2)).clone();

			circle(currentImg,Point(node->pos1[0]+windowSize/2,node->pos1[1]+windowSize/2),3,Scalar(255));
			circle(currentImg,Point(node->pos2[0]+windowSize/2,node->pos2[1]+windowSize/2),1,Scalar(255));
			sprintf(name_withInd, "%d.jpg", i);
			//cout<<namebase+name_withInd<<endl;
			imwrite(namebase+"left_"+levelName+name_withInd,currentImg);
		}

		MakeSureDirectoryPathExists((namebase+"right_"+levelName).c_str());
		for (int i=0;i<SampledPositionR.size();i++)
		{
			currentImg=samples[SampledPositionR.at(i)].imgMat(Range(samples[SampledPositionR.at(i)].pos.y-windowSize/2,
				samples[SampledPositionR.at(i)].pos.y+windowSize/2),Range(samples[SampledPositionR.at(i)].pos.x-windowSize/2,
				samples[SampledPositionR.at(i)].pos.x+windowSize/2)).clone();

			circle(currentImg,Point(node->pos1[0]+windowSize/2,node->pos1[1]+windowSize/2),3,Scalar(255));
			circle(currentImg,Point(node->pos2[0]+windowSize/2,node->pos2[1]+windowSize/2),1,Scalar(255));
			sprintf(name_withInd, "%d.jpg", i);
			//cout<<namebase+name_withInd<<endl;
			imwrite(namebase+"right_"+levelName+name_withInd,currentImg);
		}
		
		delete []proportion1;
		delete []proportion2;


	}
	leftNum=SampledPositionL.size();
	rightNum=SampledPositionR.size();

	
	//if no need to stop, split and call them recursively
	if (leftNum>min_sample_count)
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		char levelName[10];
		sprintf(levelName, "%d\\", node->nLevel+1);
		split_output(leftChild,namebase+"left_"+levelName);
	}
	else //treat it as a leaf
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		
		leftChild->num_all=leftChild->sampleInd->size();
		int i,j;
		if (leftChild->num_all==0)
		{
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					leftChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<leftChild->sampleInd->size();i++)
			{		
				if (leftChild->sampleInd->at(i)>=0)
				{
					leftChild->num_of_each_class[samples[leftChild->sampleInd->at(i)].label]++;
				}
				else
				{
					leftChild->num_of_each_class[addedSamples->at(-leftChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]/=(float)leftChild->num_all;
			}
		}
	
		char levelName[10];
		sprintf(levelName, "%d\\", node->nLevel+1);
		namebase=namebase+"left_"+levelName;
		MakeSureDirectoryPathExists(namebase.c_str());
		Mat currentImg;
		for (int i=0;i<SampledPositionL.size();i++)
		{
			currentImg=samples[SampledPositionL.at(i)].imgMat(Range(samples[SampledPositionL.at(i)].pos.y-windowSize/2,
				samples[SampledPositionL.at(i)].pos.y+windowSize/2),Range(samples[SampledPositionL.at(i)].pos.x-windowSize/2,
				samples[SampledPositionL.at(i)].pos.x+windowSize/2));
			sprintf(name_withInd, "%d.jpg", i);
			//cout<<namebase+name_withInd<<endl;
			imwrite(namebase+name_withInd,currentImg);
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<leftChild->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;

		
	}
	if (rightNum>min_sample_count)
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}

		char levelName[10];
		sprintf(levelName, "%d\\", node->nLevel+1);
		split_output(rightChild,namebase+"right_"+levelName);
	}
	else //treat it as a leaf
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		
		rightChild->num_all=rightChild->sampleInd->size();
		int i,j;
		if (rightChild->num_all==0)
		{
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					rightChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<rightChild->sampleInd->size();i++)
			{	
				if (rightChild->sampleInd->at(i)>=0)
				{
					rightChild->num_of_each_class[samples[rightChild->sampleInd->at(i)].label]++;
				}
				else
				{
					rightChild->num_of_each_class[addedSamples->at(-rightChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]/=(float)rightChild->num_all;
			}

		}

		char levelName[10];
		sprintf(levelName, "%d\\", node->nLevel+1);
		namebase=namebase+"right_"+levelName;
		MakeSureDirectoryPathExists(namebase.c_str());
		Mat currentImg;
		for (int i=0;i<SampledPositionR.size();i++)
		{
			currentImg=samples[SampledPositionR.at(i)].imgMat(Range(samples[SampledPositionR.at(i)].pos.y-windowSize/2,
				samples[SampledPositionR.at(i)].pos.y+windowSize/2),Range(samples[SampledPositionR.at(i)].pos.x-windowSize/2,
				samples[SampledPositionR.at(i)].pos.x+windowSize/2));
			sprintf(name_withInd, "%d.jpg", i);
			//cout<<namebase+name_withInd<<endl;
			imwrite(namebase+name_withInd,currentImg);
		}

	/*	for (i=0;i<labelNum;i++)
		{
			cout<<rightChild->num_of_each_class[i]<<" ";
		}
		cout<<endl;*/
	}

}


//ofstream out_p("F:\\imgdata\\Rock sec20\\training process\\process.txt",ios::out);

void RandTree::split(Node *node)
{
	//srand(time(NULL));
	cout<<node->nLevel<<" "<<node->sampleInd->size()<<"   ";
	//if (node->nLevel==5&&node->sampleInd->size()==122)
	//{
	//	cout<<"see what happens...\n";
	//}

	//out_p<<node->nLevel<<" "<<node->sampleInd->size()<<" "<<123<<endl;
	//for (int i=0;i<node->sampleInd->size();i++)
	//{
	//	out_p<<node->sampleInd->at(i)<<" "<<samplesnode->sampleInd->at(i);
	//}
	//out_p<<endl;

	//int totalNum=0;
	int i;
	for (i=1;i<node->sampleInd->size();i++)
	{
		if (samples[node->sampleInd->at(i)].label!=samples[node->sampleInd->at(i-1)].label)
		{
			break;
		}
	}

	if (node->nLevel>=max_depth||i==node->sampleInd->size())//never exceed the max depth. Also, if all are with the same labels, it is a leaf node
	{
		node->num_all=node->sampleInd->size();
		int i,j;
		if (node->num_of_each_class==NULL)
		{
			node->num_of_each_class=new float[labelNum];
			for (i=0;i<labelNum;i++)
			{
				node->num_of_each_class[i]=0;
			}
		}
		//get the total num
		for (i=0;i<node->sampleInd->size();i++)
		{		
			node->num_of_each_class[samples[node->sampleInd->at(i)].label]++;
		}
		//get the prob
		for (i=0;i<labelNum;i++)
		{
			node->num_of_each_class[i]/=(float)node->num_all;
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<node->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;


		return;
	}
	//vector<int> l_sampleList;	//current filtered sample in left child
	//vector<int> r_sampleList;	//current filtered sample in right child
	//test the accruacy, if enough, return
	

	
	vector<CvPoint> SampledPosition1,SampledPosition2;
	int curSampleNum;
	if (node->nLevel==0)//root
	{
		curSampleNum=rootSampleNum;
	}
	else
	{
		curSampleNum=100*node->nLevel;//100d
	}
	//cout<<"sampling data on level "<<node->nLevel<<endl;
	
	//cout<<"sampling done!\n";

	int splitInd;
	int currentSampleInd;
	vector<int> SampledPositionL,SampledPositionR; //the samples goes to left and right
	int leftNum,rightNum;
	//color with depth ratio
	{
		//cout<<"finding optimal split test\n";
		int i,j,l;
		double threshold;
		double * proportion1;double *proportion2;
		//cout<<labelNum<<endl;
		proportion1=new double [labelNum];
		proportion2=new double [labelNum];
		double maxGain=-100000000000000;
		int maxGainInd=-1;
		int maxThreshold=-1;
		double gain;
		
		
		while(maxGainInd==-1) //just to make sure there will be a test that both left and right are non-empty
		{
			sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d

			////binary test
			//for (i=0;i<curSampleNum;i++)
			//{
			//	SampledPositionL.clear();
			//	SampledPositionR.clear();
			//	splitInd=i;
			//	for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
			//	{
			//		currentSampleInd=node->sampleInd->at(j);
			//		//cout<<i<<endl;
			//		if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
			//			SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
			//			samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
			//			SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x))
			//			//if(1)
			//		{
			//			SampledPositionL.push_back(currentSampleInd);
			//		}
			//		else
			//		{ 
			//			SampledPositionR.push_back(currentSampleInd);
			//		}
			//	}
			//	if (SampledPositionL.size()==0||SampledPositionR.size()==0)
			//	{
			//		continue;
			//	}
			//	//caculate the entrop 
			//	for (j=0;j<labelNum;j++)
			//	{
			//		proportion1[j]=proportion2[j]=0;
			//	}
			//	for (j=0;j<SampledPositionL.size();j++)
			//	{
			//		proportion1[samples[SampledPositionL.at(j)].label]++;
			//	}
			//	for (j=0;j<SampledPositionR.size();j++)
			//	{
			//		proportion2[samples[SampledPositionR.at(j)].label]++;
			//	}

			//	//..get the probability
			//	if (SampledPositionL.size()>0)
			//	{
			//		for (j=0;j<labelNum;j++)
			//		{
			//			proportion1[j]/=(double)SampledPositionL.size();
			//		}
			//	}
			//	if (SampledPositionR.size()>0)
			//	{
			//		for (j=0;j<labelNum;j++)
			//		{
			//			proportion2[j]/=(double)SampledPositionR.size();
			//		}
			//	}

			//	//..get the entropy
			//	double entropy[2];
			//	entropy[0]=entropy[1]=0;
			//	for (j=0;j<labelNum;j++)
			//	{
			//		if (proportion1[j]!=0)
			//		{
			//			entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
			//		}
			//		if (proportion2[j]!=0)
			//		{
			//			entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
			//		}		
			//	}
	
			//	//get the gain
			//	gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
			//	if (gain>maxGain)
			//	{
			//		maxGain=gain;
			//		maxGainInd=i;
			//	}
			////	break;
			//	//if (node->nLevel==5&&node->sampleInd->size()==122)
			///*	{
			//		cout<<gain<<" "<<maxGain<<" "<<maxGainInd<<endl;
			//	}*/
			//}

			//Threshold test
			for (i=0;i<curSampleNum;i++)
			{
				splitInd=i;
				//for (threshold=0;threshold<254;threshold+=10)
				for(l=0;l<5;l++)
				{
					//threshold=(double)RandInt_cABc(0,256)/256.0f;//[0,1]
					threshold=(double)RandInt_cABc(0,512)/256.0f;//[0,2]
					//threshold=(double)RandInt_cABc(0,256*4)/256.0f-2;//[-2,2]
					if (l==0)
					{
						threshold=0;
					}
					else 
						continue;

				/*	if (trainStyle==0)
					{
						threshold=0;
						if (l>0)
						{
							break;
						}
					}*/

				////////////no threshold//////////////////
	/*			for(l=0;l<1;l++)
				{
					threshold=0;*/
					SampledPositionL.clear();
					SampledPositionR.clear();
					for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
					{
						currentSampleInd=node->sampleInd->at(j);

						if (currentSampleInd<0)
						{
							if (addedSamples->at(-currentSampleInd).imgMat.at<float>(SampledPosition1[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
								SampledPosition1[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)>
								addedSamples->at(-currentSampleInd).imgMat.at<float>(SampledPosition2[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
								SampledPosition2[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}
						}
						else
						{
							if(isevaluating)
							cout<<SampledPosition1[splitInd].y<<" "<<samples[currentSampleInd].pos.y<<" "<<SampledPosition1[splitInd].x<<" "<<
									samples[currentSampleInd].pos.x<<" "<<SampledPosition2[splitInd].y<<" "<<SampledPosition2[splitInd].x<<
									" "<<samples[currentSampleInd].imgMat.rows<<" "<<samples[currentSampleInd].imgMat.cols<<endl;
				
							if (samples[currentSampleInd].imgMat.at<float>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
								SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
								samples[currentSampleInd].imgMat.at<float>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
								SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}
						}
						//cout<<i<<endl;
				
					}
					if (SampledPositionL.size()==0||SampledPositionR.size()==0)
					{
						continue;
					}
					//caculate the entrop 
					for (j=0;j<labelNum;j++)
					{
						proportion1[j]=proportion2[j]=0;
					}
					for (j=0;j<SampledPositionL.size();j++)
					{
						if (SampledPositionL.at(j)>=0)
						{
							proportion1[samples[SampledPositionL.at(j)].label]++;
						}
						else
						{
							proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
						}
						
					}
					for (j=0;j<SampledPositionR.size();j++)
					{
						if (SampledPositionR.at(j)>=0)
						{
							proportion2[samples[SampledPositionR.at(j)].label]++;
						}
						else
						{
							proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
						}
						
					}

					//..get the probability
					if (SampledPositionL.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]/=(double)SampledPositionL.size();
						}
					}
					if (SampledPositionR.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion2[j]/=(double)SampledPositionR.size();
						}
					}

					//..get the entropy
					double entropy[2];
					entropy[0]=entropy[1]=0;
					for (j=0;j<labelNum;j++)
					{
						if (proportion1[j]!=0)
						{
							entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
						}
						if (proportion2[j]!=0)
						{
							entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
						}		
					}

					//get the gain
					gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
					//cout<<threshold<<" "<<gain<<endl;
					if (gain>maxGain)
					{
						maxGain=gain;
						maxGainInd=i;
						maxThreshold=threshold;
					}
				}
			}

			//RGB test



		}
		
		
		//finally get the left and right sample
		SampledPositionL.clear();
		SampledPositionR.clear();
		splitInd=maxGainInd;
		node->pos1[0]=SampledPosition1[splitInd].x;
		node->pos1[1]=SampledPosition1[splitInd].y;
		node->pos2[0]=SampledPosition2[splitInd].x;
		node->pos2[1]=SampledPosition2[splitInd].y;
		node->threshold=maxThreshold;
		for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
		{
			currentSampleInd=node->sampleInd->at(j);
			//cout<<i<<endl;
			if (currentSampleInd<0)
			{
				if (addedSamples->at(-currentSampleInd).imgMat.at<float>(SampledPosition1[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
					SampledPosition1[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)>
					addedSamples->at(-currentSampleInd).imgMat.at<float>(SampledPosition2[splitInd].y+addedSamples->at(-currentSampleInd).pos.y,
					SampledPosition2[splitInd].x+addedSamples->at(-currentSampleInd).pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
			else
			{
				if (samples[currentSampleInd].imgMat.at<float>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
					SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
					samples[currentSampleInd].imgMat.at<float>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
					SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
	
		}
		
		delete []proportion1;
		delete []proportion2;


	}
	leftNum=SampledPositionL.size();
	rightNum=SampledPositionR.size();

	//if no need to stop, split and call them recursively
	if (leftNum>min_sample_count)
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		split(leftChild);
	}
	else //treat it as a leaf
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		
		leftChild->num_all=leftChild->sampleInd->size();
		int i,j;
		if (leftChild->num_all==0)
		{
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					leftChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<leftChild->sampleInd->size();i++)
			{		
				if (leftChild->sampleInd->at(i)>=0)
				{
					leftChild->num_of_each_class[samples[leftChild->sampleInd->at(i)].label]++;
				}
				else
				{
					leftChild->num_of_each_class[addedSamples->at(-leftChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]/=(float)leftChild->num_all;
			}
		}
	

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<leftChild->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;

		
	}
	if (rightNum>min_sample_count)
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		split(rightChild);
	}
	else //treat it as a leaf
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		
		rightChild->num_all=rightChild->sampleInd->size();
		int i,j;
		if (rightChild->num_all==0)
		{
			//get the prob
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
			}
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					rightChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<rightChild->sampleInd->size();i++)
			{	
				if (rightChild->sampleInd->at(i)>=0)
				{
					rightChild->num_of_each_class[samples[rightChild->sampleInd->at(i)].label]++;
				}
				else
				{
					rightChild->num_of_each_class[addedSamples->at(-rightChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]/=(float)rightChild->num_all;
			}

		}

	/*	for (i=0;i<labelNum;i++)
		{
			cout<<rightChild->num_of_each_class[i]<<" ";
		}
		cout<<endl;*/
	}

}

void RandTree::split_color(Node *node)
{
	//srand(time(NULL));
	cout<<node->nLevel<<" "<<node->sampleInd->size()<<"   ";
	//if (node->nLevel==5&&node->sampleInd->size()==122)
	//{
	//	cout<<"see what happens...\n";
	//}

	//out_p<<node->nLevel<<" "<<node->sampleInd->size()<<" "<<123<<endl;
	//for (int i=0;i<node->sampleInd->size();i++)
	//{
	//	out_p<<node->sampleInd->at(i)<<" "<<samplesnode->sampleInd->at(i);
	//}
	//out_p<<endl;

	//int totalNum=0;
	int i;
	for (i=1;i<node->sampleInd->size();i++)
	{
		if (samples[node->sampleInd->at(i)].label!=samples[node->sampleInd->at(i-1)].label)
		{
			break;
		}
	}

	if (node->nLevel>=max_depth||i==node->sampleInd->size())//never exceed the max depth. Also, if all are with the same labels, it is a leaf node
	{
		node->num_all=node->sampleInd->size();
		int i,j;
		if (node->num_of_each_class==NULL)
		{
			node->num_of_each_class=new float[labelNum];
			for (i=0;i<labelNum;i++)
			{
				node->num_of_each_class[i]=0;
			}
		}
		//get the total num
		for (i=0;i<node->sampleInd->size();i++)
		{		
			node->num_of_each_class[samples[node->sampleInd->at(i)].label]++;
		}
		//get the prob
		for (i=0;i<labelNum;i++)
		{
			node->num_of_each_class[i]/=(float)node->num_all;
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<node->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;


		return;
	}
	//vector<int> l_sampleList;	//current filtered sample in left child
	//vector<int> r_sampleList;	//current filtered sample in right child
	//test the accruacy, if enough, return
	

	
	vector<CvPoint> SampledPosition1,SampledPosition2;
	int curSampleNum;
	if (node->nLevel==0)//root
	{
		curSampleNum=rootSampleNum;
	}
	else
	{
		curSampleNum=100*node->nLevel;//100d
	}
	//cout<<"sampling data on level "<<node->nLevel<<endl;
	
	//cout<<"sampling done!\n";

	int splitInd;
	int currentSampleInd;
	vector<int> SampledPositionL,SampledPositionR; //the samples goes to left and right
	int leftNum,rightNum;
	
	//maximum entropy
	{
		//cout<<"finding optimal split test\n";
		int i,j,l;
		double threshold;
		double * proportion1;double *proportion2;
		//cout<<labelNum<<endl;
		proportion1=new double [labelNum];
		proportion2=new double [labelNum];
		double maxGain=-100000000000000;
		int maxGainInd=-1;
		int maxThreshold=-1;
		double gain;
		
		
		while(maxGainInd==-1) //just to make sure there will be a test that both left and right are non-empty
		{
			sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d

			//Threshold test
			for (i=0;i<curSampleNum;i++)
			{
				splitInd=i;
				//for (threshold=0;threshold<254;threshold+=10)
				for(l=0;l<5;l++)
				{
					//threshold=(double)RandInt_cABc(0,256)/256.0f;//[0,1]
					threshold=(double)RandInt_cABc(0,512)/256.0f;//[0,2]
					//threshold=(double)RandInt_cABc(0,256*4)/256.0f-2;//[-2,2]
					if (l==0)
					{
						threshold=0;
					}
					else 
						continue;

					 
					SampledPositionL.clear();
					SampledPositionR.clear();
					for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
					{
						currentSampleInd=node->sampleInd->at(j);
						
						if(isevaluating)
							cout<<SampledPosition1[splitInd].y<<" "<<samples[currentSampleInd].pos.y<<" "<<SampledPosition1[splitInd].x<<" "<<
							samples[currentSampleInd].pos.x<<" "<<SampledPosition2[splitInd].y<<" "<<SampledPosition2[splitInd].x<<
							" "<<samples[currentSampleInd].imgMat.rows<<" "<<samples[currentSampleInd].imgMat.cols<<endl;
		
						double currentDepth=1.0f/samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
						if (samples[currentSampleInd].gradientMap.at<uchar>((double)SampledPosition1[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
							(double)SampledPosition1[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)>
							(double)samples[currentSampleInd].gradientMap.at<uchar>(SampledPosition2[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
							(double)SampledPosition2[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)+threshold)
							//if(1)
						{
							SampledPositionL.push_back(currentSampleInd);
						}
						else
						{ 
							SampledPositionR.push_back(currentSampleInd);
						}
						
						//cout<<i<<endl;
				
					}
					if (SampledPositionL.size()==0||SampledPositionR.size()==0)
					{
						continue;
					}
					//caculate the entrop 
					for (j=0;j<labelNum;j++)
					{
						proportion1[j]=proportion2[j]=0;
					}
					for (j=0;j<SampledPositionL.size();j++)
					{
						if (SampledPositionL.at(j)>=0)
						{
							proportion1[samples[SampledPositionL.at(j)].label]++;
						}
						else
						{
							proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
						}
						
					}
					for (j=0;j<SampledPositionR.size();j++)
					{
						if (SampledPositionR.at(j)>=0)
						{
							proportion2[samples[SampledPositionR.at(j)].label]++;
						}
						else
						{
							proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
						}
						
					}

					//..get the probability
					if (SampledPositionL.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]/=(double)SampledPositionL.size();
						}
					}
					if (SampledPositionR.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion2[j]/=(double)SampledPositionR.size();
						}
					}

					//..get the entropy
					double entropy[2];
					entropy[0]=entropy[1]=0;
					for (j=0;j<labelNum;j++)
					{
						if (proportion1[j]!=0)
						{
							entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
						}
						if (proportion2[j]!=0)
						{
							entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
						}		
					}

					//get the gain
					gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
					//cout<<threshold<<" "<<gain<<endl;
					if (gain>maxGain)
					{
						maxGain=gain;
						maxGainInd=i;
						maxThreshold=threshold;
					}
				}
			}

			//RGB test



		}
		
		
		//finally get the left and right sample
		SampledPositionL.clear();
		SampledPositionR.clear();
		splitInd=maxGainInd;
		node->pos1[0]=SampledPosition1[splitInd].x;
		node->pos1[1]=SampledPosition1[splitInd].y;
		node->pos2[0]=SampledPosition2[splitInd].x;
		node->pos2[1]=SampledPosition2[splitInd].y;
		node->threshold=maxThreshold;
		for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
		{
			currentSampleInd=node->sampleInd->at(j);
		
			double currentDepth=1.0f/samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
			if (samples[currentSampleInd].gradientMap.at<uchar>((double)SampledPosition1[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
				(double)SampledPosition1[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)>
				(double)samples[currentSampleInd].gradientMap.at<uchar>(SampledPosition2[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
				(double)SampledPosition2[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)+maxThreshold)
		/*	if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
				SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
				samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
				SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+maxThreshold)*/
				//if(1)
			{
				SampledPositionL.push_back(currentSampleInd);
			}
			else
			{ 
				SampledPositionR.push_back(currentSampleInd);
			}
		
	
		}
		
		delete []proportion1;
		delete []proportion2;


	}
	leftNum=SampledPositionL.size();
	rightNum=SampledPositionR.size();

	//if no need to stop, split and call them recursively
	if (leftNum>min_sample_count)
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		split_color(leftChild);
	}
	else //treat it as a leaf
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		
		leftChild->num_all=leftChild->sampleInd->size();
		int i,j;
		if (leftChild->num_all==0)
		{
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					leftChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<leftChild->sampleInd->size();i++)
			{		
				if (leftChild->sampleInd->at(i)>=0)
				{
					leftChild->num_of_each_class[samples[leftChild->sampleInd->at(i)].label]++;
				}
				else
				{
					leftChild->num_of_each_class[addedSamples->at(-leftChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]/=(float)leftChild->num_all;
			}
		}
	

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<leftChild->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;

		
	}
	if (rightNum>min_sample_count)
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		split_color(rightChild);
	}
	else //treat it as a leaf
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		
		rightChild->num_all=rightChild->sampleInd->size();
		int i,j;
		if (rightChild->num_all==0)
		{
			//get the prob
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
			}
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					rightChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<rightChild->sampleInd->size();i++)
			{	
				if (rightChild->sampleInd->at(i)>=0)
				{
					rightChild->num_of_each_class[samples[rightChild->sampleInd->at(i)].label]++;
				}
				else
				{
					rightChild->num_of_each_class[addedSamples->at(-rightChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]/=(float)rightChild->num_all;
			}

		}

	/*	for (i=0;i<labelNum;i++)
		{
			cout<<rightChild->num_of_each_class[i]<<" ";
		}
		cout<<endl;*/
	}

}



void RandTree::split_WithDepth(Node *node)
{
	//srand(time(NULL));
	cout<<node->nLevel<<" "<<node->sampleInd->size()<<"   ";
	//if (node->nLevel==5&&node->sampleInd->size()==122)
	//{
	//	cout<<"see what happens...\n";
	//}

	//out_p<<node->nLevel<<" "<<node->sampleInd->size()<<" "<<123<<endl;
	//for (int i=0;i<node->sampleInd->size();i++)
	//{
	//	out_p<<node->sampleInd->at(i)<<" "<<samplesnode->sampleInd->at(i);
	//}
	//out_p<<endl;

	//int totalNum=0;
	int i;
	for (i=1;i<node->sampleInd->size();i++)
	{
		if (samples[node->sampleInd->at(i)].label!=samples[node->sampleInd->at(i-1)].label)
		{
			break;
		}
	}

	if (node->nLevel>=max_depth||i==node->sampleInd->size())//never exceed the max depth. Also, if all are with the same labels, it is a leaf node
	{
		node->num_all=node->sampleInd->size();
		int i,j;
		if (node->num_of_each_class==NULL)
		{
			node->num_of_each_class=new float[labelNum];
			for (i=0;i<labelNum;i++)
			{
				node->num_of_each_class[i]=0;
			}
		}
		//get the total num
		for (i=0;i<node->sampleInd->size();i++)
		{		
			node->num_of_each_class[samples[node->sampleInd->at(i)].label]++;
		}
		//get the prob
		for (i=0;i<labelNum;i++)
		{
			node->num_of_each_class[i]/=(float)node->num_all;
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<node->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;


		return;
	}
	//vector<int> l_sampleList;	//current filtered sample in left child
	//vector<int> r_sampleList;	//current filtered sample in right child
	//test the accruacy, if enough, return
	

	
	vector<CvPoint> SampledPosition1,SampledPosition2;
	int curSampleNum;
	if (node->nLevel==0)//root
	{
		curSampleNum=rootSampleNum;
	}
	else
	{
		curSampleNum=100*node->nLevel;//100d
	}
	//cout<<"sampling data on level "<<node->nLevel<<endl;
	
	//cout<<"sampling done!\n";

	int splitInd;
	int currentSampleInd;
	vector<int> SampledPositionL,SampledPositionR; //the samples goes to left and right
	int leftNum,rightNum;
	{
		//cout<<"finding optimal split test\n";
		int i,j,l;
		double threshold;
		double * proportion1;double *proportion2;
		//cout<<labelNum<<endl;
		proportion1=new double [labelNum];
		proportion2=new double [labelNum];
		double maxGain=-100000000000000;
		int maxGainInd=-1;
		int maxThreshold=-1;
		double gain;
		
		
		while(maxGainInd==-1) //just to make sure there will be a test that both left and right are non-empty
		{
			sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d

			////binary test
			//for (i=0;i<curSampleNum;i++)
			//{
			//	SampledPositionL.clear();
			//	SampledPositionR.clear();
			//	splitInd=i;
			//	for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
			//	{
			//		currentSampleInd=node->sampleInd->at(j);
			//		//cout<<i<<endl;
			//		if (samples[currentSampleInd].imgMat.at<uchar>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
			//			SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
			//			samples[currentSampleInd].imgMat.at<uchar>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
			//			SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x))
			//			//if(1)
			//		{
			//			SampledPositionL.push_back(currentSampleInd);
			//		}
			//		else
			//		{ 
			//			SampledPositionR.push_back(currentSampleInd);
			//		}
			//	}
			//	if (SampledPositionL.size()==0||SampledPositionR.size()==0)
			//	{
			//		continue;
			//	}
			//	//caculate the entrop 
			//	for (j=0;j<labelNum;j++)
			//	{
			//		proportion1[j]=proportion2[j]=0;
			//	}
			//	for (j=0;j<SampledPositionL.size();j++)
			//	{
			//		proportion1[samples[SampledPositionL.at(j)].label]++;
			//	}
			//	for (j=0;j<SampledPositionR.size();j++)
			//	{
			//		proportion2[samples[SampledPositionR.at(j)].label]++;
			//	}

			//	//..get the probability
			//	if (SampledPositionL.size()>0)
			//	{
			//		for (j=0;j<labelNum;j++)
			//		{
			//			proportion1[j]/=(double)SampledPositionL.size();
			//		}
			//	}
			//	if (SampledPositionR.size()>0)
			//	{
			//		for (j=0;j<labelNum;j++)
			//		{
			//			proportion2[j]/=(double)SampledPositionR.size();
			//		}
			//	}

			//	//..get the entropy
			//	double entropy[2];
			//	entropy[0]=entropy[1]=0;
			//	for (j=0;j<labelNum;j++)
			//	{
			//		if (proportion1[j]!=0)
			//		{
			//			entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
			//		}
			//		if (proportion2[j]!=0)
			//		{
			//			entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
			//		}		
			//	}
	
			//	//get the gain
			//	gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
			//	if (gain>maxGain)
			//	{
			//		maxGain=gain;
			//		maxGainInd=i;
			//	}
			////	break;
			//	//if (node->nLevel==5&&node->sampleInd->size()==122)
			///*	{
			//		cout<<gain<<" "<<maxGain<<" "<<maxGainInd<<endl;
			//	}*/
			//}

			//Threshold test
			for (i=0;i<curSampleNum;i++)
			{
				splitInd=i;
				//for (threshold=0;threshold<254;threshold+=10)
				for(l=0;l<6;l++)
				{
					//threshold=(double)RandInt_cABc(0,256)/256.0f;//[0,1]
					threshold=(double)RandInt_cABc(0,60)/750.0f;//divided by the reference depth
					//threshold=(double)RandInt_cABc(0,256*4)/256.0f-2;//[-2,2]
					if (l==0)
					{
						threshold=0;
					}
					else 
						continue;

				/*	if (trainStyle==0)
					{
						threshold=0;
						if (l>0)
						{
							break;
						}
					}*/

				////////////no threshold//////////////////
	/*			for(l=0;l<1;l++)
				{
					threshold=0;*/
					SampledPositionL.clear();
					SampledPositionR.clear();
					for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
					{
						currentSampleInd=node->sampleInd->at(j);


						if(isevaluating)
							cout<<SampledPosition1[splitInd].y<<" "<<samples[currentSampleInd].pos.y<<" "<<SampledPosition1[splitInd].x<<" "<<
							samples[currentSampleInd].pos.x<<" "<<SampledPosition2[splitInd].y<<" "<<SampledPosition2[splitInd].x<<
							" "<<samples[currentSampleInd].imgMat.rows<<" "<<samples[currentSampleInd].imgMat.cols<<endl;

						double currentDepth=samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
						//cout<<currentDepth<<endl;
						if (samples[currentSampleInd].imgMat.at<float>((double)SampledPosition1[splitInd].y/currentDepth+samples[currentSampleInd].pos.y,
							(double)SampledPosition1[splitInd].x/currentDepth+samples[currentSampleInd].pos.x)>
							samples[currentSampleInd].imgMat.at<float>((double)SampledPosition2[splitInd].y/currentDepth+samples[currentSampleInd].pos.y,
							(double)SampledPosition2[splitInd].x/currentDepth+samples[currentSampleInd].pos.x)+threshold)
							//if(1)
						{
							SampledPositionL.push_back(currentSampleInd);
						}
						else
						{ 
							SampledPositionR.push_back(currentSampleInd);
						}
					
						//cout<<currentDepth<<" "<<SampledPosition1[splitInd].y<<" "<<SampledPosition1[splitInd].x<<endl;
				
					}
					if (SampledPositionL.size()==0||SampledPositionR.size()==0)
					{
						continue;
					}
					//caculate the entrop 
					for (j=0;j<labelNum;j++)
					{
						proportion1[j]=proportion2[j]=0;
					}
					for (j=0;j<SampledPositionL.size();j++)
					{
						if (SampledPositionL.at(j)>=0)
						{
							proportion1[samples[SampledPositionL.at(j)].label]++;
						}
						else
						{
							proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
						}
						
					}
					for (j=0;j<SampledPositionR.size();j++)
					{
						if (SampledPositionR.at(j)>=0)
						{
							proportion2[samples[SampledPositionR.at(j)].label]++;
						}
						else
						{
							proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
						}
						
					}

					//..get the probability
					if (SampledPositionL.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]/=(double)SampledPositionL.size();
						}
					}
					if (SampledPositionR.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion2[j]/=(double)SampledPositionR.size();
						}
					}

					//..get the entropy
					double entropy[2];
					entropy[0]=entropy[1]=0;
					for (j=0;j<labelNum;j++)
					{
						if (proportion1[j]!=0)
						{
							entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
						}
						if (proportion2[j]!=0)
						{
							entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
						}		
					}

					//get the gain
					gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
					//cout<<threshold<<" "<<gain<<endl;
					if (gain>maxGain)
					{
						maxGain=gain;
						maxGainInd=i;
						maxThreshold=threshold;
					}
				}
			}

			//RGB test



		}
		
		
		//finally get the left and right sample
		SampledPositionL.clear();
		SampledPositionR.clear();
		splitInd=maxGainInd;
		node->pos1[0]=SampledPosition1[splitInd].x;
		node->pos1[1]=SampledPosition1[splitInd].y;
		node->pos2[0]=SampledPosition2[splitInd].x;
		node->pos2[1]=SampledPosition2[splitInd].y;
		node->threshold=maxThreshold;
		for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
		{
			currentSampleInd=node->sampleInd->at(j);
			//cout<<i<<endl;

			double currentDepth=samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
			if (samples[currentSampleInd].imgMat.at<float>((double)SampledPosition1[splitInd].y/currentDepth+samples[currentSampleInd].pos.y,
				(double)SampledPosition1[splitInd].x/currentDepth+samples[currentSampleInd].pos.x)>
				samples[currentSampleInd].imgMat.at<float>((double)SampledPosition2[splitInd].y/currentDepth+samples[currentSampleInd].pos.y,
				(double)SampledPosition2[splitInd].x/currentDepth+samples[currentSampleInd].pos.x)+maxThreshold)
		/*	if (samples[currentSampleInd].imgMat.at<float>(SampledPosition1[splitInd].y+samples[currentSampleInd].pos.y,
				SampledPosition1[splitInd].x+samples[currentSampleInd].pos.x)>
				samples[currentSampleInd].imgMat.at<float>(SampledPosition2[splitInd].y+samples[currentSampleInd].pos.y,
				SampledPosition2[splitInd].x+samples[currentSampleInd].pos.x)+maxThreshold)*/
				//if(1)
			{
				SampledPositionL.push_back(currentSampleInd);
			}
			else
			{ 
				SampledPositionR.push_back(currentSampleInd);
			}
			
	
		}
		
		delete []proportion1;
		delete []proportion2;


	}
	leftNum=SampledPositionL.size();
	rightNum=SampledPositionR.size();

	//if no need to stop, split and call them recursively
	if (leftNum>min_sample_count)
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		split_WithDepth(leftChild);
	}
	else //treat it as a leaf
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		
		leftChild->num_all=leftChild->sampleInd->size();
		int i,j;
		if (leftChild->num_all==0)
		{
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					leftChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<leftChild->sampleInd->size();i++)
			{		
				if (leftChild->sampleInd->at(i)>=0)
				{
					leftChild->num_of_each_class[samples[leftChild->sampleInd->at(i)].label]++;
				}
				else
				{
					leftChild->num_of_each_class[addedSamples->at(-leftChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]/=(float)leftChild->num_all;
			}
		}
	

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<leftChild->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;

		
	}
	if (rightNum>min_sample_count)
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		split_WithDepth(rightChild);
	}
	else //treat it as a leaf
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		
		rightChild->num_all=rightChild->sampleInd->size();
		int i,j;
		if (rightChild->num_all==0)
		{
			//get the prob
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
			}
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					rightChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<rightChild->sampleInd->size();i++)
			{	
				if (rightChild->sampleInd->at(i)>=0)
				{
					rightChild->num_of_each_class[samples[rightChild->sampleInd->at(i)].label]++;
				}
				else
				{
					rightChild->num_of_each_class[addedSamples->at(-rightChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]/=(float)rightChild->num_all;
			}

		}

	/*	for (i=0;i<labelNum;i++)
		{
			cout<<rightChild->num_of_each_class[i]<<" ";
		}
		cout<<endl;*/
	}

}

void RandTree::split_mixture(Node *node)
{
	cout<<node->nLevel<<" "<<node->sampleInd->size()<<"   ";
	//if (node->nLevel==5&&node->sampleInd->size()==122)
	//{
	//	cout<<"see what happens...\n";
	//}

	//out_p<<node->nLevel<<" "<<node->sampleInd->size()<<" "<<123<<endl;
	//for (int i=0;i<node->sampleInd->size();i++)
	//{
	//	out_p<<node->sampleInd->at(i)<<" "<<samplesnode->sampleInd->at(i);
	//}
	//out_p<<endl;

	if (node->nLevel>=max_depth)//never exceed the max depth
	{
		node->num_all=node->sampleInd->size();
		int i,j;
		if (node->num_of_each_class==NULL)
		{
			node->num_of_each_class=new float[labelNum];
			for (i=0;i<labelNum;i++)
			{
				node->num_of_each_class[i]=0;
			}
		}
		//get the total num
		for (i=0;i<node->sampleInd->size();i++)
		{		
			node->num_of_each_class[samples[node->sampleInd->at(i)].label]++;
		}
		//get the prob
		for (i=0;i<labelNum;i++)
		{
			node->num_of_each_class[i]/=(float)node->num_all;
		}

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<node->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;


		return;
	}
	//vector<int> l_sampleList;	//current filtered sample in left child
	//vector<int> r_sampleList;	//current filtered sample in right child
	//test the accruacy, if enough, return
	

	
	vector<CvPoint> SampledPosition1,SampledPosition2;
	int curSampleNum;
	if (node->nLevel==0)//root
	{
		curSampleNum=rootSampleNum;
	}
	else
	{
		curSampleNum=100*node->nLevel;//100d
	}
	//cout<<"sampling data on level "<<node->nLevel<<endl;
	
	//cout<<"sampling done!\n";

	int splitInd;
	int currentSampleInd;
	vector<int> SampledPositionL,SampledPositionR; //the samples goes to left and right
	int leftNum,rightNum;
	{
		//cout<<"finding optimal split test\n";
		int i,j,threshold,l;
		double * proportion1;double *proportion2;
		proportion1=new double [labelNum];
		proportion2=new double [labelNum];
		double maxGain=-100000000000000;
		int maxGainInd=-1;
		int maxThreshold=-1;
		double gain;
		
		
		while(maxGainInd==-1) //just to make sure there will be a test that both left and right are non-empty
		{
			sampleWindowIndex(curSampleNum,SampledPosition1,SampledPosition2);//100d

			//Threshold test
			for (i=0;i<curSampleNum;i++)
			{
				splitInd=i;
				
				//depth first
				for(l=0;l<5;l++)
				{
					threshold=RandInt_cABc(0,254);

					if (trainStyle==1||trainStyle==2)
					{
						threshold=0;
						if (l>0)
						{
							break;
						}
					}

					SampledPositionL.clear();
					SampledPositionR.clear();
					for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
					{
						currentSampleInd=node->sampleInd->at(j);

						double currentDepth=1.0f/samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
						if (samples[currentSampleInd].imgMat.at<float>((double)SampledPosition1[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
							(double)SampledPosition1[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)>
							samples[currentSampleInd].imgMat.at<float>((double)SampledPosition2[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
							(double)SampledPosition2[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)+threshold)
							//if(1)
						{
							SampledPositionL.push_back(currentSampleInd);
						}
						else
						{ 
							SampledPositionR.push_back(currentSampleInd);
						}
					
					}
					if (SampledPositionL.size()==0||SampledPositionR.size()==0)
					{
						continue;
					}
					//caculate the entrop 
					for (j=0;j<labelNum;j++)
					{
						proportion1[j]=proportion2[j]=0;
					}
					for (j=0;j<SampledPositionL.size();j++)
					{
						if (SampledPositionL.at(j)>=0)
						{
							proportion1[samples[SampledPositionL.at(j)].label]++;
						}
						else
						{
							proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
						}
						
					}
					for (j=0;j<SampledPositionR.size();j++)
					{
						if (SampledPositionR.at(j)>=0)
						{
							proportion2[samples[SampledPositionR.at(j)].label]++;
						}
						else
						{
							proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
						}
						
					}

					//..get the probability
					if (SampledPositionL.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]/=(double)SampledPositionL.size();
						}
					}
					if (SampledPositionR.size()>0)
					{
						for (j=0;j<labelNum;j++)
						{
							proportion2[j]/=(double)SampledPositionR.size();
						}
					}

					//..get the entropy
					double entropy[2];
					entropy[0]=entropy[1]=0;
					for (j=0;j<labelNum;j++)
					{
						if (proportion1[j]!=0)
						{
							entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
						}
						if (proportion2[j]!=0)
						{
							entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
						}		
					}

					//get the gain
					gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
					if (gain>maxGain)
					{
						maxGain=gain;
						maxGainInd=i;
						maxThreshold=threshold;
					}
				}
			}

			//color test
			if(trainStyle==2)
			{
				for (i=0;i<curSampleNum;i++)
				{
					splitInd=i;
					//for (threshold=0;threshold<254;threshold+=10)
					//for(l=0;l<5;l++)
					{
						threshold=0;

						SampledPositionL.clear();
						SampledPositionR.clear();
						for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
						{
							currentSampleInd=node->sampleInd->at(j);
							
							double currentDepth=1.0f/samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
							if (samples[currentSampleInd].gradientMap.at<uchar>((double)SampledPosition1[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
								(double)SampledPosition1[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)>
								samples[currentSampleInd].gradientMap.at<uchar>((double)SampledPosition2[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
								(double)SampledPosition2[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)+threshold)
								//if(1)
							{
								SampledPositionL.push_back(currentSampleInd);
							}
							else
							{ 
								SampledPositionR.push_back(currentSampleInd);
							}

							//cout<<i<<endl;

						}
						if (SampledPositionL.size()==0||SampledPositionR.size()==0)
						{
							continue;
						}
						//caculate the entrop 
						for (j=0;j<labelNum;j++)
						{
							proportion1[j]=proportion2[j]=0;
						}
						for (j=0;j<SampledPositionL.size();j++)
						{
							if (SampledPositionL.at(j)>=0)
							{
								proportion1[samples[SampledPositionL.at(j)].label]++;
							}
							else
							{
								proportion1[addedSamples->at(-SampledPositionL.at(j)).label]++;
							}

						}
						for (j=0;j<SampledPositionR.size();j++)
						{
							if (SampledPositionR.at(j)>=0)
							{
								proportion2[samples[SampledPositionR.at(j)].label]++;
							}
							else
							{
								proportion2[addedSamples->at(-SampledPositionR.at(j)).label]++;
							}

						}

						//..get the probability
						if (SampledPositionL.size()>0)
						{
							for (j=0;j<labelNum;j++)
							{
								proportion1[j]/=(double)SampledPositionL.size();
							}
						}
						if (SampledPositionR.size()>0)
						{
							for (j=0;j<labelNum;j++)
							{
								proportion2[j]/=(double)SampledPositionR.size();
							}
						}

						//..get the entropy
						double entropy[2];
						entropy[0]=entropy[1]=0;
						for (j=0;j<labelNum;j++)
						{
							if (proportion1[j]!=0)
							{
								entropy[0]+=(-proportion1[j]*log(proportion1[j])/log(2.0));
							}
							if (proportion2[j]!=0)
							{
								entropy[1]+=(-proportion2[j]*log(proportion2[j])/log(2.0));
							}		
						}

						//get the gain
						gain=-(double)SampledPositionL.size()/(double)node->sampleInd->size()*entropy[0]-(double)SampledPositionR.size()/(double)node->sampleInd->size()*entropy[1];
						if (gain>maxGain)
						{
							maxGain=gain;
							maxGainInd=i;
							maxThreshold=-1;
						}
					}
				}

			}
			


		}
	/*	if (maxThreshold<0)
		{
			cout<<maxThreshold<<" ";
		}*/
		
		
		//finally get the left and right sample
		SampledPositionL.clear();
		SampledPositionR.clear();
		splitInd=maxGainInd;
		node->pos1[0]=SampledPosition1[splitInd].x;
		node->pos1[1]=SampledPosition1[splitInd].y;
		node->pos2[0]=SampledPosition2[splitInd].x;
		node->pos2[1]=SampledPosition2[splitInd].y;
		node->threshold=maxThreshold;
		for (j=0;j<node->sampleInd->size();j++) //if larger, go to left. Otherwise right
		{
			currentSampleInd=node->sampleInd->at(j);
			double currentDepth=1.0f/samples[currentSampleInd].imgMat.at<float>(samples[currentSampleInd].pos.y,samples[currentSampleInd].pos.x);
			//cout<<i<<endl;
			if (maxThreshold<0) //color
			{
				if (samples[currentSampleInd].gradientMap.at<uchar>((double)SampledPosition1[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
					(double)SampledPosition1[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)>
					samples[currentSampleInd].gradientMap.at<uchar>((double)SampledPosition2[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
					(double)SampledPosition2[splitInd].x*currentDepth+samples[currentSampleInd].pos.x))
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
			else	//depth
			{
				if (samples[currentSampleInd].imgMat.at<float>((double)SampledPosition1[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
					(double)SampledPosition1[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)>
					samples[currentSampleInd].imgMat.at<float>((double)SampledPosition2[splitInd].y*currentDepth+samples[currentSampleInd].pos.y,
					(double)SampledPosition2[splitInd].x*currentDepth+samples[currentSampleInd].pos.x)+maxThreshold)
					//if(1)
				{
					SampledPositionL.push_back(currentSampleInd);
				}
				else
				{ 
					SampledPositionR.push_back(currentSampleInd);
				}
			}
	
		}
		
		delete []proportion1;
		delete []proportion2;


	}
	leftNum=SampledPositionL.size();
	rightNum=SampledPositionR.size();

	//if no need to stop, split and call them recursively
	if (leftNum>min_sample_count)
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		split_mixture(leftChild);
	}
	else //treat it as a leaf
	{
		Node *leftChild=new Node;
		leftChild->nLevel=node->nLevel+1;
		node->l_child=leftChild;
		//copy(SampledPositionL.begin(),SampledPositionL.end(),leftChild->sampleInd);
		for (int i=0;i<SampledPositionL.size();i++)
		{
			leftChild->sampleInd->push_back(SampledPositionL[i]);
		}
		
		leftChild->num_all=leftChild->sampleInd->size();
		int i,j;
		if (leftChild->num_all==0)
		{
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (leftChild->num_of_each_class==NULL)
			{
				leftChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					leftChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<leftChild->sampleInd->size();i++)
			{		
				if (leftChild->sampleInd->at(i)>=0)
				{
					leftChild->num_of_each_class[samples[leftChild->sampleInd->at(i)].label]++;
				}
				else
				{
					leftChild->num_of_each_class[addedSamples->at(-leftChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				leftChild->num_of_each_class[i]/=(float)leftChild->num_all;
			}
		}
	

		//for (i=0;i<labelNum;i++)
		//{
		//	cout<<leftChild->num_of_each_class[i]<<" ";
		//}
		//cout<<endl;

		
	}
	if (rightNum>min_sample_count)
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		split_mixture(rightChild);
	}
	else //treat it as a leaf
	{
		Node *rightChild=new Node;
		rightChild->nLevel=node->nLevel+1;
		node->r_child=rightChild;
		//copy(SampledPositionR.begin(),SampledPositionR.end(),rightChild->sampleInd);
		for (int i=0;i<SampledPositionR.size();i++)
		{
			rightChild->sampleInd->push_back(SampledPositionR[i]);
		}
		
		rightChild->num_all=rightChild->sampleInd->size();
		int i,j;
		if (rightChild->num_all==0)
		{
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]=0;
			}
		}
		else
		{
			
			if (rightChild->num_of_each_class==NULL)
			{
				rightChild->num_of_each_class=new float[labelNum];
				for (i=0;i<labelNum;i++)
				{
					rightChild->num_of_each_class[i]=0;
				}
			}
			//get the total num
			for (i=0;i<rightChild->sampleInd->size();i++)
			{	
				if (rightChild->sampleInd->at(i)>=0)
				{
					rightChild->num_of_each_class[samples[rightChild->sampleInd->at(i)].label]++;
				}
				else
				{
					rightChild->num_of_each_class[addedSamples->at(-rightChild->sampleInd->at(i)).label]++;
				}
				
			}
			//get the prob
			for (i=0;i<labelNum;i++)
			{
				rightChild->num_of_each_class[i]/=(float)rightChild->num_all;
			}

		}

	/*	for (i=0;i<labelNum;i++)
		{
			cout<<rightChild->num_of_each_class[i]<<" ";
		}
		cout<<endl;*/
	}

}

int RandTree::getLabel(Node *root,Mat mat,CvPoint &pos)
{
	Node *current=root;

	while(current!=NULL)
	{
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			return current->label;
		}

	/*	Mat tmp=mat.clone();
		Point x,y;
		x.y=current->pos1[1]+pos.y;
		x.x=current->pos1[0]+pos.x;
		y.y=current->pos2[1]+pos.y;
		y.x=current->pos2[0]+pos.x;
		circle(tmp,x,3,255);
		circle(tmp,y,3,255);
		namedWindow("1");
		imshow("1",tmp);
		waitKey();*/

		if (mat.at<uchar>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
			mat.at<uchar>(current->pos2[1]+pos.y,current->pos2[0]+pos.x))
		{
			
			if (current->l_child==NULL)
			{
				return current->label;
			}
			current=current->l_child;
		}
		else
		{
			
			if (current->r_child==NULL)//leaf
			{
				return current->label;
			}
			current=current->r_child;
		}
	}
}

void RandTree::getProb(Node *root,Mat mat,CvPoint &pos,LeafNode &leafnode)
{
	Node *current=root;

	while(1)
	{
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			//if (current->label>=0)
			{
				leafnode.leaf_node=current;
				leafnode.label=current->label;
			}
			break;
		}

		if (trainStyle==0)
		{
			double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
			if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
				mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					break;
				}
				current=current->l_child;
			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					break;
				}
				current=current->r_child;
			}
		}
		else
		{
			if (mat.at<float>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
				mat.at<float>(current->pos2[1]+pos.y,current->pos2[0]+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					break;
				}
				current=current->l_child;
			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					break;
				}
				current=current->r_child;
			}
		}
	}
}

//mat: depth image
//gradient: color image
void RandTree::getProb(Node *root,Mat mat,Mat gradient,CvPoint &pos,LeafNode &leafnode)
{
	Node *current=root;

//	cout<<"begin of the tree\n";

	while(1)
	{
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			//if (current->label>=0)
			{
				leafnode.leaf_node=current;
				leafnode.label=current->label;
			}
			break;
		}

		if (trainStyle==1)
		{
			double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
			if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
				gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					break;
				}
				current=current->l_child;
			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					break;
				}
				current=current->r_child;
			}
		}
		else if (trainStyle==2)
		{
			if (current->threshold>=0)	//depth
			{
			//	cout<<current->nLevel<<" depth\n";
				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
				if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
					mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)
				//if (mat.at<uchar>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
				//	mat.at<uchar>(current->pos2[1]+pos.y,current->pos2[0]+pos.x)+current->threshold)
				{

					if (current->l_child==NULL)
					{
						break;
					}
					current=current->l_child;
				}
				else
				{

					if (current->r_child==NULL)//leaf
					{
						break;
					}
					current=current->r_child;
				}
			}
			else	//color
			{
				//cout<<current->nLevel<<" color\n";
				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
				if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
					gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+(-current->threshold-1))
				{

					if (current->l_child==NULL)
					{
						break;
					}
					current=current->l_child;
				}
				else
				{

					if (current->r_child==NULL)//leaf
					{
						break;
					}
					current=current->r_child;
				}
			}
		}
	}
}

void RandTree::getSample(string name)
{
	labelNum=sizeof(interestInd)/sizeof(int)+1;
	int id=name.find_last_of('\\');
	path=name.substr(0,id+1);
		//get dirname
	int localWindowSize=1;
	int startInd=(localWindowSize+1)/2-localWindowSize;
	int endInd=localWindowSize-(localWindowSize+1)/2;
	cout<<"loading data\n";
	int ind=name.find_last_of('\\');
	if (ind<0)
	{
		ind=name.find_last_of('/');
	}
	string dirName=name.substr(0,ind+1);

	fstream in(name.c_str(),ios::in);
	in>>shapeNum;
	//shapeNum=5;
	shape=new Shape*[shapeNum];
	char cname[500];
	//	in.getline(cname,499);
	string cfilename;
	for (int i=0;i<shapeNum;i++)
	{
		//	name="";
		if (i%1000==0)
		{
			cout<<i<<" "<<shapeNum<<endl;
		}
		shape[i]=new Shape;
		in.getline(cname,499);
		cfilename=cname;
		if(cfilename.length()<3)
		{
			i--;
			continue;
		}
		//shape[i]->getVertex(cfilename);
		shape[i]->getDpeth(cfilename);
	}
	
	//scale,affine warping 
	//generate the samples
	sampledImage=new vector<sampleImage>*[shapeNum];
	Mat tmp,dst;
	Mat currentImage;
	double width;
	Mat warp_mat(2,3,CV_32F);
	Point current_center;
	int totalAngles=8;
	double **current_pts;
	current_pts=new double *[shape[0]->ptsNum];
	for (int i=0;i<shape[0]->ptsNum;i++)
	{
		current_pts[i]=new double[2];
	}

	int i,j,k,l;

	int scaleNum=0;
	double scaleParameter=0.9;
	//int interestInd[]={0,8, 16 ,24, 42, 48, 4 ,12, 20, 28, 45, 51};
	//6,7,11,12,3,4,5,8,10,13,18,19
	//int interestInd[]={5,6, 10,11,2,3,4,7,9,12,17,18};

	cout<<"sampling\n";
	for (i=0;i<shapeNum;i++)
	{
		cout<<i<<" "<<shapeNum<<endl;
		sampledImage[i]=new vector<sampleImage>;
		//currentImage=shape[i]->colorImg;
		sampledImage[i]->push_back(sampleImage(shape[i]->colorImg,shape[i]->pts,shape[i]->ptsNum));
	}


	cout<<"feeding\n";
	//feed into the samples
	sampleNum=shapeNum*sampledImage[0]->size()*(shape[0]->ptsNum);//each feature point is a sample
	cout<<sampleNum<<endl;
	//sampleNum=shapeNum;
	//samples=new Sample[sampleNum];
	samples.resize(sampleNum);

	int tcurSID=0;
	//int,l;
	//Mat currentImage;
	for (i=0;i<shapeNum;i++)
	{
		//for(int j=0;j<1;j++)
		//cout<<i<<endl;
		for (j=0;j<shape[i]->ptsNum;j++)
		{
			//do the sampling on data
			//doSampling();
			//negalact the outline
			/*if (j>27)
			{
				continue;
			}*/

	
		/*	Point p;

			for (int ss=0;ss<shape[i]->ptsNum;ss++)
			{
				p.x=sampledImage[i]->at(k).pts[ss][0];
				p.y=sampledImage[i]->at(k).pts[ss][1];
				circle(sampledImage[i]->at(k).img,p,3,255);
			}
			namedWindow("affine");
			imshow("affine",sampledImage[i]->at(k).img);
			waitKey();*/

			//cout<<sampledImage[i]->size()<<endl;
			for (k=0;k<sampledImage[i]->size();k++)
			{
				//do not use the points which will exceed its size
				if (sampledImage[i]->at(k).pts[j][0]<=(windowSize+1)/2+1||sampledImage[i]->at(k).pts[j][0]>=sampledImage[i]->at(k).img.cols-((windowSize+1)/2)-1||
					sampledImage[i]->at(k).pts[j][1]<=(windowSize+1)/2+1||sampledImage[i]->at(k).pts[j][1]>=sampledImage[i]->at(k).img.rows-((windowSize+1)/2)-1)
				{
					cout<<sampledImage[i]->at(k).pts[j][0]<<" "<<sampledImage[i]->at(k).pts[j][1]<<
						sampledImage[i]->at(k).img.cols<<" "<<sampledImage[i]->at(k).img.rows<<endl;
					continue;
				}
				
				if(trainStyle==0)
				{
					samples[tcurSID].imgMat=sampledImage[i]->at(k).img;
				}
				else if (trainStyle==1)
				{
					samples[tcurSID].imgMat=sampledImage[i]->at(k).gradientMap;
				}
				else if (trainStyle==2)
				{
					samples[tcurSID].imgMat=sampledImage[i]->at(k).img;
				}


				samples[tcurSID].gradientMap=sampledImage[i]->at(k).gradientMap;
					
			

				//0,8,16,24,42,48
				//0 4 8 12 16 20 24 28 42 45 48 51
				//0 8 16 24 42 48 4 12 20 28 45 51
				//int interestInd[]={5,6, 10,11,2,3,4,7,9,12,17,18};
				//int interestInd[]={2,3,10,11,22,23,0,1,8,9,24,27,18,19};
				if (labelNum>2)
				{
					int mm;
					for ( mm=0;mm<sizeof(interestInd)/sizeof(int);mm++)
					{
						if (j==interestInd[mm])
						{
							samples[tcurSID].label=mm;
							break;
						}
					}
					if (mm==sizeof(interestInd)/sizeof(int))
					{
						samples[tcurSID].label=sizeof(interestInd)/sizeof(int);
					}
				}
				else if(labelNum==2)
				{
					if (j==0)
					{
						samples[tcurSID].label=0;
					}
					else
					{
						samples[tcurSID].label=1;
					}
				}

				samples[tcurSID].pos=cvPoint(sampledImage[i]->at(k).pts[j][0],sampledImage[i]->at(k).pts[j][1]);

				tcurSID++;		

			}
					
		
			
		}
		
	}

	for (int i=tcurSID;i<sampleNum;i++)
	{
		samples.pop_back();
	}
	sampleNum=samples.size();
	cout<<"real sample num: "<<sampleNum<<endl;
	//samples.resize(tcurSID-1);
	//then, get the discriptors
}

void RandTree::getNameList(string name)
{
	fstream in(name.c_str(),ios::in);
	in>>TotalShapeNum;
	//TotalShapeNum=20;
	nameList=new string[TotalShapeNum];

	//fstream in(name.c_str(),ios::in);
	char tmpstr[500];
	in.getline(tmpstr,499);
	for (int i=0;i<TotalShapeNum;i++)
	{
		in.getline(tmpstr,499);
		
		nameList[i]=tmpstr;

		//cout<<nameList[i]<<endl;
	}
	in.close();
}

void RandTree::distroyAllData()
{
	for (int i=0;i<shapeNum;i++)
	{
		for (int j=0;j<sampledImage[i]->size();j++)
		{
			sampledImage[i]->at(j).distroy(shape[0]->ptsNum);
			
		}
		//sampledImage[i]->clear();
	/*	vector< sampleImage > vtTemp;
		sampledImage[i]->swap(vtTemp);*/
		std::vector<sampleImage>().swap(*sampledImage[i]);
	}

	for (int i=0;i<samples.size();i++)
	{
		samples[i].imgMat.release();
	}
	delete []sampledImage;
	shapeNum=0;
}

void RandTree::getSample(int num)
{
	vector<int> randInd;

	if (num==-1)
	{
		randInd.resize(TotalShapeNum);
		for (int i=0;i<TotalShapeNum;i++)
		{
			randInd[i]=i+1;
		}
		num=TotalShapeNum;
	}
	else if(num>0)
		RandSample(TotalShapeNum,num,randInd);
	else if (num==-2)
	{
		RandSample(10530,1000,randInd);

		vector<int> RealRandInd;
		RandSample(901,500,RealRandInd);

		for (int i=0;i<RealRandInd.size();i++)
		{
			randInd.push_back(RealRandInd[i]+10530);
		}
		num=1500;
	}


	labelNum=sizeof(interestInd)/sizeof(int)+1;
	//int id=name.find_last_of('\\');
	//path=name.substr(0,id+1);
		//get dirname
	int localWindowSize=1;
	int startInd=(localWindowSize+1)/2-localWindowSize;
	int endInd=localWindowSize-(localWindowSize+1)/2;
	cout<<"loading data\n";
	//int ind=name.find_last_of('\\');
	//if (ind<0)
	//{
	//	ind=name.find_last_of('/');
	//}
	//string dirName=name.substr(0,ind+1);

	shapeNum=num;
	shape=new Shape*[shapeNum];
	int cind;
	for (int i=0;i<shapeNum;i++)
	{
		cind=randInd[i]-1;
		cout<<i<<" "<<nameList[cind]<<endl;
		shape[i]=new Shape;
		shape[i]->getDpeth(nameList[cind],trainStyle);

		if (i==0)
		{
			int id=nameList[cind].find_last_of('\\');
			path=nameList[cind].substr(0,id+1);
			//get dirname
		}
	}

	//fstream in(name.c_str(),ios::in);
	//in>>shapeNum;
	////shapeNum=5;
	//shape=new Shape*[shapeNum];
	//char cname[500];
	////	in.getline(cname,499);
	//string cfilename;
	//for (int i=0;i<shapeNum;i++)
	//{
	//	//	name="";
	//	if (i%1000==0)
	//	{
	//		cout<<i<<" "<<shapeNum<<endl;
	//	}
	//	shape[i]=new Shape;
	//	in.getline(cname,499);
	//	cfilename=cname;
	//	if(cfilename.length()<3)
	//	{
	//		i--;
	//		continue;
	//	}
	//	//shape[i]->getVertex(cfilename);
	//	shape[i]->getDpeth(cfilename);

	//	if (i>num)
	//	{
	//		break;
	//	}
	//}
	
	//scale,affine warping 
	//generate the samples
	sampledImage=new vector<sampleImage>*[shapeNum];
	Mat tmp,dst;
	Mat currentImage;
	double width;
	Mat warp_mat(2,3,CV_32F);
	Point current_center;
	int totalAngles=8;
	double **current_pts;
	current_pts=new double *[shape[0]->ptsNum];
	for (int i=0;i<shape[0]->ptsNum;i++)
	{
		current_pts[i]=new double[2];
	}

	int i,j,k,l;

	int scaleNum=0;
	double scaleParameter=0.9;
	//int interestInd[]={0,8, 16 ,24, 42, 48, 4 ,12, 20, 28, 45, 51};
	//6,7,11,12,3,4,5,8,10,13,18,19
	//int interestInd[]={5,6, 10,11,2,3,4,7,9,12,17,18};

	cout<<"sampling\n";
	for (i=0;i<shapeNum;i++)
	{
		cout<<i<<" "<<shapeNum<<endl;
		sampledImage[i]=new vector<sampleImage>;

		//for (float j=0.9;j<1.2;j+=0.1)
		for (float j=0.8;j<1.3;j+=0.1)
		{
			//we do not care the scale any more with depth data
			if (j!=1)
			{
				continue;
			}

			if (trainStyle==0)
			{
				Mat currentImage=Mat::zeros((int)(j*shape[i]->colorImg.rows),(int)(j*shape[i]->colorImg.cols),CV_32FC1);
				resize(shape[i]->colorImg,currentImage,currentImage.size());
				for(int tt=0;tt<shape[i]->ptsNum;tt++)
				{
					current_pts[tt][0]=shape[i]->pts[tt][0]*j;
					current_pts[tt][1]=shape[i]->pts[tt][1]*j;
				}

				sampledImage[i]->push_back(sampleImage(currentImage,current_pts,shape[i]->ptsNum));
				currentImage.release();
			}
			else if (trainStyle==1||trainStyle==2)
			{
				Mat currentImage=Mat::zeros((int)(j*shape[i]->colorImg.rows),(int)(j*shape[i]->colorImg.cols),CV_32FC1);
				resize(shape[i]->colorImg,currentImage,currentImage.size());
				for(int tt=0;tt<shape[i]->ptsNum;tt++)
				{
					current_pts[tt][0]=shape[i]->pts[tt][0]*j;
					current_pts[tt][1]=shape[i]->pts[tt][1]*j;
				}

				Mat currentColorImg=Mat::zeros((int)(j*shape[i]->colorImg.rows),(int)(j*shape[i]->colorImg.cols),CV_32FC1);
				resize(cvarrToMat(shape[i]->hostImage),currentColorImg,currentColorImg.size());

				sampledImage[i]->push_back(sampleImage(currentImage,currentColorImg,current_pts,shape[i]->ptsNum));
				currentImage.release();
			}
		

			

		
		/*	namedWindow("1");
			imshow("1",currentImage);
			waitKey();*/
		}

		shape[i]->colorImg.release();

		if (trainStyle==1||trainStyle==2)
		{
			cvReleaseImage(&shape[i]->hostImage);
		}
		//
		//sampledImage[i]->push_back(sampleImage(shape[i]->colorImg,shape[i]->pts,shape[i]->ptsNum));
	}


	cout<<"feeding\n";
	//feed into the samples
	sampleNum=shapeNum*sampledImage[0]->size()*(shape[0]->ptsNum);//each feature point is a sample
	cout<<sampleNum<<endl;
	//sampleNum=shapeNum;
	//samples=new Sample[sampleNum];
	samples.resize(sampleNum);

	int tcurSID=0;
	//int,l;
	//Mat currentImage;
	for (i=0;i<shapeNum;i++)
	{
		//for(int j=0;j<1;j++)
		//cout<<i<<endl;
		for (j=0;j<shape[i]->ptsNum;j++)
		{
			//do the sampling on data
			//doSampling();
			//negalact the outline
			/*if (j>27)
			{
				continue;
			}*/

	
		/*	Point p;

			for (int ss=0;ss<shape[i]->ptsNum;ss++)
			{
				p.x=sampledImage[i]->at(k).pts[ss][0];
				p.y=sampledImage[i]->at(k).pts[ss][1];
				circle(sampledImage[i]->at(k).img,p,3,255);
			}
			namedWindow("affine");
			imshow("affine",sampledImage[i]->at(k).img);
			waitKey();*/

			//cout<<sampledImage[i]->size()<<endl;
			for (k=0;k<sampledImage[i]->size();k++)
			{
				//do not use the points which will exceed its size
				if (sampledImage[i]->at(k).pts[j][0]<=(windowSize+1)/2+1||sampledImage[i]->at(k).pts[j][0]>=sampledImage[i]->at(k).img.cols-((windowSize+1)/2)-1||
					sampledImage[i]->at(k).pts[j][1]<=(windowSize+1)/2+1||sampledImage[i]->at(k).pts[j][1]>=sampledImage[i]->at(k).img.rows-((windowSize+1)/2)-1)
				{
					cout<<sampledImage[i]->at(k).pts[j][0]<<" "<<sampledImage[i]->at(k).pts[j][1]<<
						sampledImage[i]->at(k).img.cols<<" "<<sampledImage[i]->at(k).img.rows<<endl;
					continue;
				}

				//do not use the 0-depth samples
				if (sampledImage[i]->at(k).img.at<float>(sampledImage[i]->at(k).pts[j][1],sampledImage[i]->at(k).pts[j][0])==0)
				{
					cout<<"bad depth value \n";
					continue;
				}
				
				if(trainStyle==0)
				{
					samples[tcurSID].imgMat=sampledImage[i]->at(k).img;
				}
		/*		else if (trainStyle==1)
				{
					samples[tcurSID].imgMat=sampledImage[i]->at(k).gradientMap;
				}*/
				else if (trainStyle==1||trainStyle==2)
				{
					samples[tcurSID].imgMat=sampledImage[i]->at(k).img;
				}


				samples[tcurSID].gradientMap=sampledImage[i]->at(k).gradientMap;
					
			

				//0,8,16,24,42,48
				//0 4 8 12 16 20 24 28 42 45 48 51
				//0 8 16 24 42 48 4 12 20 28 45 51
				//int interestInd[]={5,6, 10,11,2,3,4,7,9,12,17,18};
				//int interestInd[]={2,3,10,11,22,23,0,1,8,9,24,27,18,19};
				if (labelNum>2)
				{
					int mm;
					for ( mm=0;mm<sizeof(interestInd)/sizeof(int);mm++)
					{
						if (j==interestInd[mm])
						{
							samples[tcurSID].label=mm;
							break;
						}
					}
					if (mm==sizeof(interestInd)/sizeof(int))
					{
						samples[tcurSID].label=sizeof(interestInd)/sizeof(int);
					}
				}
				else if(labelNum==2)
				{
					if (j==0)
					{
						samples[tcurSID].label=0;
					}
					else
					{
						samples[tcurSID].label=1;
					}
				}

				samples[tcurSID].pos=cvPoint(sampledImage[i]->at(k).pts[j][0],sampledImage[i]->at(k).pts[j][1]);

				tcurSID++;		

			}
					
		
			
		}
		
	}

	for (int i=tcurSID;i<sampleNum;i++)
	{
		samples.pop_back();
	}
	sampleNum=samples.size();
	cout<<"real sample num: "<<sampleNum<<endl;
	//samples.resize(tcurSID-1);
	//then, get the discriptors
}

void RandTree::predict_rt(IplImage *img)
{
	//rtrees.load("F:\\imgdata\\Video 2 Train\\Trained RTtree.txt");
	//cout<<"Train Error: "<<rtrees.get_train_error()<<endl;
	Mat AllLabel(img->height,img->width,CV_32F);
	int i,j,k,l;
	Mat m_img=cvarrToMat(img);
	float *feature;
	ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
	if (1)
	{
	
		//CvSeq* objectKeypoints=0, *objectDescriptors = 0;
		//CvMemStorage* storage4Pts = cvCreateMemStorage(0);
		//CvMemStorage* storage = cvCreateMemStorage(0);
		//objectKeypoints=cvCreateSeq( 0, /* sequence of points */
		//	sizeof(CvSeq), /* header size - no extra fields */
		//	sizeof(CvSURFPoint), /* element size */
		//	storage4Pts /* the container storage */ );
		//CvSURFParams params = cvSURFParams(500, 1);
		//vector<CvSURFPoint> pts;
		//int i,j;
		//for (i=5;i<img->width-5;i++)
		//{
		//	for(j=5;j<img->height-5;j++)
		//	{
		//		CvSURFPoint pt;
		//		pt.pt.x=i;
		//		pt.pt.y=j;
		//		pt.size=10;
		//		cvSeqPush( objectKeypoints, &pt);
		//		pts.push_back(pt);
		//	}
		//}
		//cout<<"extracting features\n";
		//cvExtractSURF( img, 0, &objectKeypoints, &objectDescriptors, storage, params,1 );



		////predict for every point
		////assign value to mat file
		//CvSeqReader obj_reader;
		//int length=(int)(objectDescriptors->elem_size/sizeof(float));
		//CvMat *data=cvCreateMat(objectDescriptors->total,length,CV_32F);
		//Mat m_data=cvarrToMat(data);
		//float* obj_ptr;
		//obj_ptr = m_data.ptr<float>(0);
		//cvStartReadSeq( objectDescriptors, &obj_reader );
		//for(j = 0; j < objectDescriptors->total; j++ )
		//{
		//	const float* descriptor = (const float*)obj_reader.ptr;
		//	CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
		//	memcpy(obj_ptr, descriptor, length*sizeof(float));
		//	obj_ptr += length;
		//}
		//get the feature set
		int size=windowSize;
		cout<<"predicting labels\n";
		CvPoint center;
		ofstream out("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::out);

		LabelResult tmp;
		for (i=size;i<m_img.cols-size;i++)
		{
			//	cout<<i<<endl;
			for (j=size;j<m_img.rows-size;j++)
			{
				center.x=i;
				center.y=j;
				pridict_prob(m_img,center,tmp);
				//Mat cdata(1,sampleDim,CV_32F,feature);
				out<<i<<" "<<j<<" "<<tmp.label<<endl;
				//out<<i<<" "<<j<<" "<<tmp.label<<" "<<tmp.prob_all[0]<<endl;
			//	delete []feature;
			}
		}
		out.close();

		//for(i=0;i<objectDescriptors->total;i++)
		//{
		//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
		//}
		//out.close();
	}
	//else
	{
		//display
		ifstream in("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
		AllLabel=0;
		int y,x;
		int label;
		int i;
		while(in)
		{
			in>>y>>x>>label;
			AllLabel.at<float>(x,y)=label;
		}

		//show all in one image
		//Mat currentLabel(img->height,img->width,CV_32F);
		////show directly
		//Mat ImageWithLabel=cvarrToMat(img).clone();
		//int currentID=2;
		//	for (currentID=0;currentID<6;currentID++)
		//	{
		//		for (i=0;i<currentLabel.rows;i++)
		//		{
		//			for (j=0;j<currentLabel.cols;j++)
		//			{
		//				currentLabel.at<float>(i,j)=(AllLabel.at<float>(i,j)==currentID);
		//			}
		//		}

		//		
		//		Point c;

		//		for (i=0;i<currentLabel.rows;i++)
		//		{
		//			for (j=0;j<currentLabel.cols;j++)
		//			{
		//				if(AllLabel.at<float>(i,j)==currentID)
		//				{
		//					c.x=j;
		//					c.y=i;
		//					circle(ImageWithLabel,c,2,255);
		//				}
		//			}
		//		}
		//	}
		//	char name[50];
		////	sprintf(name, "feature ", ID+1);
		//	namedWindow("feature");
		//	imshow("feature",ImageWithLabel);
		//	waitKey();


		//show single image

		Mat currentLabel(img->height,img->width,CV_32F);
		int currentID;
		for (currentID=0;currentID<labelNum;currentID++)
		{
			for (i=0;i<currentLabel.rows;i++)
			{
				for (j=0;j<currentLabel.cols;j++)
				{
					currentLabel.at<float>(i,j)=(AllLabel.at<float>(i,j)==currentID);
				}
			}

			//	//show directly
			//	Mat ImageWithLabel=cvarrToMat(img).clone();
			//	Point c;

			//	for (i=0;i<currentLabel.rows;i++)
			//	{
			//		for (j=0;j<currentLabel.cols;j++)
			//		{
			//			if(AllLabel.at<float>(i,j)==currentID)
			//			{
			//				c.x=j;
			//				c.y=i;
			//				circle(ImageWithLabel,c,2,255);
			//			}
			//		}
			//	}
			//	char name[50];
			//	sprintf(name, "feature %d", currentID+1);
			//	namedWindow(name);
			//	imshow(name,ImageWithLabel);
			//	//////////////////////////////////////////////////


			//local window search
			int localWindowSize=10;
			double threshold=0;
			int center[2];
			int sum;
			vector<CvPoint> filterdLabel;
			for (i=localWindowSize;i<img->width-localWindowSize;i++)
			{
				for (j=localWindowSize;j<img->height-localWindowSize;j++)
				{
					center[0]=j;center[1]=i;
					sum=0;
					for (k=center[0]-localWindowSize/2;k<center[0]+localWindowSize/2;k++)
					{
						for (l=center[1]-localWindowSize/2;l<center[1]+localWindowSize/2;l++)
						{
							if (currentLabel.at<float>(k,l)==1)
							{
								sum++;
							}
						}
					}
					if (sum>localWindowSize*localWindowSize*threshold)
					{
						filterdLabel.push_back(cvPoint(i,j));
					}
				}
			}
			cout<<"remained pts number: "<<filterdLabel.size()<<endl;
			Mat ImageWithLabel=cvarrToMat(img).clone();
			Point c;
			for (i=0;i<filterdLabel.size();i++)
			{
				c.x=filterdLabel.at(i).x;
				c.y=filterdLabel.at(i).y;
				circle(ImageWithLabel,c,1,255);
			}
			char name[50];
			sprintf(name, "feature %d", currentID+1);
			namedWindow(name);
			imshow(name,ImageWithLabel);
			waitKey();
		}
	}
}

void RandTree::predict_imgList_GPU_depth(string listName)
{

	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{

	
		in.getline(name,500,'\n');

		string saveName=name;
		saveName=saveName.substr(0,ImgName.length()-4);
		string prob_name=saveName;
		saveName+=".txt";
	/*	if (ll>_imgNum-8)
		{
			continue;
		}*/
		
		Mat m_img=imread(name,0);

		Mat colorImg=imread(name);

		Mat depthImg=Mat::ones(colorImg.rows,colorImg.cols,CV_32FC1);


		//namedWindow("1");
		//imshow("1",depthImg);
		//waitKey();

		depthImg*=0;

		///////////////////////read in the depth data//////////////////////////////

		//depth
		string depthFileName=name;
		depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		depthFileName+="_depth.txt";
		int x,y;
		double depthValue;

		//ifstream in22(depthFileName.c_str(),ios::in);

		//while(in22)
		//{
		//	in22>>x>>y>>depthValue;
		//	if (abs(depthValue)<=1)
		//	{
		//		depthImg.at<float>(y,x)=depthValue;
		//	}
		//}

		double standardDepth=750;
		ifstream in22(depthFileName.c_str(),ios::in);
		if (in22)
		{
			in22>>y>>x>>depthValue;
			in22.clear(); 
			in22.seekg(0);


			if (y==0&&x==0&&depthValue==0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						in22>>depthValue;
						if (depthValue!=0)
						{
							//depthImg.at<float>(i,j)=2980-depthValue;
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}
						else
						{
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}

					}
				}
			}
			else
			{
				double depthScale;
				string scaleFileName="G:\\face database\\facegen database depth only";
				scaleFileName+=depthFileName.substr(depthFileName.find_last_of('\\'),depthFileName.find_first_of('_')-depthFileName.find_last_of('\\'));
				scaleFileName+="_scale.txt";

				ifstream inscale(scaleFileName.c_str(),ios::in);
				inscale>>depthScale;
				inscale.close();

				//for synthesized data, need to be adjusted
				while(in22)
				{
					in22>>y>>x>>depthValue;
					if (depthValue==-1)
						depthValue=0;
					else
					{
						depthValue=(5-depthValue)/depthScale/standardDepth;
						//cout<<depthValue<<endl;
					}
					//if (abs(depthValue)<=1)
					{
						depthImg.at<float>(x,y)=depthValue;
					}
				}

			}
		}
		else //if it is in xml format, it should be exactly the depth. If there are minus values, then it should be 1200-depth
		{
			depthFileName=name;
			depthFileName=depthFileName.substr(0,depthFileName.length()-4);
			depthFileName+=".xml";
			CvMat* tmpDepth = (CvMat*)cvLoad( depthFileName.c_str());
			depthImg.release();
			depthImg=cvarrToMat(tmpDepth).clone();


			///*	Mat tmp=cvarrToMat(tmpDepth);*/
			//	for (int i=0;i<depthImg.rows;i++)
			//	{
			//		for (int j=0;j<depthImg.cols;j++)
			//		{
			//			if (depthImg.at<float>(i,j)<0)
			//			{
			//				cout<<depthImg.at<float>(i,j)<<" ";
			//			}
			//			
			//		}
			//		cout<<endl;
			//	}
			double testMaxval=500;
			int maxIdx[3]; 
			minMaxIdx(depthImg, &testMaxval, 0, maxIdx,0);
			if (testMaxval<0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)=1200-depthImg.at<float>(i,j);
						}

					}
				}
			}

			minMaxIdx(depthImg,0, &testMaxval, 0, maxIdx);
			if (testMaxval>10)	//else, already divided by the standard depth
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)/=standardDepth;
							////bad values
							//if (colorImg.at<float>(i,j)<0.1||colorImg.at<float>(i,j)>10)
							//{
							//	cout<<colorImg.at<float>(i,j)<<" ";
							//	colorImg.at<float>(i,j)=0;
							//}
						}

					}
				}
			}
		}

		if (trainStyle==0)
		{
			predict_rt_depth_GPU(depthImg,m_img,saveName);
		}
		else if (trainStyle==1)
		{
			predict_rt_color_GPU(depthImg,m_img,saveName);
		}
		else
		{
			predict_rt_depth_color_GPU(depthImg,m_img,saveName);
		}

//		predict_rt_GPU(img,saveName);

		//continue;
		int i,j;
		Mat probImg=colorImg;
		int offset;
		int currentLabelIndex;
		double blendPara=0.7;
		float currentProb;
		int currentShowingLabel=10;
		string tmpName;
		for (currentShowingLabel=0;currentShowingLabel<labelNum-1;currentShowingLabel++)
		{
			probImg=colorImg.clone();
			char name[50];
			sprintf(name, "_Label %d.jpg", currentShowingLabel);
			tmpName=prob_name+name;
			for (i=windowSize;i<m_img.cols-windowSize;i++)
			{
				//	cout<<i<<endl;
				for (j=windowSize;j<m_img.rows-windowSize;j++)
				{

					offset=j*m_img.cols+i;

					/*		if (labelResult[offset*(1+MAX_LABEL_NUMBER)]!=11)
					{
					continue;
					}
					if (labelResult[offset*(1+MAX_LABEL_NUMBER)+1+(int)labelResult[offset*(1+MAX_LABEL_NUMBER)]]<0.4)
					{
					continue;
					}
					currentLabelIndex=(float)labelResult[offset*(1+MAX_LABEL_NUMBER)]*(1000.0f/(float)(labelNum-1));*/

					currentProb=labelResult[offset*(1+MAX_LABEL_NUMBER)+1+currentShowingLabel];
				/*	if (currentProb<0.1)
					{
						continue;
					}*/
					currentLabelIndex=currentProb*1000.0f;
					if (currentLabelIndex==1000)
					{
						currentLabelIndex=999;
					}
					//circle(probImg,Point(i,j),3,Scalar(colorIndex[currentLabelIndex][2]*255,colorIndex[currentLabelIndex][1]*255,colorIndex[currentLabelIndex][0]*255));
					probImg.at<Vec3b>(j,i).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
						colorImg.at<Vec3b>(j,i).val[0]*(1-blendPara);
					probImg.at<Vec3b>(j,i).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
						colorImg.at<Vec3b>(j,i).val[1]*(1-blendPara);
					probImg.at<Vec3b>(j,i).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
						colorImg.at<Vec3b>(j,i).val[2]*(1-blendPara);
					//	delete []feature;
				}
			}
			namedWindow("labeled results");
			imshow("labeled results",probImg);
			char c=waitKey();

			if (currentShowingLabel==0)
			{
				break;
			}

		//	imwrite(tmpName,probImg);
		/*	if (c=='p')   
			{
				waitKey();
			}*/
		}
	/*	if (usingCUDA)
		{
			break;
		}*/
	}

}


void RandTree::predict_imgList_GPU(string listName)
{

	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{

	
		in.getline(name,500,'\n');
	/*	if (ll>_imgNum-8)
		{
			continue;
		}*/
		IplImage *img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorImg=imread(name);

		ImgName=name;
		Mat dst_color;
		Mat AllLabel(img->height,img->width,CV_32F);
		int i,j,k,l;
		Mat tmp,dst;

		Mat m_img=cvarrToMat(img);

		//Mat Prob_Img=m_img.clone();
		
	
		cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;

		string saveName=ImgName.substr(0,ImgName.length()-4);
		string imgName;

		imgName=saveName;
		string prob_name=saveName;
		saveName+=".txt";
		imgName+=".jpg";

		

		predict_rt_GPU(img,saveName);
		//continue;

		Mat probImg=colorImg;
		int offset;
		int currentLabelIndex;
		double blendPara=0.7;
		float currentProb;
		int currentShowingLabel=10;
		string tmpName;
		for (currentShowingLabel=0;currentShowingLabel<labelNum-1;currentShowingLabel++)
		{
			probImg=colorImg.clone();
			char name[50];
			sprintf(name, "_Label %d.jpg", currentShowingLabel);
			tmpName=prob_name+name;
			for (i=windowSize;i<m_img.cols-windowSize;i++)
			{
				//	cout<<i<<endl;
				for (j=windowSize;j<m_img.rows-windowSize;j++)
				{

					offset=j*m_img.cols+i;

					/*		if (labelResult[offset*(1+MAX_LABEL_NUMBER)]!=11)
					{
					continue;
					}
					if (labelResult[offset*(1+MAX_LABEL_NUMBER)+1+(int)labelResult[offset*(1+MAX_LABEL_NUMBER)]]<0.4)
					{
					continue;
					}
					currentLabelIndex=(float)labelResult[offset*(1+MAX_LABEL_NUMBER)]*(1000.0f/(float)(labelNum-1));*/

					currentProb=labelResult[offset*(1+MAX_LABEL_NUMBER)+1+currentShowingLabel];
				/*	if (currentProb<0.1)
					{
						continue;
					}*/
					currentLabelIndex=currentProb*1000.0f;
					if (currentLabelIndex==1000)
					{
						currentLabelIndex=999;
					}
					//circle(probImg,Point(i,j),3,Scalar(colorIndex[currentLabelIndex][2]*255,colorIndex[currentLabelIndex][1]*255,colorIndex[currentLabelIndex][0]*255));
					probImg.at<Vec3b>(j,i).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
						colorImg.at<Vec3b>(j,i).val[0]*(1-blendPara);
					probImg.at<Vec3b>(j,i).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
						colorImg.at<Vec3b>(j,i).val[1]*(1-blendPara);
					probImg.at<Vec3b>(j,i).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
						colorImg.at<Vec3b>(j,i).val[2]*(1-blendPara);
					//	delete []feature;
				}
			}
			namedWindow("labeled results");
			imshow("labeled results",probImg);
			char c=waitKey();

			//imwrite(tmpName,probImg);
		/*	if (c=='p')
			{
				waitKey();
			}*/
		}

		//char name[50];
		//sprintf(name, "_Label %d.jpg", currentShowingLabel);
		//prob_name+=name;
		//for (i=windowSize;i<m_img.cols-windowSize;i++)
		//{
		//	//	cout<<i<<endl;
		//	for (j=windowSize;j<m_img.rows-windowSize;j++)
		//	{
		//	
		//		offset=j*m_img.cols+i;
		//	
		///*		if (labelResult[offset*(1+MAX_LABEL_NUMBER)]!=11)
		//		{
		//			continue;
		//		}
		//		if (labelResult[offset*(1+MAX_LABEL_NUMBER)+1+(int)labelResult[offset*(1+MAX_LABEL_NUMBER)]]<0.4)
		//		{
		//			continue;
		//		}
		//		currentLabelIndex=(float)labelResult[offset*(1+MAX_LABEL_NUMBER)]*(1000.0f/(float)(labelNum-1));*/

		//		currentProb=labelResult[offset*(1+MAX_LABEL_NUMBER)+1+currentShowingLabel];
		//		if (currentProb<0.1)
		//		{
		//			continue;
		//		}
		//		currentLabelIndex=currentProb*1000.0f;
		//		if (currentLabelIndex==1000)
		//		{
		//			currentLabelIndex=999;
		//		}
		//		//circle(probImg,Point(i,j),3,Scalar(colorIndex[currentLabelIndex][2]*255,colorIndex[currentLabelIndex][1]*255,colorIndex[currentLabelIndex][0]*255));
		//		probImg.at<Vec3b>(j,i).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
		//			colorImg.at<Vec3b>(j,i).val[0]*(1-blendPara);
		//		probImg.at<Vec3b>(j,i).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
		//			colorImg.at<Vec3b>(j,i).val[1]*(1-blendPara);
		//		probImg.at<Vec3b>(j,i).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
		//			colorImg.at<Vec3b>(j,i).val[2]*(1-blendPara);
		//		//	delete []feature;
		//	}
		//}
		//namedWindow("labeled results");
		//imshow("labeled results",probImg);
		//imwrite(prob_name,probImg);
		//char c=waitKey(2);
		//if (c=='p')
		//{
		//	waitKey();
		//}
	
	/*	dst_color=colorImg.clone();
		output_probimg(dst_color,saveName);*/

		cvReleaseImage(&img);

	/*	if (usingCUDA)
		{
			break;
		}*/
	}

}

void RandTree::predict_rt_GPU(IplImage *img,string saveName="GPU labeling result.txt")
{

	Mat AllLabel(img->height,img->width,CV_32F);
	
	int i,j,k,l;
	Mat m_img=cvarrToMat(img);
	float *feature;
	for (i=0;i<m_img.rows;i++)
	{
		for (j=0;j<m_img.cols;j++)
		{
			host_inputImage[i*m_img.cols+j]=m_img.at<uchar>(i,j);
		}
	}
	predict_GPU(host_inputImage,m_img.cols,m_img.rows,labelResult);

	int curColorIndex;int currentLabel,currentLabelIndex;
	double blendPara=0.7;


	////string saveName="GPU labeling result.txt";
	//ofstream out(saveName.c_str(),ios::out);
	//int offset;
	//for (i=windowSize;i<m_img.cols-windowSize;i++)
	//{
	//	//	cout<<i<<endl;
	//	for (j=windowSize;j<m_img.rows-windowSize;j++)
	//	{
	//		offset=j*m_img.cols+i;
	//		
	//		out<<i<<" "<<j<<" "<<labelResult[offset*(1+MAX_LABEL_NUMBER)]<<" ";
	//		for (int k=0;k<labelNum;k++)
	//		{
	//			out<<labelResult[offset*(1+MAX_LABEL_NUMBER)+1+k]<<" ";
	//		}
	//		out<<endl;
	//		//	delete []feature;
	//	}
	//}
	//out.close();


	////ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
	////if (1)
	//{
	//	int size=windowSize;
	//	cout<<"predicting labels\n";
	//	CvPoint center;
	//	string saveName=ImgName.substr(0,ImgName.length()-4);
	//	saveName+="_eye_left.txt";
	//	ofstream out(saveName.c_str(),ios::out);

	//	LabelResult tmp;
	//	for (i=size;i<m_img.cols-size;i++)
	//	{
	//		//	cout<<i<<endl;
	//		for (j=size;j<m_img.rows-size;j++)
	//		{
	//			center.x=i;
	//			center.y=j;
	//			pridict_prob(m_img,center,tmp);
	//			//Mat cdata(1,sampleDim,CV_32F,feature);
	//			//out<<i<<" "<<j<<" "<<tmp.label<<endl;
	//			out<<i<<" "<<j<<" "<<tmp.label<<" ";
	//			for (int k=0;k<labelNum;k++)
	//			{
	//				out<<tmp.prob_all[k]<<" ";
	//			}
	//			out<<endl;
	//			//	delete []feature;
	//		}
	//	}
	//	out.close();

	//	//for(i=0;i<objectDescriptors->total;i++)
	//	//{
	//	//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
	//	//}
	//	//out.close();
	//}
	//cvReleaseImage(&img);
}


//depth only
void RandTree::predict_rt_depth_GPU(Mat &depthImage,Mat &colorImg,string saveName="GPU labeling result.txt")
{

	Mat AllLabel(depthImage.rows,depthImage.cols,CV_32FC1);

	int i,j,k,l;

	float *feature;
	for (i=0;i<depthImage.rows;i++)
	{
		for (j=0;j<depthImage.cols;j++)
		{
			host_depthImage[i*depthImage.cols+j]=depthImage.at<float>(i,j);
			host_colorImage[i*depthImage.cols+j]=colorImg.at<uchar>(i,j);
		}
	}
	predict_GPU_withDepth(host_colorImage,host_depthImage,depthImage.cols,depthImage.rows,labelResult,trainStyle);

	int curColorIndex;int currentLabel,currentLabelIndex;
	double blendPara=0.7;
}

//color only
void RandTree::predict_rt_color_GPU(Mat &depthImage,Mat &colorImg,string saveName="GPU labeling result.txt")
{

	Mat AllLabel(depthImage.rows,depthImage.cols,CV_32FC1);

	int i,j,k,l;

	float *feature;
	for (i=0;i<depthImage.rows;i++)
	{
		for (j=0;j<depthImage.cols;j++)
		{
			host_depthImage[i*depthImage.cols+j]=depthImage.at<float>(i,j);
			host_colorImage[i*depthImage.cols+j]=colorImg.at<uchar>(i,j);
		}
	}
	predict_GPU_withDepth(host_colorImage,host_depthImage,depthImage.cols,depthImage.rows,labelResult,trainStyle);

	int curColorIndex;int currentLabel,currentLabelIndex;
	double blendPara=0.7;
}

//deoth and color
void RandTree::predict_rt_depth_color_GPU(Mat &depthImage,Mat &colorImg,string saveName="GPU labeling result.txt")
{

	Mat AllLabel(depthImage.rows,depthImage.cols,CV_32FC1);

	int i,j,k,l;

	float *feature;
	for (i=0;i<depthImage.rows;i++)
	{
		for (j=0;j<depthImage.cols;j++)
		{
			host_depthImage[i*depthImage.cols+j]=depthImage.at<float>(i,j);
			host_colorImage[i*depthImage.cols+j]=colorImg.at<uchar>(i,j);
		}
	}
	predict_GPU_withDepth(host_colorImage,host_depthImage,depthImage.cols,depthImage.rows,labelResult,trainStyle);

	int curColorIndex;int currentLabel,currentLabelIndex;
	double blendPara=0.7;
}

void RandTree::predict_rt(string ImgName)
{
	IplImage *img=cvLoadImage(ImgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
	Mat AllLabel(img->height,img->width,CV_32F);
	int i,j,k,l;
	Mat m_img=cvarrToMat(img);
	float *feature;
	//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
	//if (1)
	{
		int size=windowSize;
		cout<<"predicting labels\n";
		CvPoint center;
		string saveName=ImgName.substr(0,ImgName.length()-4);
		saveName+="_eye_left.txt";
		ofstream out(saveName.c_str(),ios::out);

		LabelResult tmp;
		for (i=size;i<m_img.cols-size;i++)
		{
			//	cout<<i<<endl;
			for (j=size;j<m_img.rows-size;j++)
			{
				center.x=i;
				center.y=j;
				pridict_prob(m_img,center,tmp);
				//Mat cdata(1,sampleDim,CV_32F,feature);
				//out<<i<<" "<<j<<" "<<tmp.label<<endl;
				out<<i<<" "<<j<<" "<<tmp.label<<" ";
				for (int k=0;k<labelNum;k++)
				{
					out<<tmp.prob_all[k]<<" ";
				}
				out<<endl;
				//	delete []feature;
			}
		}
		out.close();

		//for(i=0;i<objectDescriptors->total;i++)
		//{
		//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
		//}
		//out.close();
	}
	cvReleaseImage(&img);
}


void RandTree::predict_fulltest(string ImgName)
{
	IplImage *img=cvLoadImage(ImgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
	Mat colorImg=imread(ImgName.c_str());
	Mat dst_color;
	Mat AllLabel(img->height,img->width,CV_32F);
	int i,j,k,l;
	Mat tmp,dst;

	Mat m_img=cvarrToMat(img);

	Mat Prob_Img=m_img.clone();
	double width;
	Mat warp_mat(2,3,CV_32F);
	Point current_center;
	current_center.x=m_img.cols/2;
	current_center.y=m_img.rows/2;
	
	for (int times=0;times<11;times++)
	{
		if (times==0)
		{
			dst=m_img.clone();
			dst_color=colorImg.clone();
		}
		else
		{
			warp_mat=getRotationMatrix2D(current_center,RandDouble_cABc(-9,9), RandDouble_cABc(pow(0.8,5),1));
			//dst=currentImage;
			/// Apply the Affine Transform just found to the src image
			warpAffine( m_img, dst, warp_mat, m_img.size() );
			warpAffine( colorImg, dst_color, warp_mat, m_img.size() );
		}
		

		float *feature;
		//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
		//if (1)
		{
			int size=windowSize;
			cout<<"predicting labels "<<times<<" /10\n";
			CvPoint center;
			string saveName=ImgName.substr(0,ImgName.length()-4);
			string imgName;
			char name[50];
			sprintf(name, "_test %d", times);
			saveName+=name;
			imgName=saveName;
			saveName+=".txt";
			imgName+=".jpg";
			imwrite(imgName,dst_color);
			
			ofstream out(saveName.c_str(),ios::out);

			LabelResult tmp;
			for (i=size;i<dst.cols-size;i++)
			{
				//	cout<<i<<endl;
				for (j=size;j<dst.rows-size;j++)
				{
					center.x=i;
					center.y=j;
					pridict_prob(dst,center,tmp);
					//Mat cdata(1,sampleDim,CV_32F,feature);
					//out<<i<<" "<<j<<" "<<tmp.label<<endl;
					out<<i<<" "<<j<<" "<<tmp.label<<" ";
					for (int k=0;k<labelNum;k++)
					{
						out<<tmp.prob_all[k]<<" ";
					}
					out<<endl;
					//	delete []feature;
				}
			}
			out.close();

			output_probimg(dst_color,saveName);
			
			//for(i=0;i<objectDescriptors->total;i++)
			//{
			//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
			//}
			//out.close();
		}

		dst.release();
		dst_color.release();
		warp_mat.release();
	}


	
	cvReleaseImage(&img);
}

void RandTree::predict_DepthList_fast_withLabelCenter(string listName,bool writeFile)
{
	if (usingCUDA)
	{
		//predict_imgList_GPU(listName);
		predict_imgList_GPU_depth(listName);
		return;
	}

	vector<CvPoint> *labelPos=new vector<CvPoint>[labelNum];
	vector<float> *weightPos=new vector<float>[labelNum];


	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{
		in.getline(name,500,'\n');
		
		if (ll<0)
		{
			continue;
		}	

		//IplImage *img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);

		Mat m_img=imread(name,0);

		Mat colorImg=imread(name);

		Mat depthImg=Mat::ones(colorImg.rows,colorImg.cols,CV_32FC1);

		
		//namedWindow("1");
		//imshow("1",depthImg);
		//waitKey();

		depthImg*=0;

		///////////////////////read in the depth data//////////////////////////////

		//depth
		string depthFileName=name;
		depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		depthFileName+="_depth.txt";
		int x,y;
		double depthValue;

		//ifstream in22(depthFileName.c_str(),ios::in);

		//while(in22)
		//{
		//	in22>>x>>y>>depthValue;
		//	if (abs(depthValue)<=1)
		//	{
		//		depthImg.at<float>(y,x)=depthValue;
		//	}
		//}

		double standardDepth=750;
		ifstream in22(depthFileName.c_str(),ios::in);
		if (in22)
		{
			in22>>y>>x>>depthValue;
			in22.clear(); 
			in22.seekg(0);


			if (y==0&&x==0&&depthValue==0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						in22>>depthValue;
						if (depthValue!=0)
						{
							//depthImg.at<float>(i,j)=2980-depthValue;
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}
						else
						{
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}

					}
				}
			}
			else
			{
				double depthScale;
				string scaleFileName="G:\\face database\\facegen database depth only";
				scaleFileName+=depthFileName.substr(depthFileName.find_last_of('\\'),depthFileName.find_first_of('_')-depthFileName.find_last_of('\\'));
				scaleFileName+="_scale.txt";

				ifstream inscale(scaleFileName.c_str(),ios::in);
				inscale>>depthScale;
				inscale.close();

				//for synthesized data, need to be adjusted
				while(in22)
				{
					in22>>y>>x>>depthValue;
					if (depthValue==-1)
						depthValue=0;
					else
					{
						depthValue=(5-depthValue)/depthScale/standardDepth;
						//cout<<depthValue<<endl;
					}
					//if (abs(depthValue)<=1)
					{
						depthImg.at<float>(x,y)=depthValue;
					}
				}
			
			}
		}
		else //if it is in xml format, it should be exactly the depth. If there are minus values, then it should be 1200-depth
		{
			depthFileName=name;
			depthFileName=depthFileName.substr(0,depthFileName.length()-4);
			depthFileName+=".xml";
			CvMat* tmpDepth = (CvMat*)cvLoad( depthFileName.c_str());
			depthImg.release();
			depthImg=cvarrToMat(tmpDepth).clone();


			///*	Mat tmp=cvarrToMat(tmpDepth);*/
			//	for (int i=0;i<depthImg.rows;i++)
			//	{
			//		for (int j=0;j<depthImg.cols;j++)
			//		{
			//			if (depthImg.at<float>(i,j)<0)
			//			{
			//				cout<<depthImg.at<float>(i,j)<<" ";
			//			}
			//			
			//		}
			//		cout<<endl;
			//	}
			double testMaxval=500;
			int maxIdx[3]; 
			minMaxIdx(depthImg, &testMaxval, 0, maxIdx,0);
			if (testMaxval<0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)=1200-depthImg.at<float>(i,j);
						}

					}
				}
			}

			minMaxIdx(depthImg,0, &testMaxval, 0, maxIdx);
			if (testMaxval>10)	//else, already divided by the standard depth
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)/=standardDepth;
							////bad values
							//if (colorImg.at<float>(i,j)<0.1||colorImg.at<float>(i,j)>10)
							//{
							//	cout<<colorImg.at<float>(i,j)<<" ";
							//	colorImg.at<float>(i,j)=0;
							//}
						}

					}
				}
			}
		}

		//string depthFileName=name;
		//depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		//depthFileName+="_depth.txt";
		//int x,y;
		//double depthValue;

		//ifstream in22(depthFileName.c_str(),ios::in);
		//in22>>y>>x>>depthValue;
		//in22.clear(); 
		//in22.seekg(0);

		//double standardDepth=750;
		//if (y==0&&x==0&&depthValue==0)
		//{
		//	for (int i=0;i<depthImg.rows;i++)
		//	{
		//		for (int j=0;j<depthImg.cols;j++)
		//		{
		//			
		//			in22>>depthValue;
		//			//if (depthValue!=0)
		//			{
		//				//depthImg.at<float>(i,j)=2980-depthValue;
		//				depthImg.at<float>(i,j)=depthValue/standardDepth;
		//			}
		//			/*else
		//			{
		//				depthImg.at<float>(i,j)=depthValue/standardDepth;
		//			}*/
		//			
		//		}
		//	}
		//}
		//else
		//{
		//	while(in22)
		//	{
		//		in22>>y>>x>>depthValue;
		//		if (abs(depthValue)<=1)
		//		{
		//			depthImg.at<float>(x,y)=depthValue;
		//		}
		//	}
		//}
		////////////////////////////////////////////////////////

	/*	namedWindow("1");
		imshow("1",depthImg);
		waitKey();*/
		//Mat colorImg123=imread(name);
		
		//rescale the face
	/*	float scale=0.9;
		resize(colorImg,colorImg,Size(scale*colorImg.cols,scale*colorImg.rows));
		resize(depthImg,depthImg,Size(scale*depthImg.cols,scale*depthImg.rows));
		resize(m_img,m_img,Size(scale*m_img.cols,scale*m_img.rows));*/
		
	/*		namedWindow("1");
		imshow("1",(m_img));
		waitKey();*/

	
		ImgName=name;
		Mat dst_color;
		Mat AllLabel(m_img.rows,m_img.cols,CV_32F);
		int i,j,k,l;
		Mat tmp,dst;

		
		Mat Prob_Img=m_img.clone();
		Mat gradientMap;
		Mat imgGradientMap;
		double width;
		Mat warp_mat(2,3,CV_32F);
		Point current_center;
		current_center.x=m_img.cols/2;
		current_center.y=m_img.rows/2;
		cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;

		Mat detectionResult=Mat::zeros(640*480,3+labelNum,CV_32FC1);
		int validNum=0;
		for (int times=0;times<1;times++)
		{
			if (times==0)
			{
				dst=depthImg.clone();
				dst_color=colorImg.clone();
			}
			

			//set up the image
			int x,y;
			int i;
			double *prob=new double[labelNum];
			
			int curColorIndex;int currentLabel,currentLabelIndex;
			double blendPara=0.7;

			//Mat probImg(colorImg.rows,colorImg.cols*(labelNum+1),colorImg.type());
			//for (int i=0;i<m_img.rows;i++)
			//{
			//	for (int j=0;j<m_img.cols;j++)
			//	{
			//		//if (j<probImg.cols/2)
			//		//{
			//		//cout<<i<<" "<<j<<endl;
			//		probImg.at<Vec3b>(i,j)=colorImg.at<Vec3b>(i,j);
			//		for (int k=0;k<labelNum-1;k++)
			//		{
			//			curColorIndex=0;
			//			/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
			//			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
			//			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
			//			y=i;
			//			x=j;
			//			probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
			//				colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
			//			probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
			//				colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);;
			//			probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
			//				colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);;


			//		}
			//		//}

			//	}
			//}

			float *feature;
			//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
			//if (1)
			{
				int size=windowSize;
				
	
				LabelResult tmp;
				for (i=size;i<dst.cols-size;i++)
				{
					//	cout<<i<<endl;
					for (j=size;j<dst.rows-size;j++)
					{
						CvPoint center;
						center.x=i;
						center.y=j;

						if (dst.at<float>(j,i)==0)
						{
							continue;
						}
						
						if (trainStyle==0)
						{
							pridict_prob(dst,center,tmp);
						}
						else
						{
							pridict_prob(dst,m_img,center,tmp);
						}
						
						////////////////////////////////////////////////
						if (tmp.prob_all[tmp.label]>0.4)
						{
							weightPos[tmp.label].push_back(tmp.prob_all[tmp.label]);
							labelPos[tmp.label].push_back(center);

							detectionResult.at<float>(validNum,0)=i;
							detectionResult.at<float>(validNum,1)=j;
							detectionResult.at<float>(validNum,2)=tmp.label;

							for (int km=0;km<labelNum;km++)
							{
								detectionResult.at<float>(validNum,3+km)=tmp.prob_all[km];
							}
							validNum++;
						}
					
						////////////////////////////////////////////////


						//x=i;
						//y=j;
						//for (int k=0;k<labelNum-1;k++)
						//{
						//	curColorIndex=tmp.prob_all[k]*1000;
						//	if (curColorIndex==1000)
						//	{
						//		curColorIndex=999;
						//	}
						//
						//	/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
						//	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
						//	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
						//	probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
						//		colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
						//	probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
						//		colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);;
						//	probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
						//		colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);;


						//}

						//currentLabelIndex=tmp.label*(1000.0f/(float)(labelNum-1));
						//if (currentLabelIndex==1000)
						//{
						//	currentLabelIndex=999;
						//}
						//if (tmp.prob_all[tmp.label]<0.4)
						//{
						//	currentLabelIndex=999;
						//}
						//else
						//{

						//}
						//probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
						//	colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
						//probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
						//	colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);
						//probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
						//	colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);
						//Mat cdata(1,sampleDim,CV_32F,feature);
						//out<<i<<" "<<j<<" "<<tmp.label<<endl;
						/*out<<i<<" "<<j<<" "<<tmp.label<<" ";
						for (int k=0;k<labelNum;k++)
						{
							out<<tmp.prob_all[k]<<" ";
						}
						out<<endl;*/
						//	delete []feature;
					}
				}
			
				//for(i=0;i<objectDescriptors->total;i++)
				//{
				//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
				//}
				//out.close();
			}
			//char name[50];
			//sprintf(name, "_test_%d_%d_%d_%d", times,max_num_of_trees_in_the_forest,max_depth,windowSize);
			//string prob_name=ImgName.substr(0,ImgName.length()-4);//(fileName.find_last_of("."))
			//prob_name+=name;
			//prob_name+="_prob.jpg";
			//cout<<prob_name<<endl;
			//imwrite(prob_name,probImg);

			//dst.release();
			//dst_color.release();
			//warp_mat.release();
			//probImg.release();
		}

		//get the weighted label center
		double labelCenter[30][2];
		double thresholdRatio=0.8;
	

		
		//first, get the location with maximum probability
		int maximumInd[30];
		for (int i=0;i<labelNum;i++)
		{
			double maximumProb=-1;
			for (int j=0;j<weightPos[i].size();j++)
			{
				if (weightPos[i][j]>maximumProb)
				{
					maximumProb=weightPos[i][j];
					maximumInd[i]=j;
				}
			}
		}


		//second, normalize the weights
		for (int i=0;i<labelNum;i++)
		{
			double totalWeight=0;
			int usedNum=0;
			for (int j=0;j<weightPos[i].size();j++)
			{
				if (weightPos[i][j]/weightPos[i][maximumInd[i]]>thresholdRatio)
				{
					CvPoint tmp=cvPoint(labelPos[i][j].x-labelPos[i][maximumInd[i]].x,
						labelPos[i][j].y-labelPos[i][maximumInd[i]].y);
					if (sqrtf(tmp.x*tmp.x+tmp.y*tmp.y)<windowSize)
					{
						totalWeight+=weightPos[i][j];
						usedNum++;
					}
				
				}
				
			}
			for (int j=0;j<weightPos[i].size();j++)
			{
				weightPos[i][j]/=totalWeight;
			}
		}

		//then, get the weighted center
		for (int i=0;i<labelNum;i++)
		{
			labelCenter[i][0]=labelCenter[i][1]=0;

			for (int j=0;j<weightPos[i].size();j++)
			{
				if (weightPos[i][j]/weightPos[i][maximumInd[i]]>thresholdRatio)
				{
					CvPoint tmp=cvPoint(labelPos[i][j].x-labelPos[i][maximumInd[i]].x,
						labelPos[i][j].y-labelPos[i][maximumInd[i]].y);
					if (sqrtf(tmp.x*tmp.x+tmp.y*tmp.y)<windowSize)
					{
						labelCenter[i][0]+=weightPos[i][j]*labelPos[i][j].x;
						labelCenter[i][1]+=weightPos[i][j]*labelPos[i][j].y;
					}
				
				}			
			}
		}

	
		
	/*	for (int i=0;i<labelNum;i++)
		{
			Mat resultImg=colorImg.clone();
			
			for (int j=0;j<weightPos[i].size();j++)
			{
				double ratio=weightPos[i][j]/weightPos[i][maximumInd[i]];
				if (ratio>thresholdRatio)
				{
					circle(resultImg,cvPoint((int)labelPos[i][j].x,
						(int)labelPos[i][j].y),3,CV_RGB(0,0,255));
				}
				
			}

			circle(resultImg,cvPoint((int)labelPos[i][maximumInd[i]].x,
				(int)labelPos[i][maximumInd[i]].y),3,CV_RGB(0,255,0));

			namedWindow("1");
			imshow("1",resultImg);
			waitKey();
		}*/


		

		//finally, show the image
		Mat resultImg=colorImg.clone();
		for (int i=0;i<labelNum-1;i++)
		{
			circle(resultImg,cvPoint((int)labelCenter[i][0],
				(int)labelCenter[i][1]),3,CV_RGB(0,0,255));
		}

		namedWindow("1");
		imshow("1",resultImg);
		waitKey();

	/*	depthFileName=name;
		depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		depthFileName+="_expectation.jpg";
		imwrite(depthFileName,resultImg);*/

		////output the probality result
		//depthFileName=name;
		//depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		//depthFileName+="_rt.txt";
		//ofstream out_rt(depthFileName.c_str(),ios::out);
		//for (int i=0;i<validNum;i++)
		//{
		//	for (int j=0;j<detectionResult.cols;j++)
		//	{
		//		out_rt<<detectionResult.at<float>(i,j)<<" ";
		//	}
		//	out_rt<<endl;
		//}
		//out_rt.close();

		//cvReleaseImage(&img);
	}

	


}

float RandTree::predictPoint(int x,int y,Mat &colorImg,Mat &depthImg,int index,float refProb)
{
	int size=windowSize;


	LabelResult tmp;
	int i=x;int j=y;
	CvPoint center;
	center.x=i;
	center.y=j;

	if (depthImg.at<float>(j,i)==0)
	{
		return 0;
	}

	if (trainStyle==0)
	{
		pridict_prob(depthImg,center,tmp);
	}
	else
	{
		pridict_prob(depthImg,colorImg,center,tmp);
	}
	return tmp.prob_all[index];
	//cout<<tmp.prob_all[index]<<" "<<refProb<<endl;				
	/*if (tmp.prob_all[index]<0.9*refProb)
	{
		return 0;
	}
	return 1;*/
		
}


void RandTree::getProb_missingData(Node *root,Mat mat,CvPoint &pos,LeafNode &leafnode)
{
	Node *current=root;

	while(1)
	{
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			//if (current->label>=0)
			{
				leafnode.leaf_node=current;
				leafnode.label=current->label;
			}
			break;
		}

		if (trainStyle==0)
		{
			double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
			if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
				mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					break;
				}
				current=current->l_child;
			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					break;
				}
				current=current->r_child;
			}
		}
		else
		{
			if (mat.at<float>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
				mat.at<float>(current->pos2[1]+pos.y,current->pos2[0]+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					break;
				}
				current=current->l_child;
			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					break;
				}
				current=current->r_child;
			}
		}
	}
}

//mat: depth image
//gradient: color image
void RandTree::getProb_missingData(Node *root,Mat mat,Mat gradient,CvPoint &pos,vector<LeafNode> &leafnode)
{
	Node *current=root;

	vector<Node *> currentNodes;
	currentNodes.push_back(root);

	//cout<<"begin of the tree\n";

	while(currentNodes.size()!=0)
	{
		current=currentNodes[currentNodes.size()-1];
		currentNodes.pop_back();
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			//if (current->label>=0)
			{
				LeafNode tmp;
				tmp.leaf_node=current;
				tmp.label=current->label;
				leafnode.push_back(tmp);
			}
			//break;
		}

		if (trainStyle==1)
		{
			double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
			if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
				gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					//break;
				}
				else
				{
					//current=current->l_child;
					currentNodes.push_back(current->l_child);
				}
				
			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					;//break;
				}
				else
				{
					//current=current->r_child;
					currentNodes.push_back(current->r_child);
				}
				
			}
		}
		else if (trainStyle==2)
		{
			if (current->threshold>=0)	//depth
			{
				//cout<<current->nLevel<<" depth\n";
				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
				///////////////consider the missing values here//////////////////////
				//go to both sides
				if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)==0||
					mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)==0)
				{
					if (current->l_child!=NULL)
					{
						currentNodes.push_back(current->l_child);
					}
					if (current->r_child!=NULL)
					{
						currentNodes.push_back(current->r_child);
					}
				}
				else
				{
					if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
						mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)
						//if (mat.at<uchar>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
						//	mat.at<uchar>(current->pos2[1]+pos.y,current->pos2[0]+pos.x)+current->threshold)
					{

						if (current->l_child==NULL)
						{
							//break;
						}
						else
						{
							//current=current->l_child;
							currentNodes.push_back(current->l_child);
						}
					}
					else
					{

						if (current->r_child==NULL)//leaf
						{
							;//break;
						}
						else
						{
							//current=current->r_child;
							currentNodes.push_back(current->r_child);
						}
					}
				}
			}
			else	//color
			{
				//cout<<current->nLevel<<" color\n";
				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
				if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
					gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+(-current->threshold-1))
				{

					if (current->l_child==NULL)
					{
						//break;
					}
					else
					{
						//current=current->l_child;
						currentNodes.push_back(current->l_child);
					}
				}
				else
				{

					if (current->r_child==NULL)//leaf
					{
						;//break;
					}
					else
					{
						//current=current->r_child;
						currentNodes.push_back(current->r_child);
					}
				}
			}
		}
	}
}


void RandTree::pridict_prob_missingData(Mat mat,CvPoint &pos,LabelResult& result)
{
	//LabelResult result;
	if (treeNum<1)
	{
		return ;
	}
	double *label_prob_all=new double[labelNum];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_all[i]=0;
	}
	int i,j;
	LeafNode leaf;
	for (i=0;i<treeNum;i++)
	{
		getProb_missingData(roots[i],mat,pos,leaf);
		if (leaf.leaf_node!=NULL)
		{
			for (j=0;j<labelNum;j++)
			{
				label_prob_all[j]+=leaf.leaf_node->num_of_each_class[j];
			}

		}
	}

	//find the most frequent label
	int maxInd=0;
	double maxNum=-1;
	for (i=0;i<labelNum;i++)
	{
		if (label_prob_all[i]>maxNum)
		{
			maxNum=label_prob_all[i];
			maxInd=i;
		}
	}

	//if (label_prob_all[maxInd]>threshold)
	{
		result.label=maxInd;
		result.prob=label_prob_all[maxInd]/(double)treeNum;
		//result.prob_all=new double[labelNum];
		for (i=0;i<labelNum;i++)
		{
			result.prob_all[i]=label_prob_all[i]/(double)treeNum;
		}
		//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	}
	delete []label_prob_all;

}

void RandTree::pridict_prob_missingData(Mat mat,Mat gradient,CvPoint &pos,LabelResult& result)
{
	//LabelResult result;
	if (treeNum<1)
	{
		return ;
	}
	double *label_prob_all=new double[labelNum];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_all[i]=0;
	}
	int i,j;
	int totalTreeNum=0;
	vector<LeafNode> leaf;
	for (i=0;i<treeNum;i++)
	{
		getProb_missingData(roots[i],mat,gradient,pos,leaf);
		for (int j=0;j<leaf.size();j++)
		{
			if (leaf[j].leaf_node!=NULL)
			{
				totalTreeNum++;
				for (int k=0;k<labelNum;k++)
				{
					label_prob_all[k]+=leaf[j].leaf_node->num_of_each_class[k];
				}

			}
		}
		
	}

	//find the most frequent label
	int maxInd=0;
	double maxNum=-1;
	for (i=0;i<labelNum;i++)
	{
		if (label_prob_all[i]>maxNum)
		{
			maxNum=label_prob_all[i];
			maxInd=i;
		}
	}

	//if (label_prob_all[maxInd]>threshold)
	{
		result.label=maxInd;
		result.prob=label_prob_all[maxInd]/(double)totalTreeNum;
		//result.prob_all=new double[labelNum];
		for (i=0;i<labelNum;i++)
		{
			result.prob_all[i]=label_prob_all[i]/(double)totalTreeNum;
		}
		//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	}
	delete []label_prob_all;

}

float RandTree::predictPoint_missingData(int x,int y,Mat &colorImg,Mat &depthImg,int index,float refProb)
{
	int size=windowSize;


	LabelResult tmp;
	int i=x;int j=y;
	CvPoint center;
	center.x=i;
	center.y=j;

	if (depthImg.at<float>(j,i)==0)
	{
		return 0;
	}

	if (trainStyle==0)
	{
		pridict_prob_missingData(depthImg,center,tmp);
	}
	else
	{
		pridict_prob_missingData(depthImg,colorImg,center,tmp);
	}
	return tmp.prob_all[index];
	//cout<<tmp.prob_all[index]<<" "<<refProb<<endl;				
	/*if (tmp.prob_all[index]<0.9*refProb)
	{
		return 0;
	}
	return 1;*/
		
}

float RandTree::predictPoint_missingData(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &tmp)
{
	int size=windowSize;


	
	int i=x;int j=y;
	CvPoint center;
	center.x=i;
	center.y=j;

	if (depthImg.at<float>(j,i)==0)
	{
		return 0;
	}

	if (trainStyle==0)
	{
		pridict_prob_missingData(depthImg,center,tmp);
	}
	else
	{
		pridict_prob_missingData(depthImg,colorImg,center,tmp);
	}


	//return tmp.prob_all[index];
	//cout<<tmp.prob_all[index]<<" "<<refProb<<endl;				
	/*if (tmp.prob_all[index]<0.9*refProb)
	{
		return 0;
	}
	return 1;*/
		
}


void RandTree::predict_IMG(Mat &colorIMg,Mat &depthImg,Mat *result,int startX/* =0 */,int endX/* =10000 */,int startY/* =0;int endY/* =10000 */,int endY)
{
	
	for (int i=0;i<labelNum-1;i++)
	{
		result[i]*=0;
	}
	for (int i=windowSize;i<colorIMg.rows-windowSize;i++)
	{
		for (int j=windowSize;j<colorIMg.cols-windowSize;j++)
		{
			if (depthImg.at<float>(i,j)==0)
			{
				continue;
			}
			if (i<startY||i>endY||j<startX||j>endX)
			{
				continue;
			}
			LabelResult tmp;
			predictPoint_missingData(j,i,colorIMg,depthImg,tmp);
			for (int k=0;k<labelNum-1;k++)
			{
				result[k].at<float>(i,j)=tmp.prob_all[k];
			}
		}
	}
}

void RandTree::getProb_LRprob(Node *root,Mat mat,Mat gradient,CvPoint &pos,vector<LeafNode> &leafnode)
{
	Node *current=root;

	vector<Node *> currentNodes;
	currentNodes.push_back(root);

	//cout<<"begin of the tree\n";

	while(currentNodes.size()!=0)
	{
		current=currentNodes[currentNodes.size()-1];
		currentNodes.pop_back();
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			//if (current->label>=0)
			{
				LeafNode tmp;
				tmp.leaf_node=current;
				tmp.label=current->label;
				leafnode.push_back(tmp);
			}
			//break;
		}

		if (trainStyle==1)
		{
			double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
			if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
				gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					//break;
				}
				else
				{
					//current=current->l_child;
					currentNodes.push_back(current->l_child);
				}

			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					;//break;
				}
				else
				{
					//current=current->r_child;
					currentNodes.push_back(current->r_child);
				}

			}
		}
		else if (trainStyle==2)
		{
			if (current->threshold>=0)	//depth
			{
				//cout<<current->nLevel<<" depth\n";

				///////////////consider the missing values here//////////////////////
				//go to both sides

				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);

				if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)==0||
					mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)==0)
				{
					if (current->label==1&&current->l_child!=NULL)
					{
						currentNodes.push_back(current->l_child);
					}
					else if (current->label==0&&current->r_child!=NULL)
					{
						currentNodes.push_back(current->r_child);
					}
				}
				else
				{
					if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
						mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)
						//if (mat.at<uchar>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
						//	mat.at<uchar>(current->pos2[1]+pos.y,current->pos2[0]+pos.x)+current->threshold)
					{

						if (current->l_child==NULL)
						{
							//break;
						}
						else
						{
							//current=current->l_child;
							currentNodes.push_back(current->l_child);
						}
					}
					else
					{

						if (current->r_child==NULL)//leaf
						{
							;//break;
						}
						else
						{
							//current=current->r_child;
							currentNodes.push_back(current->r_child);
						}
					}
				}
				
			}
			else	//color
			{
				//cout<<current->nLevel<<" color\n";
				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
				if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
					gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+(-current->threshold-1))
				{

					if (current->l_child==NULL)
					{
						//break;
					}
					else
					{
						//current=current->l_child;
						currentNodes.push_back(current->l_child);
					}
				}
				else
				{

					if (current->r_child==NULL)//leaf
					{
						;//break;
					}
					else
					{
						//current=current->r_child;
						currentNodes.push_back(current->r_child);
					}
				}
			}
		}
	}
}

void RandTree::pridict_prob_LRprob(Mat mat,Mat gradient,CvPoint &pos,LabelResult& result)
{
	//LabelResult result;
	if (treeNum<1)
	{
		return ;
	}
	double *label_prob_all=new double[labelNum];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_all[i]=0;
	}
	int i,j;
	int totalTreeNum=0;
	vector<LeafNode> leaf;
	for (i=0;i<treeNum;i++)
	{
		getProb_LRprob(roots[i],mat,gradient,pos,leaf);
		for (int j=0;j<leaf.size();j++)
		{
			if (leaf[j].leaf_node!=NULL)
			{
				totalTreeNum++;
				for (int k=0;k<labelNum;k++)
				{
					label_prob_all[k]+=leaf[j].leaf_node->num_of_each_class[k];
				}

			}
		}

	}

	//find the most frequent label
	int maxInd=0;
	double maxNum=-1;
	for (i=0;i<labelNum;i++)
	{
		if (label_prob_all[i]>maxNum)
		{
			maxNum=label_prob_all[i];
			maxInd=i;
		}
	}

	//if (label_prob_all[maxInd]>threshold)
	{
		result.label=maxInd;
		result.prob=label_prob_all[maxInd]/(double)totalTreeNum;
		//result.prob_all=new double[labelNum];
		for (i=0;i<labelNum;i++)
		{
			result.prob_all[i]=label_prob_all[i]/(double)totalTreeNum;
		}
		//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	}
	delete []label_prob_all;

}

float RandTree::predictPoint_LRprob(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &tmp)
{
	int size=windowSize;


	
	int i=x;int j=y;
	CvPoint center;
	center.x=i;
	center.y=j;

	if (depthImg.at<float>(j,i)==0)
	{
		return 0;
	}

	if (trainStyle==0)
	{
		//pridict_prob_missingData(depthImg,center,tmp);
	}
	else
	{
		pridict_prob_LRprob(depthImg,colorImg,center,tmp);
	}


	//return tmp.prob_all[index];
	//cout<<tmp.prob_all[index]<<" "<<refProb<<endl;				
	/*if (tmp.prob_all[index]<0.9*refProb)
	{
		return 0;
	}
	return 1;*/
		
}

void RandTree::predict_IMG_LRprob(Mat &colorIMg,Mat &depthImg,Mat *result,int startX/* =0 */,int endX/* =10000 */,int startY/* =0;int endY/* =10000 */,int endY)
{

	for (int i=0;i<labelNum-1;i++)
	{
		result[i]*=0;
	}
	for (int i=windowSize;i<colorIMg.rows-windowSize;i++)
	{
		for (int j=windowSize;j<colorIMg.cols-windowSize;j++)
		{
			if (depthImg.at<float>(i,j)==0)
			{
				continue;
			}
			if (i<startY||i>endY||j<startX||j>endX)
			{
				continue;
			}
			LabelResult tmp;
			predictPoint_LRprob(j,i,colorIMg,depthImg,tmp);
			for (int k=0;k<labelNum-1;k++)
			{
				result[k].at<float>(i,j)=tmp.prob_all[k];
			}
		}
	}
}


void RandTree::predict_DepthList_fast(string listName,bool writeFile)
{
	if (usingCUDA)
	{
		predict_imgList_GPU(listName);
		return;
	}

	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{
		in.getline(name,500,'\n');
		
		if (ll<0)
		{
			continue;
		}	

		//IplImage *img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);

		Mat m_img=imread(name,0);

		Mat colorImg=imread(name);

		Mat depthImg=Mat::ones(colorImg.rows,colorImg.cols,CV_32FC1);

		
		//namedWindow("1");
		//imshow("1",depthImg);
		//waitKey();

		depthImg*=0;

		///////////////////////read in the depth data//////////////////////////////

		//depth
		string depthFileName=name;
		depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		depthFileName+="_depth.txt";
		int x,y;
		double depthValue;

		//ifstream in22(depthFileName.c_str(),ios::in);

		//while(in22)
		//{
		//	in22>>x>>y>>depthValue;
		//	if (abs(depthValue)<=1)
		//	{
		//		depthImg.at<float>(y,x)=depthValue;
		//	}
		//}

		double standardDepth=750;
		ifstream in22(depthFileName.c_str(),ios::in);
		if (in22)
		{
			in22>>y>>x>>depthValue;
			in22.clear(); 
			in22.seekg(0);


			if (y==0&&x==0&&depthValue==0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						in22>>depthValue;
						if (depthValue!=0)
						{
							//depthImg.at<float>(i,j)=2980-depthValue;
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}
						else
						{
							depthImg.at<float>(i,j)=depthValue/standardDepth;
						}

					}
				}
			}
			else
			{
				double depthScale;
				string scaleFileName="G:\\face database\\facegen database depth only";
				scaleFileName+=depthFileName.substr(depthFileName.find_last_of('\\'),depthFileName.find_first_of('_')-depthFileName.find_last_of('\\'));
				scaleFileName+="_scale.txt";

				ifstream inscale(scaleFileName.c_str(),ios::in);
				inscale>>depthScale;
				inscale.close();

				//for synthesized data, need to be adjusted
				while(in22)
				{
					in22>>y>>x>>depthValue;
					if (depthValue==-1)
						depthValue=0;
					else
					{
						depthValue=(5-depthValue)/depthScale/standardDepth;
						//cout<<depthValue<<endl;
					}
					//if (abs(depthValue)<=1)
					{
						depthImg.at<float>(x,y)=depthValue;
					}
				}
			
			}
		}
		else //if it is in xml format, it should be exactly the depth. If there are minus values, then it should be 1200-depth
		{
			depthFileName=name;
			depthFileName=depthFileName.substr(0,depthFileName.length()-4);
			depthFileName+=".xml";
			CvMat* tmpDepth = (CvMat*)cvLoad( depthFileName.c_str());
			depthImg.release();
			depthImg=cvarrToMat(tmpDepth).clone();


			///*	Mat tmp=cvarrToMat(tmpDepth);*/
			//	for (int i=0;i<depthImg.rows;i++)
			//	{
			//		for (int j=0;j<depthImg.cols;j++)
			//		{
			//			if (depthImg.at<float>(i,j)<0)
			//			{
			//				cout<<depthImg.at<float>(i,j)<<" ";
			//			}
			//			
			//		}
			//		cout<<endl;
			//	}
			double testMaxval=500;
			int maxIdx[3]; 
			minMaxIdx(depthImg, &testMaxval, 0, maxIdx,0);
			if (testMaxval<0)
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)=1200-depthImg.at<float>(i,j);
						}

					}
				}
			}

			minMaxIdx(depthImg,0, &testMaxval, 0, maxIdx);
			if (testMaxval>10)	//else, already divided by the standard depth
			{
				for (int i=0;i<depthImg.rows;i++)
				{
					for (int j=0;j<depthImg.cols;j++)
					{
						if (depthImg.at<float>(i,j)!=0)
						{
							depthImg.at<float>(i,j)/=standardDepth;
							////bad values
							//if (colorImg.at<float>(i,j)<0.1||colorImg.at<float>(i,j)>10)
							//{
							//	cout<<colorImg.at<float>(i,j)<<" ";
							//	colorImg.at<float>(i,j)=0;
							//}
						}

					}
				}
			}
		}

		//string depthFileName=name;
		//depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		//depthFileName+="_depth.txt";
		//int x,y;
		//double depthValue;

		//ifstream in22(depthFileName.c_str(),ios::in);
		//in22>>y>>x>>depthValue;
		//in22.clear(); 
		//in22.seekg(0);

		//double standardDepth=750;
		//if (y==0&&x==0&&depthValue==0)
		//{
		//	for (int i=0;i<depthImg.rows;i++)
		//	{
		//		for (int j=0;j<depthImg.cols;j++)
		//		{
		//			
		//			in22>>depthValue;
		//			//if (depthValue!=0)
		//			{
		//				//depthImg.at<float>(i,j)=2980-depthValue;
		//				depthImg.at<float>(i,j)=depthValue/standardDepth;
		//			}
		//			/*else
		//			{
		//				depthImg.at<float>(i,j)=depthValue/standardDepth;
		//			}*/
		//			
		//		}
		//	}
		//}
		//else
		//{
		//	while(in22)
		//	{
		//		in22>>y>>x>>depthValue;
		//		if (abs(depthValue)<=1)
		//		{
		//			depthImg.at<float>(x,y)=depthValue;
		//		}
		//	}
		//}
		////////////////////////////////////////////////////////

	/*	namedWindow("1");
		imshow("1",depthImg);
		waitKey();*/
		//Mat colorImg123=imread(name);
		
		//rescale the face
	/*	float scale=0.9;
		resize(colorImg,colorImg,Size(scale*colorImg.cols,scale*colorImg.rows));
		resize(depthImg,depthImg,Size(scale*depthImg.cols,scale*depthImg.rows));
		resize(m_img,m_img,Size(scale*m_img.cols,scale*m_img.rows));*/
		
	/*		namedWindow("1");
		imshow("1",(m_img));
		waitKey();*/

		ImgName=name;
		Mat dst_color;
		Mat AllLabel(m_img.rows,m_img.cols,CV_32F);
		int i,j,k,l;
		Mat tmp,dst;

		
		Mat Prob_Img=m_img.clone();
		Mat gradientMap;
		Mat imgGradientMap;
		double width;
		Mat warp_mat(2,3,CV_32F);
		Point current_center;
		current_center.x=m_img.cols/2;
		current_center.y=m_img.rows/2;
		cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;
		for (int times=0;times<1;times++)
		{
			if (times==0)
			{
				dst=depthImg.clone();
				dst_color=colorImg.clone();
			}
			

			//set up the image
			int x,y;
			int i;
			double *prob=new double[labelNum];
			
			int curColorIndex;int currentLabel,currentLabelIndex;
			double blendPara=0.7;

			Mat probImg(colorImg.rows,colorImg.cols*(labelNum+1),colorImg.type());
			for (int i=0;i<m_img.rows;i++)
			{
				for (int j=0;j<m_img.cols;j++)
				{
					//if (j<probImg.cols/2)
					//{
					//cout<<i<<" "<<j<<endl;
					probImg.at<Vec3b>(i,j)=colorImg.at<Vec3b>(i,j);
					for (int k=0;k<labelNum-1;k++)
					{
						curColorIndex=0;
						/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
						probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
						probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
						y=i;
						x=j;
						probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
						probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);;
						probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);;


					}
					//}

				}
			}

			float *feature;
			//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
			//if (1)
			{
				int size=windowSize;
				CvPoint center;
	
				LabelResult tmp;
				for (i=size;i<dst.cols-size;i++)
				{
					//	cout<<i<<endl;
					for (j=size;j<dst.rows-size;j++)
					{
						center.x=i;
						center.y=j;

						if (dst.at<float>(j,i)==0)
						{
							continue;
						}
						
						if (trainStyle==0)
						{
							pridict_prob(dst,center,tmp);
						}
						else
						{
							pridict_prob(dst,m_img,center,tmp);
						}
						
						
						x=i;
						y=j;
						for (int k=0;k<labelNum-1;k++)
						{
							curColorIndex=tmp.prob_all[k]*1000;
							if (curColorIndex==1000)
							{
								curColorIndex=999;
							}
						
							/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
							probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
							probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
							probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
								colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
							probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
								colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);;
							probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
								colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);;


						}

						currentLabelIndex=tmp.label*(1000.0f/(float)(labelNum-1));
						if (currentLabelIndex==1000)
						{
							currentLabelIndex=999;
						}
						if (tmp.prob_all[tmp.label]<0.4)
						{
							currentLabelIndex=999;
						}
						probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
						probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);
						probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);
						//Mat cdata(1,sampleDim,CV_32F,feature);
						//out<<i<<" "<<j<<" "<<tmp.label<<endl;
						/*out<<i<<" "<<j<<" "<<tmp.label<<" ";
						for (int k=0;k<labelNum;k++)
						{
							out<<tmp.prob_all[k]<<" ";
						}
						out<<endl;*/
						//	delete []feature;
					}
				}
			
				//for(i=0;i<objectDescriptors->total;i++)
				//{
				//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
				//}
				//out.close();
			}
			char name[50];
			sprintf(name, "_test_%d_%d_%d_%d", times,max_num_of_trees_in_the_forest,max_depth,windowSize);
			string prob_name=ImgName.substr(0,ImgName.length()-4);//(fileName.find_last_of("."))
			prob_name+=name;
			prob_name+="_prob.jpg";
			cout<<prob_name<<endl;
			imwrite(prob_name,probImg);

			dst.release();
			dst_color.release();
			warp_mat.release();
			probImg.release();
		}



		//cvReleaseImage(&img);
	}

}


void RandTree::predict_DepthImgList(string listName)
{
	if (usingCUDA)
	{
		predict_imgList_GPU(listName);
		return;
	}

	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{
		Mat depthImg=Mat::ones(480,640,CV_32FC1);
		depthImg*=-1;
		in.getline(name,500,'\n');
		string depthFileName=name;
		depthFileName=depthFileName.substr(0,depthFileName.length()-4);
		depthFileName+="_depth.txt";
		int x,y;
		double depthValue;
		ifstream in22(depthFileName.c_str(),ios::in);

		while(in22)
		{
			in22>>y>>x>>depthValue;
			if (abs(depthValue)<=1)
			{
				depthImg.at<float>(x,y)=depthValue;
			}
		}

		IplImage *img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorImg=imread(name);

		ImgName=name;
		Mat dst_color;
		Mat AllLabel(img->height,img->width,CV_32F);
		int i,j,k,l;
		Mat tmp,dst;

		Mat m_img=cvarrToMat(img);

		Mat Prob_Img=m_img.clone();
		double width;
		Mat warp_mat(2,3,CV_32F);
		Point current_center;
		current_center.x=m_img.cols/2;
		current_center.y=m_img.rows/2;
		cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;
		for (int times=0;times<1;times++)
		{
			if (times==0)
			{
				dst=depthImg.clone();
				dst_color=colorImg.clone();
			}
			else
			{
				warp_mat=getRotationMatrix2D(current_center,RandDouble_cABc(-9,9), RandDouble_cABc(pow(0.8,5),1));
				//dst=currentImage;
				/// Apply the Affine Transform just found to the src image
				warpAffine( m_img, dst, warp_mat, m_img.size() );
				warpAffine( colorImg, dst_color, warp_mat, m_img.size() );
			}


			float *feature;
			//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
			//if (1)
			{
				int size=windowSize/2;
				CvPoint center;
				string saveName=ImgName.substr(0,ImgName.length()-4);
				string imgName;
				char name[50];
				sprintf(name, "_test_%d_%d", times,windowSize);
				//saveName+=name;
				imgName=saveName;
				saveName+=".txt";
				imgName+=".jpg";
				//imwrite(imgName,dst_color);

				ofstream out(saveName.c_str(),ios::out);

				LabelResult tmp;
				for (i=size;i<dst.cols-size;i++)
				{
					//	cout<<i<<endl;
					for (j=size;j<dst.rows-size;j++)
					{
						center.x=i;
						center.y=j;
						pridict_prob(dst,center,tmp);
						//Mat cdata(1,sampleDim,CV_32F,feature);
						//out<<i<<" "<<j<<" "<<tmp.label<<endl;
						out<<i<<" "<<j<<" "<<tmp.label<<" ";
						for (int k=0;k<labelNum;k++)
						{
							out<<tmp.prob_all[k]<<" ";
						}
						out<<endl;
						//	delete []feature;
					}
				}
				out.close();

				output_probimg(dst_color,saveName);

				//for(i=0;i<objectDescriptors->total;i++)
				//{
				//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
				//}
				//out.close();
			}

			dst.release();
			dst_color.release();
			warp_mat.release();
		}



		cvReleaseImage(&img);
	}

}


void RandTree::predict_imgList(string listName)
{
	if (usingCUDA)
	{
		predict_imgList_GPU(listName);
		return;
	}

	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{
		in.getline(name,500,'\n');
		IplImage *img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorImg=imread(name);

		ImgName=name;
		Mat dst_color;
		Mat AllLabel(img->height,img->width,CV_32F);
		int i,j,k,l;
		Mat tmp,dst;

		Mat m_img=cvarrToMat(img);

		Mat Prob_Img=m_img.clone();
		double width;
		Mat warp_mat(2,3,CV_32F);
		Point current_center;
		current_center.x=m_img.cols/2;
		current_center.y=m_img.rows/2;
		cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;
		for (int times=0;times<1;times++)
		{
			if (times==0)
			{
				dst=m_img.clone();
				dst_color=colorImg.clone();
			}
			else
			{
				warp_mat=getRotationMatrix2D(current_center,RandDouble_cABc(-9,9), RandDouble_cABc(pow(0.8,5),1));
				//dst=currentImage;
				/// Apply the Affine Transform just found to the src image
				warpAffine( m_img, dst, warp_mat, m_img.size() );
				warpAffine( colorImg, dst_color, warp_mat, m_img.size() );
			}


			float *feature;
			//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
			//if (1)
			{
				int size=windowSize;
				CvPoint center;
				string saveName=ImgName.substr(0,ImgName.length()-4);
				string imgName;
				char name[50];
				sprintf(name, "_test_%d_%d", times,windowSize);
				//saveName+=name;
				imgName=saveName;
				saveName+=".txt";
				imgName+=".jpg";
				//imwrite(imgName,dst_color);

				ofstream out(saveName.c_str(),ios::out);

				LabelResult tmp;
				for (i=size;i<dst.cols-size;i++)
				{
					//	cout<<i<<endl;
					for (j=size;j<dst.rows-size;j++)
					{
						center.x=i;
						center.y=j;
						pridict_prob(dst,center,tmp);
						//Mat cdata(1,sampleDim,CV_32F,feature);
						//out<<i<<" "<<j<<" "<<tmp.label<<endl;
						out<<i<<" "<<j<<" "<<tmp.label<<" ";
						for (int k=0;k<labelNum;k++)
						{
							out<<tmp.prob_all[k]<<" ";
						}
						out<<endl;
						//	delete []feature;
					}
				}
				out.close();

				output_probimg(dst_color,saveName);

				//for(i=0;i<objectDescriptors->total;i++)
				//{
				//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
				//}
				//out.close();
			}

			dst.release();
			dst_color.release();
			warp_mat.release();
		}



		cvReleaseImage(&img);
	}
	
}

void RandTree::predict_imgList_fast(string listName,bool writeFile)
{
	if (usingCUDA)
	{
		predict_imgList_GPU(listName);
		return;
	}

	ifstream in(listName.c_str(),ios::in);
	int _imgNum;
	in>>_imgNum;
	char name[500];
	string ImgName;
	in.getline(name,500,'\n');
	for (int ll=0;ll<_imgNum;ll++)
	{
		in.getline(name,500,'\n');
		if (ll<0)
		{
			continue;
		}

		string saveName;
		ofstream out;
		if (writeFile)
		{
			saveName=name;
			saveName=saveName.substr(0,ImgName.length()-4);
			saveName+="_prob.txt";
			out.open(saveName.c_str(),ios::out);
		}
	

		IplImage *img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);
		Mat colorImg=imread(name);

		ImgName=name;
		Mat dst_color;
		Mat AllLabel(img->height,img->width,CV_32F);
		int i,j,k,l;
		Mat tmp,dst;

		Mat m_img=cvarrToMat(img);

		Mat Prob_Img=m_img.clone();
		Mat gradientMap;
		Mat imgGradientMap;
		double width;
		Mat warp_mat(2,3,CV_32F);
		Point current_center;
		current_center.x=m_img.cols/2;
		current_center.y=m_img.rows/2;
		cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;
		for (int times=0;times<1;times++)
		{
			if (times==0)
			{
				dst=m_img.clone();
				dst_color=colorImg.clone();

				if (1)
				{
					CvMat *x_kernel=cvCreateMat(3,3,CV_64FC1);
					CvMat *y_kernel=cvCreateMat(3,3,CV_64FC1);
					//CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=-1;
					//CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=1;
					//CV_MAT_ELEM(*x_kernel,double,1,0)=-2;CV_MAT_ELEM(*x_kernel,double,1,2)=2;
					//CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;

					CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=0;
					CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=0;
					CV_MAT_ELEM(*x_kernel,double,1,0)=-0.5;CV_MAT_ELEM(*x_kernel,double,1,2)=0.5;
					CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;
					cvTranspose(x_kernel,y_kernel);

					Mat g_x,g_y;
					g_x=cvCreateMat(dst.rows,dst.cols,CV_64FC1);
					g_y=cvCreateMat(dst.rows,dst.cols,CV_64FC1);

					Mat d_img=dst.clone();
					d_img.convertTo(d_img,CV_64FC1);

					filter2D(d_img,g_x,g_x.depth(),cvarrToMat(x_kernel));  
					filter2D(d_img,g_y,g_x.depth(),cvarrToMat(y_kernel)); 
					//cvFilter2D(img,g_y,y_kernel,cvPoint(-1,-1));  
					gradientMap=cvCreateMat(dst.rows,dst.cols,CV_64FC1);
					pow(g_x,2.0f,g_x);
					pow(g_y,2.0f,g_y);

					//cout<<g_x.depth()<<" "<<gradientMap.depth()<<endl;
					sqrt(g_x+g_y,gradientMap);

					double meanV=mean(gradientMap).val[0];
					for (int i=0;i<gradientMap.rows;i++)
					{
						for (int j=0;j<gradientMap.cols;j++)
						{
							if (gradientMap.at<double>(i,j)<=meanV)
							{
								gradientMap.at<double>(i,j)=0;
							}
							else
							{
								gradientMap.at<double>(i,j)=1;
							}
						}
					}

					//eliminating the isolated values
					Mat labels=gradientMap.clone()*0;
					vector<Point> pointList;
					vector<Point> pointSaveList;
					int tx,ty;
					//int totalNum;
					for (int i=0;i<gradientMap.rows;i++)
					{
						for (int j=0;j<gradientMap.cols;j++)
						{
							pointList.clear();
							pointSaveList.clear();
							if (gradientMap.at<double>(i,j)==1&&labels.at<double>(i,j)==0)
							{
								//	totalNum=1;
								/*		namedWindow("1");
								imshow("1",gradientMap);
								waitKey();*/

								pointList.push_back(Point(i,j));
								pointSaveList.push_back(Point(i,j));
								labels.at<double>(i,j)=1;
								while (pointList.size()!=0)
								{
									Point tmpP=pointList.at(pointList.size()-1);
									//cout<<tmpP.x<<" "<<tmpP.y<<endl;
									pointList.pop_back();



									tx=tmpP.x-1;
									ty=tmpP.y-1;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x;
									ty=tmpP.y-1;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x+1;
									ty=tmpP.y-1;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x-1;
									ty=tmpP.y;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x+1;
									ty=tmpP.y;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x-1;
									ty=tmpP.y+1;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x;
									ty=tmpP.y+1;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}

									tx=tmpP.x+1;
									ty=tmpP.y+1;
									if (tx>=0&&tx<gradientMap.rows&&ty>=0&&ty<gradientMap.cols&&gradientMap.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
									{
										pointList.push_back(Point(tx,ty));
										labels.at<double>(tx,ty)=1;
										pointSaveList.push_back(Point(tx,ty));
										//	totalNum++;
									}
								}
								if (pointSaveList.size()<100)
								{
									for (int k=0;k<pointSaveList.size();k++)
									{
										Point tmp=pointSaveList.at(k);
										gradientMap.at<double>(tmp.x,tmp.y)=0;
									}
								}

								/*	namedWindow("1");
								imshow("1",gradientMap);
								waitKey();*/
							}


						}
					}
					labels.release();
					
					if(trainStyle==1)
					{
						for (int m=0;m<dst.rows;m++)
						{
							for (int n=0;n<dst.cols;n++)
							{
								dst.at<uchar>(m,n)=gradientMap.at<double>(m,n)*255;
							}
						}
					}
					else if (trainStyle==2)
					{
						imgGradientMap=dst.clone();
						for (int m=0;m<imgGradientMap.rows;m++)
						{
							for (int n=0;n<imgGradientMap.cols;n++)
							{
								imgGradientMap.at<uchar>(m,n)=gradientMap.at<double>(m,n)*255;
							}
						}
					}
				
					//m_img=gradientMap;
				}
			}
			

			//set up the image
			int x,y;
			int i;
			double *prob=new double[labelNum];
			
			int curColorIndex;int currentLabel,currentLabelIndex;
			double blendPara=0.7;

			Mat probImg(colorImg.rows,colorImg.cols*(labelNum+1),colorImg.type());
			for (int i=0;i<m_img.rows;i++)
			{
				for (int j=0;j<m_img.cols;j++)
				{
					//if (j<probImg.cols/2)
					//{
					probImg.at<Vec3b>(i,j)=colorImg.at<Vec3b>(i,j);
					for (int k=0;k<labelNum-1;k++)
					{
						curColorIndex=0;
						/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
						probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
						probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
						y=i;
						x=j;
						probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
						probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);;
						probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);;


					}
					//}

				}
			}

			float *feature;
			//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
			//if (1)
			{
				int size=windowSize;
				CvPoint center;
	
				LabelResult tmp;
				for (i=size;i<dst.cols-size;i++)
				{
					//	cout<<i<<endl;
					for (j=size;j<dst.rows-size;j++)
					{
						center.x=i;
						center.y=j;
						if (trainStyle==0||trainStyle==1)
						{
							pridict_prob(dst,center,tmp);
						}
						else if (trainStyle==2)
						{
							pridict_prob(dst,imgGradientMap,center,tmp);
						}
						
						//eliminate the non-salient high response
						if ((gradientMap.at<double>(j,i)==0&&gradientMap.at<double>(j-1,i)==0&&
							gradientMap.at<double>(j+1,i)==0&&gradientMap.at<double>(j,i+1)==0&&
							gradientMap.at<double>(j+1,i+1)==0&&gradientMap.at<double>(j-1,i+1)==0&&
							gradientMap.at<double>(j,i-1)==0&&
							gradientMap.at<double>(j+1,i-1)==0&&gradientMap.at<double>(j-1,i-1)==0))
						{
							tmp.label=labelNum-1;
							for (int k=0;k<labelNum-1;k++)
							{
								tmp.prob_all[k]=0;
							}
						}
						x=i;
						y=j;
						for (int k=0;k<labelNum-1;k++)
						{
							curColorIndex=tmp.prob_all[k]*1000;
							if (curColorIndex==1000)
							{
								curColorIndex=999;
							}
						
							/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
							probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
							probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
							probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
								colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
							probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
								colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);;
							probImg.at<Vec3b>(y,(k+1)*m_img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
								colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);;


						}

						currentLabelIndex=tmp.label*(1000.0f/(float)(labelNum-1));
						if (currentLabelIndex==1000)
						{
							currentLabelIndex=999;
						}
						probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[0]*(1-blendPara);
						probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[1]*(1-blendPara);
						probImg.at<Vec3b>(y,(labelNum)*m_img.cols+x).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
							colorImg.at<Vec3b>(y,x).val[2]*(1-blendPara);
						//Mat cdata(1,sampleDim,CV_32F,feature);

						if (tmp.prob_all[labelNum-1]<0.6)
						{
							out<<i<<" "<<j<<" "<<tmp.label<<" ";
							for (int k=0;k<labelNum;k++)
							{
								out<<tmp.prob_all[k]<<" ";
							}
							out<<endl;
						}
						//out<<i<<" "<<j<<" "<<tmp.label<<endl;
						/*out<<i<<" "<<j<<" "<<tmp.label<<" ";
						for (int k=0;k<labelNum;k++)
						{
							out<<tmp.prob_all[k]<<" ";
						}
						out<<endl;*/
						//	delete []feature;
					}
				}
			
				//for(i=0;i<objectDescriptors->total;i++)
				//{
				//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
				//}
				//out.close();
			}
			char name[50];
			sprintf(name, "_test_%d_%d_%d_%d", times,max_num_of_trees_in_the_forest,max_depth,windowSize);
			string prob_name=ImgName.substr(0,ImgName.length()-4);//(fileName.find_last_of("."))
			prob_name+=name;
			prob_name+="_prob.jpg";
			cout<<prob_name<<endl;
			imwrite(prob_name,probImg);

			if (writeFile)
			{
				out.close();
			}

			dst.release();
			dst_color.release();
			warp_mat.release();
			probImg.release();
		}



		cvReleaseImage(&img);
	}

}

void RandTree::predict_img_transform(string ImgName,double angle,double scale,bool isshow,bool isflip)
{
	const char *name=ImgName.c_str();
	IplImage *img;
	Mat colorImg;
	if (!isflip)
	{
		img=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);
		colorImg=imread(name);
	}
	else
	{
		IplImage *img1=cvLoadImage(name,CV_LOAD_IMAGE_GRAYSCALE);
		img=cvCreateImage(cvGetSize(img1),img1->depth,img1->nChannels);
		cvFlip(img1,img,1);
		Mat color1=imread(name);
		flip(color1,colorImg,1);
		angle*=-1;
	}
	
	

	Mat dst_color;
	Mat AllLabel(img->height,img->width,CV_32F);
	int i,j,k,l;
	Mat tmp,dst;

	Mat m_img=cvarrToMat(img);

	Mat Prob_Img=m_img.clone();
	double width;
	Mat warp_mat(2,3,CV_32F);
	Point current_center;
	current_center.x=m_img.cols/2;
	current_center.y=m_img.rows/2;
//	cout<<"predicting labels "<<ll<<"/"<<_imgNum<<endl;
	for (int times=1;times<2;times++)
	{
		if (times==0)
		{
			dst=m_img.clone();
			dst_color=colorImg.clone();
		}
		else
		{
			warp_mat=getRotationMatrix2D(current_center,angle, scale);
			//dst=currentImage;
			/// Apply the Affine Transform just found to the src image
			warpAffine( m_img, dst, warp_mat, m_img.size() );
			warpAffine( colorImg, dst_color, warp_mat, m_img.size() );
		}


		float *feature;
		//ifstream in1("F:\\Projects\\Facial feature points detection\\matlab\\imgLabel_mine.txt",ios::in);
		//if (1)
		{
			int size=windowSize;
			CvPoint center;
			string saveName=ImgName.substr(0,ImgName.length()-4);
			string imgName;
			char name[50];
			sprintf(name, "_test %d", times);
			saveName+=name;
			imgName=saveName;
			saveName+=".txt";
			imgName+=".jpg";
			//imwrite(imgName,dst_color);

			ofstream out(saveName.c_str(),ios::out);

			LabelResult tmp;
			for (i=size;i<dst.cols-size;i++)
			{
				//	cout<<i<<endl;
				for (j=size;j<dst.rows-size;j++)
				{
					center.x=i;
					center.y=j;
					pridict_prob(dst,center,tmp);
					//Mat cdata(1,sampleDim,CV_32F,feature);
					//out<<i<<" "<<j<<" "<<tmp.label<<endl;
					out<<i<<" "<<j<<" "<<tmp.label<<" ";
					for (int k=0;k<labelNum;k++)
					{
						out<<tmp.prob_all[k]<<" ";
					}
					out<<endl;
					//	delete []feature;
				}
			}
			out.close();

			output_probimg(dst_color,saveName,isshow);

			//for(i=0;i<objectDescriptors->total;i++)
			//{
			//	out<<pts.at(i).pt.x<<" "<<pts.at(i).pt.y<<" "<<rtrees.predict(m_data.row(i))<<endl;
			//}
			//out.close();
		}

		dst.release();
		dst_color.release();
		warp_mat.release();
	}



	cvReleaseImage(&img);

}

void RandTree::output_probimg(Mat img,string fileName,bool isshow)
{
	/*namedWindow("prob");
	imshow("prob",img);
	waitKey();*/
	
	//Mat probImg(img.rows,img.cols*(labelNum+1),img.type());
	//for (int i=0;i<img.rows;i++)
	//{
	//	for (int j=0;j<img.cols;j++)
	//	{
	//		//if (j<probImg.cols/2)
	//		//{
	//			probImg.at<Vec3b>(i,j)=img.at<Vec3b>(i,j);
	//		//}
	//		
	//	}
	//}

	int x,y;
	int i;
	double *prob=new double[labelNum];
	ifstream in(fileName.c_str(),ios::in);
	int curColorIndex;int currentLabel,currentLabelIndex;
	double blendPara=0.7;

	Mat probImg(img.rows,img.cols*(labelNum+1),img.type());
	for (int i=0;i<img.rows;i++)
	{
		for (int j=0;j<img.cols;j++)
		{
			//if (j<probImg.cols/2)
			//{
			probImg.at<Vec3b>(i,j)=img.at<Vec3b>(i,j);
			for (int k=0;k<labelNum-1;k++)
			{
				curColorIndex=0;
				/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
				probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
				probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
				y=i;
				x=j;
				probImg.at<Vec3b>(y,(k+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
					img.at<Vec3b>(y,x).val[0]*(1-blendPara);
				probImg.at<Vec3b>(y,(k+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
					img.at<Vec3b>(y,x).val[1]*(1-blendPara);;
				probImg.at<Vec3b>(y,(k+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
					img.at<Vec3b>(y,x).val[2]*(1-blendPara);;


			}
			//}

		}
	}

	/*namedWindow("1");
	imshow("1",probImg);
	waitKey();*/


	while(in)
	{
		in>>x>>y>>currentLabel;
		for (i=0;i<labelNum;i++)
		{
			in>>prob[i];
		}
		for (i=0;i<labelNum-1;i++)
		{
			curColorIndex=prob[i]*1000;
			if (curColorIndex==1000)
			{
				curColorIndex=999;
			}
		/*	probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0;
			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0;
			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0;*/
			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[0]=colorIndex[curColorIndex][2]*255.0*blendPara+
				img.at<Vec3b>(y,x).val[0]*(1-blendPara);
			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[1]=colorIndex[curColorIndex][1]*255.0*blendPara+
				img.at<Vec3b>(y,x).val[1]*(1-blendPara);;
			probImg.at<Vec3b>(y,(i+1)*img.cols+x).val[2]=colorIndex[curColorIndex][0]*255.0*blendPara+
				img.at<Vec3b>(y,x).val[2]*(1-blendPara);;


		}

		currentLabelIndex=currentLabel*(1000.0f/(float)(labelNum-1));
		if (currentLabelIndex==1000)
		{
			currentLabelIndex=999;
		}
		probImg.at<Vec3b>(y,(labelNum)*img.cols+x).val[0]=colorIndex[currentLabelIndex][2]*255.0*blendPara+
			img.at<Vec3b>(y,x).val[0]*(1-blendPara);
		probImg.at<Vec3b>(y,(labelNum)*img.cols+x).val[1]=colorIndex[currentLabelIndex][1]*255.0*blendPara+
			img.at<Vec3b>(y,x).val[1]*(1-blendPara);
		probImg.at<Vec3b>(y,(labelNum)*img.cols+x).val[2]=colorIndex[currentLabelIndex][0]*255.0*blendPara+
			img.at<Vec3b>(y,x).val[2]*(1-blendPara);
	}
	delete []prob;


	//char name_withSize[50];
	//sprintf(name_withSize, "_prob.jpg", windowSize);
	string prob_name=fileName.substr(0,fileName.find_last_of("."));//(fileName.find_last_of("."))
	prob_name+="_prob.jpg";
	cout<<prob_name<<endl;
	imwrite(prob_name,probImg);

	if (isshow)
	{
		namedWindow("prob");
		imshow("prob",probImg);
		waitKey();
	}

}

void RandTree::save(int cnum)
{
	char name_withSize[50];
	sprintf(name_withSize, "trainedTree_%d_%d_%d_%d_treeNum_%d.txt", max_num_of_trees_in_the_forest,max_depth,windowSize,labelNum,cnum);
	//string name=path+"trainedTree.txt";
	string name=path+name_withSize;
	out.open(name.c_str(),ios::out);
	out<<labelNum<<endl;
	out<<treeNum<<endl;
	out<<windowSize<<" "<<trainStyle<<endl;
	for (int i=0;i<cnum;i++)
	{
		out<<TREE_START<<endl;
		out<<ROOT_NODE<<endl;
		saveTree(roots[i],out);
		out<<END_TREE<<endl;

		//cout<<TREE_START<<endl;
		//cout<<ROOT_NODE<<endl;
		//outputTree(roots[i]);
		//cout<<END_TREE<<endl;
	}
	out.close();
}

void RandTree::save()
{
	char name_withSize[50];
	sprintf(name_withSize, "trainedTree_%d_%d_%d_%d.txt", max_num_of_trees_in_the_forest,max_depth,windowSize,labelNum);
	//string name=path+"trainedTree.txt";
	string name=path+name_withSize;
	out.open(name.c_str(),ios::out);
	out<<labelNum<<endl;
	out<<treeNum<<endl;
	out<<windowSize<<" "<<trainStyle<<endl;
	for (int i=0;i<treeNum;i++)
	{
		out<<TREE_START<<endl;
		out<<ROOT_NODE<<endl;
		saveTree(roots[i],out);
		out<<END_TREE<<endl;

		//cout<<TREE_START<<endl;
		//cout<<ROOT_NODE<<endl;
		//outputTree(roots[i]);
		//cout<<END_TREE<<endl;
	}
}

void RandTree::save(string name)
{
	//string name=path+"trainedTree.txt";
	out.open(name.c_str(),ios::out);
	out<<labelNum<<endl;
	out<<treeNum<<endl;
	out<<windowSize<<" "<<trainStyle<<endl;
	for (int i=0;i<treeNum;i++)
	{
		out<<TREE_START<<endl;
		out<<ROOT_NODE<<endl;
		saveTree(roots[i],out);
		out<<END_TREE<<endl;
	}
	out.close();
}

void RandTree::load(string name)
{
	in_file.open(name.c_str(),ios::in);
	int tmp;
	int curTreeNum=0;
	in_file>>labelNum>>treeNum>>windowSize>>trainStyle;
	roots=new Node *[treeNum];
	while(in_file)
	{
		in_file>>tmp;
		if (tmp==TREE_START)
		{
			roots[curTreeNum]=new Node;
			loadTree(roots[curTreeNum],in_file);
			curTreeNum++;
		}
	}
	in_file.close();

	cout<<"trained results loaded\n";

	if (usingCUDA)
	{
		setData_RandomizedTrees_combination(max_depth,min_sample_count,regression_accuracy,
			max_num_of_trees_in_the_forest,windowSize,labelNum,roots,treeNum,true);
	}


}


void RandTree::load_prob(string name,int type)
{
	in_file.open(name.c_str(),ios::in);
	int tmp;
	int curTreeNum=0;
	in_file>>labelNum>>treeNum>>windowSize>>trainStyle;
	roots=new Node *[treeNum];
	while(in_file)
	{
		in_file>>tmp;
		if (tmp==TREE_START)
		{
			roots[curTreeNum]=new Node;
			loadTree_prob(roots[curTreeNum],in_file);
			curTreeNum++;
		}
	}
	in_file.close();

	cout<<"labelNum "<<labelNum<<" trained results loaded\n";

	if (usingCUDA)
	{
		setData_RandomizedTrees_combination(max_depth,min_sample_count,regression_accuracy,
			treeNum,windowSize,labelNum,roots,treeNum,true,type);
	}


}

void RandTree::loadTree_prob(Node *node,ifstream &in)
{
	int tmp;
	int mark;
	in_file>>mark;
	//	cout<<mark<<endl;
	if (mark==END_TREE||mark==LEFT_NULL||mark==RIGHT_NULL)
	{
		return;
	}
	Node *cur;
	switch (mark)
	{
	case ROOT_NODE:
	case LEFT_NODE:
	case RIGHT_NODE:
		{
			in_file>>node->nLevel;
			in_file>>node->pos1[0]>>node->pos1[1]>>node->pos2[0]>>node->pos2[1]>>node->threshold;
			in_file>>node->label;
			//	in_file>>tmp;
			//loadTree(node,in_file);
			break;
		}

		//case LEFT_NODE:
		//	{
		//		Node *cur0=new Node;
		//		cur=cur0;
		//		node->l_child=cur;
		//		in_file>>cur->nLevel;
		//		in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1];
		//		//in_file>>tmp;
		//		loadTree(cur,in_file);
		//		break;
		//	}
		//
		//case RIGHT_NODE:
		//	{
		//		Node *cur1=new Node;
		//		cur=cur1;
		//		node->r_child=cur;
		//		in_file>>cur->nLevel;
		//		in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1];
		//		//in_file>>tmp;
		//		loadTree(cur,in_file);
		//		break;
		//	}
		//	
		//case LEFT_NULL:
		//case RIGHT_NULL:
		//	//in_file>>tmp;
		//	{
		//		//loadTree(node,in_file);
		//		return;
		//	}		

	case LEAF_NODE_LEFT:
	case LEAF_NODE_RIGHT:
		{
			cur=node;
			//node->l_child=cur;
			in_file>>cur->nLevel;
			in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1]>>cur->threshold;
			in_file>>cur->label;
			if (cur->num_of_each_class==NULL)
			{
				cur->num_of_each_class=new float[labelNum];
			}
			for (int i=0;i<labelNum;i++)
			{
				in_file>>cur->num_of_each_class[i];
			}
			//loadTree(node,in_file);
			return;
		}
		//case LEAF_NODE_RIGHT:
		//	{
		//		Node *cur3=new Node;
		//		cur=cur3;
		//		node->r_child=cur;
		//		in_file>>cur->nLevel;
		//		in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1];
		//		if (cur->num_of_each_class==NULL)
		//		{
		//			cur->num_of_each_class=new float[labelNum];
		//		}
		//		for (int i=0;i<labelNum;i++)
		//		{
		//			in_file>>cur->num_of_each_class[i];
		//		}
		//		break;
		//	}
	default:
		break;
	}

	Node *l_child=new Node;
	node->l_child=l_child;
	loadTree_prob(l_child,in_file);
	Node *r_child=new Node;
	node->r_child=r_child;
	loadTree_prob(r_child,in_file);

}

void RandTree::outputTree(Node *node)
{
	int i;
	cout<<node->nLevel<<" ";
	cout<<node->pos1[0]<<" "<<node->pos1[1]<<" "<<
		node->pos2[0]<<" "<<node->pos2[1]<<endl;
	if (node->l_child==NULL&&node->r_child==NULL)//leaf
	{
		for (i=0;i<labelNum;i++)
		{
			cout<<node->num_of_each_class[i]<<" ";
		}
		cout<<endl;
	}
	else  //coutput left and right children
	{
		if (node->l_child==NULL)
		{
			cout<<LEFT_NULL<<endl;
		}
		else
		{
			if (node->l_child->l_child==NULL&&node->l_child->r_child==NULL)//left node is a leaf
			{
				cout<<LEAF_NODE_LEFT<<endl;
			}
			else
				cout<<LEFT_NODE<<endl;
			outputTree(node->l_child);
		}
		if (node->r_child==NULL)
		{
			cout<<RIGHT_NULL<<endl;
		}
		else
		{
			if (node->r_child->l_child==NULL&&node->r_child->r_child==NULL)//left node is a leaf
			{
				cout<<LEAF_NODE_RIGHT<<endl;
			}
			else
				cout<<RIGHT_NODE<<endl;
			outputTree(node->r_child);
		}
	}
}


void RandTree::saveTree(Node *node,ofstream &out)
{
	int i;
	out<<node->nLevel<<" ";
	out<<node->pos1[0]<<" "<<node->pos1[1]<<" "<<
		node->pos2[0]<<" "<<node->pos2[1]<<" "<<node->threshold<<endl;
	if (node->l_child==NULL&&node->r_child==NULL)//leaf
	{
		for (i=0;i<labelNum;i++)
		{
			out<<node->num_of_each_class[i]<<" ";
		}
		out<<endl;
	}
	else  //output left and right children
	{
		if (node->l_child==NULL)
		{
			out<<LEFT_NULL<<endl;
		}
		else
		{
			if (node->l_child->l_child==NULL&&node->l_child->r_child==NULL)//left node is a leaf
			{
				out<<LEAF_NODE_LEFT<<endl;
			}
			else
				out<<LEFT_NODE<<endl;
			saveTree(node->l_child,out);
		}
		if (node->r_child==NULL)
		{
			out<<RIGHT_NULL<<endl;
		}
		else
		{
			if (node->r_child->l_child==NULL&&node->r_child->r_child==NULL)//left node is a leaf
			{
				out<<LEAF_NODE_RIGHT<<endl;
			}
			else
				out<<RIGHT_NODE<<endl;
			saveTree(node->r_child,out);
		}
	}
}

void RandTree::loadTree(Node *node,ifstream &in_file)
{
	int tmp;
	int mark;
	in_file>>mark;
//	cout<<mark<<endl;
	if (mark==END_TREE||mark==LEFT_NULL||mark==RIGHT_NULL)
	{
		return;
	}
	Node *cur;
	switch (mark)
	{
		case ROOT_NODE:
		case LEFT_NODE:
		case RIGHT_NODE:
			{
				in_file>>node->nLevel;
				in_file>>node->pos1[0]>>node->pos1[1]>>node->pos2[0]>>node->pos2[1]>>node->threshold;
				//	in_file>>tmp;
				//loadTree(node,in_file);
				break;
			}
			
		//case LEFT_NODE:
		//	{
		//		Node *cur0=new Node;
		//		cur=cur0;
		//		node->l_child=cur;
		//		in_file>>cur->nLevel;
		//		in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1];
		//		//in_file>>tmp;
		//		loadTree(cur,in_file);
		//		break;
		//	}
		//
		//case RIGHT_NODE:
		//	{
		//		Node *cur1=new Node;
		//		cur=cur1;
		//		node->r_child=cur;
		//		in_file>>cur->nLevel;
		//		in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1];
		//		//in_file>>tmp;
		//		loadTree(cur,in_file);
		//		break;
		//	}
		//	
		//case LEFT_NULL:
		//case RIGHT_NULL:
		//	//in_file>>tmp;
		//	{
		//		//loadTree(node,in_file);
		//		return;
		//	}		

		case LEAF_NODE_LEFT:
		case LEAF_NODE_RIGHT:
			{
				cur=node;
				//node->l_child=cur;
				in_file>>cur->nLevel;
				in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1]>>cur->threshold;
				if (cur->num_of_each_class==NULL)
				{
					cur->num_of_each_class=new float[labelNum];
				}
				for (int i=0;i<labelNum;i++)
				{
					in_file>>cur->num_of_each_class[i];
				}
				//loadTree(node,in_file);
				return;
			}
		//case LEAF_NODE_RIGHT:
		//	{
		//		Node *cur3=new Node;
		//		cur=cur3;
		//		node->r_child=cur;
		//		in_file>>cur->nLevel;
		//		in_file>>cur->pos1[0]>>cur->pos1[1]>>cur->pos2[0]>>cur->pos2[1];
		//		if (cur->num_of_each_class==NULL)
		//		{
		//			cur->num_of_each_class=new float[labelNum];
		//		}
		//		for (int i=0;i<labelNum;i++)
		//		{
		//			in_file>>cur->num_of_each_class[i];
		//		}
		//		break;
		//	}
		default:
			break;
	}

	Node *l_child=new Node;
	node->l_child=l_child;
	loadTree(l_child,in_file);
	Node *r_child=new Node;
	node->r_child=r_child;
	loadTree(r_child,in_file);



}


void RandTree::showTree(Node *root,Mat mat,CvPoint &pos)
{
	ababdantsize=10;
	showMaxLayer=7;
	layerNum=new int[showMaxLayer];
	for (int i=0;i<showMaxLayer;i++)
	{
		layerNum[i]=0;
	}
	int subWindowTotalSize=windowSize+ababdantsize*2;
	//cout<<subWindowTotalSize*pow(2.0,showMaxLayer-1)<<endl;
	finalImage=Mat::zeros(subWindowTotalSize*showMaxLayer,subWindowTotalSize*pow(2.0,showMaxLayer-1),CV_8UC3);
	//namedWindow("show part");
	//imshow("show part",finalImage);
	//waitKey();
	showTree_recur(root,mat,pos);

	imwrite("F:\\imgdata\\Rock sec20\\training process\\mouth corner\\tree visualization.jpg",finalImage);

	namedWindow("show part");
	imshow("show part",finalImage);
	waitKey();

}

void RandTree::showTree(Node *root,Mat mat,Mat gradient,CvPoint &pos)
{
	ababdantsize=10;
	showMaxLayer=7;
	layerNum=new int[showMaxLayer];
	for (int i=0;i<showMaxLayer;i++)
	{
		layerNum[i]=0;
	}
	int subWindowTotalSize=windowSize+ababdantsize*2;
	//cout<<subWindowTotalSize*pow(2.0,showMaxLayer-1)<<endl;
	finalImage=Mat::zeros(subWindowTotalSize*showMaxLayer,subWindowTotalSize*pow(2.0,showMaxLayer-1),CV_8UC3);
	//namedWindow("show part");
	//imshow("show part",finalImage);
	//waitKey();
	showTree_recur(root,mat,gradient,pos);

	imwrite("F:\\imgdata\\Rock sec20\\training process\\mouth corner\\tree visualization.jpg",finalImage);

	namedWindow("show part");
	imshow("show part",finalImage);
	waitKey();

}

void RandTree::showTree_recur(Node *root,Mat mat,Mat gradient,CvPoint &pos)
{
	if (root->nLevel>=showMaxLayer)
	{
		return;
	}
	if (root->threshold<0)
	{
		cout<<root->threshold<<" ";
	}
		

	//get the subImage

	int subWindowTotalSize=windowSize+ababdantsize*2;
	
	int mid=(windowSize)/2;
	int leftPos=0-mid;
	int rightPos=windowSize-1-mid;
	//0~windowsize-1 - mid
	//Mat subImage=Mat::zeros(windowSize,windowSize,CV_8UC3);
	//subImage=mat(Range(pos.y+leftPos,pos.x+rightPos),Range(pos.x+leftPos,pos.x+rightPos));
	//cout<<pos.x<<" "<<pos.y<<" "<<leftPos<<" "<<rightPos<<endl;
	Mat cur=mat(Rect(pos.x+leftPos,pos.y+leftPos,windowSize,windowSize)).clone();
	Mat cur_gray;
	Mat tmpImg;
	if (root->threshold>=0)
	{
		cvtColor(mat,cur_gray,CV_RGB2GRAY);
	}
	else
	{
		cur_gray=gradient;
		tmpImg=gradient(Rect(pos.x+leftPos,pos.y+leftPos,windowSize,windowSize));
		cur*=0;
		for (int i=0;i<cur.rows;i++)
		{
			for (int j=0;j<cur.cols;j++)
			{
				cur.at<Vec3b>(i,j).val[0]=cur.at<Vec3b>(i,j).val[1]=cur.at<Vec3b>(i,j).val[2]=tmpImg.at<uchar>(i,j);
			}
		}
	}
	

	//namedWindow("1");
	//imshow("1",finalImage);
	//waitKey();
	
	Point c(root->pos1[0]+mid,root->pos1[1]+mid);
	Point c1(root->pos2[0]+mid,root->pos2[1]+mid);

	int selected=-1;

	if (root->threshold>=0)
	{
		if (cur_gray.at<uchar>(root->pos1[1]+pos.y,root->pos1[0]+pos.x)>
			cur_gray.at<uchar>(root->pos2[1]+pos.y,root->pos2[0]+pos.x)+root->threshold)
		{
			selected=0;
		}
		else
		{
			selected=1;
		}
	}
	else
	{
		if (cur_gray.at<uchar>(root->pos1[1]+pos.y,root->pos1[0]+pos.x)>
			cur_gray.at<uchar>(root->pos2[1]+pos.y,root->pos2[0]+pos.x))
		{
			selected=0;
		}
		else
		{
			selected=1;
		}
	}
	
	

	//Point c(-15+mid,mid);
	//Point c1(root->pos2[0]+mid,root->pos2[1]+mid);
	Scalar s(0,0,255);
	Scalar s66(255,255,0);
	circle(cur,c,4,s);
	circle(cur,c1,2,s66);

	//circle(cur,Point(mid,mid),2,s);

	//cur.at<Vec3b>(c.y,c.x)=0;

	Mat fullSubImage;
	
	fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	fullSubImage(Rect(ababdantsize,ababdantsize,windowSize,windowSize))+=
		cur;
	//if (root->threshold>=0)
	//{
	//	fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	//	fullSubImage(Rect(ababdantsize,ababdantsize,windowSize,windowSize))+=
	//		cur;
	//}
	//else
	//{
	///*namedWindow("1");
	//	imshow("1",cur_gray);
	//	waitKey();*/
	//
	//	fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	//	for (int i=0;i<fullSubImage.rows;i++)
	//	{
	//		for (int j=0;j<fullSubImage.cols;j++ )
	//		{
	//			fullSubImage.at<Vec3b>(i,j).val[0]=tmpImg.at<uchar>(i,j);
	//		}
	//	}
	//}
	

	float center[2];
	//cout<<(pow(2.0,root->nLevel+1)-1)*subWindowTotalSize/2<<" "<<layerNum[root->nLevel]*subWindowTotalSize<<endl;
	//center[0]=finalImage.cols/2-(pow(2.0,root->nLevel)-1)*subWindowTotalSize/2+layerNum[root->nLevel]*subWindowTotalSize;
	//if(root->nLevel<showMaxLayer-1)
		center[0]=subWindowTotalSize*pow(2.0,showMaxLayer-2-root->nLevel)+layerNum[root->nLevel]*pow(2.0,showMaxLayer-1-root->nLevel)*subWindowTotalSize;
	//else
		//center[0]=subWindowTotalSize/2+(layerNum[root->nLevel])*subWindowTotalSize;
	center[1]=(2*root->nLevel+1)*subWindowTotalSize/2;
	//cout<<root->nLevel<<" "<<layerNum[root->nLevel]<<" "<<center[0]<<" "<<center[1]<<endl;
	finalImage(Rect(center[0]-subWindowTotalSize/2,center[1]-subWindowTotalSize/2,subWindowTotalSize,subWindowTotalSize))+=fullSubImage;

	if (root->nLevel<showMaxLayer-1)
	{
		Point s1,s2,s3;
		s1.y=center[1]+windowSize/2+1;
		s1.x=center[0];
		s2.y=s3.y=center[1]+windowSize/2+ababdantsize*2-1;
		s2.x=subWindowTotalSize*pow(2.0,showMaxLayer-3-root->nLevel)+(layerNum[root->nLevel+1])*pow(2.0,showMaxLayer-2-root->nLevel)*subWindowTotalSize;
		s3.x=subWindowTotalSize*pow(2.0,showMaxLayer-3-root->nLevel)+(layerNum[root->nLevel+1]+1)*pow(2.0,showMaxLayer-2-root->nLevel)*subWindowTotalSize;
		//if (s3.x==s2.x)
		//{
		//	cout<<1<<endl;
		//}
		Scalar Choosen_color(120,0,120);
		Scalar s_color(120,120,120);
		if (selected==0)
		{
			line(finalImage,s1,s2,Choosen_color);
		}
		else
		{
			line(finalImage,s1,s2,s_color);
		}
		if (selected==1)
		{
			line(finalImage,s1,s3,Choosen_color);
		}
		else
		{
			line(finalImage,s1,s3,s_color);
		}
	//	cout<<s2.x<<" "<<s3.x<<endl;
		
	}


	layerNum[root->nLevel]++;

	//if(root->nLevel==showMaxLayer-1)
	//cout<<root->nLevel<<" "<<layerNum[root->nLevel]<<endl;	

	//
	//namedWindow("show part");
	//imshow("show part",finalImage);
	//waitKey();

	if (root->l_child!=NULL)
	{
		showTree_recur(root->l_child,mat,gradient,pos);
	}
	else
	{
		layerNum[root->nLevel+1]++;
	}
	if (root->r_child!=NULL)
	{
		showTree_recur(root->r_child,mat,gradient,pos);
	}
	else
	{
		layerNum[root->nLevel+1]++;
	}
	

}

void RandTree::showTree_recur(Node *root,Mat mat,CvPoint &pos)
{
	if (root->nLevel>=showMaxLayer)
	{
		return;
	}

	//get the subImage

	int subWindowTotalSize=windowSize+ababdantsize*2;
	
	int mid=(windowSize)/2;
	int leftPos=0-mid;
	int rightPos=windowSize-1-mid;
	//0~windowsize-1 - mid
	//Mat subImage=Mat::zeros(windowSize,windowSize,CV_8UC3);
	//subImage=mat(Range(pos.y+leftPos,pos.x+rightPos),Range(pos.x+leftPos,pos.x+rightPos));
	//cout<<pos.x<<" "<<pos.y<<" "<<leftPos<<" "<<rightPos<<endl;
	Mat cur=mat(Rect(pos.x+leftPos,pos.y+leftPos,windowSize,windowSize)).clone();
	Mat cur_gray;
	cvtColor(mat,cur_gray,CV_RGB2GRAY);

	/*namedWindow("1");
	imshow("1",cur_gray);
	waitKey();*/
	
	Point c(root->pos1[0]+mid,root->pos1[1]+mid);
	Point c1(root->pos2[0]+mid,root->pos2[1]+mid);

	int selected=-1;

	if (cur_gray.at<uchar>(root->pos1[1]+pos.y,root->pos1[0]+pos.x)>
		cur_gray.at<uchar>(root->pos2[1]+pos.y,root->pos2[0]+pos.x)+root->threshold)
	{
		selected=0;
	}
	else
	{
		selected=1;
	}
	

	//Point c(-15+mid,mid);
	//Point c1(root->pos2[0]+mid,root->pos2[1]+mid);
	Scalar s(0,0,255);
	Scalar s66(255,255,0);
	circle(cur,c,2,s);
	circle(cur,c1,2,s66);

	//circle(cur,Point(mid,mid),2,s);

	//cur.at<Vec3b>(c.y,c.x)=0;

	Mat fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	fullSubImage(Rect(ababdantsize,ababdantsize,windowSize,windowSize))+=
		cur;

	float center[2];
	//cout<<(pow(2.0,root->nLevel+1)-1)*subWindowTotalSize/2<<" "<<layerNum[root->nLevel]*subWindowTotalSize<<endl;
	//center[0]=finalImage.cols/2-(pow(2.0,root->nLevel)-1)*subWindowTotalSize/2+layerNum[root->nLevel]*subWindowTotalSize;
	//if(root->nLevel<showMaxLayer-1)
		center[0]=subWindowTotalSize*pow(2.0,showMaxLayer-2-root->nLevel)+layerNum[root->nLevel]*pow(2.0,showMaxLayer-1-root->nLevel)*subWindowTotalSize;
	//else
		//center[0]=subWindowTotalSize/2+(layerNum[root->nLevel])*subWindowTotalSize;
	center[1]=(2*root->nLevel+1)*subWindowTotalSize/2;
	//cout<<root->nLevel<<" "<<layerNum[root->nLevel]<<" "<<center[0]<<" "<<center[1]<<endl;
	finalImage(Rect(center[0]-subWindowTotalSize/2,center[1]-subWindowTotalSize/2,subWindowTotalSize,subWindowTotalSize))+=fullSubImage;

	if (root->nLevel<showMaxLayer-1)
	{
		Point s1,s2,s3;
		s1.y=center[1]+windowSize/2+1;
		s1.x=center[0];
		s2.y=s3.y=center[1]+windowSize/2+ababdantsize*2-1;
		s2.x=subWindowTotalSize*pow(2.0,showMaxLayer-3-root->nLevel)+(layerNum[root->nLevel+1])*pow(2.0,showMaxLayer-2-root->nLevel)*subWindowTotalSize;
		s3.x=subWindowTotalSize*pow(2.0,showMaxLayer-3-root->nLevel)+(layerNum[root->nLevel+1]+1)*pow(2.0,showMaxLayer-2-root->nLevel)*subWindowTotalSize;
		//if (s3.x==s2.x)
		//{
		//	cout<<1<<endl;
		//}
		Scalar Choosen_color(120,0,120);
		Scalar s_color(120,120,120);
		if (selected==0)
		{
			line(finalImage,s1,s2,Choosen_color);
		}
		else
		{
			line(finalImage,s1,s2,s_color);
		}
		if (selected==1)
		{
			line(finalImage,s1,s3,Choosen_color);
		}
		else
		{
			line(finalImage,s1,s3,s_color);
		}
	//	cout<<s2.x<<" "<<s3.x<<endl;
		
	}


	layerNum[root->nLevel]++;

	//if(root->nLevel==showMaxLayer-1)
	//cout<<root->nLevel<<" "<<layerNum[root->nLevel]<<endl;	

	//
	//namedWindow("show part");
	//imshow("show part",finalImage);
	//waitKey();

	if (root->l_child!=NULL)
	{
		showTree_recur(root->l_child,mat,pos);
	}
	else
	{
		layerNum[root->nLevel+1]++;
	}
	if (root->r_child!=NULL)
	{
		showTree_recur(root->r_child,mat,pos);
	}
	else
	{
		layerNum[root->nLevel+1]++;
	}
	

}

//not very good because we cannot always get the places of interesting points
void RandTree::imageFeaturePts(string imgname)
{
	//const char* filename=imgname.c_str();
	//IplImage* imgRGB = cvLoadImage(filename); 
	//IplImage* imgRGB2 = cvLoadImage(filename); 
	//IplImage* imgGrey = cvLoadImage(filename,CV_LOAD_IMAGE_GRAYSCALE); 

	//Mat img=cvarrToMat(imgGrey);

	//// detecting keypoints
	//SurfFeatureDetector detector(1000);
	//vector<KeyPoint> keypoints1;
	//detector.detect(img, keypoints1);

	//for (int i=0;i<keypoints1.size();i++)
	//{
	//	cvCircle(imgRGB2, keypoints1[i].pt, 3,cvScalar(255,0,0));
	//}

	//cvNamedWindow("SURF", CV_WINDOW_AUTOSIZE );
	//cvShowImage( "SURF", imgRGB2 );
	//cvWaitKey();

	//return;
	//


	//if (imgGrey==NULL){//image validation
	//	cout << "No valid image input."<<endl; 
	//	char c=getchar();
	//	return;
	//} 
	//int w=imgGrey->width;
	//int h=imgGrey->height;

	//IplImage* eig_image = cvCreateImage(cvSize(w, h),IPL_DEPTH_32F, 1);
	//IplImage* temp_image = cvCreateImage(cvSize(w, h),IPL_DEPTH_32F, 1); 

	//const int MAX_CORNERS = 500;//estimate a corner number
	//CvPoint2D32f corners[MAX_CORNERS] = {0};// coordinates of corners
	////CvPoint2D32f* corners = new CvPoint2D32f[ MAX_CORNERS ]; //another method of declaring an array
	//int corner_count = MAX_CORNERS; 
	//double quality_level = 0.01;//threshold for the eigenvalues
	//double min_distance = 5;//minimum distance between two corners
	//int eig_block_size = 5;//window size
	//int use_harris = false;//use 'harris method' or not

	////----------initial guess by cvGoodFeaturesToTrack---------------
	//cvGoodFeaturesToTrack(imgGrey,
	//	eig_image,   // output                 
	//	temp_image,
	//	corners,
	//	&corner_count,
	//	quality_level,
	//	min_distance,
	//	NULL,
	//	eig_block_size,
	//	use_harris);


	//int r=2; //rectangle size
	//int lineWidth=1; // rectangle line width
	////-----draw good feature corners on the original RGB image---------
	//for (int i=0;i<corner_count;i++){
	//	cvRectangle(imgRGB2, cvPoint(corners[i].x-r,corners[i].y-r), 
	//		cvPoint(corners[i].x+r,corners[i].y+r), cvScalar(255,0,0),lineWidth);
	//}

	//int half_win_size=3;//the window size will be 3+1+3=7
	//int iteration=20;
	//double epislon=0.1; 
	//cvFindCornerSubPix(
	//	imgGrey,
	//	corners,
	//	corner_count,
	//	cvSize(half_win_size,half_win_size),
	//	cvSize(-1,-1),//no ignoring the neighbours of the center corner
	//	cvTermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,iteration,epislon)
	//	);

	////------draw subpix corners on another original RGB image------------
	//for (int i=0;i<corner_count;i++){
	//	cvRectangle(imgRGB, cvPoint(corners[i].x-r,corners[i].y-r), 
	//		cvPoint(corners[i].x+r,corners[i].y+r), cvScalar(0,0,255),lineWidth);
	//}

	////to display a coordinate of the third corner
	//cout<<"x="<<corners[2].x;
	//cout<<",y="<<corners[2].y<<endl;

	//cvNamedWindow("cvFindCornerSubPix", CV_WINDOW_AUTOSIZE );
	//cvShowImage( "cvFindCornerSubPix", imgRGB );
	//cvNamedWindow("cvGoodFeaturesToTrack", CV_WINDOW_AUTOSIZE );
	//cvShowImage( "cvGoodFeaturesToTrack", imgRGB2 );
	//cvWaitKey();
}

void RandTree::RGB2YIQ(Mat &src,Mat &dst)
{
	Mat matrix=Mat::ones(3,3,CV_64FC1);
	matrix.at<double>(0,0)=0.299;
	matrix.at<double>(0,1)=0.587;
	matrix.at<double>(0,2)=0.114;
	matrix.at<double>(1,0)=0.596;
	matrix.at<double>(1,1)=-0.274;
	matrix.at<double>(1,2)=-0.322;
	matrix.at<double>(2,0)=0.211;
	matrix.at<double>(2,1)=-0.523;
	matrix.at<double>(2,2)=0.312;

	dst=matrix*src;

}

void RandTree::getGradientMap(Mat &img,Mat &dst)
{
	CvMat *x_kernel=cvCreateMat(3,3,CV_64FC1);
		CvMat *y_kernel=cvCreateMat(3,3,CV_64FC1);
		//CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=-1;
		//CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=1;
		//CV_MAT_ELEM(*x_kernel,double,1,0)=-2;CV_MAT_ELEM(*x_kernel,double,1,2)=2;
		//CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;

		CV_MAT_ELEM(*x_kernel,double,0,0)=CV_MAT_ELEM(*x_kernel,double,2,0)=0;
		CV_MAT_ELEM(*x_kernel,double,0,2)=CV_MAT_ELEM(*x_kernel,double,2,2)=0;
		CV_MAT_ELEM(*x_kernel,double,1,0)=-0.5;CV_MAT_ELEM(*x_kernel,double,1,2)=0.5;
		CV_MAT_ELEM(*x_kernel,double,0,1)=CV_MAT_ELEM(*x_kernel,double,1,1)=CV_MAT_ELEM(*x_kernel,double,2,1)=0;
		cvTranspose(x_kernel,y_kernel);

		Mat g_x,g_y;
		g_x=cvCreateMat(dst.rows,dst.cols,CV_64FC1);
		g_y=cvCreateMat(dst.rows,dst.cols,CV_64FC1);

		Mat d_img=img.clone();
		d_img.convertTo(d_img,CV_64FC1);

		filter2D(d_img,g_x,g_x.depth(),cvarrToMat(x_kernel));  
		filter2D(d_img,g_y,g_x.depth(),cvarrToMat(y_kernel)); 
		//cvFilter2D(img,g_y,y_kernel,cvPoint(-1,-1));  
		Mat gradientMap_double=cvCreateMat(dst.rows,dst.cols,CV_64FC1);
		pow(g_x,2.0f,g_x);
		pow(g_y,2.0f,g_y);

		//cout<<g_x.depth()<<" "<<gradientMap_double.depth()<<endl;
		sqrt(g_x+g_y,gradientMap_double);

		double meanV=mean(gradientMap_double).val[0];
		for (int i=0;i<gradientMap_double.rows;i++)
		{
			for (int j=0;j<gradientMap_double.cols;j++)
			{
				if (gradientMap_double.at<double>(i,j)<=meanV)
				{
					gradientMap_double.at<double>(i,j)=0;
				}
				else
				{
					gradientMap_double.at<double>(i,j)=1;
				}
			}
		}

		//eliminating the isolated values
		Mat labels=gradientMap_double.clone()*0;
		vector<Point> pointList;
		vector<Point> pointSaveList;
		int tx,ty;
		//int totalNum;
		for (int i=0;i<gradientMap_double.rows;i++)
		{
			for (int j=0;j<gradientMap_double.cols;j++)
			{
				pointList.clear();
				pointSaveList.clear();
				if (gradientMap_double.at<double>(i,j)==1&&labels.at<double>(i,j)==0)
				{
				//	totalNum=1;
			/*		namedWindow("1");
					imshow("1",gradientMap_double);
					waitKey();*/

					pointList.push_back(Point(i,j));
					pointSaveList.push_back(Point(i,j));
					labels.at<double>(i,j)=1;
					while (pointList.size()!=0)
					{
						Point tmpP=pointList.at(pointList.size()-1);
						//cout<<tmpP.x<<" "<<tmpP.y<<endl;
						pointList.pop_back();
							
						
						
						tx=tmpP.x-1;
						ty=tmpP.y-1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x;
						ty=tmpP.y-1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x+1;
						ty=tmpP.y-1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x-1;
						ty=tmpP.y;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x+1;
						ty=tmpP.y;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x-1;
						ty=tmpP.y+1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x;
						ty=tmpP.y+1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}

						tx=tmpP.x+1;
						ty=tmpP.y+1;
						if (tx>=0&&tx<gradientMap_double.rows&&ty>=0&&ty<gradientMap_double.cols&&gradientMap_double.at<double>(tx,ty)==1&&labels.at<double>(tx,ty)==0)
						{
							pointList.push_back(Point(tx,ty));
							labels.at<double>(tx,ty)=1;
							pointSaveList.push_back(Point(tx,ty));
						//	totalNum++;
						}
					}
					if (pointSaveList.size()<100)
					{
						for (int k=0;k<pointSaveList.size();k++)
						{
							Point tmp=pointSaveList.at(k);
							gradientMap_double.at<double>(tmp.x,tmp.y)=0;
						}
					}

					/*namedWindow("1");
					imshow("1",gradientMap_double);
					waitKey();*/
				}

			
			}
		}

		
		for (int m=0;m<dst.rows;m++)
		{
			for (int n=0;n<dst.cols;n++)
			{
				dst.at<uchar>(m,n)=gradientMap_double.at<double>(m,n)*255;
			}
		}

}


void RandTree::showTree_both(Node *root,Mat &mat,Mat &gradient,CvPoint &pos)
{
	ababdantsize=10;
	showMaxLayer=7;
	layerNum=new int[showMaxLayer];
	for (int i=0;i<showMaxLayer;i++)
	{
		layerNum[i]=0;
	}
	int subWindowTotalSize=windowSize+ababdantsize*2;
	//cout<<subWindowTotalSize*pow(2.0,showMaxLayer-1)<<endl;
	finalImage=Mat::zeros(subWindowTotalSize*showMaxLayer,subWindowTotalSize*pow(2.0,showMaxLayer-1),CV_8UC3);
	//namedWindow("show part");
	//imshow("show part",finalImage);
	//waitKey();
	showTree_recur_both(root,mat,gradient,pos);

	imwrite("D:\\Fuhao\\exp results\\tree visualization.jpg",finalImage);

	namedWindow("show part");
	imshow("show part",finalImage);
	waitKey();

}

void RandTree::showTree_both_pathOnly(Node *root,Mat &mat,Mat &gradient,CvPoint &pos)
{
	//go over all the nodes until leaf, and visualize the whole path
	Node *current=root;

	vector<Node *> currentNodes;
	currentNodes.push_back(root);

	//cout<<"begin of the tree\n";
	Scalar s(0,0,255);
	Scalar s66(255,255,0);
	Mat totoalImg=Mat::zeros(windowSize,windowSize*max_depth,CV_8UC3);
	while(currentNodes.size()!=0)
	{
		//copy the image
		Mat curImg=gradient(Rect(pos.x-windowSize/2,pos.y-windowSize/2,windowSize,windowSize)).clone();
		cvtColor(curImg,curImg,CV_GRAY2BGR);
	
		current=currentNodes[currentNodes.size()-1];
		currentNodes.pop_back();
		if (current->l_child==NULL&&current->r_child==NULL)//leaf
		{
			//if (current->label>=0)
			{
				LeafNode tmp;
				tmp.leaf_node=current;
				tmp.label=current->label;
				//leafnode.push_back(tmp);
			}
			//break;
		}

		if (trainStyle==1)
		{
			double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
			if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
				gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+current->threshold)
			{

				if (current->l_child==NULL)
				{
					//break;
				}
				else
				{
					//current=current->l_child;
					currentNodes.push_back(current->l_child);
				}

			}
			else
			{

				if (current->r_child==NULL)//leaf
				{
					;//break;
				}
				else
				{
					//current=current->r_child;
					currentNodes.push_back(current->r_child);
				}

			}

			circle(curImg,Point(current->pos1[1]*curDepth+windowSize/2,current->pos1[0]*curDepth+windowSize/2),3,s);
			circle(curImg,Point(current->pos2[1]*curDepth+windowSize/2,current->pos2[0]*curDepth+windowSize/2),1,s66);

			totoalImg(Range(0,windowSize),Range(current->nLevel*windowSize,(current->nLevel+1)*windowSize))+=curImg;

			char name_withSize[50];
			sprintf(name_withSize, "Using depth_%d", current->nLevel);
			namedWindow(name_withSize);
			imshow(name_withSize,curImg);
			waitKey();
		}
		else if (trainStyle==2)
		{
			if (current->threshold>=0)	//depth
			{
				//cout<<current->nLevel<<" depth\n";

				///////////////consider the missing values here//////////////////////
				//go to both sides

				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);

				if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)==0||
					mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)==0)
				{
					if (current->label==1&&current->l_child!=NULL)
					{
						currentNodes.push_back(current->l_child);
					}
					else if (current->label==0&&current->r_child!=NULL)
					{
						currentNodes.push_back(current->r_child);
					}
				}
				else
				{
					if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
						mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)
						//if (mat.at<uchar>(current->pos1[1]+pos.y,current->pos1[0]+pos.x)>
						//	mat.at<uchar>(current->pos2[1]+pos.y,current->pos2[0]+pos.x)+current->threshold)
					{

						if (current->l_child==NULL)
						{
							//break;
						}
						else
						{
							//current=current->l_child;
							currentNodes.push_back(current->l_child);
						}
					}
					else
					{

						if (current->r_child==NULL)//leaf
						{
							;//break;
						}
						else
						{
							//current=current->r_child;
							currentNodes.push_back(current->r_child);
						}
					}
				}

				if (current->nLevel<max_depth&&(current->l_child!=NULL||current->r_child!=NULL))
				{
				
				circle(curImg,Point(current->pos1[1]*curDepth+windowSize/2,current->pos1[0]*curDepth+windowSize/2),3,s);
				circle(curImg,Point(current->pos2[1]*curDepth+windowSize/2,current->pos2[0]*curDepth+windowSize/2),1,s66);

				char name_withSize[50];
				sprintf(name_withSize, "Depth_%d", current->nLevel);

				int fontFace = FONT_HERSHEY_PLAIN;
				putText(curImg,name_withSize,Point(5,windowSize-5),fontFace,0.5,Scalar(0,255,255));


				totoalImg(Range(0,windowSize),Range(current->nLevel*windowSize,(current->nLevel+1)*windowSize))+=curImg;
				namedWindow("totalImg");
				imshow("totalImg",totoalImg);
				waitKey();
				}
				/*char name_withSize[50];
				sprintf(name_withSize, "Depth_%d", current->nLevel);

				int fontFace = FONT_HERSHEY_PLAIN;
				putText(curImg,name_withSize,Point(5,windowSize-5),fontFace,0.5,Scalar(0,255,255));

				namedWindow(name_withSize);
				imshow(name_withSize,curImg);
				waitKey();*/
			}
			else	//color
			{
				//cout<<current->nLevel<<" color\n";
				double curDepth=1.0f/mat.at<float>(pos.y,pos.x);
				if (gradient.at<uchar>(current->pos1[1]*curDepth+pos.y,current->pos1[0]*curDepth+pos.x)>
					gradient.at<uchar>(current->pos2[1]*curDepth+pos.y,current->pos2[0]*curDepth+pos.x)+(-current->threshold-1))
				{

					if (current->l_child==NULL)
					{
						//break;
					}
					else
					{
						//current=current->l_child;
						currentNodes.push_back(current->l_child);
					}
				}
				else
				{

					if (current->r_child==NULL)//leaf
					{
						;//break;
					}
					else
					{
						//current=current->r_child;
						currentNodes.push_back(current->r_child);
					}
				}

				if (current->nLevel<max_depth&&(current->l_child!=NULL||current->r_child!=NULL))
				{
					circle(curImg,Point(current->pos1[1]*curDepth+windowSize/2,current->pos1[0]*curDepth+windowSize/2),3,s);
					circle(curImg,Point(current->pos2[1]*curDepth+windowSize/2,current->pos2[0]*curDepth+windowSize/2),1,s66);
					char name_withSize[50];
					sprintf(name_withSize, "Color_%d", current->nLevel);

					int fontFace = FONT_HERSHEY_PLAIN;
					putText(curImg,name_withSize,Point(5,windowSize-5),fontFace,0.5,Scalar(0,255,255));

					totoalImg(Range(0,windowSize),Range(current->nLevel*windowSize,(current->nLevel+1)*windowSize))+=curImg;
					namedWindow("totalImg");
					imshow("totalImg",totoalImg);
					waitKey();
				}
				

				/*char name_withSize[50];
				sprintf(name_withSize, "Color_%d", current->nLevel);

				int fontFace = FONT_HERSHEY_PLAIN;
				putText(curImg,name_withSize,Point(5,windowSize-5),fontFace,0.5,Scalar(0,255,255));

				namedWindow(name_withSize);
				imshow(name_withSize,curImg);
				waitKey();*/
			}
		}
	}
	int x=rand()%500;
	char name_withSize[50];
	sprintf(name_withSize, "D:\\Fuhao\\exp results\\process tracking\\%d.jpg", x);
	imwrite(name_withSize,totoalImg);

}

void RandTree::showTree_recur_both(Node *root,Mat &mat,Mat &gradient,CvPoint &pos)
{
	if (root->nLevel>=showMaxLayer)
	{
		return;
	}
	if (root->threshold<0)
	{
		cout<<root->threshold<<" ";
	}
		

	//get the subImage

	int subWindowTotalSize=windowSize+ababdantsize*2;
	
	int mid=(windowSize)/2;
	int leftPos=0-mid;
	int rightPos=windowSize-1-mid;
	//0~windowsize-1 - mid
	//Mat subImage=Mat::zeros(windowSize,windowSize,CV_8UC3);
	//subImage=mat(Range(pos.y+leftPos,pos.x+rightPos),Range(pos.x+leftPos,pos.x+rightPos));
	//cout<<pos.x<<" "<<pos.y<<" "<<leftPos<<" "<<rightPos<<endl;
	Mat cur=mat(Rect(pos.x+leftPos,pos.y+leftPos,windowSize,windowSize)).clone();
	Mat cur_gray;
	Mat tmpImg;
	if (root->threshold>=0)
	{
		cvtColor(mat,cur_gray,CV_RGB2GRAY);
	}
	else
	{
		cur_gray=gradient;
		tmpImg=gradient(Rect(pos.x+leftPos,pos.y+leftPos,windowSize,windowSize));
		cur*=0;
		for (int i=0;i<cur.rows;i++)
		{
			for (int j=0;j<cur.cols;j++)
			{
				cur.at<Vec3b>(i,j).val[0]=cur.at<Vec3b>(i,j).val[1]=cur.at<Vec3b>(i,j).val[2]=tmpImg.at<uchar>(i,j);
			}
		}
	}
	

	//namedWindow("1");
	//imshow("1",finalImage);
	//waitKey();
	
	Point c(root->pos1[0]+mid,root->pos1[1]+mid);
	Point c1(root->pos2[0]+mid,root->pos2[1]+mid);

	int selected=-1;

	if (root->threshold>=0)
	{
		if (cur_gray.at<uchar>(root->pos1[1]+pos.y,root->pos1[0]+pos.x)>
			cur_gray.at<uchar>(root->pos2[1]+pos.y,root->pos2[0]+pos.x)+root->threshold)
		{
			selected=0;
		}
		else
		{
			selected=1;
		}
	}
	else
	{
		if (cur_gray.at<uchar>(root->pos1[1]+pos.y,root->pos1[0]+pos.x)>
			cur_gray.at<uchar>(root->pos2[1]+pos.y,root->pos2[0]+pos.x))
		{
			selected=0;
		}
		else
		{
			selected=1;
		}
	}
	
	

	//Point c(-15+mid,mid);
	//Point c1(root->pos2[0]+mid,root->pos2[1]+mid);
	Scalar s(0,0,255);
	Scalar s66(255,255,0);
	circle(cur,c,4,s);
	circle(cur,c1,2,s66);

	//circle(cur,Point(mid,mid),2,s);

	//cur.at<Vec3b>(c.y,c.x)=0;

	Mat fullSubImage;
	
	fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	fullSubImage(Rect(ababdantsize,ababdantsize,windowSize,windowSize))+=
		cur;
	//if (root->threshold>=0)
	//{
	//	fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	//	fullSubImage(Rect(ababdantsize,ababdantsize,windowSize,windowSize))+=
	//		cur;
	//}
	//else
	//{
	///*namedWindow("1");
	//	imshow("1",cur_gray);
	//	waitKey();*/
	//
	//	fullSubImage=Mat::zeros(subWindowTotalSize,subWindowTotalSize,mat.type());
	//	for (int i=0;i<fullSubImage.rows;i++)
	//	{
	//		for (int j=0;j<fullSubImage.cols;j++ )
	//		{
	//			fullSubImage.at<Vec3b>(i,j).val[0]=tmpImg.at<uchar>(i,j);
	//		}
	//	}
	//}
	

	float center[2];
	//cout<<(pow(2.0,root->nLevel+1)-1)*subWindowTotalSize/2<<" "<<layerNum[root->nLevel]*subWindowTotalSize<<endl;
	//center[0]=finalImage.cols/2-(pow(2.0,root->nLevel)-1)*subWindowTotalSize/2+layerNum[root->nLevel]*subWindowTotalSize;
	//if(root->nLevel<showMaxLayer-1)
		center[0]=subWindowTotalSize*pow(2.0,showMaxLayer-2-root->nLevel)+layerNum[root->nLevel]*pow(2.0,showMaxLayer-1-root->nLevel)*subWindowTotalSize;
	//else
		//center[0]=subWindowTotalSize/2+(layerNum[root->nLevel])*subWindowTotalSize;
	center[1]=(2*root->nLevel+1)*subWindowTotalSize/2;
	//cout<<root->nLevel<<" "<<layerNum[root->nLevel]<<" "<<center[0]<<" "<<center[1]<<endl;
	finalImage(Rect(center[0]-subWindowTotalSize/2,center[1]-subWindowTotalSize/2,subWindowTotalSize,subWindowTotalSize))+=fullSubImage;

	if (root->nLevel<showMaxLayer-1)
	{
		Point s1,s2,s3;
		s1.y=center[1]+windowSize/2+1;
		s1.x=center[0];
		s2.y=s3.y=center[1]+windowSize/2+ababdantsize*2-1;
		s2.x=subWindowTotalSize*pow(2.0,showMaxLayer-3-root->nLevel)+(layerNum[root->nLevel+1])*pow(2.0,showMaxLayer-2-root->nLevel)*subWindowTotalSize;
		s3.x=subWindowTotalSize*pow(2.0,showMaxLayer-3-root->nLevel)+(layerNum[root->nLevel+1]+1)*pow(2.0,showMaxLayer-2-root->nLevel)*subWindowTotalSize;
		//if (s3.x==s2.x)
		//{
		//	cout<<1<<endl;
		//}
		Scalar Choosen_color(120,0,120);
		Scalar s_color(120,120,120);
		if (selected==0)
		{
			line(finalImage,s1,s2,Choosen_color);
		}
		else
		{
			line(finalImage,s1,s2,s_color);
		}
		if (selected==1)
		{
			line(finalImage,s1,s3,Choosen_color);
		}
		else
		{
			line(finalImage,s1,s3,s_color);
		}
	//	cout<<s2.x<<" "<<s3.x<<endl;
		
	}


	layerNum[root->nLevel]++;

	//if(root->nLevel==showMaxLayer-1)
	//cout<<root->nLevel<<" "<<layerNum[root->nLevel]<<endl;	

	//
	//namedWindow("show part");
	//imshow("show part",finalImage);
	//waitKey();

	if (root->l_child!=NULL)
	{
		showTree_recur_both(root->l_child,mat,gradient,pos);
	}
	else
	{
		layerNum[root->nLevel+1]++;
	}
	if (root->r_child!=NULL)
	{
		showTree_recur_both(root->r_child,mat,gradient,pos);
	}
	else
	{
		layerNum[root->nLevel+1]++;
	}
	

}


void RandTree::predictPoint(int x,int y,Mat &colorImg,Mat &depthImg,LabelResult &tmp)
{
	int size=windowSize;


//	LabelResult tmp;
	int i=x;int j=y;
	CvPoint center;
	center.x=i;
	center.y=j;

	if (depthImg.at<float>(j,i)==0)
	{
		/*for (int i=0;i<labelNum;i++)
		{
			tmp.prob_all[i]=0;
		}*/
		return;
	}

	//LabelResult resultColor,resultDepth;
	if (trainStyle==0)
	{
		pridict_prob(depthImg,center,tmp);
	}
	else
	{
		pridict_prob(depthImg,colorImg,center,tmp);
	}
}

void RandTree::predict_IMG_Both(Mat &colorIMg,Mat &depthImg,Mat *result,RandTree *rt_color,
	RandTree *rt_depth,int startX/* =0 */,int endX/* =10000 */,int startY/* =0 */,int endY/* =10000 */)
{
	cout<<startX<<" "<<endX<<" "<<startY<<" "<<endY<<endl;
	for (int i=0;i<labelNum-1;i++)
	{
	//	cout<<"label "<<i<<endl;
		result[i]*=0;
	}
	for (int i=windowSize;i<colorIMg.rows-windowSize;i++)
	{
		for (int j=windowSize;j<colorIMg.cols-windowSize;j++)
		{
			if (depthImg.at<float>(i,j)==0)
			{
				continue;
			}
			if (i<startY||i>endY||j<startX||j>endX)
			{
				continue;
			}
			LabelResult tmpColor,tmpDepth;
			rt_color->predictPoint(j,i,colorIMg,depthImg,tmpColor);
			rt_depth->predictPoint(j,i,colorIMg,depthImg,tmpDepth);
			
			for (int k=0;k<labelNum-1;k++)
			{
				result[k].at<float>(i,j)=0.5*(tmpColor.prob_all[k]+tmpDepth.prob_all[k]);
				//result[k].at<float>(i,j)=tmpColor.prob_all[k];
			}
		}
	}
}