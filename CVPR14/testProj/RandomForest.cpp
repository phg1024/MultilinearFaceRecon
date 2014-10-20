#include "RandomForest.h"
#include "GRandom.h"
#define pi 3.1415926

Mat LeafInfo::getLBF()
{
	Mat curLBF=LBF;
	return curLBF;
}

RandomForests::RandomForests(int _sampleNum)
{
	sampleNum=_sampleNum;
	showSingleStep=false;
}

//int leafNum;
void RandomForests::train(vector<ShapePair> &shapes,int ind, float radius,int _treeNum,int depth)
{
	treeNum=_treeNum;
	nodes.resize(treeNum);
	tree_depth=depth;

	vector<int> fullSampleInd;
	for(int i=0;i<shapes.size();i++)
		fullSampleInd.push_back(i);

	//generate a index pair pool
	
	
	genPoints(IndexPairs_global,radius,sampleNum*3);
	difVal_global.resize(shapes.size());
	for(int i=0;i<difVal_global.size();i++)
		difVal_global[i].resize(IndexPairs_global.size());
	for(int i=0;i<difVal_global.size();i++)
	{
		for(int j=0;j<IndexPairs_global.size();j++)
		{
			float f1=shapes[i].getCurFeature(ind,IndexPairs_global[j].u);
			float f2=shapes[i].getCurFeature(ind,IndexPairs_global[j].v);
			difVal_global[i][j]=f1-f2;
		}
		
	//	cout<<" "<<difVal[i]<<" ";
	}
	//cout<<"obtaining features finished for "<<ind<<" ";
	//#pragma omp parallel for
	for(int i=0;i<treeNum;i++)
	{
		//cout<<"tree "<<i<<" out of "<<treeNum<<endl;
		//leafNum=0;
		train_eachTree(shapes,fullSampleInd,ind,radius,0, &nodes[i]);

		//cout<<"leafNum "<<leafNum<<endl;
		//int tmp;
		//cin>>tmp;
	}

	transformFormat();

	IndexPairs_global.clear();
	for(int i=0;i<difVal_global.size();i++)
		difVal_global[i].clear();
	difVal_global.clear();
}

void RandomForests::train_eachTree(vector<ShapePair> &shapes, vector<int> sampleInd, int ind, float radius,int depth, TreeNode* node, Point2f lastMu)
{
	if(depth==tree_depth||sampleInd.size()<3)
	{
		//save the info and return
		node->leafInfo=new LeafInfo;
		node->leafInfo->sampleNum=sampleInd.size();
		Shape s(shapes[0].n);
		for(int i=0;i<sampleInd.size();i++)
		{
			s+=shapes[sampleInd[i]].dS();
			//check
			if(0)
			{
				Mat tmp=shapes[sampleInd[i]].orgImg.clone();
				circle(tmp,shapes[sampleInd[i]].pts[ind],2,Scalar(255));
				circle(tmp,shapes[sampleInd[i]].gtShape.pts[ind],2,Scalar(255),-1);
				imshow("curPts",tmp);
				shapes[sampleInd[i]].visualizePts("curFullPts");
				cout<<s.ptsVec.at<float>(2*ind)<<" "<<s.ptsVec.at<float>(2*ind+1)<<endl;
				cout<<sampleInd[i]<<endl;
				waitKey();
			}
		}
		s=s/sampleInd.size();

		if(0)
		{
			Mat tmp=shapes[sampleInd[0]].orgImg.clone();
			imshow("curPts",tmp);
			cout<<s.ptsVec.at<float>(2*ind)<<" "<<s.ptsVec.at<float>(2*ind+1)<<endl;
			waitKey();
		}
		
		node->leafInfo->ptsDif=Point2f(s.ptsVec.at<float>(2*ind),s.ptsVec.at<float>(2*ind+1));

		//if(depth<tree_depth)
			//cout<<"leaf "<<depth<<" sampleSize: "<<sampleInd.size()<<endl;
		//leafNum++;
		return;
	}


	//generate sampleNum samples from IndexPairs_global
	int fullIndexPairNum=IndexPairs_global.size();
	vector<int> usedIndexPairInd;
	RandSample_V1(fullIndexPairNum,sampleNum,usedIndexPairInd);

	vector<FeaturePair> curIndexPairs;
	for(int i=0;i<usedIndexPairInd.size();i++)
	{
		curIndexPairs.push_back(IndexPairs_global[usedIndexPairInd[i]]);
	}
	//then, regenerate the threshold
	for(int i=0;i<curIndexPairs.size();i++)
	{
		curIndexPairs[i].threshold=RandDouble_c01c()*200;
	}

	vector<float> difVal(sampleNum);
	vector<Point2f> leftMus,rightMus;
	leftMus.resize(sampleNum);rightMus.resize(sampleNum);
	for(int i=0;i<difVal.size();i++)
		difVal[i]=getVars(shapes,ind,sampleInd, curIndexPairs[i],usedIndexPairInd[i],leftMus[i],rightMus[i],lastMu);
	

	//find the one with maximum variance
	int maxInd=findBestSplit(difVal);

	//save current info
	node->index_pair=curIndexPairs[maxInd];
	//split and update shapes
	vector<int> leftNodeIndex,rightNodeIndex;
	getInd(sampleInd,usedIndexPairInd[maxInd],curIndexPairs[maxInd],leftNodeIndex,rightNodeIndex);
	//getInd(shapes,ind,sampleInd,curIndexPairs[maxInd],leftNodeIndex,rightNodeIndex);

	node->left=new TreeNode;
	node->right=new TreeNode;

	//check here
	if(0)
	{
		cout<<leftNodeIndex.size()<<" "<<rightNodeIndex.size()<<endl;
		int num=leftNodeIndex.size();
		num=num<rightNodeIndex.size()?num:rightNodeIndex.size();
		for(int i=0;i<num;i++)
		{
			//imshow("leftImg",shapes[leftNodeIndex[i]].orgImg);
			//imshow("rightImg",shapes[rightNodeIndex[i]].orgImg);

			shapes[leftNodeIndex[i]].visualizePts("leftImg");
			shapes[rightNodeIndex[i]].visualizePts("rightImg");
			waitKey();
		}
	}
	/*for(int i=0;i<curIndexPairs.size();i++)
		if(i%20==0)
			cout<<difVal[i]<<" ";*/
	//cout<<endl;
	//cout<<"depth  "<<depth<<" Left: "<<leftNodeIndex.size()<<" right: "<<rightNodeIndex.size()<<" curDif: "<<difVal[maxInd]<<" "<<curIndexPairs[maxInd].threshold<<endl;
	train_eachTree(shapes,leftNodeIndex,ind,radius,depth+1,node->left,leftMus[maxInd]);
	train_eachTree(shapes,rightNodeIndex,ind,radius,depth+1,node->right,rightMus[maxInd]);
}



void RandomForests::getInd(vector<ShapePair> &shapes,int ptsInd, vector<int> sampleInd, FeaturePair &indexPair,
		vector<int> &leftNodeIndex, vector<int> &rightNodeIndex)
{
	leftNodeIndex.clear();
	rightNodeIndex.clear();

	for(int i=0;i<sampleInd.size();i++)
	{
		int cInd=sampleInd[i];
		bool isLeft=indexPairTest(shapes[cInd],ptsInd,indexPair);

		if(isLeft)
			leftNodeIndex.push_back(cInd);
		else
			rightNodeIndex.push_back(cInd);
	}
}

void RandomForests::getInd(vector<int> sampleInd, int indexPairInd, FeaturePair &indexPair,
		vector<int> &leftNodeIndex, vector<int> &rightNodeIndex)
{
	leftNodeIndex.clear();
	rightNodeIndex.clear();

	for(int i=0;i<sampleInd.size();i++)
	{
		int cInd=sampleInd[i];
		float curDif=difVal_global[cInd][indexPairInd];
		if(curDif>indexPair.threshold)
			leftNodeIndex.push_back(cInd);
		else
			rightNodeIndex.push_back(cInd);
	}
}

int RandomForests::findBestSplit(vector<float> &res)
{
	int resInd=-1;
	float maxVal=0;
	for(int i=0;i<res.size();i++)
	{
		if(maxVal<res[i])
		{
			maxVal=res[i];
			resInd=i;
		}
	}
	return resInd;
}

bool RandomForests::indexPairTest(Shape &s, int ptsInd, FeaturePair &indexPair)
{
	Mat curImg=s.orgImg;
	float f1=s.getCurFeature(ptsInd,indexPair.u);
	float f2=s.getCurFeature(ptsInd,indexPair.v);

	return f1-f2>indexPair.threshold;
}

bool RandomForests::indexPairTest(Mat &img, Shape &s,int ptsInd, TreeNode *node)
{
	Mat curImg=img;

	/*if(showSingleStep)
	{
		Mat cImg=img.clone();

		

		Point2f pts_in=node->index_pair.u;
		Mat RS_local=s.RS_local;

		cout<<pts_in<<endl;
		cout<<RS_local<<endl;

		imshow("cImg1",cImg);
		waitKey();

		float tx_pure=pts_in.x;
		float ty_pure=pts_in.y;
		float ctx=tx_pure*RS_local.at<float>(0,0)+ty_pure*RS_local.at<float>(1,0);
		float cty=tx_pure*RS_local.at<float>(0,1)+ty_pure*RS_local.at<float>(1,1);

		imshow("cImg2",cImg);
		waitKey();

		Point2f inputPts;
		inputPts.x=s.pts[ptsInd].x+ctx;
		inputPts.y=s.pts[ptsInd].y+cty;

		circle(cImg,inputPts,1,Scalar(255));

		imshow("cImg",cImg);
		waitKey();
	}*/
	
	float f1=s.getCurFeature_GivenImg(img, ptsInd,node->index_pair.u);
	float f2=s.getCurFeature_GivenImg(img, ptsInd,node->index_pair.v);

	return f1-f2>node->index_pair.threshold;
}




void RandomForests::genPoints(vector<FeaturePair> &curPoints,float radius, int givenSampleNum)
{
	srand(cv::getTickCount()); 
	curPoints.resize(givenSampleNum);
	//randomly generate points
	for(int i=0;i<givenSampleNum;i++)
	{
		curPoints[i].u=generatePts(radius);
		curPoints[i].v=curPoints[i].u;
		while(curPoints[i].v==curPoints[i].u)
			curPoints[i].v=generatePts(radius);
		curPoints[i].threshold=RandDouble_c01c()*200;//[0,200]
	}

	//check here
	if(0)
	{
		Mat img=refShape->orgImg.clone();
		for(int i=0;i<sampleNum;i++)
		{
			Point cur=refShape->pts[36];
			Point cur1=cur;
			cur.x+=curPoints[i].u.x;
			cur.y+=curPoints[i].u.y;

			cur1.x+=curPoints[i].v.x;
			cur1.y+=curPoints[i].v.y;

			circle(img,cur,2,Scalar(255));
			circle(img,cur1,2,Scalar(255));
		}
		imshow("curSample",img);
		waitKey();
	}
}

Point2f RandomForests::generatePts(float radius)
{
	double a=RandDouble_o01c();
		double b=RandDouble_o01c();
		if(b<a)
		{
			double c=a;
			a=b;
			b=c;
		}
		Point2f u(b*radius*cos(2*pi*a/b),b*radius*sin(2*pi*a/b));	
		return u;
}

float RandomForests::getVar(vector<ShapePair> &shapes,int ind,vector<int> &sampleInd, FeaturePair& indexPair)
{
	//split
	vector<int> leftInd,rightInd;

	//cout<<"checking variance\n";
//	while(1)
	{
		getInd(shapes,ind,sampleInd,indexPair,leftInd,rightInd);

		/*if(leftInd.size()==0||rightInd.size()==0)
			indexPair.threshold=RandDouble_c01c()*200;
		else
			break;*/
	}

	//calculate variance
	float varLeft=getMeanShape(shapes,leftInd,ind);
	float varRight=getMeanShape(shapes,rightInd,ind);

	//cout<<varLeft<<" "<<leftInd.size()<<" "<<varRight<<" "<<rightInd.size()<<" "<<varLeft*leftInd.size()+varRight*rightInd.size()<<endl;
	return varLeft*leftInd.size()+varRight*rightInd.size();
}


float RandomForests::getVars(vector<ShapePair> &shapes,int ptsInd, vector<int> &sampleInd,  FeaturePair& indexPair, int indexPairInd, Point2f&leftMu, Point2f &rightMu, Point2f &parentMu)
{
	//split
	vector<int> leftInd,rightInd;
	
	getInd(sampleInd, indexPairInd, indexPair,
		leftInd, rightInd);

	//calculate variance
	float varLeft=getMeanShape_mu(shapes,leftInd,ptsInd,leftMu);

	float varRight;
	if(parentMu.x==-1000)
		varRight=getMeanShape_mu(shapes,rightInd,ptsInd,rightMu);
	else
	{
		rightMu=parentMu*(float)sampleInd.size()-leftMu*(float)leftInd.size();

		if(rightInd.size()>0)
		{
			rightMu.x/=(float)rightInd.size();
			rightMu.y/=(float)rightInd.size();
			varRight=rightMu.x*rightMu.x+rightMu.y*rightMu.y;
		}
		else
			varRight=0;
	}

	//cout<<varLeft<<" "<<leftInd.size()<<" "<<varRight<<" "<<rightInd.size()<<" "<<varLeft*leftInd.size()+varRight*rightInd.size()<<endl;
	return varLeft*leftInd.size()+varRight*rightInd.size();
}

float RandomForests::getMeanShape(vector<ShapePair> &shapes,vector<int> &ind,int ptsInd)
{
	Point2f pointMean(0,0);
	for(int i=0;i<ind.size();i++)
		pointMean+=shapes[ind[i]].dS().pts[ptsInd];

	if(ind.size()>0)
	{
		pointMean.x/=ind.size();
		pointMean.y/=ind.size();
	}
	
	
	

	return pointMean.x*pointMean.x+pointMean.y*pointMean.y;
}

float RandomForests::getMeanShape_mu(vector<ShapePair> &shapes,vector<int> &ind,int ptsInd, Point2f&mu)
{
	Point2f pointMean(0,0);
	for(int i=0;i<ind.size();i++)
		pointMean+=shapes[ind[i]].dS().pts[ptsInd];

	if(ind.size()>0)
	{
		pointMean.x/=ind.size();
		pointMean.y/=ind.size();
	}

	mu.x=pointMean.x;
	mu.y=pointMean.y;
	return pointMean.x*pointMean.x+pointMean.y*pointMean.y;
}

LBFFeature RandomForests::getLBF(vector<LeafInfo *> &leafInfo)
{
	LBFFeature LBF_Feature;
	LBF_Feature.totalNum=totalLeafNum;
	LBF_Feature.onesInd.clear();
	int curLeafNum=0;
	for(int i=0;i<leafInfo.size();i++)
	{
		LBF_Feature.onesInd.push_back(leafInfo[i]->oneInd+curLeafNum);
		curLeafNum+=LeafNumEachForest[i];
	}
	return LBF_Feature;

}

LBFFeature RandomForests::predict(Mat &img, Shape &s, int ptsInd)
{
	vector<LeafInfo *> leafInfo;
	leafInfo.resize(nodes.size());
	//find the leaf for each tree
	for(int i=0;i<nodes.size();i++)
	{
		if(showSingleStep)
			cout<<"leaves "<<i<<" ";
		LeafInfo *leafNode=getLeaf(img,s,ptsInd,&nodes[i]);
		leafInfo[i]=leafNode;
	}

	//obtain the final LBF
	LBFFeature LBF_Feature=getLBF(leafInfo);
	return LBF_Feature;

	if(showSingleStep)
		cout<<"leaves found\n";

	//and then add the delta if it is fully local
	int totalNum=0;
	Point2d dS(0,0);

	for(int i=0;i<nodes.size();i++)
	{
		int curNum=leafInfo[i]->sampleNum;
		dS.x+=leafInfo[i]->ptsDif.x*curNum;
		dS.y+=leafInfo[i]->ptsDif.y*curNum;
		totalNum+=curNum;
	}
	dS.x/=totalNum;
	dS.y/=totalNum;

	s.ptsVec.at<float>(2*ptsInd)+=dS.x;
	s.ptsVec.at<float>(2*ptsInd+1)+=dS.y;

	//check
	if(0)
	{
		Mat curImg=img.clone();
		circle(curImg,s.pts[ptsInd],2,Scalar(255));
		circle(curImg,Point2f(s.pts[ptsInd].x+dS.x,
			s.pts[ptsInd].y+dS.y),2,Scalar(255),-1);
		imshow("curDs",curImg);
		cout<<s.pts[ptsInd]<<" "<<dS<<endl;
		waitKey();

	}

	s.pts[ptsInd].x+=dS.x;
	s.pts[ptsInd].y+=dS.y;

	
}

LeafInfo *RandomForests::getLeaf(Mat &img, Shape &s, int ptsInd, TreeNode *node)
{
	
	while(node->leafInfo==NULL) //not leaf node
	{
		bool isLeft=indexPairTest(img,s,ptsInd,node);
		
		if(isLeft)
			node=node->left;
		else
			node=node->right;
	}

	return node->leafInfo;
}

void RandomForests::setRefShape(Shape *s)
{
	refShape=s;
}

void RandomForests::transformFormat()
{
	TreeVectors.clear();
	TreeVectors.resize(nodes.size());

	totalLeafNum=0;

	LeafNumEachForest.resize(nodes.size());
	for(int i=0;i<TreeVectors.size();i++)
	{
		vector<TreeNode *> curNodes;
		curNodes.push_back(&nodes[i]);

		//go over all nodes
		while(!curNodes.empty())
		{
			TreeNode *curNode=curNodes.back();
			curNode->curInd=TreeVectors[i].size();
			TreeVectors[i].push_back(curNode);
			curNodes.pop_back();
			if(curNode->left!=NULL)
				curNodes.push_back(curNode->left);
			if(curNode->right!=NULL)
				curNodes.push_back(curNode->right);
		}

		//index all nodes
		//int curLeafNum=0;
		vector<int> leafInd;
		for(int j=0;j<TreeVectors[i].size();j++)
		{
			if(TreeVectors[i][j]->leafInfo!=NULL)
			{
				//TreeVectors[i][j].leafInfo->leafInd=curLeafNum;
				leafInd.push_back(j);
				//curLeafNum++;
			}
		}

		totalLeafNum+=leafInd.size();
		LeafNumEachForest[i]=leafInd.size();
		//set LBF features
		for(int j=0;j<leafInd.size();j++)
		{
			TreeVectors[i][leafInd[j]]->leafInfo->fullNum=leafInd.size();
			TreeVectors[i][leafInd[j]]->leafInfo->oneInd=j;
			Mat LBF=Mat::zeros(1,leafInd.size(),CV_32FC1);
			LBF.at<float>(j)=1;
			TreeVectors[i][leafInd[j]]->leafInfo->LBF=LBF;
		}

		//set ind link
		for(int j=0;j<TreeVectors[i].size();j++)
		{
			if(TreeVectors[i][j]->leafInfo==NULL)
			{
				if(TreeVectors[i][j]->left!=NULL)
					TreeVectors[i][j]->leftInd=TreeVectors[i][j]->left->curInd;
				if(TreeVectors[i][j]->right!=NULL)
					TreeVectors[i][j]->rightInd=TreeVectors[i][j]->right->curInd;
			}
		}
	}


}

void TreeNode::save(ofstream &out)
{
	int isLeaf=0;
	if(leafInfo!=NULL)
		isLeaf=1;
	out.write((char *) &isLeaf,sizeof(int));

	if(isLeaf)
	{
		out.write((char *) &leafInfo->sampleNum,sizeof(int));
		out.write((char *) &leafInfo->ptsDif.x,sizeof(float));
		out.write((char *) &leafInfo->ptsDif.y,sizeof(float));
		out.write((char *) &leafInfo->fullNum,sizeof(int));
		out.write((char *) &leafInfo->oneInd,sizeof(int));
		//out.write((char *) &leafInfo->LBF.cols,sizeof(int));
		//out.write((char *) &leafInfo->LBF.data,sizeof(float)*leafInfo->LBF.cols);
	}
	else
	{
		out.write((char *) &leftInd,sizeof(int));
		out.write((char *) &rightInd,sizeof(int));
		out.write((char *) &index_pair.u.x,sizeof(float));
		out.write((char *) &index_pair.u.y,sizeof(float));
		out.write((char *) &index_pair.v.x,sizeof(float));
		out.write((char *) &index_pair.v.y,sizeof(float));
		out.write((char *) &index_pair.threshold,sizeof(float));
	}
}

void TreeNode::load(ifstream &in)
{
	int isLeaf=0;
	in.read((char *) &isLeaf,sizeof(int));

	if(isLeaf)
	{
		leafInfo=new LeafInfo();
		in.read((char *) &leafInfo->sampleNum,sizeof(int));
		in.read((char *) &leafInfo->ptsDif.x,sizeof(float));
		in.read((char *) &leafInfo->ptsDif.y,sizeof(float));

		in.read((char *) &leafInfo->fullNum,sizeof(int));
		in.read((char *) &leafInfo->oneInd,sizeof(int));

		/*int LBFDims;
		in.read((char *) &LBFDims,sizeof(int));
		leafInfo->LBF=Mat::zeros(1,LBFDims,CV_32FC1);
		in.read((char *) &leafInfo->LBF.data,sizeof(float)*leafInfo->LBF.cols);*/
	}
	else
	{
		in.read((char *) &leftInd,sizeof(int));
		in.read((char *) &rightInd,sizeof(int));
		in.read((char *) &index_pair.u.x,sizeof(float));
		in.read((char *) &index_pair.u.y,sizeof(float));
		in.read((char *) &index_pair.v.x,sizeof(float));
		in.read((char *) &index_pair.v.y,sizeof(float));
		in.read((char *) &index_pair.threshold,sizeof(float));
	}
}

void RandomForests::save(ofstream &out)
{
	//transformFormat();

	int treeTotalNum=TreeVectors.size();
	out.write((char *) &treeTotalNum,sizeof(int));

	for(int i=0;i<treeTotalNum;i++)
	{
		int nodeNum=TreeVectors[i].size();
		out.write((char *) &nodeNum,sizeof(int));

		for(int j=0;j<nodeNum;j++)
		{
			TreeVectors[i][j]->save(out);
		}
	}
}



void RandomForests::load(ifstream &in)
{
	in.read((char *) &treeNum,sizeof(int));
	TreeVectors.resize(treeNum);

	//cout<<treeNum<<" 1 ";
	for(int i=0;i<treeNum;i++)
	{
		int nodeNum;
		in.read((char *) &nodeNum,sizeof(int));
		//cout<<" "<<nodeNum<<" ";
		TreeVectors[i].resize(nodeNum);
		for(int j=0;j<nodeNum;j++)
		{
			TreeVectors[i][j]=new TreeNode;
			TreeVectors[i][j]->load(in);
		}
	}

	//then, restore the tree sturcture
	//set the links
	//cout<<"2 ";
	for(int i=0;i<treeNum;i++)
	{
		//TreeNode *nodePointer=TreeVectors[i][0];
		for(int j=0;j<TreeVectors[i].size();j++)
		{
			if(TreeVectors[i][j]->leafInfo==NULL)
			{
				if(TreeVectors[i][j]->leftInd!=-1)
				{
					TreeVectors[i][j]->left=(TreeVectors[i][TreeVectors[i][j]->leftInd]);
					//cout<<TreeVectors[i][0]+TreeVectors[i][j]->leftInd<<" "<<(TreeVectors[i][TreeVectors[i][j]->leftInd])<<endl;
				}
				if(TreeVectors[i][j]->rightInd!=-1)
					TreeVectors[i][j]->right=(TreeVectors[i][TreeVectors[i][j]->rightInd]);
			}
		}
	}
	//cout<<"3 ";
	nodes.clear();
	nodes.resize(treeNum);
	for(int i=0;i<treeNum;i++)
		nodes[i]=*TreeVectors[i][0];

	//obtain totalLeafNum and LeafNumEachForest

	totalLeafNum=0;
	LeafNumEachForest.resize(nodes.size());
	for(int i=0;i<TreeVectors.size();i++)
	{
		vector<int> leafInd;
		for(int j=0;j<TreeVectors[i].size();j++)
		{
			if(TreeVectors[i][j]->leafInfo!=NULL)
			{
				//TreeVectors[i][j].leafInfo->leafInd=curLeafNum;
				leafInd.push_back(j);
				//curLeafNum++;
			}
		}
		totalLeafNum+=leafInd.size();
		LeafNumEachForest[i]=leafInd.size();
	}
}

void RandomForests::printTrees()
{
	treeNum=nodes.size();
	for(int i=0;i<treeNum;i++)
	{
		printSingleTree(&nodes[i]);
		cout<<endl;
	}
}

void RandomForests::printSingleTree(TreeNode *curNode)
{
	if(curNode->leafInfo==NULL)
	{
		cout<<" ["<<curNode->index_pair.u<<" "<<curNode->index_pair.v<<" "<<curNode->index_pair.threshold<<"] ";
		if(curNode->left!=NULL)
			printSingleTree(curNode->left);
		if(curNode->right!=NULL)
			printSingleTree(curNode->right);
	}
	else
	{
		cout<<" Leaf: ["<< curNode->leafInfo->sampleNum<<" "<<curNode->leafInfo->ptsDif<<" "<<curNode->leafInfo->LBF<<"] ";
	}
}

Mat RandomForests::visualize() //only draw 2 depth
{

	Mat res=Mat::zeros(51*nodes.size(),7*51,CV_8UC3);
	
	for(int i=0;i<nodes.size();i++)
	{
		drawRes(&nodes[i],res,i);
	}

	return res;
}

void RandomForests::drawRes(TreeNode *node,Mat &res,int treeInd)
{
	int unitSize=res.cols/7;
	Point2f c=Point2f((unitSize-1)/2,(unitSize-1)/2);
	Point2f tl=Point2f(0,treeInd*unitSize);
	drawRes_node(node,res,tl, c);

	res.row(treeInd*unitSize).setTo(Scalar(255,255,255));

	if(node->left!=NULL)
		drawRes_node(node->left,res,Point2f(tl.x+1*unitSize,tl.y), c);
	if(node->right!=NULL)
		drawRes_node(node->right,res,Point2f(tl.x+2*unitSize,tl.y), c);

	if(node->left->left!=NULL)
		drawRes_node(node->left->left,res,Point2f(tl.x+3*unitSize,tl.y), c);
	if(node->left->right!=NULL)
		drawRes_node(node->left->right,res,Point2f(tl.x+4*unitSize,tl.y), c);

	if(node->right->left!=NULL)
		drawRes_node(node->right->left,res,Point2f(tl.x+5*unitSize,tl.y), c);
	if(node->right->right!=NULL)
		drawRes_node(node->right->right,res,Point2f(tl.x+6*unitSize,tl.y), c);

}

void RandomForests::drawRes_node(TreeNode *node,Mat &res, Point2f tl, Point2f c)
{
	if(node->leafInfo==NULL)
	{
		Point2f pt1=c+node->index_pair.u;
		Point2f pt2=c+node->index_pair.v;
		circle(res,tl+pt1,2,Scalar(0,255,0));
		circle(res,tl+pt2,2,Scalar(255,255,0));
	}
}