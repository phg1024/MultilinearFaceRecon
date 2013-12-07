#include "geoHashing.h"
#include <fstream>
#include "omp.h"
#include <math.h>
#include "definationCPU.h"
using namespace std;
const float e=2.7173;
string nameList_hash[1800];


distancePir distanceVec[1800],resortVec[1800];

bool comparator_hash ( const distancePir& l, const distancePir& r)
{ return l.first > r.first; }

bool comparator1 ( const distancePir& l, const distancePir& r)
{ return l.first < r.first; }

GeoHashing::GeoHashing(Mat &shapes,float _dsize)
{
	ptsNum=shapes.cols/2;
	basisNum=ptsNum*(ptsNum-1)/2;

	dSize=_dsize;

	basisTabel=new basisPair[basisNum];
	discretWindow=_dsize;
	buildBasisTabel(ptsNum,basisTabel);

	table=new HashTable();
	dataTabel=new vector<vector<Property>>;

	int size[] = {2200, 2200}; 
	keyIndexTable=Mat::zeros(2200,2200,CV_16S)-1;
	smallestX=-1200;
	smallestY=-1200;

	//read in the name
	// Peihong commented out the following block
	/*
	char cname[2000];
	ifstream in("D:\\Fuhao\\face dataset\\lfpw\\train_78\\selected\\imgList.txt",ios::in);
	int totalNum;
	in>>totalNum;
	in.getline(cname,800);
	for (int i=0;i<totalNum;i++)
	{
		in.getline(cname,800);
		nameList_hash[i]=cname;
	}
	*/

	originalTrainingData=shapes;


	//definition here
	rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	


	/*ptsNum=shapes.cols/2;
	shapeNum=shapes.rows;
	data=shapes;
	int num=ptsNum*(ptsNum-1)/2;
	hashTabel=new SingleHashTabel*[shapeNum];

	for (int i=0;i<shapeNum;i++)
	{
		hashTabel[i]=new SingleHashTabel [num];
	}

	disSize=_dsize;*/
}

void GeoHashing::buildBasisTabel(int featureNum,basisPair* pairList)
{
	int cnum;
	cnum=0;
	for (int i=0;i<featureNum-1;i++)
	{
		for (int j=i+1;j<featureNum;j++)
		{
			pairList[cnum].id1=i;
			pairList[cnum].id2=j;
			cnum++;
		}
	}
}
//
void GeoHashing::buildHashTabel(int pairNum,basisPair * pairList,Mat &data)
{
	//buildSingleTabel(data,2,0,0.25);
	shapeNum=data.rows;

	int cind1,cind2;
	for (int i=0;i<basisNum;i++)
	{
		buildSingleTabel(data,i,dSize);
		//cout<<"current entry number: "<<dataTabel->size()<<endl;
	}

	
	

	//GeoHash::iterator iter=table->mappings->begin();
	//map<float,float> basis;
	//map<float,float>::iterator basisIter;
	//for (int i=0;i<dataTabel->size();i++)
	//{
	//	basis=iter->first;
	//	basisIter=basis.begin();
	//	int x_ref=basisIter->first-smallestX;
	//	int y_ref=basisIter->second-smallestY;//<<" "<<tmpEntry[iter->second][0]<<" "<<tmpEntry[iter->second][1]<<endl;
	//	keyIndexTable.at<int>(x_ref,y_ref)=iter->second;
	//	
	////	cout<<x_ref<<" "<<y_ref<<" "<<keyIndexTable.at<int>(x_ref,y_ref)<<" "<<i<<endl;
	//	iter++;

	//	
	//}

	cout<<"hash table built!\n";	

	//cout<<table->mappings->size()<<" "<<dataTabel->size()<<endl;

	//GeoHash::iterator iter=table->mappings->begin();
	//map<float,float> basis;
	//map<float,float>::iterator basisIter;
	//
	//{
	//	basis=iter->first;
	//	basisIter=basis.begin();
	//	cout<<iter->second<<" "<<basisIter->first<<" "<<basisIter->second<<endl;//<<" "<<tmpEntry[iter->second][0]<<" "<<tmpEntry[iter->second][1]<<endl;
	//}
}

void GeoHashing::loadTable(char *name)
{
	ifstream in(name,ios::in);

}
//
void GeoHashing::buildSingleTabel(Mat &data,int basisID,float discretNum)
{
	int id1=basisTabel[basisID].id1;
	int id2=basisTabel[basisID].id2;

	shapeNum=data.rows;
	
	Mat oringin;
	oringin.create(shapeNum,2,CV_64FC1);

	oringin.col(0)=(data.col(id1)+data.col(id2))/2;
	oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
	Mat x=oringin.clone();
	x.col(0)=oringin.col(0)-data.col(id1);
	x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	Mat y=x*rotation;

	/*cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
	cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
	cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
*/
	Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
	int cnum;
	for (int i=0;i<shapeNum;i++)
	{
		if (norm(x.row(i))<0.00000001)	//do not consider the overlapped points
		{
			continue;
		}
		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
			if (j==id1||j==id2)
			{
				continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(i,j)-oringin.at<double>(i,0);
			tmp.at<double>(1,cnum)=data.at<double>(i,j+ptsNum)-oringin.at<double>(i,1);

			//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
			cnum++;
		}

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(i)*tmp;
		newXY.row(1)=y.row(i)*tmp;
		newXY/=(x.at<double>(i,0)*x.at<double>(i,0)+x.at<double>(i,1)*x.at<double>(i,1));


	/*	for (int j=0;j<newXY.cols;j++)
		{
			cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		}
		continue;*/
		//then find the most nearest value to the exact value
		float discretized[2];
		vector<int> usedInd;
		for (int j=0;j<newXY.cols;j++)
		{
			/*cout<<oringin.at<double>(j,0)<<" "<<oringin.at<double>(j,1)<<endl;
			cout<<x.at<double>(j,0)<<" "<<x.at<double>(j,1)<<endl;
			cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;*/
			discretized[0]=discretValue(newXY.at<double>(0,j),discretNum);
			discretized[1]=discretValue(newXY.at<double>(1,j),discretNum);
			//save into the hashTabel

			//try to insert the current location
			bool isInsert=table->insert(discretized);

			//then, add in the value
			Property tmpProperty(i,basisID);
			GeoHash::iterator iter=table->isInside(discretized);
			int cid=iter->second;

			if (cid<dataTabel->size())
			{
					if (find(usedInd.begin(),usedInd.end(),cid)==usedInd.end())
					{
						dataTabel->at(cid).push_back(tmpProperty);		
						usedInd.push_back(cid);
					}
							
			}
			else
			{
				vector<Property> tmpIn;
				tmpIn.push_back(tmpProperty);
				dataTabel->push_back(tmpIn);

			//	tmpEntry[dataTabel->size()-1][0]=discretized[0];
			//	tmpEntry[dataTabel->size()-1][1]=discretized[1];
			}


			////if no position in the hashTabel
			//if (iter==table->mappings->end())
			//{
			//	table->insert(discretized);
			//	int csize=table->mappings->size();
			//	dataTabel.push_back(hashEntry(basisID,i,))
			//}
			//else
			//{
			//	int cnum=iter-table->mappings->begin();
			//	basisTabel->propertyList
			//}
		}
	}


}
//
float GeoHashing::discretValue(float input,float disceteNum)
{
	int num=floor(input/disceteNum);
	float residual=input-num*disceteNum;

	if (residual>disceteNum/2)
	{
		return (num+1)*disceteNum;
	}
	return num*disceteNum;
}

void GeoHashing::saveTable(char *name)
{
	ofstream out(name,ios::out);
	GeoHash::iterator iter=table->mappings->begin();
	map<float,float> basis;
	map<float,float>::iterator basisIter;
	for (int i=0;i<table->mappings->size();i++)
	{
		basis=iter->first;
		basisIter=basis.begin();
		out<<basisIter->first<<" "<<basisIter->second<<endl;//<<" "<<tmpEntry[iter->second][0]<<" "<<tmpEntry[iter->second][1]<<endl;
		//cout<<dataTabel->at(i).size()<<endl;
		
		/*int cind=iter->second;
		for (int j=0;j<dataTabel->at(cind).size();j++)
		{
			out<<dataTabel->at(cind).at(j).ExampleID<<" "<<dataTabel->at(cind).at(j).basisID<<" ";
		}
		out<<endl;*/

		iter++;

		
	}
	out.close();
}

void GeoHashing::showTabel()
{
	GeoHash::iterator iter=table->mappings->begin();
	map<float,float> basis;
	map<float,float>::iterator basisIter;
	for (int i=0;i<dataTabel->size();i++)
	{
		basis=iter->first;
		basisIter=basis.begin();
		cout<<basisIter->first<<" "<<basisIter->second<<endl;;//<<" "<<tmpEntry[iter->second][0]<<" "<<tmpEntry[iter->second][1]<<endl;
		//cout<<dataTabel->at(i).size()<<endl;
	
		int cind=iter->second;
		for (int j=0;j<dataTabel->at(cind).size();j++)
		{
			cout<<dataTabel->at(cind).at(j).ExampleID<<" "<<dataTabel->at(cind).at(j).basisID<<" ";
		}
		cout<<endl;

		iter++;

		continue;
	}
}

void GeoHashing::buildHashTabelVec(int pairNum,basisPair * pairList,Mat &data)
{
	//buildSingleTabel(data,2,0,0.25);

	int cind1,cind2;
	for (int i=0;i<basisNum;i++)
	{
		buildSingleTabelVec(data,i,dSize);
		//cout<<"current entry number: "<<dataTabel->size()<<endl;
	}
	voteNum=new Mat[basisNum];
	cout<<"hash table built!\n";	
}

bool GeoHashing::vote_countAllVec_old(Mat &data,vector<int> &exampleCandidate,vector<int>&KNNID,int thresNum,int nnnum,vector<Point2f>*candidatePts,Mat *img,char *name)
{
//	LONGLONG   t1,t2; 
//	LONGLONG   persecond; 
//	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);

	//vector<int> usedIndex;
	//initialVote(data,usedIndex,candidatePts,img);
	
	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	
	
	for(int i=0;i<basisNum;i++)
		voteNum[i]=Mat::zeros(1,shapeNum,CV_32S);

	Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);
	
//	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);

	#pragma omp parallel for
	for (int i=0;i<basisNum;i++)
	{
		//exampleCandidate.clear();
		

		int id1=basisTabel[i].id1;
		int id2=basisTabel[i].id2;
		Mat oringin;
		oringin.create(1,2,CV_64FC1);

		oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
		Mat x=oringin.clone();
		x.col(0)=oringin.col(0)-data.col(id1);
		x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

		/*if (norm(x)>200)
		{
			continue;
		}*/

		Mat y=x*rotation;

		/*if (i==0)
		{
			cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
			cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
			cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
		}*/
		
		
		Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
		int cnum;

		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
				if (j==id1||j==id2)
			{
			continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(0,j)-oringin.at<double>(0,0);
			tmp.at<double>(1,cnum)=data.at<double>(0,j+ptsNum)-oringin.at<double>(0,1);
			
			cnum++;
		}

	/*	if (i==0)
		{
			cout<<"tmp values:\n";
			for (int l=0;l<tmp.cols;l++)
			{
				cout<<tmp.at<double>(0,l)<<" "<<tmp.at<double>(1,l)<<endl;
			}
		}*/

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(0)*tmp;
		newXY.row(1)=y.row(0)*tmp;
		newXY/=(x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(0,1)*x.at<double>(0,1));

		float discretized[2];
	//	vector<int> usedInd;
		for (int j=0;j<newXY.cols;j++)
		{
			//cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		


			discretized[0]=discretValue(newXY.at<double>(0,j),dSize);
			discretized[1]=discretValue(newXY.at<double>(1,j),dSize);

		/*	if (i==0)
			{
				cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
			}*/
			
			GeoHash::iterator iter=table->isInside(discretized);
			if (iter==table->mappings->end())
			{
				continue;
			}

			int cind=iter->second;

			vector<int> *curLink=dataTabelVec[cind]->at(i);

			if (curLink==NULL)
			{
				continue;
			}

			for (int k=0;k<curLink->size();k++)
			{
				voteNum[i].at<int>(0,curLink->at(k))++;
			}
		}
	}

	for (int i=0;i<basisNum;i++)
	{
		voteAll+=voteNum[i];
	}

	

	//sort and see
	for (int i=0;i<shapeNum;i++)
	{
		distanceVec[i].first=voteAll.at<int>(0,i);
		distanceVec[i].second=i;
	}
	sort(distanceVec,distanceVec+shapeNum,comparator_hash);
	
	//search for best basis for the initial KNN
	int curBestId,curBestNum;
	//int bestNumEach[100];
	int bestBasisEach[100];
	for (int i=0;i<nnnum;i++)
	{
		curBestNum=0;
		for (int j=0;j<basisNum;j++)
		{
			if (voteNum[j].at<int>(0,distanceVec[i].second)>curBestNum)
			{
				curBestId=j;
				curBestNum=voteNum[j].at<int>(0,distanceVec[i].second);
			}
		}
		//bestNumEach[i]=curBestNum;
		bestBasisEach[i]=curBestId;

		//cout<<curBestNum<<endl;
	}
	

	/*cout<<"ind\n";
	for (int i=0;i<50;i++)
	{
		cout<<distanceVec[i].second<<endl;
	}*/

	//reSort(data,bestBasisEach,exampleCandidate,nnnum,img,name,candidatePts);
	bool isS;
	isS=reSort_vec(data,bestBasisEach,exampleCandidate,nnnum,img,name,candidatePts);
	if (!isS)
	{
		return false;
	}

	KNNID.clear();
	for (int i=0;i<nnnum;i++)
	{
		KNNID.push_back(distanceVec[resortVec[i].second].second);
	}
	return true;
}


void GeoHashing::vote_countAllVec(Mat &data,Mat &dataOldFormat,vector<int> &exampleCandidate,vector<int>&KNNID,int thresNum,int nnnum,vector<Point2f>*candidatePts,Mat *img,char *name)
{
//	LONGLONG   t1,t2; 
//	LONGLONG   persecond; 
//	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);

	//vector<int> usedIndex;
	//initialVote(data,usedIndex,candidatePts,img);
	
	//Mat rotation=Mat::zeros(2,2,CV_64FC1);
	//rotation.at<double>(0,0)=0;
	//rotation.at<double>(0,1)=1;
	//rotation.at<double>(1,0)=-1;
	//rotation.at<double>(1,1)=0;	
	Mat *voteNum=new Mat[basisNum];
	for(int i=0;i<basisNum;i++)
		voteNum[i]=Mat::zeros(1,shapeNum,CV_32S);

	/*int **voteNumPointer=new int *[basisNum];
	for (int i=0;i<basisNum;i++)
	{
		voteNumPointer[i]=(int *)voteNum[i].data;
	}*/

	Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);

	//for(int i=0;i<basisNum;i++)
	//	voteNum[i]*=0;
	//voteAll*=0;

//	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);

	#pragma omp parallel for
	for (int i=0;i<basisNum;i++)
	{
		//exampleCandidate.clear();
		

		int id1=basisTabel[i].id1;
		int id2=basisTabel[i].id2;
		Mat oringin=(data.col(id1)+data.col(id2))*0.5;


	
		Mat x=Mat::zeros(2,2,CV_64FC1);
		x.at<double>(0,0)=oringin.at<double>(0,0)-data.at<double>(0,id1);
		x.at<double>(0,1)=oringin.at<double>(1,0)-data.at<double>(1,id1);
		x.at<double>(1,0)=-x.at<double>(0,1);
		x.at<double>(1,1)=x.at<double>(0,0);
		/*if (norm(x)>200)
		{
			continue;
		}*/
	
		
	/*	if (i==0)
		{
			cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(1,0)<<endl;
			cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
			cout<<"y: "<<x.at<double>(1,0)<<" "<<x.at<double>(1,1)<<endl;
		}*/


		Mat tmp=data.clone();
		tmp.row(0)-=oringin.at<double>(0,0);
		tmp.row(1)-=oringin.at<double>(1,0);

		/*if (i==0)
		{
			cout<<"tmp values:\n";
			for (int l=0;l<tmp.cols;l++)
			{
				if (l==id1||l==id2)
				{
					continue;
				}
				cout<<tmp.at<double>(0,l)<<" "<<tmp.at<double>(1,l)<<endl;
			}
		}*/
		//int cnum;

		//cnum=0;
		//for (int j=0;j<ptsNum;j++)
		//{
		//		if (j==id1||j==id2)
		//	{
		//	continue;
		//	}

		//	tmp.at<double>(0,cnum)=data.at<double>(0,j)-oringin.at<double>(0,0);
		//	tmp.at<double>(1,cnum)=data.at<double>(0,j+ptsNum)-oringin.at<double>(0,1);

		//	//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
		//	cnum++;
		//}

		Mat newXY=x*tmp;
		newXY/=(x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(0,1)*x.at<double>(0,1));

		float discretized[2];
		//continue;


	//	vector<int> usedInd;

		//before:0.87, after: 1.86
		//no parrllel: before: 1.25,after: 4.91
		for (int j=0;j<newXY.cols;j++)
		{
			
			if (j==id1||j==id2)
			{
				continue;
			}
		/*	if (i==0)
			{
				cout<<j<<" "<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
			}*/
			

			discretized[0]=discretValue(newXY.at<double>(0,j),dSize);
			discretized[1]=discretValue(newXY.at<double>(1,j),dSize);

			//time consuming

			map<float,float> basis;
			basis.insert(make_pair(discretized[0],discretized[1]));
			GeoHash::iterator iter;
			iter = table->mappings->find(basis);

			//GeoHash::iterator iter=table->isInside(discretized);
			if (iter==table->mappings->end())
			{
				continue;
			}
			
			int cind=iter->second;

			

			vector<int> *curLink=dataTabelVec[cind]->at(i);

			if (curLink==NULL)
			{
				continue;
			}

		
			
			//time consuming
			for (int k=0;k<curLink->size();k++)
			{
				voteNum[i].at<int>(0,curLink->at(k))++;
					//voteNumPointer[i][curLink->at(k)]++;
			}
		}
	}
	
	for (int i=0;i<basisNum;i++)
	{
		voteAll+=voteNum[i];
	}
	//return;
	

	//sort and see
	#pragma omp parallel for
	for (int i=0;i<shapeNum;i++)
	{
		distanceVec[i].first=voteAll.at<int>(0,i);
		distanceVec[i].second=i;
	}
	sort(distanceVec,distanceVec+shapeNum,comparator_hash);

	//search for best basis for the initial KNN
	
	//int bestNumEach[100];
	int bestBasisEach[100];
	int curBestNum;
	int curBestId;
	//#pragma omp parallel for
	for (int i=0;i<nnnum;i++)
	{
		curBestNum=0;		
		for (int j=0;j<basisNum;j++)
		{
			if (voteNum[j].at<int>(0,distanceVec[i].second)>curBestNum)
			{
				curBestId=j;
				curBestNum=voteNum[j].at<int>(0,distanceVec[i].second);
			}
		}
		//bestNumEach[i]=curBestNum;
		bestBasisEach[i]=curBestId;

		//cout<<curBestNum<<endl;
	}
	//return;
	
	/*cout<<"ind\n";
	for (int i=0;i<50;i++)
	{
		cout<<distanceVec[i].second<<endl;
	}*/
	//////->about 0.3 ms
	//reSort(data,bestBasisEach,exampleCandidate,nnnum,img,name,candidatePts);
	reSort_vec(dataOldFormat,bestBasisEach,exampleCandidate,nnnum,img,name,candidatePts);

	KNNID.clear();
	for (int i=0;i<nnnum;i++)
	{
		KNNID.push_back(distanceVec[resortVec[i].second].second);
	}

}



void GeoHashing::buildSingleTabelVec(Mat &data,int basisID,float discretNum)
{
	int id1=basisTabel[basisID].id1;
	int id2=basisTabel[basisID].id2;

	shapeNum=data.rows;
	
	Mat oringin;
	oringin.create(shapeNum,2,CV_64FC1);

	oringin.col(0)=(data.col(id1)+data.col(id2))/2;
	oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
	Mat x=oringin.clone();
	x.col(0)=oringin.col(0)-data.col(id1);
	x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	Mat y=x*rotation;

	/*cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
	cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
	cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
*/
	Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
	int cnum;
	for (int i=0;i<shapeNum;i++)
	{
		if (norm(x.row(i))<0.00000001)	//do not consider the overlapped points
		{
			continue;
		}
		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
			if (j==id1||j==id2)
			{
				continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(i,j)-oringin.at<double>(i,0);
			tmp.at<double>(1,cnum)=data.at<double>(i,j+ptsNum)-oringin.at<double>(i,1);

			//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
			cnum++;
		}

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(i)*tmp;
		newXY.row(1)=y.row(i)*tmp;
		newXY/=(x.at<double>(i,0)*x.at<double>(i,0)+x.at<double>(i,1)*x.at<double>(i,1));


	/*	for (int j=0;j<newXY.cols;j++)
		{
			cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		}
		continue;*/
		//then find the most nearest value to the exact value
		float discretized[2];
		vector<int> usedInd;
		for (int j=0;j<newXY.cols;j++)
		{
			/*cout<<oringin.at<double>(j,0)<<" "<<oringin.at<double>(j,1)<<endl;
			cout<<x.at<double>(j,0)<<" "<<x.at<double>(j,1)<<endl;
			cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;*/
			discretized[0]=discretValue(newXY.at<double>(0,j),discretNum);
			discretized[1]=discretValue(newXY.at<double>(1,j),discretNum);
			//save into the hashTabel

			//try to insert the current location
			bool isInsert=table->insert(discretized);

			//then, add in the value
			Property tmpProperty(i,basisID);
			GeoHash::iterator iter=table->isInside(discretized);
			int cid=iter->second;

			if (cid<dataTabelVec.size())
			{
					if (find(usedInd.begin(),usedInd.end(),cid)==usedInd.end())
					{
						//dataTabel->at(cid).push_back(tmpProperty);	
						if (dataTabelVec[cid]->at(basisID)==NULL)
						{
							dataTabelVec[cid]->at(basisID)=new vector<int>;
						}
						dataTabelVec[cid]->at(basisID)->push_back(i);
						usedInd.push_back(cid);
					}
							
			}
			else
			{
				vector<vector<int>*> *vecForCurPose=new vector<vector<int>*>;
				vecForCurPose->resize(basisNum);
				for (int i=0;i<basisNum;i++)
				{
					vecForCurPose->at(i)=NULL;
				}
				vecForCurPose->at(basisID)=new vector<int>;
				vecForCurPose->at(basisID)->push_back(i);
				dataTabelVec.push_back(vecForCurPose);

			//	tmpEntry[dataTabel->size()-1][0]=discretized[0];
			//	tmpEntry[dataTabel->size()-1][1]=discretized[1];
			}


			////if no position in the hashTabel
			//if (iter==table->mappings->end())
			//{
			//	table->insert(discretized);
			//	int csize=table->mappings->size();
			//	dataTabel.push_back(hashEntry(basisID,i,))
			//}
			//else
			//{
			//	int cnum=iter-table->mappings->begin();
			//	basisTabel->propertyList
			//}
		}
	}


}



void GeoHashing::vote(Mat &data,vector<int> &exampleCandidate,int thresNum)
{
	LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	
	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);

	//#pragma omp parallel for
	for (int i=0;i<basisNum;i++)
	{
		exampleCandidate.clear();
		int voteNum[800];
		for (int j=0;j<shapeNum;j++)
		{
			voteNum[j]=0;
		}

		int id1=basisTabel[i].id1;
		int id2=basisTabel[i].id2;
		Mat oringin;
		oringin.create(1,2,CV_64FC1);

		oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
		Mat x=oringin.clone();
		x.col(0)=oringin.col(0)-data.col(id1);
		x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

	

		Mat y=x*rotation;

		/*cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
		cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
		cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
		*/
		Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
		int cnum;

		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
				if (j==id1||j==id2)
			{
			continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(0,j)-oringin.at<double>(0,0);
			tmp.at<double>(1,cnum)=data.at<double>(0,j+ptsNum)-oringin.at<double>(0,1);

			//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
			cnum++;
		}

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(0)*tmp;
		newXY.row(1)=y.row(0)*tmp;
		newXY/=(x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(0,1)*x.at<double>(0,1));

		//continue;;
		/*	for (int j=0;j<newXY.cols;j++)
		{
		cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		}
		continue;*/
		//then find the most nearest value to the exact value
		//int tmoVal=0;
		float discretized[2];
		for (int j=0;j<newXY.cols;j++)
		{
			//cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		


			discretized[0]=discretValue(newXY.at<double>(0,j),dSize);
			discretized[1]=discretValue(newXY.at<double>(1,j),dSize);

			
			GeoHash::iterator iter=table->isInside(discretized);
			if (iter==table->mappings->end())
			{
				break;
			}

			int cind=iter->second;

			//continue;
			//#pragma omp parallel for
			
			vector<Property> *tmpProp=&(dataTabel->at(cind));
			for (int k=0;k<tmpProp->size();k++)
			{
				if (tmpProp->at(k).basisID==i)
				{
					voteNum[tmpProp->at(k).ExampleID]++;
					//voteNum[6]++;
				}
			}
			

		}
		
		//then, check the number
		for (int j=0;j<shapeNum;j++)
		{
			/*if (j==604)
			{
				cout<<"vote for 604 from basis "<<i<<" :"<<voteNum[j]<<endl;
			}*/
			if (voteNum[j]>thresNum)
			{
				exampleCandidate.push_back(j);

				///////////////align them and see/////////////////////
				if (j==0)
				{
					continue;
				}
				int originalPtsNum=originalTrainingData.cols/2;
				Mat originTrain=oringin.clone();
				Mat curOrigData=originalTrainingData.row(j);
				originTrain.col(0)=(curOrigData.col(id1)+curOrigData.col(id2))/2;
				originTrain.col(1)=(curOrigData.col(id1+ptsNum)+curOrigData.col(id2+ptsNum))/2;

				Mat x_new=originTrain.clone();
				x_new.col(0)=originTrain.col(0)-curOrigData.col(id1);
				x_new.col(1)=originTrain.col(1)-curOrigData.col(id1+ptsNum);

				float theta=acos((x_new.dot(x))/norm(x)/norm(x_new));

				double scale=norm(x)/norm(x_new);
				cout<<"base: "<<i<<"  exampleID: "<<j<<endl;
				//scale=1;

				Mat ttt=curOrigData.clone();
				curOrigData.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
				curOrigData.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
				curOrigData.colRange(0,ptsNum)=(curOrigData.colRange(0,ptsNum)-originTrain.at<double>(0,0))*scale+oringin.at<double>(0,0);
				curOrigData.colRange(ptsNum,ptsNum*2)=(curOrigData.colRange(ptsNum,ptsNum*2)-originTrain.at<double>(0,1))*scale+(oringin.at<double>(0,1));

				Mat img=imread(nameList_hash[j-1]);
				for (int k=0;k<ptsNum;k++)
				{
					circle(img,Point(curOrigData.at<double>(0,k),curOrigData.at<double>(0,k+ptsNum)),5,Scalar(0,0,255));
					circle(img,Point(data.at<double>(0,k),data.at<double>(0,k+ptsNum)),2,Scalar(0,255,0));
				}
				circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),1,Scalar(0,255,0));
				circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),1,Scalar(0,255,0));

				circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),3,Scalar(0,255,0));
				circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),3,Scalar(0,255,0));
				namedWindow("1");
				imshow("1",img);
				waitKey();
			}
		}


	


		//if (exampleCandidate.size()>0)
		//{
		//	cout<<basisTabel[i].id1<<" "<<basisTabel[i].id2<<" ";

		//	for (int n=0;n<exampleCandidate.size();n++)
		//	{
		//		if (exampleCandidate[n]==0)
		//		{
		//			continue;
		//		}
		//		cout<<exampleCandidate[n]<<" "<<nameList_hash[exampleCandidate[n]-1]<<endl;
		//		Mat tmpImg=imread(nameList_hash[exampleCandidate[n]-1]);
		//		namedWindow("1");
		//		imshow("1",tmpImg);
		//		waitKey();
		//	}
		//	cout<<endl;
		//}
		
	}
	QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	double   time=(t2-t1)*1000/persecond; 
	cout<<"total time: "<<time<<"ms "<<"searching time: "<<time/basisNum<<" ms"<<endl;


}

void GeoHashing::initialVote(Mat &data,vector<int> &usedCandidateID,vector<Point> *candidatePts,Mat *_img)
{
	LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	
	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	
	Mat *voteNum=new Mat[basisNum];
	for(int i=0;i<basisNum;i++)
		voteNum[i]=Mat::zeros(1,shapeNum,CV_32S);

	Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);

	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);

	//#pragma omp parallel for
	for (int i=0;i<basisNum;i++)
	{
		int id1=basisTabel[i].id1;
		int id2=basisTabel[i].id2;
		Mat oringin;
		oringin.create(1,2,CV_64FC1);

		oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
		Mat x=oringin.clone();
		x.col(0)=oringin.col(0)-data.col(id1);
		x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

		/*if (norm(x)>200)
		{
			continue;
		}*/

		Mat y=x*rotation;

		/*cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
		cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
		cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
		*/
		Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
		int cnum;

		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
				if (j==id1||j==id2)
			{
			continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(0,j)-oringin.at<double>(0,0);
			tmp.at<double>(1,cnum)=data.at<double>(0,j+ptsNum)-oringin.at<double>(0,1);

			//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
			cnum++;
		}

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(0)*tmp;
		newXY.row(1)=y.row(0)*tmp;
		newXY/=(x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(0,1)*x.at<double>(0,1));

		//continue;;
		/*	for (int j=0;j<newXY.cols;j++)
		{
		cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		}
		continue;*/
		//then find the most nearest value to the exact value
		//int tmoVal=0;
		//cout<<dSize<<endl;
		float discretized[2];
	//	vector<int> usedInd;
		for (int j=0;j<newXY.cols;j++)
		{
			//cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		


			discretized[0]=discretValue(newXY.at<double>(0,j),dSize);
			discretized[1]=discretValue(newXY.at<double>(1,j),dSize);

			
			GeoHash::iterator iter=table->isInside(discretized);
			if (iter==table->mappings->end())
			{
				break;
			}

			int cind=iter->second;

			/*if (find(usedInd.begin(),usedInd.end(),cind)==usedInd.end())
			{
				usedInd.push_back(cind);
			}
			else
			{
				continue;
			}*/
			//cout<<discretized[0]<<" "<<discretized[1]<<" "<<keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY)<<" "<<iter->second<<endl;
		//	if (discretized[0]<smallestX||discretized[1]<smallestY)
		/*	{
				continue;
			}
			int cind=keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY);*/


			//continue;
			//#pragma omp parallel for
			vector<Property> *tmpProp=&(dataTabel->at(cind));
			for (int k=0;k<tmpProp->size();k++)
			{
				if (tmpProp->at(k).basisID==i)
				{
					voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)++;
				/*	if (voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)>1)
					{
						voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)=1;
					}*/
					//break;
					//voteNum[6]++;
				}
			}
			

		}

		//for (int j=0;j<shapeNum;j++)
		//{
		//	if (voteNum[i].at<int>(0,j)>ptsNum-2)
		//	{
		//		//voteNum[i].at<int>(0,j)=ptsNum-2;
		//		cout<<voteNum[i].at<int>(0,j)<<" "<<"basis "<<i<<" shape"<<j<<endl;
		//	}
		//}
		
		continue;		
	}
	QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	double   time=(t2-t1)*1000/persecond; 
	cout<<"total time: "<<time<<"ms "<<"searching time: "<<time/basisNum<<" ms"<<endl;
	

	//////////////////////////////////////////////////
	//search for best basis for each example
	int curBestId,curBestNum;
	int bestNumEach[1000];
	int bestBasisEach[1000];
	for (int i=0;i<shapeNum;i++)
	{
		curBestNum=0;
		for (int j=0;j<basisNum;j++)
		{
			if (voteNum[j].at<int>(0,i)>curBestNum)
			{
				curBestId=j;
				curBestNum=voteNum[j].at<int>(0,i);
			}
		}
		bestNumEach[i]=curBestNum;
		bestBasisEach[i]=curBestId;

		//cout<<curBestNum<<endl;
	}

	//Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);
	for (int i=0;i<basisNum;i++)
	{
		voteAll+=voteNum[i];
	}



	//sort and see
	for (int i=0;i<shapeNum;i++)
	{
		distanceVec[i].first=voteAll.at<int>(0,i);
		distanceVec[i].second=i;
	}
	sort(distanceVec,distanceVec+shapeNum,comparator_hash);

	int nearestInd=0;
	Mat tmp;

	////////////////sort the first 10/////////////////////////
	int nnNum=10;
		float disVec[1000];
	
	for (int i=0;i<nnNum;i++)
	{
		tmp=originalTrainingData.row(distanceVec[i].second).clone();
		alignTools.refineAlign(data,tmp,basisTabel[bestBasisEach[i]].id1,basisTabel[bestBasisEach[i]].id2,disVec[i]);

		//tmp=alignTools.alignedShape;
	/*	Mat img=i_img->clone();
		namedWindow("1");
		for (int j=0;j<ptsNum;j++)
		{
			circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
			circle(img,Point(tmp.at<double>(0,j),tmp.at<double>(0,j+ptsNum)),3,Scalar(255,0,0));
		}
		imshow("1",img);
		waitKey();*/

	}

	//resort
	for (int i=0;i<nnNum;i++)
	{
		resortVec[i].first=disVec[i];
		resortVec[i].second=i;
	}

	cout<<"resort result: \n";
	for (int i=0;i<10;i++)
	{
		cout<<resortVec[i].first<<endl;
	}


	sort(resortVec,resortVec+nnNum,comparator1);

	//get the inlier index
	nearestInd=resortVec[0].second;
	/////////////////////////////////////////////////////



	//align the first example and check the id
	tmp=originalTrainingData.row(distanceVec[nearestInd].second).clone();
	float tmpDis;
	alignTools.refineAlign(data,tmp,basisTabel[bestBasisEach[nearestInd]].id1,basisTabel[bestBasisEach[nearestInd]].id2,tmpDis);

	usedCandidateID.resize(ptsNum);
	for (int i=0;i<ptsNum;i++)
	{
		if (candidatePts[i].size()==1)
		{
			usedCandidateID[i]=0;
		}
		else //select the nearest one
		{
			float minDis=10000000;
			int minID=-1;
			float curDis;
			for (int j=0;j<candidatePts[i].size();j++)
			{
				curDis=(tmp.at<double>(0,j)-candidatePts[i][j].x)*(tmp.at<double>(0,j)-candidatePts[i][j].x)+
					(tmp.at<double>(0,j+ptsNum)-candidatePts[i][j].y)*(tmp.at<double>(0,j+ptsNum)-candidatePts[i][j].y);
				if (curDis<minDis)
				{
					minDis=curDis;
					minID=j;
				}
			}
			usedCandidateID[i]=minID;
		}
	}
	

	for (int i=0;i<ptsNum;i++)
	{
		data.at<double>(0,i)=candidatePts[i][usedCandidateID[i]].x;
		data.at<double>(0,i+ptsNum)=candidatePts[i][usedCandidateID[i]].y;
	}

		Mat img=(*_img).clone();
		for (int i=0;i<ptsNum;i++)
		{
			circle(img,candidatePts[i][usedCandidateID[i]],3,255);
			circle(img,Point(tmp.at<double>(0,i),tmp.at<double>(0,i+ptsNum)),5,Scalar(255,255,0));
		}
		namedWindow("initial align");
		imshow("initial align",img);
		waitKey();
	

}


void GeoHashing::vote_countAll(Mat &data,vector<int> &exampleCandidate,vector<int>&KNNID,int thresNum,int nnnum,vector<Point2f>*candidatePts,Mat *img,char *name)
{
	/*LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);*/

	//vector<int> usedIndex;
	//initialVote(data,usedIndex,candidatePts,img);
	
	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	
	Mat *voteNum=new Mat[basisNum];
	for(int i=0;i<basisNum;i++)
		voteNum[i]=Mat::zeros(1,shapeNum,CV_32S);

	Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);

	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1);

	#pragma omp parallel for
	for (int i=0;i<basisNum;i++)
	{
		//exampleCandidate.clear();
		

		int id1=basisTabel[i].id1;
		int id2=basisTabel[i].id2;
		Mat oringin;
		oringin.create(1,2,CV_64FC1);

		oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
		Mat x=oringin.clone();
		x.col(0)=oringin.col(0)-data.col(id1);
		x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

		/*if (norm(x)>200)
		{
			continue;
		}*/

		Mat y=x*rotation;

		/*cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
		cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
		cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
		*/
		Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
		int cnum;

		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
				if (j==id1||j==id2)
			{
			continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(0,j)-oringin.at<double>(0,0);
			tmp.at<double>(1,cnum)=data.at<double>(0,j+ptsNum)-oringin.at<double>(0,1);

			//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
			cnum++;
		}

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(0)*tmp;
		newXY.row(1)=y.row(0)*tmp;
		newXY/=(x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(0,1)*x.at<double>(0,1));

		//continue;;
		/*	for (int j=0;j<newXY.cols;j++)
		{
		cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		}
		continue;*/
		//then find the most nearest value to the exact value
		//int tmoVal=0;
		//cout<<dSize<<endl;
		float discretized[2];
	//	vector<int> usedInd;
		for (int j=0;j<newXY.cols;j++)
		{
			//cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		


			discretized[0]=discretValue(newXY.at<double>(0,j),dSize);
			discretized[1]=discretValue(newXY.at<double>(1,j),dSize);

			
			GeoHash::iterator iter=table->isInside(discretized);
			if (iter==table->mappings->end())
			{
				break;
			}

			int cind=iter->second;

			/*if (find(usedInd.begin(),usedInd.end(),cind)==usedInd.end())
			{
				usedInd.push_back(cind);
			}
			else
			{
				continue;
			}*/
			//cout<<discretized[0]<<" "<<discretized[1]<<" "<<keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY)<<" "<<iter->second<<endl;
		//	if (discretized[0]<smallestX||discretized[1]<smallestY)
		/*	{
				continue;
			}
			int cind=keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY);*/


			//continue;
			//#pragma omp parallel for
			vector<Property> *tmpProp=&(dataTabel->at(cind));
			for (int k=0;k<tmpProp->size();k++)
			{
				if (tmpProp->at(k).basisID==i)
				{
					voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)++;
				/*	if (voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)>1)
					{
						voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)=1;
					}*/
					//break;
					//voteNum[6]++;
				}
			}
			

		}

		//for (int j=0;j<shapeNum;j++)
		//{
		//	if (voteNum[i].at<int>(0,j)>ptsNum-2)
		//	{
		//		//voteNum[i].at<int>(0,j)=ptsNum-2;
		//		cout<<voteNum[i].at<int>(0,j)<<" "<<"basis "<<i<<" shape"<<j<<endl;
		//	}
		//}
		
		continue;		
	}
	/*QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	double   time=(t2-t1)*1000/persecond; 
	cout<<"total time: "<<time<<"ms "<<"searching time: "<<time/basisNum<<" ms"<<endl;*/


	//char name1[500];
	//int tmppp1=dSize*100;
	//sprintf(name1, "D:\\Fuhao\\Facial feature points detection\\GeoHash\\result comparison\\votes_%d.txt",tmppp1);
	//ofstream outvote(name1,ios::out);
	//for (int i=0;i<basisNum;i++)
	//{
	//	for (int j=0;j<shapeNum;j++)
	//	{
	//		outvote<<voteNum[i].at<int>(0,j)<<" ";
	//	}
	//	outvote<<endl;
	//}
	//outvote.close();

	//analyze the best basis or the best example
	
	//int bestBasis,bestExample;
	//int bestNum=0;
	//for (int i=0;i<basisNum;i++)
	//{
	//	for (int j=0;j<shapeNum;j++)
	//	{
	//		if (voteNum[i].at<int>(0,j)>bestNum)
	//		{
	//			bestNum=voteNum[i].at<int>(0,j);
	//			bestBasis=i;
	//			bestExample=j;
	//		}
	//	}
	//}

	//cout<<"nearest neighbor: "<<bestExample<<" with number "<<bestNum<<endl;
	//Mat img=imread(nameList_hash[bestExample-1]);
	//namedWindow("1");
	//imshow("1",img);
	//waitKey();

	//////////////////////////////////////
	//get the best basis and sort
	//int bestBasis,bestExample;
	//int bestNum=0;
	//for (int i=0;i<basisNum;i++)
	//{
	//	for (int j=0;j<shapeNum;j++)
	//	{
	//		if (voteNum[i].at<int>(0,j)>bestNum)
	//		{
	//			bestNum=voteNum[i].at<int>(0,j);
	//			bestBasis=i;
	//			bestExample=j;
	//		}
	//	}
	//}
	//for (int i=0;i<shapeNum;i++)
	//{
	//	distanceVec[i].first=voteNum[bestBasis].at<int>(0,i);
	//	distanceVec[i].second=i;
	//}
	//sort(distanceVec,distanceVec+shapeNum,comparator);

	//for (int i=0;i<shapeNum;i++)
	//{

	//	cout<<distanceVec[i].first<<" ";
	//	int j=distanceVec[i].second;
	//	if (j==0)
	//	{
	//		continue;
	//	}

	//	//cout<<j<<endl;

	//	Mat img=imread(nameList_hash[j-1]);
	//	namedWindow("1");
	//	imshow("1",img);
	//	waitKey();
	//}
	//////////////////////////////////////
	
	//////////////////////////////////////////////////
	//search for best basis for each example
	int curBestId,curBestNum;
	int bestNumEach[2000];
	int bestBasisEach[2000];
	for (int i=0;i<shapeNum;i++)
	{
		curBestNum=0;
		for (int j=0;j<basisNum;j++)
		{
			if (voteNum[j].at<int>(0,i)>curBestNum)
			{
				curBestId=j;
				curBestNum=voteNum[j].at<int>(0,i);
			}
		}
		bestNumEach[i]=curBestNum;
		bestBasisEach[i]=curBestId;

		//cout<<curBestNum<<endl;
	}

	//cout<<"best num and basisID\n";
	//for (int i=0;i<50;i++)
	//{
	//	cout<<bestNumEach[i]<<" "<<bestBasisEach[i]<<endl;
	//}

	//for (int i=0;i<shapeNum;i++)
	//{
	//	distanceVec[i].first=bestNumEach[i];
	//	distanceVec[i].second=i;
	//}
	//sort(distanceVec,distanceVec+shapeNum,comparator);

	//reSort(data,bestBasisEach);

	//for (int i=0;i<shapeNum;i++)
	//{
	//	int j=distanceVec[resortVec[i].second].second;
	//	if (j==0)
	//	{
	//		continue;
	//	}

	//	cout<<distanceVec[i].first<<" ";

	//	//align the feature and see
	//	int id1=basisTabel[bestBasisEach[i]].id1;
	//	int id2=basisTabel[bestBasisEach[i]].id2;
	//	Mat oringin;
	//	oringin.create(1,2,CV_64FC1);

	//	oringin.col(0)=(data.col(id1)+data.col(id2))/2;
	//	oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;


	//	Mat x=oringin.clone();
	//	x.col(0)=oringin.col(0)-data.col(id1);
	//	x.col(1)=oringin.col(1)-data.col(id1+ptsNum);
	//	Mat y=x*rotation;

	//	int originalPtsNum=originalTrainingData.cols/2;
	//	
	//	Mat originTrain=oringin.clone();
	//	Mat curOrigData=originalTrainingData.row(distanceVec[i].second);
	//	originTrain.col(0)=(curOrigData.col(id1)+curOrigData.col(id2))/2;
	//	originTrain.col(1)=(curOrigData.col(id1+ptsNum)+curOrigData.col(id2+ptsNum))/2;

	//	Mat x_new=originTrain.clone();
	//	x_new.col(0)=originTrain.col(0)-curOrigData.col(id1);
	//	x_new.col(1)=originTrain.col(1)-curOrigData.col(id1+ptsNum);

	//	float theta=acos((x_new.dot(x))/norm(x)/norm(x_new));

	//	double scale=norm(x)/norm(x_new);
	//	cout<<"base: "<<i<<"  exampleID: "<<j<<endl;
	//	//scale=1;

	//	Mat ttt=curOrigData.clone();
	//	curOrigData.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
	//	curOrigData.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
	//	curOrigData.colRange(0,ptsNum)=(curOrigData.colRange(0,ptsNum)-originTrain.at<double>(0,0))*scale+oringin.at<double>(0,0);
	//	curOrigData.colRange(ptsNum,ptsNum*2)=(curOrigData.colRange(ptsNum,ptsNum*2)-originTrain.at<double>(0,1))*scale+(oringin.at<double>(0,1));

	//	//Mat img=imread(nameList_hash[j-1]);
	//	Mat img=imread(inputImgName);
	//	for (int k=0;k<ptsNum;k++)
	//	{
	//		circle(img,Point(curOrigData.at<double>(0,k),curOrigData.at<double>(0,k+ptsNum)),5,Scalar(0,0,255));
	//		circle(img,Point(data.at<double>(0,k),data.at<double>(0,k+ptsNum)),2,Scalar(0,255,0));
	//	}
	//	circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),1,Scalar(0,255,0));
	//	circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),1,Scalar(0,255,0));

	//	circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),3,Scalar(0,255,0));
	//	circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),3,Scalar(0,255,0));

	//	Mat imgCur=imread(nameList_hash[j-1]);
	//	namedWindow("Input");
	//	imshow("Input",img);
	//	namedWindow("Training Example");
	//	imshow("Training Example",imgCur);
	//	waitKey();


	//	/*Mat img=imread(nameList_hash[j-1]);
	//	namedWindow("1");
	//	imshow("1",img);
	//	waitKey();*/
	//}
	////////////////////////////////////////////////////////

	

	//Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);
	for (int i=0;i<basisNum;i++)
	{
		voteAll+=voteNum[i];
	}



	//sort and see
	for (int i=0;i<shapeNum;i++)
	{
		distanceVec[i].first=voteAll.at<int>(0,i);
		distanceVec[i].second=i;
	}
	sort(distanceVec,distanceVec+shapeNum,comparator_hash);

	/*cout<<"ind\n";
	for (int i=0;i<50;i++)
	{
		cout<<distanceVec[i].second<<endl;
	}*/

	reSort(data,bestBasisEach,exampleCandidate,nnnum,img,name,candidatePts);

	KNNID.clear();
	for (int i=0;i<nnnum;i++)
	{
		KNNID.push_back(distanceVec[resortVec[i].second].second);
	}

	//cout<<distanceVec[0].second<<" "<<distanceVec[1].second<<endl;
	//char name[500];
	//int tmppp=dSize*100;
	//cout<<tmppp<<endl;
	//sprintf(name, "D:\\Fuhao\\Facial feature points detection\\GeoHash\\result comparison\\geoHash_%d.txt",tmppp);
	//ofstream out(name,ios::out);
	//
	//out<<nnnum<<endl;
	//for (int i=0;i<nnnum;i++)
	//{
	//	if (distanceVec[resortVec[i].second].second>0)
	//	{
	//		out<<nameList_hash[distanceVec[resortVec[i].second].second-1]<<endl;
	//	}
	//	
	//}
	//out.close();

	////cout<<nameList_hash[14]<<endl;
	//for (int i=0;i<shapeNum;i++)
	//{
	//	int j=distanceVec[i].second;
	//	if (j==0)
	//	{
	//		continue;
	//	}

	//	cout<<j<<endl;

	//	Mat img=imread(nameList_hash[j-1]);
	//	namedWindow("1");
	//	imshow("1",img);
	//	waitKey();
	//}


}


bool GeoHashing::reSort_vec(Mat &detectedFeatures,int *basisID,vector<int> &inlierList,int nnNum,Mat *i_img,char *name,
	vector<Point2f>*candidatePts)
{
	//go over and exclude the bad values
	bool enoughPoints;
	//get the inlier index
	vector<int> inlierSet;
	enoughPoints=alignTools.getInlier(detectedFeatures,originalTrainingData.row(distanceVec[0].second).clone(),
		basisTabel[basisID[0]].id1,basisTabel[basisID[0]].id2,inlierSet,i_img);
	if (!enoughPoints)
	{
		return false;
	}

	int realUsedNum=10;
	float disVec[15];

	
	
	#pragma omp parallel for
	for (int i=0;i<realUsedNum;i++)
	{
		Mat tmp;
		//tmp=originalTrainingData.row(distanceVec[i].second).clone();
		tmp=originalTrainingData.row(distanceVec[i].second);

	/*	if (i==0)
		{
			for (int j=0;j<tmp.cols;j++)
			{
				cout<<tmp.at<double>(0,j)<<" ";
			}
		}*/
		//alignTools.refineAlign(detectedFeatures,tmp,basisTabel[basisID[i]].id1,basisTabel[basisID[i]].id2,disVec[i],i_img);
		alignTools.refineAlign_noChange(detectedFeatures,tmp,inlierSet,disVec[i],i_img);

	/*	if (i==0)
		{
			for (int j=0;j<tmp.cols;j++)
			{
				cout<<tmp.at<double>(0,j)<<" ";
			}
		}*/

		//tmp=alignTools.alignedShape;
	/*	Mat img=i_img->clone();
		namedWindow("current aligned shape");
		for (int j=0;j<ptsNum;j++)
		{
			circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
			circle(img,Point(tmp.at<double>(0,j),tmp.at<double>(0,j+ptsNum)),3,Scalar(255,0,0));
		}
		imshow("current aligned shape",img);
		waitKey();*/

	}

	//resort
	for (int i=0;i<nnNum;i++)
	{
		resortVec[i].first=disVec[i];
		resortVec[i].second=i;
	}



	//only sort the first 10
	sort(resortVec,resortVec+realUsedNum,comparator1);

	/*cout<<"resort result: \n";
	for (int i=0;i<10;i++)
	{
		cout<<resortVec[i].first<<endl;
	}*/

	
	int cind=resortVec[0].second;
	Mat tmp=originalTrainingData.row(distanceVec[cind].second).clone();
	//alignTools.refineAlign(detectedFeatures,tmp,basisTabel[basisID[cind]].id1,basisTabel[basisID[cind]].id2,disVec[cind]);
	alignTools.refineAlign(detectedFeatures,tmp,inlierSet,disVec[cind]);
	

	inlierList.clear();
	float curDis;
	for (int i=0;i<ptsNum;i++)
	{
		float miniumDis=100000;
		int minID=-1;
		for (int j=0;j<candidatePts[i].size();j++)
		{
			curDis=sqrtf((tmp.at<double>(0,i)-candidatePts[i][j].x)*(tmp.at<double>(0,i)-candidatePts[i][j].x)+
				(tmp.at<double>(0,ptsNum+i)-candidatePts[i][j].y)*(tmp.at<double>(0,ptsNum+i)-candidatePts[i][j].y));
			if (curDis<miniumDis)
			{
				miniumDis=curDis;
				minID=j;
			}
		}

		if (miniumDis<5)
		{
			inlierList.push_back(i);
			detectedFeatures.at<double>(0,i)=candidatePts[i][minID].x;
			detectedFeatures.at<double>(0,ptsNum+i)=candidatePts[i][minID].y;
		}		
	}
	return true;
}


void GeoHashing::reSort(Mat &detectedFeatures,int *basisID,vector<int> &inlierList,int nnNum,Mat *i_img,char *name,
	vector<Point2f>*candidatePts)
{
	//go over and exclude the bad values

	//get the inlier index
	vector<int> inlierSet;
	alignTools.getInlier(detectedFeatures,originalTrainingData.row(distanceVec[0].second).clone(),
		basisTabel[basisID[0]].id1,basisTabel[basisID[0]].id2,inlierSet,i_img);

	float disVec[1000];
	Mat tmp;
	for (int i=0;i<nnNum;i++)
	{
		tmp=originalTrainingData.row(distanceVec[i].second).clone();
		//alignTools.refineAlign(detectedFeatures,tmp,basisTabel[basisID[i]].id1,basisTabel[basisID[i]].id2,disVec[i],i_img);
		alignTools.refineAlign(detectedFeatures,tmp,inlierSet,disVec[i],i_img);

		//tmp=alignTools.alignedShape;
	/*	Mat img=i_img->clone();
		namedWindow("current aligned shape");
		for (int j=0;j<ptsNum;j++)
		{
			circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
			circle(img,Point(tmp.at<double>(0,j),tmp.at<double>(0,j+ptsNum)),3,Scalar(255,0,0));
		}
		imshow("current aligned shape",img);
		waitKey();*/

	}

	//resort
	for (int i=0;i<nnNum;i++)
	{
		resortVec[i].first=disVec[i];
		resortVec[i].second=i;
	}



	//only sort the first 10
	sort(resortVec,resortVec+10,comparator1);

	/*cout<<"resort result: \n";
	for (int i=0;i<10;i++)
	{
		cout<<resortVec[i].first<<endl;
	}*/

	
	int cind=resortVec[0].second;
	tmp=originalTrainingData.row(distanceVec[cind].second).clone();
	//alignTools.refineAlign(detectedFeatures,tmp,basisTabel[basisID[cind]].id1,basisTabel[basisID[cind]].id2,disVec[cind]);
	alignTools.refineAlign(detectedFeatures,tmp,inlierSet,disVec[cind]);
	

	inlierList.clear();
	float curDis;
	for (int i=0;i<ptsNum;i++)
	{
		float miniumDis=100000;
		int minID=-1;
		for (int j=0;j<candidatePts[i].size();j++)
		{
			curDis=sqrtf((tmp.at<double>(0,i)-candidatePts[i][j].x)*(tmp.at<double>(0,i)-candidatePts[i][j].x)+
				(tmp.at<double>(0,ptsNum+i)-candidatePts[i][j].y)*(tmp.at<double>(0,ptsNum+i)-candidatePts[i][j].y));
			if (curDis<miniumDis)
			{
				miniumDis=curDis;
				minID=j;
			}
		}

		if (miniumDis<20)
		{
			inlierList.push_back(i);
			detectedFeatures.at<double>(0,i)=candidatePts[i][minID].x;
			detectedFeatures.at<double>(0,ptsNum+i)=candidatePts[i][minID].y;
		}		
	}

	
	//namedWindow("1");

	//Mat imgORG=(*i_img).clone();
	////namedWindow("1");
	//for (int j=0;j<ptsNum;j++)
	//{
	//	circle(imgORG,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
	//	circle(imgORG,Point(tmp.at<double>(0,j),tmp.at<double>(0,j+ptsNum)),3,Scalar(255,0,0));
	//}
	//for (int j=0;j<inlierList.size();j++)
	//{
	//	circle(imgORG,Point(detectedFeatures.at<double>(0,inlierList[j]),detectedFeatures.at<double>(0,inlierList[j]+ptsNum)),2,Scalar(0,0,255));
	//}
	//imwrite(name,imgORG);
	//namedWindow("inlier modes");
	//imshow("inlier modes",imgORG);
	//waitKey();

	//inlierList.clear();
	//for (int i=0;i<ptsNum;i++)
	//{
	//	float curDis=sqrtf((tmp.at<double>(0,i)-detectedFeatures.at<double>(0,i))*(tmp.at<double>(0,i)-detectedFeatures.at<double>(0,i))+
	//		(tmp.at<double>(0,ptsNum+i)-detectedFeatures.at<double>(0,ptsNum+i))*(tmp.at<double>(0,ptsNum+i)-detectedFeatures.at<double>(0,ptsNum+i)));
	//	if (curDis<5)
	//	{
	//		inlierList.push_back(i);
	//	}
	//}

	


	//imwrite("nearestNeighbor.jpg",imgORG);
	return;
	/*Mat img=imread(inputImgName);
	for (int j=0;j<ptsNum;j++)
	{
		circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
	}*/
	Mat img;
	img=imread(inputImgName);
	for (int j=0;j<ptsNum;j++)
	{
		circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
	}
	for (int i=0;i<nnNum;i++)
	{
		tmp=originalTrainingData.row(distanceVec[resortVec[i].second].second).clone();
		alignTools.refineAlign(detectedFeatures,tmp,basisTabel[basisID[resortVec[i].second]].id1,basisTabel[basisID[resortVec[i].second]].id2,disVec[i]);

		//tmp=alignTools.alignedShape;
	/*	img=imread(inputImgName);
		for (int j=0;j<ptsNum;j++)
		{
			circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
		}*/
	namedWindow("1");
		for (int j=0;j<ptsNum;j++)
		{
			//circle(img,Point(detectedFeatures.at<double>(0,j),detectedFeatures.at<double>(0,j+ptsNum)),2,Scalar(0,255,0));
			circle(img,Point(tmp.at<double>(0,j),tmp.at<double>(0,j+ptsNum)),3,Scalar(255,0,0));
		}
	/*	imshow("1",img);
		
		if(distanceVec[resortVec[i].second].second>0)
		{
			Mat img1=imread(nameList_hash[distanceVec[resortVec[i].second].second-1]);
			namedWindow("curImg");
			imshow("curImg",img1);
		}
	
		cout<<resortVec[i].first<<" ";
		waitKey();*/
	}

	/*imwrite("30NN.jpg",img);
	ofstream out("30NN_Dis.txt",ios::out);
	for (int i=0;i<nnNum;i++)
	{
		out<<resortVec[i].first<<endl;
	}
	out.close();*/
}

float GeoHashing::getWeight(float u,float v,float basisDis,float s,float x,float y,float sigma)
{
	float prob=1;

	float tao=(x*x+y*y+3)*sigma*sigma;


	float dis2Mean=(u-x)*(u-x)+(v-y)*(v-y);
	prob+=(u*u+v*v+3)*(u*u+v*v+3)*basisDis/(12*s*tao)*pow(e,-dis2Mean/(tao/basisDis));

	return log(prob);
}


void GeoHashing::vote_weighted(Mat &data,vector<int> &exampleCandidate,int thresNum)
{
	LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	
	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	
	Mat *voteNum=new Mat[basisNum];
	for(int i=0;i<basisNum;i++)
		voteNum[i]=Mat::zeros(1,shapeNum,CV_32S);

	Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);

	QueryPerformanceCounter((LARGE_INTEGER   *)&t1);

	//#pragma omp parallel for
	for (int i=0;i<basisNum;i++)
	{
		exampleCandidate.clear();
		

		int id1=basisTabel[i].id1;
		int id2=basisTabel[i].id2;
		Mat oringin;
		oringin.create(1,2,CV_64FC1);

		oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;

	
		Mat x=oringin.clone();
		x.col(0)=oringin.col(0)-data.col(id1);
		x.col(1)=oringin.col(1)-data.col(id1+ptsNum);

	

		Mat y=x*rotation;

		/*cout<<"oringin: "<<oringin.at<double>(0,0)<<" "<<oringin.at<double>(0,1)<<endl;
		cout<<"x: "<<x.at<double>(0,0)<<" "<<x.at<double>(0,1)<<endl;
		cout<<"y: "<<y.at<double>(0,0)<<" "<<y.at<double>(0,1)<<endl;
		*/
		Mat tmp=Mat::zeros(2,ptsNum-2,CV_64FC1);
		int cnum;

		cnum=0;
		for (int j=0;j<ptsNum;j++)
		{
				if (j==id1||j==id2)
			{
			continue;
			}

			tmp.at<double>(0,cnum)=data.at<double>(0,j)-oringin.at<double>(0,0);
			tmp.at<double>(1,cnum)=data.at<double>(0,j+ptsNum)-oringin.at<double>(0,1);

			//cout<<tmp.at<double>(0,cnum)<<" "<<tmp.at<double>(1,cnum)<<endl;
			cnum++;
		}

		Mat newXY=tmp.clone();
		newXY.row(0)=x.row(0)*tmp;
		newXY.row(1)=y.row(0)*tmp;
		newXY/=(x.at<double>(0,0)*x.at<double>(0,0)+x.at<double>(0,1)*x.at<double>(0,1));

		//continue;;
		/*	for (int j=0;j<newXY.cols;j++)
		{
		cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		}
		continue;*/
		//then find the most nearest value to the exact value
		//int tmoVal=0;
		float basisDis=norm(x)*norm(x)*4;

		float discretized[2];
	//	vector<int> usedInd;
		for (int j=0;j<newXY.cols;j++)
		{
			//cout<<newXY.at<double>(0,j)<<" "<<newXY.at<double>(1,j)<<endl;
		


			discretized[0]=discretValue(newXY.at<double>(0,j),dSize);
			discretized[1]=discretValue(newXY.at<double>(1,j),dSize);

			
		

			/*if (find(usedInd.begin(),usedInd.end(),cind)==usedInd.end())
			{
				usedInd.push_back(cind);
			}
			else
			{
				continue;
			}*/
			//cout<<discretized[0]<<" "<<discretized[1]<<" "<<keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY)<<" "<<iter->second<<endl;
		//	if (discretized[0]<smallestX||discretized[1]<smallestY)
		/*	{
				continue;
			}
			int cind=keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY);*/

			//calculate the weights
			//if(j==0)
			//{
				//cout<<"basis distance: "<<basisDis<<endl;

			GeoHash::iterator iter;
			int cind;

			/*iter=table->isInside(discretized);
			if (iter==table->mappings->end())
			{
				break;
			}

			cind=iter->second;*/

				int windowSize=5;
				int curEntry[2];
				for (int mm=-windowSize;mm<windowSize+1;mm++)
				{
					for (int nn=-windowSize;nn<windowSize+1;nn++)
					{
						curEntry[0]=discretized[0]+mm;
						curEntry[1]=discretized[1]+nn;
						
						iter=table->isInside(discretized);
						if (iter==table->mappings->end())
						{
							break;
						}
						cout<<getWeight(discretized[0]+mm,discretized[1]+nn,basisDis,21,
							newXY.at<double>(0,j),newXY.at<double>(1,j),4)<<" ";
					}
					//cout<<endl;
				}
				float curWeight=getWeight(discretized[0],discretized[1],basisDis,21,
					newXY.at<double>(0,j),newXY.at<double>(1,j),4);
				cout<<"weight for current bin: "<<curWeight<<endl<<endl;
			//}
		


			//continue;
			//#pragma omp parallel for
			vector<Property> *tmpProp=&(dataTabel->at(cind));
			for (int k=0;k<tmpProp->size();k++)
			{
				if (tmpProp->at(k).basisID==i)
				{
					voteNum[i].at<int>(0,tmpProp->at(k).ExampleID)++;
				

				}
			}
			

		}

		//for (int j=0;j<shapeNum;j++)
		//{
		//	if (voteNum[i].at<int>(0,j)>ptsNum-2)
		//	{
		//		//voteNum[i].at<int>(0,j)=ptsNum-2;
		//		cout<<voteNum[i].at<int>(0,j)<<" "<<"basis "<<i<<" shape"<<j<<endl;
		//	}
		//}
		
		continue;
		//then, check the number
		//for (int j=0;j<shapeNum;j++)
		//{
		//	if (j==604)
		//	{
		//		cout<<"vote for 604 from basis "<<i<<" :"<<voteNum[j]<<endl;
		//	}
		//	if (voteNum[j]>thresNum)
		//	{
		//		exampleCandidate.push_back(j);

		//		///////////////align them and see/////////////////////
		//		if (j==0)
		//		{
		//			continue;
		//		}
		//		int originalPtsNum=originalTrainingData.cols/2;
		//		Mat originTrain=oringin.clone();
		//		Mat curOrigData=originalTrainingData.row(j);
		//		originTrain.col(0)=(curOrigData.col(id1)+curOrigData.col(id2))/2;
		//		originTrain.col(1)=(curOrigData.col(id1+ptsNum)+curOrigData.col(id2+ptsNum))/2;

		//		Mat x_new=originTrain.clone();
		//		x_new.col(0)=originTrain.col(0)-curOrigData.col(id1);
		//		x_new.col(1)=originTrain.col(1)-curOrigData.col(id1+ptsNum);

		//		float theta=acos((x_new.dot(x))/norm(x)/norm(x_new));

		//		double scale=norm(x)/norm(x_new);
		//		cout<<"base: "<<i<<"  exampleID: "<<j<<endl;
		//		//scale=1;

		//		Mat ttt=curOrigData.clone();
		//		curOrigData.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
		//		curOrigData.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
		//		curOrigData.colRange(0,ptsNum)=(curOrigData.colRange(0,ptsNum)-originTrain.at<double>(0,0))*scale+oringin.at<double>(0,0);
		//		curOrigData.colRange(ptsNum,ptsNum*2)=(curOrigData.colRange(ptsNum,ptsNum*2)-originTrain.at<double>(0,1))*scale+(oringin.at<double>(0,1));

		//		Mat img=imread(nameList_hash[j-1]);
		//		for (int k=0;k<ptsNum;k++)
		//		{
		//			circle(img,Point(curOrigData.at<double>(0,k),curOrigData.at<double>(0,k+ptsNum)),5,Scalar(0,0,255));
		//			circle(img,Point(data.at<double>(0,k),data.at<double>(0,k+ptsNum)),2,Scalar(0,255,0));
		//		}
		//		circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),1,Scalar(0,255,0));
		//		circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),1,Scalar(0,255,0));

		//		circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),3,Scalar(0,255,0));
		//		circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),3,Scalar(0,255,0));
		//		namedWindow("1");
		//		imshow("1",img);
		//		waitKey();
		//	}
		//}


	


		//if (exampleCandidate.size()>0)
		//{
		//	cout<<basisTabel[i].id1<<" "<<basisTabel[i].id2<<" ";

		//	for (int n=0;n<exampleCandidate.size();n++)
		//	{
		//		if (exampleCandidate[n]==0)
		//		{
		//			continue;
		//		}
		//		cout<<exampleCandidate[n]<<" "<<nameList_hash[exampleCandidate[n]-1]<<endl;
		//		Mat tmpImg=imread(nameList_hash[exampleCandidate[n]-1]);
		//		namedWindow("1");
		//		imshow("1",tmpImg);
		//		waitKey();
		//	}
		//	cout<<endl;
		//}
		
	}
	QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	double   time=(t2-t1)*1000/persecond; 
	cout<<"total time: "<<time<<"ms "<<"searching time: "<<time/basisNum<<" ms"<<endl;
	
	
	char name1[500];
	int tmppp1=dSize*100;
	sprintf(name1, "D:\\Fuhao\\Facial feature points detection\\GeoHash\\result comparison\\votes_%d.txt",tmppp1);
	ofstream outvote(name1,ios::out);
	for (int i=0;i<basisNum;i++)
	{
		for (int j=0;j<shapeNum;j++)
		{
			outvote<<voteNum[i].at<int>(0,j)<<" ";
		}
		outvote<<endl;
	}
	outvote.close();

	//analyze the best basis or the best example
	
	//int bestBasis,bestExample;
	//int bestNum=0;
	//for (int i=0;i<basisNum;i++)
	//{
	//	for (int j=0;j<shapeNum;j++)
	//	{
	//		if (voteNum[i].at<int>(0,j)>bestNum)
	//		{
	//			bestNum=voteNum[i].at<int>(0,j);
	//			bestBasis=i;
	//			bestExample=j;
	//		}
	//	}
	//}

	//cout<<"nearest neighbor: "<<bestExample<<" with number "<<bestNum<<endl;
	//Mat img=imread(nameList_hash[bestExample-1]);
	//namedWindow("1");
	//imshow("1",img);
	//waitKey();

	//////////////////////////////////////
	//get the best basis and sort
	//int bestBasis,bestExample;
	//int bestNum=0;
	//for (int i=0;i<basisNum;i++)
	//{
	//	for (int j=0;j<shapeNum;j++)
	//	{
	//		if (voteNum[i].at<int>(0,j)>bestNum)
	//		{
	//			bestNum=voteNum[i].at<int>(0,j);
	//			bestBasis=i;
	//			bestExample=j;
	//		}
	//	}
	//}
	//for (int i=0;i<shapeNum;i++)
	//{
	//	distanceVec[i].first=voteNum[bestBasis].at<int>(0,i);
	//	distanceVec[i].second=i;
	//}
	//sort(distanceVec,distanceVec+shapeNum,comparator);

	//for (int i=0;i<shapeNum;i++)
	//{

	//	cout<<distanceVec[i].first<<" ";
	//	int j=distanceVec[i].second;
	//	if (j==0)
	//	{
	//		continue;
	//	}

	//	//cout<<j<<endl;

	//	Mat img=imread(nameList_hash[j-1]);
	//	namedWindow("1");
	//	imshow("1",img);
	//	waitKey();
	//}
	//////////////////////////////////////
	
	//////////////////////////////////////////////////
	//search for best basis for each example
	int curBestId,curBestNum;
	int bestNumEach[1000];
	int bestBasisEach[1000];
	for (int i=0;i<shapeNum;i++)
	{
		curBestNum=0;
		for (int j=0;j<basisNum;j++)
		{
			if (voteNum[j].at<int>(0,i)>curBestNum)
			{
				curBestId=j;
				curBestNum=voteNum[j].at<int>(0,i);
			}
		}
		bestNumEach[i]=curBestNum;
		bestBasisEach[i]=curBestId;

		//cout<<curBestNum<<endl;
	}
	for (int i=0;i<shapeNum;i++)
	{
		distanceVec[i].first=bestNumEach[i];
		distanceVec[i].second=i;
	}
	sort(distanceVec,distanceVec+shapeNum,comparator_hash);

	vector<int> inlier;
	reSort(data,bestBasisEach,inlier);

	for (int i=0;i<shapeNum;i++)
	{
		int j=distanceVec[resortVec[i].second].second;
		if (j==0)
		{
			continue;
		}

		cout<<distanceVec[i].first<<" ";

		//align the feature and see
		int id1=basisTabel[bestBasisEach[i]].id1;
		int id2=basisTabel[bestBasisEach[i]].id2;
		Mat oringin;
		oringin.create(1,2,CV_64FC1);

		oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;


		Mat x=oringin.clone();
		x.col(0)=oringin.col(0)-data.col(id1);
		x.col(1)=oringin.col(1)-data.col(id1+ptsNum);
		Mat y=x*rotation;

		int originalPtsNum=originalTrainingData.cols/2;
		
		Mat originTrain=oringin.clone();
		Mat curOrigData=originalTrainingData.row(distanceVec[i].second);
		originTrain.col(0)=(curOrigData.col(id1)+curOrigData.col(id2))/2;
		originTrain.col(1)=(curOrigData.col(id1+ptsNum)+curOrigData.col(id2+ptsNum))/2;

		Mat x_new=originTrain.clone();
		x_new.col(0)=originTrain.col(0)-curOrigData.col(id1);
		x_new.col(1)=originTrain.col(1)-curOrigData.col(id1+ptsNum);

		float theta=acos((x_new.dot(x))/norm(x)/norm(x_new));

		double scale=norm(x)/norm(x_new);
		cout<<"base: "<<i<<"  exampleID: "<<j<<endl;
		//scale=1;

		Mat ttt=curOrigData.clone();
		curOrigData.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
		curOrigData.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
		curOrigData.colRange(0,ptsNum)=(curOrigData.colRange(0,ptsNum)-originTrain.at<double>(0,0))*scale+oringin.at<double>(0,0);
		curOrigData.colRange(ptsNum,ptsNum*2)=(curOrigData.colRange(ptsNum,ptsNum*2)-originTrain.at<double>(0,1))*scale+(oringin.at<double>(0,1));

		//Mat img=imread(nameList_hash[j-1]);
		Mat img=imread(inputImgName);
		for (int k=0;k<ptsNum;k++)
		{
			circle(img,Point(curOrigData.at<double>(0,k),curOrigData.at<double>(0,k+ptsNum)),5,Scalar(0,0,255));
			circle(img,Point(data.at<double>(0,k),data.at<double>(0,k+ptsNum)),2,Scalar(0,255,0));
		}
		circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),1,Scalar(0,255,0));
		circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),1,Scalar(0,255,0));

		circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),3,Scalar(0,255,0));
		circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),3,Scalar(0,255,0));

		Mat imgCur=imread(nameList_hash[j-1]);
		namedWindow("Input");
		imshow("Input",img);
		namedWindow("Training Example");
		imshow("Training Example",imgCur);
		waitKey();


		/*Mat img=imread(nameList_hash[j-1]);
		namedWindow("1");
		imshow("1",img);
		waitKey();*/
	}
	////////////////////////////////////////////////////////

	

	//Mat voteAll=Mat::zeros(1,shapeNum,CV_32S);
	for (int i=0;i<basisNum;i++)
	{
		voteAll+=voteNum[i];
	}
	//sort and see
	for (int i=0;i<shapeNum;i++)
	{
		distanceVec[i].first=voteAll.at<int>(0,i);
		distanceVec[i].second=i;
	}
	sort(distanceVec,distanceVec+shapeNum,comparator_hash);

	//vector<int> inlier;
	reSort(data,bestBasisEach,inlier);

	cout<<distanceVec[0].second<<" "<<distanceVec[1].second<<endl;
	char name[500];
	int tmppp=dSize*100;
	cout<<tmppp<<endl;
	sprintf(name, "D:\\Fuhao\\Facial feature points detection\\GeoHash\\result comparison\\geoHash_%d.txt",tmppp);
	ofstream out(name,ios::out);
	int nnnum=30;
	out<<nnnum<<endl;
	for (int i=0;i<nnnum;i++)
	{
		if (distanceVec[i].second>0)
		{
			out<<nameList_hash[distanceVec[i].second-1]<<endl;
		}
		
	}
	out.close();

	//cout<<nameList_hash[14]<<endl;
	for (int i=0;i<shapeNum;i++)
	{
		int j=distanceVec[i].second;
		if (j==0)
		{
			continue;
		}

		cout<<j<<endl;

		Mat img=imread(nameList_hash[j-1]);
		namedWindow("1");
		imshow("1",img);
		waitKey();



		//	//align the feature and see
		//	int id1=basisTabel[bestBasisEach[i]].id1;
		//	int id2=basisTabel[bestBasisEach[i]].id2;
		//	Mat oringin;
		//	oringin.create(1,2,CV_64FC1);

		//	oringin.col(0)=(data.col(id1)+data.col(id2))/2;
		//	oringin.col(1)=(data.col(ptsNum+id1)+data.col(ptsNum+id2))/2;


		//	Mat x=oringin.clone();
		//	x.col(0)=oringin.col(0)-data.col(id1);
		//	x.col(1)=oringin.col(1)-data.col(id1+ptsNum);
		//	Mat y=x*rotation;

		//	int originalPtsNum=originalTrainingData.cols/2;
		//	
		//	Mat originTrain=oringin.clone();
		//	Mat curOrigData=originalTrainingData.row(distanceVec[i].second);
		//	originTrain.col(0)=(curOrigData.col(id1)+curOrigData.col(id2))/2;
		//	originTrain.col(1)=(curOrigData.col(id1+ptsNum)+curOrigData.col(id2+ptsNum))/2;

		//	Mat x_new=originTrain.clone();
		//	x_new.col(0)=originTrain.col(0)-curOrigData.col(id1);
		//	x_new.col(1)=originTrain.col(1)-curOrigData.col(id1+ptsNum);

		//	float theta=acos((x_new.dot(x))/norm(x)/norm(x_new));

		//	double scale=norm(x)/norm(x_new);
		//	cout<<"base: "<<i<<"  exampleID: "<<j<<endl;
		//	//scale=1;

		//	Mat ttt=curOrigData.clone();
		//	curOrigData.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
		//	curOrigData.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
		//	curOrigData.colRange(0,ptsNum)=(curOrigData.colRange(0,ptsNum)-originTrain.at<double>(0,0))*scale+oringin.at<double>(0,0);
		//	curOrigData.colRange(ptsNum,ptsNum*2)=(curOrigData.colRange(ptsNum,ptsNum*2)-originTrain.at<double>(0,1))*scale+(oringin.at<double>(0,1));

		//	//Mat img=imread(nameList_hash[j-1]);
		//	Mat img=imread(inputImgName);
		//	for (int k=0;k<ptsNum;k++)
		//	{
		//		circle(img,Point(curOrigData.at<double>(0,k),curOrigData.at<double>(0,k+ptsNum)),5,Scalar(0,0,255));
		//		circle(img,Point(data.at<double>(0,k),data.at<double>(0,k+ptsNum)),2,Scalar(0,255,0));
		//	}
		//	circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),1,Scalar(0,255,0));
		//	circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),1,Scalar(0,255,0));

		//	circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),3,Scalar(0,255,0));
		//	circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),3,Scalar(0,255,0));

		//	Mat imgCur=imread(nameList_hash[j-1]);
		//	namedWindow("Input");
		//	imshow("Input",img);
		//	namedWindow("Training Example");
		//	imshow("Training Example",imgCur);
		//	waitKey();
		
	}


}