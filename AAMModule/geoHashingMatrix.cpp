#include "geoHashingMatrix.h"
#include <fstream>
#include "omp.h"
using namespace std;

string nameListMatrix[1000];

typedef std::pair<int,int> distancePir;
distancePir  distanceVecMatrix[1000];
bool comparatorMatrix ( const distancePir& l, const distancePir& r)
{ return l.first > r.first; }

geoHashingMatrix::geoHashingMatrix(Mat &shapes,float _dsize)
{
	ptsNum=shapes.cols/2;
	basisNum=ptsNum*(ptsNum-1)/2;

	dSize=_dsize;

	basisTabel=new basisPairMatrix[basisNum];
	discretWindow=_dsize;
	buildBasisTabel(ptsNum,basisTabel);

	table=new HashTable();
	dataTabel=new vector<Mat>;

	//read in the name
	char cname[800];
	ifstream in("D:\\Fuhao\\face dataset\\train_larger database\\imgList.txt",ios::in);
	int totalNum;
	in>>totalNum;
	in.getline(cname,800);
	for (int i=0;i<totalNum;i++)
	{
		in.getline(cname,800);
		nameListMatrix[i]=cname;
	}

	keyIndexTable=Mat::zeros(2200,2200,CV_32S)-1;
	smallestX=-1200;
	smallestY=-1200;

	originalTrainingData=shapes;
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

void geoHashingMatrix::buildBasisTabel(int featureNum,basisPairMatrix* pairList)
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
void geoHashingMatrix::buildHashTabel(int pairNum,basisPairMatrix * pairList,Mat &data)
{
	//buildSingleTabel(data,2,0,0.25);

	int cind1,cind2;
	for (int i=0;i<basisNum;i++)
	{
		//cout<<i<<" "<<basisNum<<endl;
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

	//	//	cout<<x_ref<<" "<<y_ref<<" "<<keyIndexTable.at<int>(x_ref,y_ref)<<" "<<i<<endl;
	//	iter++;
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

void geoHashingMatrix::loadTable(char *name)
{
	ifstream in(name,ios::in);

}
//
void geoHashingMatrix::buildSingleTabel(Mat &data,int basisID,float discretNum)
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
			//Property tmpProperty(i,basisID);
			GeoHash::iterator iter=table->isInside(discretized);
			int cid=iter->second;
			int curValue=basisID*shapeNum+i;
			if (cid<dataTabel->size())
			{
				Mat tmpp=dataTabel->at(cid);
				tmpp.at<int>(basisID,i)++;
				//dataTabel->at(cid).push_back(curValue);
			}
			else
			{
				Mat tmpIn=Mat::zeros(basisNum,shapeNum,CV_32S);
				tmpIn.at<int>(basisID,i)=tmpIn.at<int>(basisID,i)+1;
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
float geoHashingMatrix::discretValue(float input,float disceteNum)
{
	int num=floor(input/disceteNum);
	float residual=input-num*disceteNum;

	if (residual>disceteNum/2)
	{
		return (num+1)*disceteNum;
	}
	return num*disceteNum;
}

void geoHashingMatrix::saveTable(char *name)
{
	//ofstream out(name,ios::out);
	//GeoHash::iterator iter=table->mappings->begin();
	//map<float,float> basis;
	//map<float,float>::iterator basisIter;
	//for (int i=0;i<table->mappings->size();i++)
	//{
	//	basis=iter->first;
	//	basisIter=basis.begin();
	//	out<<basisIter->first<<" "<<basisIter->second<<endl;//<<" "<<tmpEntry[iter->second][0]<<" "<<tmpEntry[iter->second][1]<<endl;
	//	//cout<<dataTabel->at(i).size()<<endl;
	//	
	//	int cind=iter->second;
	//	for (int j=0;j<dataTabel->at(cind).size();j++)
	//	{
	//		out<<dataTabel->at(cind).at(j)<<" ";
	//	}
	//	out<<endl;

	//	iter++;

	//	
	//}
	//out.close();
}

void geoHashingMatrix::showTabel()
{
	//GeoHash::iterator iter=table->mappings->begin();
	//map<float,float> basis;
	//map<float,float>::iterator basisIter;
	//for (int i=0;i<dataTabel->size();i++)
	//{
	//	basis=iter->first;
	//	basisIter=basis.begin();
	//	cout<<basisIter->first<<" "<<basisIter->second<<endl;;//<<" "<<tmpEntry[iter->second][0]<<" "<<tmpEntry[iter->second][1]<<endl;
	//	//cout<<dataTabel->at(i).size()<<endl;
	//
	//	int cind=iter->second;
	//	for (int j=0;j<dataTabel->at(cind).size();j++)
	//	{
	//		cout<<dataTabel->at(cind).at(j)<<" ";
	//	}
	//	cout<<endl;

	//	iter++;

	//	continue;
	//}
}


void geoHashingMatrix::vote(Mat &data,vector<int> &exampleCandidate,int thresNum)
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
	/*	int voteNum[800];
		for (int j=0;j<shapeNum;j++)
		{
			voteNum[j]=0;
		}*/
		Mat voteNum=Mat::zeros(1,shapeNum,CV_32S);

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
			
			//vector<int> *tmpProp=&(dataTabel->at(cind));
			//for (int k=0;k<tmpProp->size();k++)
			//{
			//	int curVal=tmpProp->at(k);
			//	int basisID=curVal/shapeNum;
			//	int shapeID=curVal%shapeNum;
			//	if (basisID==i)
			//	{
			//		voteNum[shapeID]++;
			//		//voteNum[6]++;
			//	}
			//}

			Mat ttt=(dataTabel->at(cind));
			voteNum+=ttt.row(i);
			

		}
		
		////then, check the number
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

		//		Mat img=imread(nameListMatrix[j-1]);
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
		//		cout<<exampleCandidate[n]<<" "<<nameListMatrix[exampleCandidate[n]-1]<<endl;
		//		Mat tmpImg=imread(nameListMatrix[exampleCandidate[n]-1]);
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

void geoHashingMatrix::vote_countAll(Mat &data,vector<int> &exampleCandidate,int thresNum)
{
	LONGLONG   t1,t2; 
	LONGLONG   persecond; 
	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);
	
	Mat rotation=Mat::zeros(2,2,CV_64FC1);
	rotation.at<double>(0,0)=0;
	rotation.at<double>(0,1)=1;
	rotation.at<double>(1,0)=-1;
	rotation.at<double>(1,1)=0;

	

	//int *voteNum=new int[shapeNum*basisNum];
	//for (int j=0;j<shapeNum*basisNum;j++)
	//{
	//	voteNum[j]=0;
	//}
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

		/*	if (discretized[0]<smallestX||discretized[1]<smallestY)
			{
				continue;
			}
			int cind=keyIndexTable.at<int>(discretized[0]-smallestX,discretized[1]-smallestY);*/

	
			//continue;
			//#pragma omp parallel for
			
			//vector<int> *tmpProp=&(dataTabel->at(cind));
			//for (int k=0;k<tmpProp->size();k++)
			//{
			//	/*int curVal=tmpProp->at(k);
			//	int basisID=curVal/shapeNum;
			//	int shapeID=curVal%shapeNum;*/

			//	//if (basisID==i)
			//	//{
			//	//	voteNum[shapeID]++;
			//	//	//voteNum[6]++;
			//	//}
			//	int curVal=tmpProp->at(k);
			//	voteNum[curVal]++;
			//}

			Mat ttt=(dataTabel->at(cind));
			voteNum[i]=voteNum[i]+ttt.row(i);
			

		}
		
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

		//		Mat img=imread(nameListMatrix[j-1]);
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
		//		cout<<exampleCandidate[n]<<" "<<nameListMatrix[exampleCandidate[n]-1]<<endl;
		//		Mat tmpImg=imread(nameListMatrix[exampleCandidate[n]-1]);
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

	for (int i=0;i<basisNum;i++)
	{
		voteAll+=voteNum[i];
	}


	//analyze the best basis or the best example
	int bestBasis,bestExample;
	int bestNum=0;
	for (int i=0;i<basisNum;i++)
	{
		for (int j=0;j<shapeNum;j++)
		{
			if (voteNum[i].at<int>(0,j)>bestNum)
			{
				bestNum=voteNum[i].at<int>(0,j);
				bestBasis=i;
				bestExample=j;
			}
		}
	}

	cout<<"nearest neighbor: "<<bestExample<<" with number "<<bestNum<<endl;
	Mat img=imread(nameListMatrix[bestExample-1]);
	namedWindow("1");
	imshow("1",img);
	waitKey();
	//

	//sort and see
	for (int i=0;i<shapeNum;i++)
	{
		 distanceVecMatrix[i].first=voteAll.at<int>(0,i);
		 distanceVecMatrix[i].second=i;
	}
	sort( distanceVecMatrix, distanceVecMatrix+shapeNum,comparatorMatrix);

	cout<< distanceVecMatrix[0].second<<" "<< distanceVecMatrix[1].second<<endl;
	char name[500];
	int tmppp=dSize*100;
	cout<<tmppp<<endl;
	sprintf(name, "D:\\Fuhao\\Facial feature points detection\\GeoHash\\result comparison\\geoHash_%d.txt",tmppp);
	ofstream out(name,ios::out);
	int nnnum=30;
	out<<nnnum<<endl;
	for (int i=0;i<nnnum;i++)
	{
		if ( distanceVecMatrix[i].second>0)
		{
			out<<nameListMatrix[ distanceVecMatrix[i].second-1]<<endl;
		}
		
	}
	out.close();

	//cout<<nameListMatrix[14]<<endl;
	for (int i=0;i<shapeNum;i++)
	{
		int j= distanceVecMatrix[i].second;
		if (j==0)
		{
			continue;
		}

		cout<<j<<endl;

		Mat img=imread(nameListMatrix[j-1]);
		namedWindow("1");
		imshow("1",img);
		waitKey();

			//int originalPtsNum=originalTrainingData.cols/2;
			//Mat originTrain=oringin.clone();
			//Mat curOrigData=originalTrainingData.row(j);
			//originTrain.col(0)=(curOrigData.col(id1)+curOrigData.col(id2))/2;
			//originTrain.col(1)=(curOrigData.col(id1+ptsNum)+curOrigData.col(id2+ptsNum))/2;

			//Mat x_new=originTrain.clone();
			//x_new.col(0)=originTrain.col(0)-curOrigData.col(id1);
			//x_new.col(1)=originTrain.col(1)-curOrigData.col(id1+ptsNum);

			//float theta=acos((x_new.dot(x))/norm(x)/norm(x_new));

			//double scale=norm(x)/norm(x_new);
			//cout<<"base: "<<i<<"  exampleID: "<<j<<endl;
			////scale=1;

			//Mat ttt=curOrigData.clone();
			//curOrigData.colRange(0,ptsNum)=cos(theta)*ttt.colRange(0,ptsNum)-sin(theta)*ttt.colRange(ptsNum,ptsNum*2);
			//curOrigData.colRange(ptsNum,ptsNum*2)=sin(theta)*ttt.colRange(0,ptsNum)+cos(theta)*ttt.colRange(ptsNum,ptsNum*2);
			//curOrigData.colRange(0,ptsNum)=(curOrigData.colRange(0,ptsNum)-originTrain.at<double>(0,0))*scale+oringin.at<double>(0,0);
			//curOrigData.colRange(ptsNum,ptsNum*2)=(curOrigData.colRange(ptsNum,ptsNum*2)-originTrain.at<double>(0,1))*scale+(oringin.at<double>(0,1));

			//Mat img=imread(nameListMatrix[j-1]);
			//for (int k=0;k<ptsNum;k++)
			//{
			//	circle(img,Point(curOrigData.at<double>(0,k),curOrigData.at<double>(0,k+ptsNum)),5,Scalar(0,0,255));
			//	circle(img,Point(data.at<double>(0,k),data.at<double>(0,k+ptsNum)),2,Scalar(0,255,0));
			//}
			//circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),1,Scalar(0,255,0));
			//circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),1,Scalar(0,255,0));

			//circle(img,Point(data.at<double>(0,id1),data.at<double>(0,id1+ptsNum)),3,Scalar(0,255,0));
			//circle(img,Point(data.at<double>(0,id2),data.at<double>(0,id2+ptsNum)),3,Scalar(0,255,0));
			//namedWindow("1");
			//imshow("1",img);
			//waitKey();
		
	}


}