#include "shape.h"

#include <iostream>
using namespace std;

Shape::Shape()
{
	init();
}

Shape::Shape(Shape *input)
{
	init();
	setHostImage(input->hostImage);
	getVertex(input->ptsForMatlab,input->ptsNum,input->width,input->height);
}

void Shape::init()
{
	pts=triangles=NULL;
	ptsForMatlab=NULL;
	affineTable=NULL;
	hostImage=NULL;
	mask=NULL;
	mask_withindex=NULL;
//	sl_engine=new SL_Basis;
}

Shape::~Shape()
{
	if(pts!=NULL)
	{
		for (int i=0;i<ptsNum;i++)
		{
			delete []pts[i];
		}
		delete []pts;
	}
	if(triangles!=NULL)
	{
		for (int i=0;i<triangleNum;i++)
		{
			delete []triangles[i];
		}
		delete []triangles;
	}
	if (ptsForMatlab!=NULL)
	{
		delete []ptsForMatlab;
	}
	cvReleaseImage(&hostImage);
	//delete sl_engine;
}

void Shape::setPtsNum(int num)
{
	ptsNum=num;
	if (ptsForMatlab!=NULL)
	{
		delete []ptsForMatlab;
	}
	ptsForMatlab=new double[ptsNum*2];
	if (pts!=NULL)
	{
		for (int i=0;i<ptsNum;i++)
		{
			delete []pts[i];
		}
		delete []pts;
	}
	pts=new double *[ptsNum];
	for (int i=0;i<ptsNum;i++)
	{
		pts[i]=new double[2];
	}
}

void Shape::setTriangleNum(int num)
{
	triangleNum=num;
	if (triangles==NULL)
	{
		triangles=new double *[triangleNum];
		for (int i=0;i<3;i++)
		{
			triangles[i]=new double[3];
		}
	}
}

void Shape::setPts(int num,double **input)
{
	setPtsNum(num);
	for (int i=0;i<ptsNum;i++)
	{
		pts[i][0]=input[i][0];
		pts[i][1]=input[i][1];
	}
}

void Shape::setTriangles(int num, double **input)
{
	setTriangleNum(num);
	for (int i=0;i<triangleNum;i++)
	{
		for (int j=0;j<3;j++)
		{
			triangles[i][j]=input[i][j];
		}
	}
}

void Shape::setHostImage(IplImage *img)
{
	hostImage=cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
	cvCopyImage(img,hostImage);
}

void Shape::getVertex(double *InputPts,int ptsNum,int width,int height)
{
	setPtsNum(ptsNum);
	for (int i=0;i<ptsNum;i++)
	{
		pts[i][0]=InputPts[i];
		pts[i][1]=InputPts[ptsNum+i];
		if (pts[i][0]<1)
		{
			pts[i][0]*=width;
			pts[i][1]*=height;
		}	
		ptsForMatlab[i]=pts[i][0];
		ptsForMatlab[ptsNum+i]=pts[i][1];
	}
	minx=maxx=pts[0][0];
	miny=maxy=pts[0][1];
	for (int i=1;i<ptsNum;i++)
	{
		minx=min(minx,pts[i][0]);
		miny=min(miny,pts[i][1]);
		maxx=max(maxx,pts[i][0]);
		maxy=max(maxy,pts[i][1]);
	}

}

void Shape::getDpeth(string ptsName,int trainStyle)
{

	ifstream in(ptsName.c_str(),ios::in);
	int num=0;

	in>>width>>height>>num>>num;

	//cout<<"point num: "<<num<<endl;
	setPtsNum(num);

	for (int i=0;i<ptsNum;i++)
	{
		in>>pts[i][0]>>pts[i][1];
		if (pts[i][0]<1)
		{
			pts[i][0]*=width;
			pts[i][1]*=height;
		}
		ptsForMatlab[i]=pts[i][0];
		ptsForMatlab[ptsNum+i]=pts[i][1];
	}
	in.close();


	minx=maxx=pts[0][0];
	miny=maxy=pts[0][1];
	for (int i=1;i<ptsNum;i++)
	{
		minx=min(minx,pts[i][0]);
		miny=min(miny,pts[i][1]);
		maxx=max(maxx,pts[i][0]);
		maxy=max(maxy,pts[i][1]);
	}

	colorImg=Mat::ones(height,width,CV_32FC1);
	colorImg*=0;

	
	//depth
	string depthFileName=ptsName.substr(0,ptsName.length()-4);
	depthFileName+="_depth.txt";
	int x,y;
	double depthValue;

	//ifstream in22(depthFileName.c_str(),ios::in);

	//while(in22)
	//{
	//	in22>>x>>y>>depthValue;
	//	if (abs(depthValue)<=1)
	//	{
	//		colorImg.at<float>(y,x)=depthValue;
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
			for (int i=0;i<colorImg.rows;i++)
			{
				for (int j=0;j<colorImg.cols;j++)
				{
					in22>>depthValue;
					if (depthValue!=0)
					{
						//colorImg.at<float>(i,j)=2980-depthValue;
						colorImg.at<float>(i,j)=depthValue/standardDepth;
					}
					else
					{
						colorImg.at<float>(i,j)=depthValue/standardDepth;
					}

				}
			}
		}
		else
		{
			//for synthesized data, need to be adjusted
			//read in the scale
			double depthScale;
			string scaleFileName="G:\\face database\\facegen database depth only";
			scaleFileName+=ptsName.substr(ptsName.find_last_of('\\'),ptsName.find_first_of('_')-ptsName.find_last_of('\\'));
			scaleFileName+="_scale.txt";

			ifstream inscale(scaleFileName.c_str(),ios::in);
			inscale>>depthScale;
			inscale.close();

			while(in22)
			{
				in22>>y>>x>>depthValue;
				if (depthValue==-1)
					depthValue=0;
				else
					depthValue=(5-depthValue)/depthScale/standardDepth;
				//if (abs(depthValue)<=1)
				{
					colorImg.at<float>(x,y)=depthValue;
				}
			}
		}
	}
	else //if it is in xml format, it should be exactly the depth. If there are minus values, then it should be 1200-depth
	{
		depthFileName=ptsName.substr(0,ptsName.length()-4);
		depthFileName+=".xml";
		CvMat* tmpDepth = (CvMat*)cvLoad( depthFileName.c_str());
		colorImg.release();
		colorImg=cvarrToMat(tmpDepth).clone();

		
	///*	Mat tmp=cvarrToMat(tmpDepth);*/
	//	for (int i=0;i<colorImg.rows;i++)
	//	{
	//		for (int j=0;j<colorImg.cols;j++)
	//		{
	//			if (colorImg.at<float>(i,j)<0)
	//			{
	//				cout<<colorImg.at<float>(i,j)<<" ";
	//			}
	//			
	//		}
	//		cout<<endl;
	//	}
		double testMaxval=500;
		int maxIdx[3]; 
		minMaxIdx(colorImg, &testMaxval, 0, maxIdx,0);
		if (testMaxval<0)
		{
			for (int i=0;i<colorImg.rows;i++)
			{
				for (int j=0;j<colorImg.cols;j++)
				{
					if (colorImg.at<float>(i,j)!=0)
					{
						colorImg.at<float>(i,j)=1200-colorImg.at<float>(i,j);
					}

				}
			}
		}

		minMaxIdx(colorImg,0, &testMaxval, 0, maxIdx);
		if (testMaxval>10)	//else, already divided by the standard depth
		{
			for (int i=0;i<colorImg.rows;i++)
			{
				for (int j=0;j<colorImg.cols;j++)
				{
					if (colorImg.at<float>(i,j)!=0)
					{
						colorImg.at<float>(i,j)/=standardDepth;
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
	
	//namedWindow("1");
	//imshow("1",colorImg);
	//waitKey();

	//if not using depth only, read in the color images
	if (trainStyle!=0)
	{
		string imgName=ptsName;

		imgName.replace(imgName.end()-3,imgName.end(),"jpg");
		ifstream testIfExist(imgName.c_str(),ios::in);
		if (testIfExist)
		{
			testIfExist.close();
			hostImage=cvLoadImage(imgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
		}
		else
		{
			imgName=ptsName;
			imgName.replace(imgName.end()-3,imgName.end(),"bmp");
			hostImage=cvLoadImage(imgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
		}
	}

	//if (hostImage==NULL)
	//{
	/*	imgName.replace(imgName.end()-3,imgName.end(),"jpg");
		hostImage=cvLoadImage(imgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);*/
	//	
	//	for (int i=0;i<ptsNum;i++)
	//	{
	//		cvCircle(hostImage,cvPoint(pts[i][0],pts[i][1]),3,CV_RGB(0,0,255));
	//	}
	//	namedWindow("1");
	//	imshow("1",cvarrToMat(hostImage));
	//	waitKey();
	//	//colorImg=imread(imgName.c_str());
	//}
}

//void Shape::getVertex(string imgName)
void Shape::getVertex(string ptsName)
{
	

	string imgName=ptsName;
	imgName.replace(imgName.end()-3,imgName.end(),"jpg");
	hostImage=cvLoadImage(imgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
	colorImg=imread(imgName.c_str());

	if (hostImage==NULL)
	{
		imgName.replace(imgName.end()-3,imgName.end(),"bmp");
		hostImage=cvLoadImage(imgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
		colorImg=imread(imgName.c_str());
	}
	if (hostImage==NULL)
	{
		imgName.replace(imgName.end()-3,imgName.end(),"png");
		hostImage=cvLoadImage(imgName.c_str(),CV_LOAD_IMAGE_GRAYSCALE);
		colorImg=imread(imgName.c_str());
	}
	

	

	//img=shape->hostImage;
	//CvSize imgsize=cvGetSize(hostImage);

	ifstream in(ptsName.c_str(),ios::in);
	//in>>width>>height;
	//double aa,bb;
	int num=0;
	//while (in)
	//{
	//	in>>aa>>bb;
	//	num++;
	//}
	////	in.close();
	//setPtsNum(num-1);

	////int ptsNum=shape->ptsNum;


	////double **pts=new double *[ptsNum];
	////for (int i=0;i<ptsNum;i++)
	////{
	////	pts[i]=new double[2];
	////}

	//in.clear();
	//in.seekg(0);
	in>>width>>height>>num>>num;

	setPtsNum(num);

	for (int i=0;i<ptsNum;i++)
	{
		in>>pts[i][0]>>pts[i][1];
		if (pts[i][0]<1)
		{
			pts[i][0]*=width;
			pts[i][1]*=height;
		}
		ptsForMatlab[i]=pts[i][0];
		ptsForMatlab[ptsNum+i]=pts[i][1];
	}
	in.close();

	
	minx=maxx=pts[0][0];
	miny=maxy=pts[0][1];
	for (int i=1;i<ptsNum;i++)
	{
		minx=min(minx,pts[i][0]);
		miny=min(miny,pts[i][1]);
		maxx=max(maxx,pts[i][0]);
		maxy=max(maxy,pts[i][1]);
	}
	//CvMat pointMat=cvMat( 1, ptsNum, CV_64FC1, 0 );
	//for (int i=0;i<ptsNum;i++)
	//{
	//	CV_MAT_ELEM(pointMat,)
	//	points[i].x=pts[i][0];
	//	points[i].y=pts[i][1];
	//}
	//cout<<pointMat.rows<<" "<<pointMat.cols<<endl;
	//hullMat=cvCreateMat(1,ptsNum,CV_32SC2);
	//cvConvexHull2( &pointMat, hullMat, CV_CLOCKWISE, 0 );

	//return pts;
}

void Shape::centerPts(int style)
{
	//style=0,pts
	//style=1,ptsForMatlab
	double center[2];
	if (style==0)
	{
		center[0]=0;
		center[1]=0;
		for (int i=0;i<ptsNum;i++)
		{
			center[0]+=pts[i][0];
			center[1]+=pts[i][1];
		}
		center[0]/=ptsNum;center[1]/=ptsNum;

		for (int i=0;i<ptsNum;i++)
		{
			pts[i][0]-=center[0];
			pts[i][1]-=center[1];
		}
	}
	else if (style==1)
	{
		center[0]=0;
		center[1]=0;
		for (int i=0;i<ptsNum;i++)
		{
			center[0]+=ptsForMatlab[i];
			center[1]+=ptsForMatlab[ptsNum+i];
		}
		center[0]/=ptsNum;center[1]/=ptsNum;

		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]-=center[0];
			ptsForMatlab[ptsNum+i]-=center[1];
			pts[i][0]-=center[0];
			pts[i][1]-=center[1];
		}
	}
	else if (style==2)
	{
		center[0]=0;
		center[1]=0;
		for (int i=0;i<ptsNum;i++)
		{
			center[0]+=ptsForMatlab[i];
			center[1]+=ptsForMatlab[ptsNum+i];
		}
		center[0]/=ptsNum;center[1]/=ptsNum;

		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]-=center[0];
			ptsForMatlab[ptsNum+i]-=center[1];
		}
	}

	//update bound
	minx-=center[0];
	maxx-=center[0];
	miny-=center[1];
	maxy-=center[1];
	minx=maxx=ptsForMatlab[0];
	miny=maxy=ptsForMatlab[ptsNum+0];
	for (int i=1;i<ptsNum;i++)
	{
		minx=min(minx,ptsForMatlab[i]);
		miny=min(miny,ptsForMatlab[i]);
		maxx=max(maxx,ptsForMatlab[ptsNum+i]);
		maxy=max(maxy,ptsForMatlab[ptsNum+i]);
	}
}

void Shape::scale(double s,int style,bool centerd)
{
	if (style==1)
	{
		double center[2]={0,0};
		if(centerd)
		{
			for (int i=0;i<ptsNum;i++)
			{
				center[0]+=ptsForMatlab[i];
				center[1]+=ptsForMatlab[ptsNum+i];
			}
			center[0]/=ptsNum;center[1]/=ptsNum;

			for (int i=0;i<ptsNum;i++)
			{
				ptsForMatlab[i]-=center[0];
				ptsForMatlab[ptsNum+i]-=center[1];
				pts[i][0]-=center[0];
				pts[i][1]-=center[1];
			}

		}


		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]*=s;
			ptsForMatlab[ptsNum+i]*=s;
			pts[i][0]*=s;
			pts[i][1]*=s;
		}
		if (centerd)
		{
			for (int i=0;i<ptsNum;i++)
			{
				ptsForMatlab[i]+=center[0];
				ptsForMatlab[ptsNum+i]+=center[1];
				pts[i][0]+=center[0];
				pts[i][1]+=center[1];
			}

		}

	}
	else if (style==2)
	{
		double center[2]={0,0};
		if(centerd)
		{
			for (int i=0;i<ptsNum;i++)
			{
				center[0]+=ptsForMatlab[i];
				center[1]+=ptsForMatlab[ptsNum+i];
			}
			center[0]/=ptsNum;center[1]/=ptsNum;

			for (int i=0;i<ptsNum;i++)
			{
				ptsForMatlab[i]-=center[0];
				ptsForMatlab[ptsNum+i]-=center[1];
			}

		}


		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]*=s;
			ptsForMatlab[ptsNum+i]*=s;
		}
		if (centerd)
		{
			for (int i=0;i<ptsNum;i++)
			{
				ptsForMatlab[i]+=center[0];
				ptsForMatlab[ptsNum+i]+=center[1];
			}

		}

	}
	//update bound
	minx=maxx=ptsForMatlab[0];
	miny=maxy=ptsForMatlab[ptsNum+0];
	for (int i=1;i<ptsNum;i++)
	{
		minx=min(minx,ptsForMatlab[i]);
		miny=min(miny,ptsForMatlab[i]);
		maxx=max(maxx,ptsForMatlab[ptsNum+i]);
		maxy=max(maxy,ptsForMatlab[ptsNum+i]);
	}
}

void Shape::normalize(int style)
{
	//style=0,pts
	//style=1,ptsForMatlab

	if (style==1)
	{
		double sum[2]={0,0};
		for (int i=0;i<ptsNum;i++)
		{
			sum[0]+=ptsForMatlab[i]*ptsForMatlab[i];
			sum[1]+=ptsForMatlab[ptsNum+i]*ptsForMatlab[ptsNum+i];
		}
		scaleParameter=sqrt(sum[0]+sum[1]);

		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]/=scaleParameter;
			ptsForMatlab[ptsNum+i]/=scaleParameter;
			pts[i][0]/=scaleParameter;
			pts[i][1]/=scaleParameter;
		}

	}
	else if (style==0)
	{
		double sum[2]={0,0};
		for (int i=0;i<ptsNum;i++)
		{
			sum[0]+=pts[i][0]*pts[i][0];
			sum[1]+=pts[i][1]*pts[i][1];
		}
		scaleParameter=sqrt(sum[0]+sum[1]);

		for (int i=0;i<ptsNum;i++)
		{
			pts[i][0]/=scaleParameter;
			pts[i][1]/=scaleParameter;
		}
	}
	else if (style==2)
	{
		double sum[2]={0,0};
		for (int i=0;i<ptsNum;i++)
		{
			sum[0]+=ptsForMatlab[i]*ptsForMatlab[i];
			sum[1]+=ptsForMatlab[ptsNum+i]*ptsForMatlab[ptsNum+i];
		}
		scaleParameter=sqrt(sum[0]+sum[1]);

		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]/=scaleParameter;
			ptsForMatlab[ptsNum+i]/=scaleParameter;
		}
	}
	//update bound
	minx=maxx=ptsForMatlab[0];
	miny=maxy=ptsForMatlab[ptsNum+0];
	for (int i=1;i<ptsNum;i++)
	{
		minx=min(minx,ptsForMatlab[i]);
		miny=min(miny,ptsForMatlab[i]);
		maxx=max(maxx,ptsForMatlab[ptsNum+i]);
		maxy=max(maxy,ptsForMatlab[ptsNum+i]);
	}
}

void Shape::translate(double tx,double ty,int style)
{
	if(style==1)
	{
		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]-=tx;
			ptsForMatlab[ptsNum+i]-=ty;
			pts[i][0]-=tx;
			pts[i][1]-=ty;
		}

	}
	else 	if(style==2)
	{
		for (int i=0;i<ptsNum;i++)
		{
			ptsForMatlab[i]-=tx;
			ptsForMatlab[ptsNum+i]-=ty;
		}

	}
	minx-=tx;
	maxx-=tx;
	miny-=ty;
	maxy-=ty;
}

void Shape::getPtsIndex()
{
	pts_Index=new double *[ptsNum];
	for (int i=0;i<ptsNum;i++)
	{
		pts_Index[i]=new double[2];
	}
	double thres=1.5;
	for (int i=0;i<ptsNum;i++)
	{
		for (int j=0;j<width;j++)
		{
			for (int k=0;k<height;k++)
			{
				if (sqrt((pts[i][0]-j)*(pts[i][0]-j)+(pts[i][1]-k)*(pts[i][1]-k))<thres)
				{
					pts_Index[i][0]=j;
					pts_Index[i][1]=k;
				}
			}
		}
	}
}

double *Shape::getcenter()
{
	double *center=new double[2];
	center[0]=center[1]=0;
	for (int i=0;i<ptsNum;i++)
	{
		center[0]+=ptsForMatlab[i];
		center[1]+=ptsForMatlab[ptsNum+i];
	}
	center[0]/=ptsNum;
	center[1]/=ptsNum;
	//cout<<center[0]<<" "<<center[1]<<endl;
	return center;
}

void Shape::getMask(CvMat *triangleList)
{
	pix_num=0;
	Shape *dst=this;
	mask=cvCreateMat(height,width,CV_64FC1);
	mask_withindex=cvCreateMat(height,width,CV_64FC1);
	double newPoint[2];   //new position in src,2d 
	CvMat *point=cvCreateMat(3,1,CV_64FC1);
	CvMat *m=cvCreateMat(3,3,CV_64FC1);
	CvMat *m_inv=cvCreateMat(3,3,CV_64FC1);
	CvMat *weight=cvCreateMat(3,1,CV_64FC1);
	double x0,x1,x2,x3,y0,y1,y2,y3;
	double alpha,beta,gamma;
	int lastTriangleInd=0;

	////here we use all the images
	//for (int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		CV_MAT_ELEM(*mask,double,j,i)=1;
	//	}
	//}
	//return;
	//IplImage *dst
	int triangleInd;

	marginMask=cvCreateMat(height,width,CV_64FC1);

	
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			triangleInd=-1;
			//判断是否在某个三角形内
			x0=i;
			y0=j;

			//CV_MAT_ELEM(*point,double,0,0)=x0;
			//CV_MAT_ELEM(*point,double,1,0)=y0;
			//CV_MAT_ELEM(*point,double,2,0)=1;

			//caculate last triangle
			x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][0];
			x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][0];
			x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][0];

			y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][1];
			y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][1];
			y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][1];

			if (((x0<x1&&x0<x2&&x0<x3)||(x0>x1&&x0>x2&&x0>x3)||
				(y0<y1&&y0<y3&&y0<y3)||(y0>y1&&y0>y2&&y0>y3)))
			{
				;
			}
			else
			{
				////设定3*3矩阵
				//CV_MAT_ELEM(*m,double,0,0)=x1;
				//CV_MAT_ELEM(*m,double,0,1)=x2;
				//CV_MAT_ELEM(*m,double,0,2)=x3;

				//CV_MAT_ELEM(*m,double,1,0)=y1;
				//CV_MAT_ELEM(*m,double,1,1)=y2;
				//CV_MAT_ELEM(*m,double,1,2)=y3;

				//CV_MAT_ELEM(*m,double,2,0)=1;
				//CV_MAT_ELEM(*m,double,2,1)=1;
				//CV_MAT_ELEM(*m,double,2,2)=1;

				//cvInv(m,m_inv);
				//cvMatMul(m_inv,point,weight);

				//caculate alpha beta and gamma
				double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
				beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
				gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
				alpha=1-beta-gamma;
				if (alpha>=0&&beta>=0&&gamma>=0) //find the right triangles
				{
					triangleInd=lastTriangleInd;
					//break;
				}
			}



			if(triangleInd==-1)
			{
				for (int k=0;k<triangleList->rows;k++)
				{
					x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][0];
					x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][0];
					x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][0];

					y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][1];
					y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][1];
					y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][1];

					if (((x0<x1&&x0<x2&&x0<x3)||(x0>x1&&x0>x2&&x0>x3)||
						(y0<y1&&y0<y2&&y0<y3)||(y0>y1&&y0>y2&&y0>y3)))
					{
						continue;
					}				

					////设定3*3矩阵
					//CV_MAT_ELEM(*m,double,0,0)=x1;
					//CV_MAT_ELEM(*m,double,0,1)=x2;
					//CV_MAT_ELEM(*m,double,0,2)=x3;

					//CV_MAT_ELEM(*m,double,1,0)=y1;
					//CV_MAT_ELEM(*m,double,1,1)=y2;
					//CV_MAT_ELEM(*m,double,1,2)=y3;

					//CV_MAT_ELEM(*m,double,2,0)=1;
					//CV_MAT_ELEM(*m,double,2,1)=1;
					//CV_MAT_ELEM(*m,double,2,2)=1;

					//cvInv(m,m_inv);
					//cvMatMul(m_inv,point,weight);

					//caculate alpha beta and gamma
					double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
					beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
					gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
					alpha=1-beta-gamma;
					if (alpha>=0&&beta>=0&&gamma>=0) //find the right triangles
					{
						triangleInd=k;
						break;
					}

					//if (getWeights(point,dst,triangleList,k,alpha,beta,gamma)) //find the right triangles
					//{
					//	triangleInd=k;
					//	break;
					//}					
				}

			}

			if(triangleInd!=-1)// 如果找到三角形，则进行插值
			{
				CV_MAT_ELEM(*mask,double,j,i)=1;
				CV_MAT_ELEM(*mask_withindex,double,j,i)=pix_num;
				pix_num++;
				//	cvSet2D(dstImg,j,i,cvGet2D(img,newPoint[1],newPoint[0]));
				lastTriangleInd=triangleInd;

			}
			else
			{
				CV_MAT_ELEM(*mask,double,j,i)=0;
				CV_MAT_ELEM(*mask_withindex,double,j,i)=-1;
			}


		}
	}
	inv_mask=new int*[pix_num];
	for (int i=0;i<pix_num;i++)
	{
		inv_mask[i]=new int[2];
	}
	int tmpind=0;
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			if (CV_MAT_ELEM(*mask,double,j,i)==1)
			{
				inv_mask[tmpind][0]=i;
				inv_mask[tmpind][1]=j;
				tmpind++;
			}
		}
	}

//	return pix_num;
}

void Shape::getMargin()
{
	marginMask=cvCreateMat(height,width,CV_64FC1);
	//get margin
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			if (CV_MAT_ELEM(*mask,double,j,i)==1&&i>0&&i<width-1&&j>0&&j<height-1)
			{
				if (CV_MAT_ELEM(*mask,double,i-1,j)==0||
					CV_MAT_ELEM(*mask,double,i+1,j)==0||
					CV_MAT_ELEM(*mask,double,i,j-1)==0||
					CV_MAT_ELEM(*mask,double,i,j+1)==0||
					CV_MAT_ELEM(*mask,double,i-1,j-1)==0||
					CV_MAT_ELEM(*mask,double,i-1,j+1)==0||
					CV_MAT_ELEM(*mask,double,i+1,j-1)==0||
					CV_MAT_ELEM(*mask,double,i+1,j+1)==0)
				{
					CV_MAT_ELEM(*marginMask,double,j,i)=1;
				}
			}
			else
				CV_MAT_ELEM(*marginMask,double,j,i)=0;
		}
	}
}

void Shape::getTabel(CvMat *triangleList)
{
	affineTable=new affineParameters**[width];
	for (int i=0;i<width;i++)
	{
		affineTable[i]=new affineParameters *[height];
		for (int j=0;j<height;j++)
		{
			affineTable[i][j]=new affineParameters;
		}
	}
	double x0,x1,x2,x3,y0,y1,y2,y3;
	double alpha,beta,gamma;
	int lastTriangleInd=0;
	CvMat *point=cvCreateMat(3,1,CV_64FC1);

	////here we use all the images
	//for (int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		CV_MAT_ELEM(*mask,double,j,i)=1;
	//	}
	//}
	//return;
	//IplImage *dst
	int triangleInd;




	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
	/*		if (i==6&&j==102)
			{
				cout<<1<<endl;
			}*/
			triangleInd=-1;
			alpha=beta=gamma=0;
			//判断是否在某个三角形内
			x0=i;
			y0=j;

			CV_MAT_ELEM(*point,double,0,0)=x0;
			CV_MAT_ELEM(*point,double,1,0)=y0;
			CV_MAT_ELEM(*point,double,2,0)=1;

			//caculate last triangle
			x1=pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][0];
			x2=pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][0];
			x3=pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][0];

			y1=pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][1];
			y2=pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][1];
			y3=pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][1];

			if (((x0<x1&&x0<x2&&x0<x3)||(x0>x1&&x0>x2&&x0>x3)||
				(y0<y1&&y0<y3&&y0<y3)||(y0>y1&&y0>y2&&y0>y3)))
			{
				;
			}
			else
			{
				////设定3*3矩阵
				//CV_MAT_ELEM(*m,double,0,0)=x1;
				//CV_MAT_ELEM(*m,double,0,1)=x2;
				//CV_MAT_ELEM(*m,double,0,2)=x3;

				//CV_MAT_ELEM(*m,double,1,0)=y1;
				//CV_MAT_ELEM(*m,double,1,1)=y2;
				//CV_MAT_ELEM(*m,double,1,2)=y3;

				//CV_MAT_ELEM(*m,double,2,0)=1;
				//CV_MAT_ELEM(*m,double,2,1)=1;
				//CV_MAT_ELEM(*m,double,2,2)=1;

				//cvInv(m,m_inv);
				//cvMatMul(m_inv,point,weight);

				//caculate alpha beta and gamma
				double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
				beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
				gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
				alpha=1-beta-gamma;
				if (alpha>=0&&beta>=0&&gamma>=0) //find the right triangles
				{
					triangleInd=lastTriangleInd;
					//break;
				}
			}



			if(triangleInd==-1)
			{
				for (int k=0;k<triangleList->rows;k++)
				{
					x1=pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][0];
					x2=pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][0];
					x3=pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][0];

					y1=pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][1];
					y2=pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][1];
					y3=pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][1];

					if (((x0<x1&&x0<x2&&x0<x3)||(x0>x1&&x0>x2&&x0>x3)||
						(y0<y1&&y0<y2&&y0<y3)||(y0>y1&&y0>y2&&y0>y3)))
					{
						continue;
					}				

					////设定3*3矩阵
					//CV_MAT_ELEM(*m,double,0,0)=x1;
					//CV_MAT_ELEM(*m,double,0,1)=x2;
					//CV_MAT_ELEM(*m,double,0,2)=x3;

					//CV_MAT_ELEM(*m,double,1,0)=y1;
					//CV_MAT_ELEM(*m,double,1,1)=y2;
					//CV_MAT_ELEM(*m,double,1,2)=y3;

					//CV_MAT_ELEM(*m,double,2,0)=1;
					//CV_MAT_ELEM(*m,double,2,1)=1;
					//CV_MAT_ELEM(*m,double,2,2)=1;

					//cvInv(m,m_inv);
					//cvMatMul(m_inv,point,weight);

					//caculate alpha beta and gamma
					double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
					beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
					gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
					alpha=1-beta-gamma;
					if (alpha>=0&&beta>=0&&gamma>=0) //find the right triangles
					{
						triangleInd=k;
	
						break;
					}

					//if (getWeights(point,dst,triangleList,k,alpha,beta,gamma)) //find the right triangles
					//{
					//	triangleInd=k;
					//	break;
					//}					
				}

			}
		

			affineTable[i][j]->triangleInd=triangleInd;
			affineTable[i][j]->alpha=alpha;
			affineTable[i][j]->beta=beta;
			affineTable[i][j]->gamma=gamma;
		}
	}

}

void Shape::getTabel_strong(CvMat *triangleList)
{
	affineTable_strong=new affineParameters***[width];
	for (int i=0;i<width;i++)
	{
		affineTable_strong[i]=new affineParameters **[height];
		for (int j=0;j<height;j++)
		{
			affineTable_strong[i][j]=new affineParameters *[triangleList->rows];
			for (int k=0;k<triangleList->rows;k++)
			{
				affineTable_strong[i][j][k]=new affineParameters;
			}
		}
	}
	double x0,x1,x2,x3,y0,y1,y2,y3;
	double alpha,beta,gamma,fenmu;
	int lastTriangleInd=0;
	CvMat *point=cvCreateMat(3,1,CV_64FC1);

	////here we use all the images
	//for (int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		CV_MAT_ELEM(*mask,double,j,i)=1;
	//	}
	//}
	//return;
	//IplImage *dst
	int triangleInd;




	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			x0=i;
			y0=j;
				for (int k=0;k<triangleList->rows;k++)
				{
					x1=pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][0];
					x2=pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][0];
					x3=pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][0];

					y1=pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][1];
					y2=pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][1];
					y3=pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][1];

					fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
					beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
					gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
					alpha=1-beta-gamma;


					//affineTable_strong[i][j]->triangleInd=triangleInd;
					affineTable_strong[i][j][k]->alpha=alpha;
					affineTable_strong[i][j][k]->beta=beta;
					affineTable_strong[i][j][k]->gamma=gamma;
					
					//if (getWeights(point,dst,triangleList,k,alpha,beta,gamma)) //find the right triangles
					//{
					//	triangleInd=k;
					//	break;
					//}					
				}

			}

		}
}

//pts=triangles=NULL;
//ptsForMatlab=NULL;
//affineTable=NULL;
//setHostImage(input->hostImage);
//getVertex(input->ptsForMatlab,input->ptsNum,input->width,input->height);
//mask
//
void Shape::save(ofstream &out)
{
	SL_Basis slEngine;
	if (hostImage!=NULL)
	{
		out<<1<<endl;
		slEngine.saveMatrix(out,hostImage);
	}
	else
		out<<0<<endl;
	out<<pix_num<<" "<<ptsNum<<" "<<width<<" "<<height<<endl;
	out<<minx<<" "<<miny<<" "<<maxx<<" "<<maxy<<endl;
	slEngine.saveMatrix(out,ptsForMatlab,ptsNum*2);
	slEngine.saveMatrix(out,mask);

	//affinetabel
	if (affineTable!=NULL)
	{
		out<<1<<" "<<width<<" "<<height<<endl;
		for (int i=0;i<width;i++)
		{
			for (int j=0;j<height;j++)
			{
				out<<affineTable[i][j]->alpha<<" "<<
					affineTable[i][j]->beta<<" "<<
					affineTable[i][j]->gamma<<" "<<
					affineTable[i][j]->triangleInd<<endl;
			}
		}
	}
	else
		out<<0<<endl;

	slEngine.saveMatrix(out,pts,ptsNum,2);
	slEngine.saveMatrix(out,pts_Index,ptsNum,2);
	slEngine.saveMatrix(out,inv_mask,pix_num,2);
	slEngine.saveMatrix(out,mask_withindex);
}

void Shape::load(ifstream &in)
{
	init();
	SL_Basis slEngine;
	int isImage;
	in>>isImage;
	if (isImage)
	{
		hostImage=slEngine.loadMatrix(in,hostImage);
	}
	in>>pix_num>>ptsNum>>width>>height;
	in>>minx>>miny>>maxx>>maxy;

	ptsForMatlab=slEngine.loadMatrix(in,ptsForMatlab);

	//setHostImage(hostImage);
	//getVertex(ptsForMatlab,ptsNum,width,height);


	mask=slEngine.loadMatrix(in,mask);

	//read affine table
	int istabel;
	in>>istabel;
	if (istabel==1)
	{
		int tw,th;
		in>>tw>>th;
		affineTable=new affineParameters**[tw];
		for (int i=0;i<tw;i++)
		{
			affineTable[i]=new affineParameters *[th];
			for (int j=0;j<th;j++)
			{
				affineTable[i][j]=new affineParameters;
			}
		}
		for(int i=0;i<tw;i++)
		{
			for (int j=0;j<th;j++)
			{
				in>>affineTable[i][j]->alpha>>
					affineTable[i][j]->beta>>
					affineTable[i][j]->gamma>>
					affineTable[i][j]->triangleInd;
			}
		}
	}

	pts=slEngine.loadMatrix(in,pts);
	pts_Index=slEngine.loadMatrix(in,pts_Index);
	inv_mask=slEngine.loadMatrix(in,inv_mask);
	mask_withindex=slEngine.loadMatrix(in,mask_withindex);

	getWeightTabel(affineTable);

	////set the parameters for pts,ptsindex,tabel etc...
	//pts=new double *[ptsNum];
	//for (int i=0;i<ptsNum;i++)
	//{
	//	pts[i]=new double[2];
	//}
	//for (int i=0;i<ptsNum;i++)
	//{
	//	pts[i][0]=ptsForMatlab[i];
	//	pts[i][1]=ptsForMatlab[ptsNum+i];
	//}
	//getPtsIndex();

}

void Shape::save(string name)
{
	ofstream out(name.c_str(),ios::out);
	out<<width<<" "<<height;
	for (int i=0;i<ptsNum;i++)
	{
		out<<ptsForMatlab[i]/width<<" "<<ptsForMatlab[ptsNum+i]/height<<endl;
	}
	out.close();

	if (hostImage!=NULL)
	{
		string imgName=name;
		imgName.replace(imgName.end()-3,imgName.end(),"jpg"); 
		cvSaveImage(imgName.c_str(),hostImage);
	}
	out.close();

}

void Shape::getWeightTabel(affineParameters ***affineTable)
{
	//Mat m_triangleList=cvarrToMat(triangleList);
	weightTabel=new double *[pix_num];
	for (int i=0;i<pix_num;i++)
	{
		weightTabel[i]=new double [3];
	}
	//weightTabel.create(pix_num,3,CV_64FC1);
//	indexTabel.create(3,pix_num*2,CV_64FC1);
	int i,j;
	int ind=0;
	int triangleInd;
	for (i=minx;i<=maxx;i++)
	{
		for (j=miny;j<=maxy;j++)
		{
			triangleInd=affineTable[i][j]->triangleInd;
			if (triangleInd!=-1)
			{
				weightTabel[ind][0]=affineTable[i][j]->alpha;
				weightTabel[ind][1]=affineTable[i][j]->beta;
				weightTabel[ind][2]=affineTable[i][j]->gamma;

			/*	weightTabel.at<double>(ind,0)=affineTable[i][j]->alpha;
				weightTabel.at<double>(ind,1)=affineTable[i][j]->beta;
				weightTabel.at<double>(ind,2)=affineTable[i][j]->gamma;*/
		//		indexTabel.at<double>(0,ind)=(int);
				ind++;
			}
		}
	}
}