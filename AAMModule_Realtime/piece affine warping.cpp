#include "Piece Affine Warping.h"


PieceAffineWarpping::PieceAffineWarpping()
{
	lastTriangleInd=0;
	Interpolation=false;
}

IplImage * PieceAffineWarpping::piecewiseAffineWarping(IplImage *img,IplImage *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,affineParameters ***affineTable)
{
	if(!drawon)//if no preferred dst image,create one
		dstImg=cvCreateImage(cvSize(dst->width,dst->height),img->depth,img->nChannels);
	m_dstImg=cvarrToMat(dstImg);
	m_img=cvarrToMat(img);

	m_dstImg=0;
	//for (int i=0;i<dst->width;i++)
	//{
	//	for (int j=0;j<dst->height;j++)
	//	{
	//		cvSet2D(dstImg,j,i,cvScalar(0,0,0));
	//	}
	//}
	//cvNamedWindow("Warped Function");
	//cvShowImage("Warped Function",dstImg);
	//cvWaitKey();
	//CvMat *newPoint=cvCreateMat(3,1,CV_64FC1);
	
	CvMat *point=cvCreateMat(3,1,CV_64FC1);
	//CvMat *m=cvCreateMat(3,3,CV_64FC1);
	//CvMat *m_inv=cvCreateMat(3,3,CV_64FC1);
	//CvMat *weight=cvCreateMat(3,1,CV_64FC1);
	
	Mat m_triangleList=cvarrToMat(triangleList);
	//Mat point(3,1,CV_64FC1);
	//Mat m(3,3,CV_64FC1);
	//Mat m_inv(3,3,CV_64FC1);
	//Mat weight(3,1,CV_64FC1);
	
	int triangleInd;
	double alpha,beta,gamma;
	double x0,x1,x2,x3,y0,y1,y2,y3;
	double newPoint[2];   //new position in src,2d 

	

	//IplImage *nimg=cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
	//cvCopy(img,nimg);
	//int loopI=dst->maxx-dst->minx+1;
	//int loopJ=dst->maxy-dst->miny+1;

	int i,j,k;

	double intX,intY;
	double ratioX,ratioY;
	double tpx1,tpx2;
//	#pragma omp parallel for //firstprivate(triangleInd) 
	for (i=dst->minx;i<=dst->maxx;i++)
	{
		for (j=dst->miny;j<=dst->maxy;j++)
		{
	//for(int m=0;m<loopI*loopJ;m++)
	//{
	//	//#pragma omp critical

	//	{

			
		/*	i=m/loopJ+dst->minx;
			j=m%loopJ+dst->miny;*/
			//cout<<i<<" "<<j<<endl;
		
			{
				if (affineTable!=NULL)
				{
					//#pragma omp critical
					{
						triangleInd=affineTable[i][j]->triangleInd;
						if (triangleInd!=-1)
						{
							alpha=affineTable[i][j]->alpha;
							beta=affineTable[i][j]->beta;
							gamma=affineTable[i][j]->gamma;
						}
						
					}
				}
			
				else
				{
					triangleInd=-1;
					//判断是否在某个三角形内
					x0=i;
					y0=j;

					CV_MAT_ELEM(*point,double,0,0)=x0;
					CV_MAT_ELEM(*point,double,1,0)=y0;
					CV_MAT_ELEM(*point,double,2,0)=1;

					//caculate last triangle
					/*		x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][0];
					x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][0];
					x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][0];

					y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][1];
					y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][1];
					y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][1];*/

					x1=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,0)][0];
					x2=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,1)][0];
					x3=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,2)][0];

					y1=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,0)][1];
					y2=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,1)][1];
					y3=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,2)][1];

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
						//alpha=CV_MAT_ELEM(*weight,double,0,0);
						//beta=CV_MAT_ELEM(*weight,double,1,0);
						//gamma=CV_MAT_ELEM(*weight,double,2,0);


						//caculate alpha beta and gamma
						/*	double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
						beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
						gamma=((y0-y1)*(x2-x1)-(x0-y1)*(y2-y1))/fenmu;*/
						double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
						beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
						gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
						alpha=1-beta-gamma;
						//cout<<alpha<<" "<<beta<<" "<<gamma<<endl;

						////caculate alpha beta and gamma

						//cout<<alpha<<" "<<beta<<" "<<gamma<<endl;



						if (alpha>=0&&beta>=0&&gamma>=0) //find the right triangles
						{
							triangleInd=lastTriangleInd;
							//break;
						}
					}



					if(triangleInd==-1)
					{
						for (k=0;k<triangleList->rows;k++)
						{
							/*	x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][0];
							x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][0];
							x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][0];

							y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][1];
							y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][1];
							y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][1];*/

							x1=dst->pts[(int)m_triangleList.at<double>(k,0)][0];
							x2=dst->pts[(int)m_triangleList.at<double>(k,1)][0];
							x3=dst->pts[(int)m_triangleList.at<double>(k,2)][0];

							y1=dst->pts[(int)m_triangleList.at<double>(k,0)][1];
							y2=dst->pts[(int)m_triangleList.at<double>(k,1)][1];
							y3=dst->pts[(int)m_triangleList.at<double>(k,2)][1];


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

							//////caculate alpha beta and gamma
							//alpha=CV_MAT_ELEM(*weight,double,0,0);
							//beta=CV_MAT_ELEM(*weight,double,1,0);
							//gamma=CV_MAT_ELEM(*weight,double,2,0);

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
				}
				//#pragma omp barrier  
				if(triangleInd!=-1)// 如果找到三角形，则进行插值
				{
					//x1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][0];
					//x2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][0];
					//x3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][0];

					//y1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][1];
					//y2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][1];
					//y3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][1];

					////newPoint[0]=(alpha*x1+beta*x2+gamma*x3);
					////newPoint[1]=(alpha*y1+beta*y2+gamma*y3);

					////	cout<<newPoint[0]<<" "<<newPoint[1]<<endl;
					//cvSet2D(dstImg,j,i,cvGet2D(img,alpha*x1+beta*x2+gamma*x3,alpha*y1+beta*y2+gamma*y3));
					//lastTriangleInd=triangleInd;

					x1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][0];
					x2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][0];
					x3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][0];

					y1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][1];
					y2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][1];
					y3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][1];

					newPoint[0]=(alpha*x1+beta*x2+gamma*x3);
					newPoint[1]=(alpha*y1+beta*y2+gamma*y3);

					//if(Interpolation)
					if(1)
					{
						intX=(int)newPoint[0];
						intY=(int)newPoint[1];
						ratioX=(newPoint[0]-intX);
						ratioY=newPoint[1]-intY;
						tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
							m_img.at<uchar>(intY,intX+1);
						tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
							m_img.at<uchar>(intY+1,intX+1);
						m_dstImg.at<uchar>(j,i)=(1-ratioY)*tpx1+ratioY*tpx2;

					}
					

					//	cout<<newPoint[0]<<" "<<newPoint[1]<<endl;
			/*		cvSet2D(dstImg,j,i,cvGet2D(img,alpha*y1+beta*y2+gamma*y3,
						alpha*x1+beta*x2+gamma*x3));*/
				//	m_dstImg.at<Vec3b>(j,i)=m_img.at<Vec3b>(alpha*y1+beta*y2+gamma*y3,alpha*x1+beta*x2+gamma*x3);
					//m_dstImg.at<Vec3b>(j,i)=m_img.at<Vec3b>(newPoint[1],newPoint[0]);
					else
					{
						m_dstImg.at<uchar>(j,i)=m_img.at<uchar>(newPoint[1],newPoint[0]);
					}
					
					lastTriangleInd=triangleInd;

					//cvDrawCircle(nimg,cvPoint(newPoint[0],newPoint[1]),3,cvScalar(255,255,255));
					
				}


			}
		}
		//cvNamedWindow("pts");
		//cvShowImage("pts",nimg);
		////cvWaitKey();
		//namedWindow("1");
		//imshow("1",m_dstImg);
		//waitKey();

		
	}
	return dstImg;
}


IplImage * PieceAffineWarpping::piecewiseAffineWarping(IplImage *img,IplImage *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,double **weightTabel,int **indexTabel)
{
	if(!drawon)//if no preferred dst image,create one
		dstImg=cvCreateImage(cvSize(dst->width,dst->height),img->depth,img->nChannels);
	m_dstImg=cvarrToMat(dstImg);
	m_img=cvarrToMat(img);

	m_dstImg=0;
	//for (int i=0;i<dst->width;i++)
	//{
	//	for (int j=0;j<dst->height;j++)
	//	{
	//		cvSet2D(dstImg,j,i,cvScalar(0,0,0));
	//	}
	//}
	//cvNamedWindow("Warped Function");
	//cvShowImage("Warped Function",dstImg);
	//cvWaitKey();
	//CvMat *newPoint=cvCreateMat(3,1,CV_64FC1);
	
	//CvMat *point=cvCreateMat(3,1,CV_64FC1);
	//CvMat *m=cvCreateMat(3,3,CV_64FC1);
	//CvMat *m_inv=cvCreateMat(3,3,CV_64FC1);
	//CvMat *weight=cvCreateMat(3,1,CV_64FC1);
	double newPoint[2];   //new position in src,2d 
	int pix_num=dst->pix_num;
	int i,j;
	int intX,intY;
	double ratioX,ratioY;
	double tpx1,tpx2;
	for (i=0;i<pix_num;i++)
	{
		newPoint[0]=newPoint[1]=0;
		for (j=0;j<3;j++)
		{
			newPoint[0]+=weightTabel[i][j]*src->pts[indexTabel[j][i]][0];
			newPoint[1]+=weightTabel[i][j]*src->pts[indexTabel[j][i]][1];
		}
		if(Interpolation)
		{
			intX=(int)newPoint[0];
			intY=(int)newPoint[1];
			ratioX=(newPoint[0]-intX);
			ratioY=newPoint[1]-intY;
			tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
				m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
				m_img.at<uchar>(intY+1,intX+1);
			m_dstImg.at<uchar>(dst->inv_mask[i][1],dst->inv_mask[i][0])=(1-ratioY)*tpx1+ratioY*tpx2;

		}
		else
		{
			m_dstImg.at<uchar>(dst->inv_mask[i][1],dst->inv_mask[i][0])=m_img.at<uchar>(newPoint[1],newPoint[0]);
		}
	}
	
	return dstImg;
}

extern "C" void test_PAWarping(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight);


IplImage * PieceAffineWarpping::piecewiseAffineWarping_GPU(IplImage *img,IplImage *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,double **weightTabel,int **indexTabel)
{
	int MAX_COUNT_NUM_LOCAL= 100000;
	int MAX_PIXEL_NUM_LOCAL=1000000;
	if(!drawon)//if no preferred dst image,create one
		dstImg=cvCreateImage(cvSize(dst->width,dst->height),img->depth,img->nChannels);
	m_dstImg=cvarrToMat(dstImg);
	m_img=cvarrToMat(img);

	int pix_num=dst->pix_num;
	int i,j;
	int intX,intY;
	double ratioX,ratioY;
	double tpx1,tpx2;

	m_dstImg=0;

	float *warpTabel;
	float *triangle_indexTabel;
	float *pts_pos;

	float *inputImg,*outputImg;
	int width,height;
	int tWidth,tHeight;

	width=img->width;height=img->height;
	tWidth=dstImg->width;tHeight=dstImg->height;


	warpTabel=new float[MAX_PIXEL_NUM_LOCAL*3];
	triangle_indexTabel=new float[MAX_PIXEL_NUM_LOCAL*3];
	pts_pos=new float[MAX_COUNT_NUM_LOCAL];
	//pts_posY=new float[MAX_COUNT_NUM_LOCAL];
	

	inputImg=new float[MAX_PIXEL_NUM_LOCAL];
	outputImg=new float[MAX_PIXEL_NUM_LOCAL];

	int cNum;
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			cNum=i*tWidth+j;
			warpTabel[3*cNum+0]=0;
			warpTabel[3*cNum+1]=0;
			warpTabel[3*cNum+2]=0;
			triangle_indexTabel[3*cNum+0]=-1;
			triangle_indexTabel[3*cNum+1]=-1;
			triangle_indexTabel[3*cNum+2]=-1;
		}
	}
	
	for (i=0;i<pix_num;i++)
	{
		cNum=dst->inv_mask[i][1]*tWidth+dst->inv_mask[i][0];
		//cout<<cNum<<endl;
		for (j=0;j<3;j++)
		{
			warpTabel[cNum*3+j]=weightTabel[i][j];
			triangle_indexTabel[cNum*3+j]=indexTabel[j][i];
		}
	}

	for (i=0;i<src->ptsNum;i++)
	{
		pts_pos[i]=src->pts[i][0];
		pts_pos[src->ptsNum+i]=src->pts[i][1];
	}
	
	for (i=0;i<height;i++)
	{
		for (j=0;j<width;j++)
		{
			inputImg[i*width+j]=m_img.at<uchar>(i,j);
		}
	}

	

	test_PAWarping(warpTabel,triangle_indexTabel,pts_pos,src->ptsNum,inputImg,width,height,outputImg,tWidth,tHeight);
	
	////cpu version
	//int cPixelInd;
	//int offset;
	//for (i=0;i<tHeight;i++)
	//{
	//	for (j=0;j<tWidth;j++)
	//	{
	//		offset=i*tWidth+j;
	//		
	//		int ind1=triangle_indexTabel[3*offset];
	//		if (ind1!=-1)
	//		{
	//			int ind2=triangle_indexTabel[3*offset+1];
	//			int ind3=triangle_indexTabel[3*offset+2];



	//			cPixelInd=CV_MAT_ELEM(*dst->mask_withindex,double,i,j);
	//		/*	cout<<warpTabel[offset*3+0]<<" "<<warpTabel[offset*3+1]<<
	//				" "<<warpTabel[offset*3+2]<<endl;
	//			
	//			cout<<weightTabel[cPixelInd][0]<<" "<<
	//				weightTabel[cPixelInd][1]<<" "<<
	//				weightTabel[cPixelInd][2]<<endl;*/

	//			//cout<<pts_posX[ind1]<<" "<<pts_posX[ind2]<<" "<<pts_posX[ind3]<<endl;
	//			//cout<<src->pts[indexTabel[0][cPixelInd]][0]<<" "<<
	//			//	src->pts[indexTabel[1][cPixelInd]][0]<<" "<<
	//			//	src->pts[indexTabel[2][cPixelInd]][0]<<endl;

	//			//cout<<offset<<" "<<dst->inv_mask[cPixelInd][1]*tWidth+dst->inv_mask[cPixelInd][0]<<endl;
	//			


	//			float x=0;
	//			float y=0;
	//			x=warpTabel[offset*3+0]*pts_posX[ind1]+
	//				warpTabel[offset*3+1]*pts_posX[ind2]+
	//				warpTabel[offset*3+2]*pts_posX[ind3];
	//			y=warpTabel[offset*3+0]*pts_posY[ind1]+
	//				warpTabel[offset*3+1]*pts_posY[ind2]+
	//				warpTabel[offset*3+2]*pts_posY[ind3]; 
	//			

	//			/*cout<<x<<" "<<y<<endl;
	//			cout<<weightTabel[cPixelInd][0]*src->pts[indexTabel[0][cPixelInd]][0]+
	//				weightTabel[cPixelInd][1]*src->pts[indexTabel[1][cPixelInd]][0]+
	//				weightTabel[cPixelInd][2]*src->pts[indexTabel[2][cPixelInd]][0]<<endl;*/

	//			intX=(int)x;
	//			intY=(int)y;
	//			ratioX=(x-intX);
	//			ratioY=y-intY;
	//		/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
	//				m_img.at<uchar>(intY,intX+1);
	//			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
	//				m_img.at<uchar>(intY+1,intX+1);*/
	//			tpx1=(1-ratioX)*inputImg[(intY*width+intX)]+ratioX*
	//				inputImg[(intY*width+(intX+1))];
	//			tpx2=(1-ratioX)*inputImg[((intY+1)*width+intX)]+ratioX*
	//				inputImg[((intY+1)*width+(intX+1))];

	//			outputImg[offset]=(1-ratioY)*tpx1+ratioY*tpx2;//<<" "<<(int)m_img.at<uchar>(y,x)<<endl;
	//			//outputImg[offset]=inputImg[offset];
	//		}
	//		else
	//		{
	//			outputImg[offset]=0;
	//		}

	//		
	//	}
	//}

	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			m_dstImg.at<uchar>(i,j)=outputImg[i*tWidth+j];
		}
	}


	delete []warpTabel;
	delete []triangle_indexTabel;
	delete []pts_pos;
	//delete []pts_posY;
	delete []inputImg;
	delete []outputImg;

	return dstImg;
}


CvMat * PieceAffineWarpping::piecewiseAffineWarping(CvMat *img,CvMat *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,double **weightTabel,int **indexTabel)
{
	if(!drawon)//if no preferred dst image,create one
		dstImg=cvCreateMat(img->cols,img->cols,img->type);
	m_dstImg=cvarrToMat(dstImg);
	m_img=cvarrToMat(img);

	m_dstImg=0;

	double newPoint[2];   //new position in src,2d 
	int pix_num=dst->pix_num;
	int i,j;
	int intX,intY;
	double ratioX,ratioY;
	double tpx1,tpx2;
	for (i=0;i<pix_num;i++)
	{
		newPoint[0]=newPoint[1]=0;
		for (j=0;j<3;j++)
		{
			newPoint[0]+=weightTabel[i][j]*src->pts[indexTabel[j][i]][0];
			newPoint[1]+=weightTabel[i][j]*src->pts[indexTabel[j][i]][1];
		}
		if(1)
		{
			intX=(int)newPoint[0];
			intY=(int)newPoint[1];
			ratioX=(newPoint[0]-intX);
			ratioY=newPoint[1]-intY;
			tpx1=(1-ratioX)*m_img.at<double>(intY,intX)+ratioX*
				m_img.at<double>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<double>(intY+1,intX)+ratioX*
				m_img.at<double>(intY+1,intX+1);
			m_dstImg.at<double>(dst->inv_mask[i][1],dst->inv_mask[i][0])=(1-ratioY)*tpx1+ratioY*tpx2;

		}
		else
		{
			m_dstImg.at<double>(dst->inv_mask[i][1],dst->inv_mask[i][0])=m_img.at<double>(newPoint[1],newPoint[0]);
		}
	}
	return dstImg;
}


CvMat * PieceAffineWarpping::piecewiseAffineWarping(CvMat *img,CvMat *dstImg,Shape *src,Shape *dst,CvMat *triangleList,bool drawon,affineParameters ***affineTable)
{
	if(!drawon)//if no preferred dst image,create one
		dstImg=cvCreateMat(img->cols,img->cols,img->type);
	m_dstImg=cvarrToMat(dstImg);
	m_img=cvarrToMat(img);

	m_dstImg=0;
	//for (int i=0;i<dst->width;i++)
	//{
	//	for (int j=0;j<dst->height;j++)
	//	{
	//		cvSet2D(dstImg,j,i,cvScalar(0,0,0));
	//	}
	//}
	//cvNamedWindow("Warped Function");
	//cvShowImage("Warped Function",dstImg);
	//cvWaitKey();
	//CvMat *newPoint=cvCreateMat(3,1,CV_64FC1);
	
	CvMat *point=cvCreateMat(3,1,CV_64FC1);
	//CvMat *m=cvCreateMat(3,3,CV_64FC1);
	//CvMat *m_inv=cvCreateMat(3,3,CV_64FC1);
	//CvMat *weight=cvCreateMat(3,1,CV_64FC1);
	
	Mat m_triangleList=cvarrToMat(triangleList);
	//Mat point(3,1,CV_64FC1);
	//Mat m(3,3,CV_64FC1);
	//Mat m_inv(3,3,CV_64FC1);
	//Mat weight(3,1,CV_64FC1);
	
	int triangleInd;
	double alpha,beta,gamma;
	double x0,x1,x2,x3,y0,y1,y2,y3;
	double newPoint[2];   //new position in src,2d 

	

	//IplImage *nimg=cvCreateImage(cvGetSize(img),img->depth,img->nChannels);
	//cvCopy(img,nimg);
	//int loopI=dst->maxx-dst->minx+1;
	//int loopJ=dst->maxy-dst->miny+1;

	int i,j,k;

	double intX,intY;
	double ratioX,ratioY;
	double tpx1,tpx2;
//	#pragma omp parallel for //firstprivate(triangleInd) 
	for (i=dst->minx;i<=dst->maxx;i++)
	{
		for (j=dst->miny;j<=dst->maxy;j++)
		{
	//for(int m=0;m<loopI*loopJ;m++)
	//{
	//	//#pragma omp critical

	//	{

			
		/*	i=m/loopJ+dst->minx;
			j=m%loopJ+dst->miny;*/
			//cout<<i<<" "<<j<<endl;
		
			{
				if (affineTable!=NULL)
				{
					//#pragma omp critical
					{
						triangleInd=affineTable[i][j]->triangleInd;
						if (triangleInd!=-1)
						{
							alpha=affineTable[i][j]->alpha;
							beta=affineTable[i][j]->beta;
							gamma=affineTable[i][j]->gamma;
						}
						
					}
				}
			
				else
				{
					triangleInd=-1;
					//判断是否在某个三角形内
					x0=i;
					y0=j;

					CV_MAT_ELEM(*point,double,0,0)=x0;
					CV_MAT_ELEM(*point,double,1,0)=y0;
					CV_MAT_ELEM(*point,double,2,0)=1;

					//caculate last triangle
					/*		x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][0];
					x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][0];
					x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][0];

					y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0)][1];
					y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1)][1];
					y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2)][1];*/

					x1=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,0)][0];
					x2=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,1)][0];
					x3=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,2)][0];

					y1=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,0)][1];
					y2=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,1)][1];
					y3=dst->pts[(int)m_triangleList.at<double>(lastTriangleInd,2)][1];

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
						//alpha=CV_MAT_ELEM(*weight,double,0,0);
						//beta=CV_MAT_ELEM(*weight,double,1,0);
						//gamma=CV_MAT_ELEM(*weight,double,2,0);


						//caculate alpha beta and gamma
						/*	double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
						beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
						gamma=((y0-y1)*(x2-x1)-(x0-y1)*(y2-y1))/fenmu;*/
						double fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
						beta=((x0-x1)*(y3-y1)-(y0-y1)*(x3-x1))/fenmu;
						gamma=((y0-y1)*(x2-x1)-(x0-x1)*(y2-y1))/fenmu;
						alpha=1-beta-gamma;
						//cout<<alpha<<" "<<beta<<" "<<gamma<<endl;

						////caculate alpha beta and gamma

						//cout<<alpha<<" "<<beta<<" "<<gamma<<endl;



						if (alpha>=0&&beta>=0&&gamma>=0) //find the right triangles
						{
							triangleInd=lastTriangleInd;
							//break;
						}
					}



					if(triangleInd==-1)
					{
						for (k=0;k<triangleList->rows;k++)
						{
							/*	x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][0];
							x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][0];
							x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][0];

							y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][1];
							y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][1];
							y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][1];*/

							x1=dst->pts[(int)m_triangleList.at<double>(k,0)][0];
							x2=dst->pts[(int)m_triangleList.at<double>(k,1)][0];
							x3=dst->pts[(int)m_triangleList.at<double>(k,2)][0];

							y1=dst->pts[(int)m_triangleList.at<double>(k,0)][1];
							y2=dst->pts[(int)m_triangleList.at<double>(k,1)][1];
							y3=dst->pts[(int)m_triangleList.at<double>(k,2)][1];


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

							//////caculate alpha beta and gamma
							//alpha=CV_MAT_ELEM(*weight,double,0,0);
							//beta=CV_MAT_ELEM(*weight,double,1,0);
							//gamma=CV_MAT_ELEM(*weight,double,2,0);

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
				}
				//#pragma omp barrier  
				if(triangleInd!=-1)// 如果找到三角形，则进行插值
				{
					//x1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][0];
					//x2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][0];
					//x3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][0];

					//y1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][1];
					//y2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][1];
					//y3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][1];

					////newPoint[0]=(alpha*x1+beta*x2+gamma*x3);
					////newPoint[1]=(alpha*y1+beta*y2+gamma*y3);

					////	cout<<newPoint[0]<<" "<<newPoint[1]<<endl;
					//cvSet2D(dstImg,j,i,cvGet2D(img,alpha*x1+beta*x2+gamma*x3,alpha*y1+beta*y2+gamma*y3));
					//lastTriangleInd=triangleInd;

					x1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][0];
					x2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][0];
					x3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][0];

					y1=src->pts[(int)m_triangleList.at<double>(triangleInd,0)][1];
					y2=src->pts[(int)m_triangleList.at<double>(triangleInd,1)][1];
					y3=src->pts[(int)m_triangleList.at<double>(triangleInd,2)][1];

					newPoint[0]=(alpha*x1+beta*x2+gamma*x3);
					newPoint[1]=(alpha*y1+beta*y2+gamma*y3);

					if(Interpolation)
					{
						intX=(int)newPoint[0];
						intY=(int)newPoint[1];
						ratioX=(newPoint[0]-intX);
						ratioY=newPoint[1]-intY;
						tpx1=(1-ratioX)*m_img.at<double>(intY,intX)+ratioX*
							m_img.at<double>(intY,intX+1);
						tpx2=(1-ratioX)*m_img.at<double>(intY+1,intX)+ratioX*
							m_img.at<double>(intY+1,intX+1);
						m_dstImg.at<double>(j,i)=(1-ratioY)*tpx1+ratioY*tpx2;

					}
					

					//	cout<<newPoint[0]<<" "<<newPoint[1]<<endl;
			/*		cvSet2D(dstImg,j,i,cvGet2D(img,alpha*y1+beta*y2+gamma*y3,
						alpha*x1+beta*x2+gamma*x3));*/
				//	m_dstImg.at<Vec3b>(j,i)=m_img.at<Vec3b>(alpha*y1+beta*y2+gamma*y3,alpha*x1+beta*x2+gamma*x3);
					//m_dstImg.at<Vec3b>(j,i)=m_img.at<Vec3b>(newPoint[1],newPoint[0]);
					else
					{
						m_dstImg.at<double>(j,i)=m_img.at<double>(newPoint[1],newPoint[0]);
					}
					
					lastTriangleInd=triangleInd;

					//cvDrawCircle(nimg,cvPoint(newPoint[0],newPoint[1]),3,cvScalar(255,255,255));
					
				}


			}
		}
		//cvNamedWindow("pts");
		//cvShowImage("pts",nimg);
		////cvWaitKey();
		//namedWindow("1");
		//imshow("1",m_dstImg);
		//waitKey();

		
	}
	return dstImg;
}

bool PieceAffineWarpping::getWeights(CvMat *point,Shape *dst,CvMat *triangleList,int k,double &alpha,double &beta,double &gamma)
{
	CvMat *m=cvCreateMat(3,3,CV_64FC1);
	CvMat *m_inv=cvCreateMat(3,3,CV_64FC1);
	CvMat *weight=cvCreateMat(3,1,CV_64FC1);

	double x0,x1,x2,x3,y0,y1,y2,y3;
	//double alpha,beta,gamma;
	x1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][0];
	x2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][0];
	x3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][0];

	y1=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,0)][1];
	y2=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,1)][1];
	y3=dst->pts[(int)CV_MAT_ELEM(*triangleList,double,k,2)][1];


	//设定3*3矩阵
	CV_MAT_ELEM(*m,double,0,0)=x1;
	CV_MAT_ELEM(*m,double,0,1)=x2;
	CV_MAT_ELEM(*m,double,0,2)=x3;

	CV_MAT_ELEM(*m,double,1,0)=y1;
	CV_MAT_ELEM(*m,double,1,1)=y2;
	CV_MAT_ELEM(*m,double,1,2)=y3;

	CV_MAT_ELEM(*m,double,2,0)=1;
	CV_MAT_ELEM(*m,double,2,1)=1;
	CV_MAT_ELEM(*m,double,2,2)=1;

	cvInv(m,m_inv);
	cvMatMul(m_inv,point,weight);

	//calculate alpha beta and gamma
	alpha=CV_MAT_ELEM(*weight,double,0,0);
	beta=CV_MAT_ELEM(*weight,double,1,0);
	gamma=CV_MAT_ELEM(*weight,double,2,0);

	cvReleaseMat(&m);
	cvReleaseMat(&m_inv);
	cvReleaseMat(&weight);
//	cvReleaseMat(&point);
	if (alpha>=0&&beta>=0&&gamma>=0)
		return true;
	return false;
}

//void PieceAffineWarpping::setTriangulation(Shape *shape,CvSubdiv2D *current,CvSubdiv2D *ref)
//{
//	;
//}