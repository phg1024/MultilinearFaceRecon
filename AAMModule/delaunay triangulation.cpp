#include "delaunay triangulation.h"

Delaunay_Tri::Delaunay_Tri()
{
	active_facet_color = CV_RGB( 255, 0, 0 );
	delaunay_color  = CV_RGB( 0,0,0);
	voronoi_color = CV_RGB(0, 180, 0);
	bkgnd_color = CV_RGB(255,255,255);

	shape=new Shape();

	triangleIndex=new int *[1000];
	for (int i=0;i<600;i++)
	{
		triangleIndex[i]=new int[3];
	}

	triangleNum=0;
}

Delaunay_Tri::~Delaunay_Tri()
{
	for (int i=0;i<600;i++)
	{
		delete []triangleIndex[i];
	}
	delete []triangleIndex;
	cvReleaseMat(&TriangleIndex);
}

void Delaunay_Tri::draw_subdiv( IplImage* img, CvSubdiv2D* subdiv,
				 CvScalar delaunay_color, CvScalar voronoi_color )
{
	CvSeqReader  reader;
	int i, total = subdiv->edges->total;
	int elem_size = subdiv->edges->elem_size;

	cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );

	for( i = 0; i < total; i++ )
	{
		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

		if( CV_IS_SET_ELEM( edge ))
		{
		//	draw_subdiv_edge( img, (CvSubdiv2DEdge)edge + 1, voronoi_color );
			draw_subdiv_edge( img, (CvSubdiv2DEdge)edge, delaunay_color );
			//draw_subdiv_facet(img,edge);
		}

		CV_NEXT_SEQ_ELEM( elem_size, reader );
	}

	////draw convex hull
	//for (i=0;i<hullcount-1;i++)
	//{
	//	//CvPoint p0=CV_MAT_ELEM(*hullMat,CV_32SC2,0,i);
	//	//CvPoint p1=CV_MAT_ELEM(*hullMat,CV_32SC2,0,i+1);
	//	//cout<<ind<<" "<<ind_next<<endl;
	//	//cvLine(img,cvPoint(pts[ind][0],pts[ind][1]),cvPoint(pts[ind_next][0],pts[ind_next][1]), CV_RGB( 0, 255, 0 ),3);
	//	cvLine(img,hullPoints[hull[i]],hullPoints[hull[i+1]], CV_RGB( 0, 255, 0 ),3);
	//}
}

void Delaunay_Tri::draw_subdiv_point( IplImage* img, CvPoint2D32f fp, CvScalar color )
{
	cvCircle( img, cvPoint(cvRound(fp.x), cvRound(fp.y)), 3, color, CV_FILLED, 8, 0 );
}

void Delaunay_Tri::paint_voronoi( CvSubdiv2D* subdiv, IplImage* img )
{
	CvSeqReader  reader;
	int i, total = subdiv->edges->total;
	int elem_size = subdiv->edges->elem_size;

	cvCalcSubdivVoronoi2D( subdiv );

	cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );

	for( i = 0; i < total; i++ )
	{
		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

		if( CV_IS_SET_ELEM( edge ))
		{
			CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge;
			// left
			draw_subdiv_facet( img, cvSubdiv2DRotateEdge( e, 1 ));

			// right
			draw_subdiv_facet( img, cvSubdiv2DRotateEdge( e, 3 ));
		}

		CV_NEXT_SEQ_ELEM( elem_size, reader );
	}
}

void Delaunay_Tri::draw_subdiv_facet( IplImage* img, CvSubdiv2DEdge edge )
{
	CvSubdiv2DEdge t = edge;
	int i, count = 0;
	CvPoint* buf = 0;

	// count number of edges in facet
	do
	{
		count++;
		t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
	} while (t != edge );

	buf = (CvPoint*)malloc( count * sizeof(buf[0]));

	// gather points
	t = edge;
	for( i = 0; i < count; i++ )
	{
		CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
		if( !pt ) break;
		buf[i] = cvPoint( cvRound(pt->pt.x), cvRound(pt->pt.y));
		t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
	}

	if( i == count )
	{
		CvSubdiv2DPoint* pt = cvSubdiv2DEdgeDst( cvSubdiv2DRotateEdge( edge, 1 ));
		cvFillConvexPoly( img, buf, count, CV_RGB(rand()&255,rand()&255,rand()&255), CV_AA, 0 );
		cvPolyLine( img, &buf, &count, 1, 1, CV_RGB(0,0,0), 1, CV_AA, 0);
		draw_subdiv_point( img, pt->pt, CV_RGB(0,0,0));
	}
	free( buf );
}



void Delaunay_Tri::draw_subdiv_edge( IplImage* img, CvSubdiv2DEdge edge, CvScalar color )
{
	CvSubdiv2DPoint* org_pt;
	CvSubdiv2DPoint* dst_pt;
	CvPoint2D32f org;
	CvPoint2D32f dst;
	CvPoint iorg, idst;

	org_pt = cvSubdiv2DEdgeOrg(edge);
	dst_pt = cvSubdiv2DEdgeDst(edge);

	if( org_pt && dst_pt )
	{
		org = org_pt->pt;
		dst = dst_pt->pt;
		if (org.x<0||dst.x<0||org.x>cvGetSize(img).width||dst.x>cvGetSize(img).width||
			org.y<0||dst.y<0||org.y>cvGetSize(img).height||dst.y>cvGetSize(img).height)
		{
			return;
		}
		iorg = cvPoint( cvRound( org.x ), cvRound( org.y ));
		idst = cvPoint( cvRound( dst.x ), cvRound( dst.y ));

		cvLine( img, iorg, idst, color, 1, CV_AA, 0 );
	}
}

CvSubdiv2D* Delaunay_Tri::init_delaunay( CvMemStorage* storage,
						  CvRect rect )
{
	CvSubdiv2D* subdiv;

	subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),
		sizeof(CvSubdiv2DPoint),
		sizeof(CvQuadEdge2D),
		storage );
	cvInitSubdivDelaunay2D( subdiv, rect );

	return subdiv;
}

void Delaunay_Tri::run_triangulation(Shape *inputShape)
{
//	shape->pts=getVertex(imgName);
	shape=inputShape;
	triangulation(shape->ptsForMatlab);
}

void Delaunay_Tri::getConvexHull()
{

	//get the convex hull
	int ptsNum=shape->ptsNum;
	double **pts=shape->pts;
	hullPoints = (CvPoint*)malloc( ptsNum * sizeof(hullPoints[0]));
	hull = (int*)malloc( ptsNum * sizeof(hull[0]));
	CvMat pointMat=cvMat( 1, ptsNum, CV_32SC2, hullPoints );
	CvMat hullMat = cvMat( 1, ptsNum, CV_32SC1, hull );
	for (int i=0;i<ptsNum;i++)
	{
		hullPoints[i].x=pts[i][0];
		hullPoints[i].y=pts[i][1];
	}
	//	cout<<pointMat.rows<<" "<<pointMat.cols<<endl;
	//	hullMat=cvCreateMat(1,ptsNum,CV_32SC2);
	cvConvexHull2( &pointMat, &hullMat, CV_CLOCKWISE, 0 );
	hullcount = hullMat.cols;
}


void Delaunay_Tri::findTriangles()
{
	CvSeqReader  reader;
	int i, total = subdiv->edges->total;
	//cout<<total<<endl;
	int elem_size = subdiv->edges->elem_size;
	cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );
	for(i = 0; i < total; i++ )
	{
		CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);
		//cvSubdiv2DEdgeOrg(edge);
		//cout<<i<<" "<<total<<endl;

		if( CV_IS_SET_ELEM( edge ))
		{
			CvSubdiv2DEdge e = (CvSubdiv2DEdge)edge;

			//cvNamedWindow("current triangle");

			//IplImage *img=cvCreateImage(cvGetSize(shape->hostImage),shape->hostImage->depth,shape->hostImage->nChannels);
			//cvCopyImage(shape->hostImage,img);
			//for (int kk=0;kk<shape->ptsNum;kk++)
			//{
			//	cvCircle(img,cvPoint(shape->pts[kk][0],shape->pts[kk][1]),1,delaunay_color);
			//}
			//draw_subdiv_edge(img,e,voronoi_color);
			//cvShowImage("current triangle",img);
			//cvWaitKey();


			
			FindTriangleFromEdge(e);

			CvSubdiv2DEdge e1 = cvSubdiv2DRotateEdge((CvSubdiv2DEdge)edge,2);

			FindTriangleFromEdge(e1);

		//	CvSubdiv2DEdge e2 = cvSubdiv2DRotateEdge(e1,2);

		//	FindTriangleFromEdge(e2);
		}
		CV_NEXT_SEQ_ELEM( elem_size, reader );
	}

	//feed into the Mat files
	TriangleIndex=cvCreateMat(triangleNum,3,CV_64FC1);
	for (int i=0;i<triangleNum;i++)
	{
		for (int j=0;j<3;j++)
		{
			CV_MAT_ELEM(*TriangleIndex,double,i,j)=triangleIndex[i][j];
		}
	}
}

//void draw_subdiv_point( IplImage* img, CvPoint2D32f fp, CvScalar color )
//{
//	cvCircle( img, cvPoint(cvRound(fp.x), cvRound(fp.y)), 3, color, CV_FILLED, 8, 0 );
//}
//
//
//void draw_subdiv_edge( IplImage* img, CvSubdiv2DEdge edge, CvScalar color )
//{
//	CvSubdiv2DPoint* org_pt;
//	CvSubdiv2DPoint* dst_pt;
//	CvPoint2D32f org;
//	CvPoint2D32f dst;
//	CvPoint iorg, idst;
//
//	org_pt = cvSubdiv2DEdgeOrg(edge);
//	dst_pt = cvSubdiv2DEdgeDst(edge);
//
//	if( org_pt && dst_pt )
//	{
//		org = org_pt->pt;
//		dst = dst_pt->pt;
//
//		iorg = cvPoint( cvRound( org.x ), cvRound( org.y ));
//		idst = cvPoint( cvRound( dst.x ), cvRound( dst.y ));
//
//		cvLine( img, iorg, idst, color, 1, CV_AA, 0 );
//	}
//}



bool Delaunay_Tri::FindTriangleFromEdge(CvSubdiv2DEdge e)
{
	CvSubdiv2DEdge t = e;
	CvPoint buf[3];
	CvPoint *pBuf = buf;
	int iPointNum = 3;

	int j,k;

	//IplImage *img=cvCreateImage(cvGetSize(shape->hostImage),shape->hostImage->depth,shape->hostImage->nChannels);
	//cvCopyImage(shape->hostImage,img);
	////for (int kk=0;kk<shape->ptsNum;kk++)
	////{
	////	cvCircle(img,cvPoint(shape->pts[kk][0],shape->pts[kk][1]),1,delaunay_color);
	////}
	//draw_subdiv_edge(img,e,voronoi_color);


	for(j = 0; j < iPointNum; j++ )
	{
		CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
		if( !pt ) break;
		buf[j] = cvPoint( cvRound(pt->pt.x), cvRound(pt->pt.y));
	//	buf[j] = cvPoint( (pt->id), (pt->pt.y));
		t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );

		//draw_subdiv_edge(img,t,voronoi_color);
	//	cvShowImage("current triangle",img);
	//	cout<<j<<" "<<buf[0].x<<" "<<buf[1].x<<" "<<buf[2].x<<endl;
		
	//	cvWaitKey();

	}

	if (j == iPointNum)
	{
		AddTriangle(buf);  
	}

	//for(k = 0; k < iPointNum; k++ )
	//{
	//	CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
	//	if( !pt ) break;
	//	buf[k] = cvPoint( cvRound(pt->pt.x), cvRound(pt->pt.y));
	//	//	buf[j] = cvPoint( (pt->id), (pt->pt.y));
	//	t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_RIGHT );
	//}

	//if (k == iPointNum)
	//{

	//	AddTriangle(buf);  
	//}

	if (j == iPointNum||k==iPointNum)
	{

	//	AddTriangle(buf);        // Ìí¼ÓÈý½ÇÐÎ

		return true;  
	}

	return false; 
}

void Delaunay_Tri::AddTriangle(CvPoint *buf)
{
	double threshold=0.5;
	int foundnum=0;
	int tmpindex[3];
	for (int times=0;times<3;times++)
	{
		//if (buf[times].x>=0)
		//{
		//	triangleIndex[triangleNum][times]=buf[times].x;
		//	foundnum++;
		//}
		//triangleIndex[triangleNum][times]=buf[times].x;
		for (int i=0;i<shape->ptsNum;i++)
		{
		//	cout<<abs(shape->pts[i][0]-buf[times].x)<<" "<<abs(shape->pts[i][1]-buf[times].y)<<endl;
			if (abs(shape->ptsForMatlab[i]-buf[times].x)<threshold&&abs(shape->ptsForMatlab[shape->ptsNum+i]-buf[times].y)<threshold)
			{
				tmpindex[times]=i;
				foundnum++;
				break;
			}
		}
	}
	int sortedIndex[3];
	if(foundnum==3)
	{
	/*	cout<<triangleNum<<" index: "<<triangleIndex[triangleNum][0]<<" "<<
			triangleIndex[triangleNum][1]<<" "<<
			triangleIndex[triangleNum][2]<<endl;*/

		//sort the three index
		sortedIndex[0]=min(tmpindex[0],min(tmpindex[1],tmpindex[2]));
		sortedIndex[2]=max(tmpindex[0],max(tmpindex[1],tmpindex[2]));
		for (int i=0;i<3;i++)
		{
			if (tmpindex[i]!=sortedIndex[0]&&tmpindex[i]!=sortedIndex[2])
			{
				sortedIndex[1]=tmpindex[i];
				break;
			}
		}
		//and then decide whether or not to insert
		int i;
		for (i=0;i<triangleNum;i++)
		{
			if (sortedIndex[0]==triangleIndex[i][0]&&sortedIndex[1]==triangleIndex[i][1]&&
				sortedIndex[2]==triangleIndex[i][2])
			{
				break;
			}
		}
		if (i==triangleNum)
		//	if (i=)
		{
			for (int j=0;j<3;j++)
			{
				triangleIndex[triangleNum][j]=sortedIndex[j];
			}
			triangleNum++;

			//cvNamedWindow("current triangle");

			//IplImage *img=cvCreateImage(cvGetSize(shape->hostImage),shape->hostImage->depth,shape->hostImage->nChannels);
			//cvCopyImage(shape->hostImage,img);
			//for (int kk=0;kk<shape->ptsNum;kk++)
			//{
			//	cvCircle(img,cvPoint(shape->pts[kk][0],shape->pts[kk][1]),1,delaunay_color);
			//}
			//cvLine(img,buf[0],buf[1],voronoi_color,2);
			//cvLine(img,buf[0],buf[2],voronoi_color,2);
			//cvLine(img,buf[1],buf[2],voronoi_color,2);
			//cvShowImage("current triangle",img);
			//cvWaitKey();
		}
		
	}
}

void Delaunay_Tri::setPtsNum(int num)
{
	shape->setPtsNum(num);
}



void Delaunay_Tri::triangulation(double *pts)
{
	//CvSize imgSize=cvGetSize(img);
	//CvRect rect = { 0, 0, imgSize.width, imgSize.height };

	int ptsNum=shape->ptsNum;
	double minx,miny,maxx,maxy;
	minx=shape->minx;
	miny=shape->miny;
	maxx=shape->maxx;
	maxy=shape->maxy;
	//CvRect rect = { minx-1, miny-1, maxx-minx+2, maxy-miny+2};
	CvRect rect = {0, 0, shape->width, shape->height};
	CvMemStorage* storage;
	//CvSubdiv2D* subdiv;
	storage = cvCreateMemStorage(0);
	subdiv = init_delaunay( storage, rect );

	//CvSubdiv2D* subdiv;
	subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),
		sizeof(CvSubdiv2DPoint),
		sizeof(CvQuadEdge2D),
		storage );
	cvInitSubdivDelaunay2D( subdiv, rect );

	//ptsNum-=3;
	//set the points set
	IplImage *img=shape->hostImage;
	for (int i=0;i<ptsNum;i++)
	{
		//cout<<i<<" "<<pts[i]<<" "<<pts[i+ptsNum]<<endl;
		CvPoint2D32f fp;
		fp.x=pts[i];
		fp.y=pts[i+ptsNum];
		cvSubdivDelaunay2DInsert( subdiv, fp );
	//	draw_subdiv_point(img,fp,CV_RGB( 255, 255, 0 ));
	}

	//triangulation

	cvCreateSubdivDelaunay2D(rect,storage);

	//get the triangles
	findTriangles();

	//save the triangle list for each points
	triangleList=new int *[ptsNum];
	listNum=new int[ptsNum];
	for (int i=0;i<ptsNum;i++)
	{
		triangleList[i]=new int[10];
		for (int j=0;j<10;j++)
		{
			triangleList[i][j]=-1;
		}
	}

	for (int i=0;i<ptsNum;i++)
	{
		listNum[i]=0;
	}
	for (int i=0;i<triangleNum;i++)
	{
		for (int j=0;j<3;j++)
		{
			int cid=triangleIndex[i][j];
			triangleList[cid][listNum[cid]]=i;
			listNum[cid]++;
		}
		
	}

	////try to locate points
	//for (int i=0;i<ptsNum;i++)
	//{
	//	CvPoint2D32f fp;
	//	fp.x=pts[i][0];
	//	fp.y=pts[i][1];
	//	CvSubdiv2DEdge e;
	//	CvSubdiv2DEdge e0 = 0;
	//	CvSubdiv2DPoint* p = 0;

	//	cvSubdiv2DLocate( subdiv, fp, &e0, &p );

	//	if( e0 )
	//	{
	//		e = e0;
	//		do
	//		{
	//			draw_subdiv_edge( img, e, voronoi_color );
	//			e = cvSubdiv2DGetEdge(e,CV_NEXT_AROUND_LEFT);
	//		}
	//		while( e != e0 );
	//	}

	//	//draw_subdiv_point( img, fp, voronoi_color );
	//	cvShowImage( "results", img );
	//	cvWaitKey();
	//}
//
//	cvNamedWindow("results");
//
	//draw triangles
//	double **pts=shape->pts;

	//////////////////Display//////////////////////////
//	//draw edges
//	draw_subdiv( img, subdiv, delaunay_color, voronoi_color );
//		cvNamedWindow("results");
//	cout<<triangleNum<<endl;
//	for (int i=0;i<triangleNum;i++)
//	{
//		int index[3]={triangleIndex[i][0],triangleIndex[i][1],triangleIndex[i][2]};
//		cout<<index[0]<<" "<<index[1]<<" "<<index[2]<<endl;
//		cvLine(img,cvPoint(pts[index[0]][0],pts[index[0]][1]),cvPoint(pts[index[1]][0],pts[index[1]][1]),voronoi_color);
//		cvLine(img,cvPoint(pts[index[0]][0],pts[index[0]][1]),cvPoint(pts[index[2]][0],pts[index[2]][1]),voronoi_color);
//		cvLine(img,cvPoint(pts[index[1]][0],pts[index[1]][1]),cvPoint(pts[index[2]][0],pts[index[2]][1]),voronoi_color);
//		cvShowImage( "results", img );
//		cvWaitKey(3);
//	}
////	
//	cvShowImage( "results", img );
//	cvWaitKey(0);
}
