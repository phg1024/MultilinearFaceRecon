#ifndef DELAUNAY_TRIANGULATION_H
#define DELAUNAY_TRIANGULATION_H

#include "opencv\cv.h"
#include "opencv\cxcore.h"
#include "opencv\highgui.h"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/highgui/highgui.hpp"
//#include "opencv\cxerror.h"

//#pragma comment(lib,"opencv_highgui230.lib")
//#pragma comment(lib,"cv230.lib") 
//#pragma comment(lib,"cvaux230.lib") 
//#pragma comment(lib,"opencv_core230.lib") 

#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#include "shape.h"

class Delaunay_Tri
{
	public:
		Delaunay_Tri();
		~Delaunay_Tri();
		void draw_subdiv_point( IplImage* img, CvPoint2D32f fp, CvScalar color );
		void draw_subdiv_edge( IplImage* img, CvSubdiv2DEdge edge, CvScalar color );
		void draw_subdiv( IplImage* img, CvSubdiv2D* subdiv,
			CvScalar delaunay_color, CvScalar voronoi_color );
		void paint_voronoi( CvSubdiv2D* subdiv, IplImage* img );
		
		/////////////////MAIN FUNCTION////////////////
		void triangulation(double *pts);

		//double ** getVertex(string imgName);
		void draw_subdiv_facet( IplImage* img, CvSubdiv2DEdge edge );

		///int ptsNum;
		Shape *shape;

		CvScalar active_facet_color, delaunay_color, voronoi_color, bkgnd_color;

		void setPtsNum(int);
		CvSubdiv2D* init_delaunay( CvMemStorage* storage,
			CvRect rect );

		bool FindTriangleFromEdge(CvSubdiv2DEdge e);
		int* hull;
		int hullcount;
		CvPoint* hullPoints;
		void getConvexHull();

		void run_triangulation(Shape *inputShape);
		void findTriangles();
		void AddTriangle(CvPoint *);

		int **triangleIndex;
		int triangleNum;

		int **triangleList;
		int *listNum;
		//record the triangle index for each vertex and 
		//the number of vertex
		CvMat *TriangleIndex;
		
		//double **pts;
		//IplImage *img;

		//save the triangles
		CvSubdiv2D* subdiv;
};

#endif