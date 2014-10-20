#include "FaceFounder.h"
#include "test.h"

FaceDetector::FaceDetector(bool isShowFace)
{
	//libFace = LibFace(DETECT, ".");
	scale=0.7753;//0.7753
	showFace=isShowFace;
}

Rect FaceDetector::findFace(IplImage *img) //0.7753
{
	
	 vector<Face> result;  

	 result = libFace.detectFaces(img, cvSize(0,0));

	if(result.size()==0)
	{
		Rect rect;
		rect.x=-1;
		return rect;
	}
	// result = libFace.detectFaces("D:\\Fuhao\\face dataset new\\faceRegression\\testset\\image_0001.png"); //result = libFace.detectFaces("123");
	 printf("faces: %d\n",result.size());
	 if(result.size()!=0&&showFace)
	 {
		 Mat tmp=cvarrToMat(img).clone();
		 for(int i=0;i<result.size();i++)
		 {
			 rectangle(tmp,Point(result[i].getX1(),
				 result[i].getY1() ),Point(result[i].getX2(),
				 result[i].getY2() ),Scalar(255),2);

			 Rect rect;
				rect=getRect(result[i]);

			 rectangle(tmp,rect,Scalar(128),2);
		 }
		 imshow("Faces",tmp);
		// waitKey();

	 }

	 Rect rect;
	 if(!result.empty())
	 {
		rect=getEnlargedRect(result[0]);
	 }
	 return rect;
}


vector<Rect> FaceDetector::findFaceFull(IplImage *img) //0.7753
{
	
	 vector<Face> result;  

	
	 result = libFace.detectFaces(img, cvSize(0,0));

	if(result.size()==0)
	{
		 vector<Rect> rectList;
		//rect.x=-1;
		return rectList;
	}
	// result = libFace.detectFaces("D:\\Fuhao\\face dataset new\\faceRegression\\testset\\image_0001.png"); //result = libFace.detectFaces("123");
	
	 if(result.size()!=0&&showFace)
	 {
		  printf("faces: %d\n",result.size());
		 Mat tmp=cvarrToMat(img).clone();
		 for(int i=0;i<result.size();i++)
		 {
			 rectangle(tmp,Point(result[i].getX1(),
				 result[i].getY1() ),Point(result[i].getX2(),
				 result[i].getY2() ),Scalar(255),2);

			 Rect rect;
				rect=getRect(result[i]);

			 rectangle(tmp,rect,Scalar(128),2);
		 }
		 imshow("Faces",tmp);
		// waitKey();

	 }

	 vector<Rect> rectList;
	 if(!result.empty())
	 {
		 for(int i=0;i<result.size();i++)
			 rectList.push_back(getEnlargedRect(result[i]));
	 }
	 return rectList;
}


Mat FaceDetector::getCurFace(Rect &rect,IplImage *img)
{
	Mat imgMat=cvarrToMat(img);
	Mat res=Mat::zeros(rect.width,rect.height,imgMat.type());

	Rect goodRect=rect;
	int initialEx=rect.x+rect.width;
	int initialEy=rect.y+rect.height;

	goodRect.x=goodRect.x>=0?goodRect.x:0;
	goodRect.y=goodRect.y>=0?goodRect.y:0;

	goodRect.width=(initialEx)<imgMat.cols?(initialEx-goodRect.x):(imgMat.cols-1-goodRect.x);
	goodRect.height=(initialEy)<imgMat.rows?(initialEy-goodRect.y):(imgMat.rows-1-goodRect.y);

	curST=rect.tl();

	Rect resRect=goodRect;
	resRect.x-=rect.x;
	resRect.y-=rect.y;

	res(resRect)+=imgMat(goodRect);

	return res;
}

Rect FaceDetector::findFaceGT(IplImage *img,Rect &gtRect) //0.7753
{
	
	 vector<Face> result;  

	
	 result = libFace.detectFaces(img, cvSize(0,0));

	
	// result = libFace.detectFaces("D:\\Fuhao\\face dataset new\\faceRegression\\testset\\image_0001.png"); //result = libFace.detectFaces("123");
	 //printf("faces: %d\n",result.size());
	 if(!result.empty()&&0)
	 {
		 Mat tmp=cvarrToMat(img).clone();
		 tmp.convertTo(tmp,CV_GRAY2BGR);
		 for(int i=0;i<result.size();i++)
		 {
			 rectangle(tmp,Point(result[i].getX1(),
				 result[i].getY1() ),Point(result[i].getX2(),
				 result[i].getY2() ),Scalar(255),2);

			 Rect rect=getRect(result[i]);
			

			 rectangle(tmp,rect,Scalar(255),2);
		 }
		 rectangle(tmp,gtRect,Scalar(128),2);
		 imshow("Faces",tmp);
		// waitKey();

	 }


	 Rect rect;
	 rect.x=-1;
	 if(result.empty())
	 {
		 if(showFace&&0)
		 {
			 Mat tmp=cvarrToMat(img).clone();

			 rectangle(tmp,gtRect,Scalar(128),2);
			 imshow("Faces",tmp);
			 waitKey();
		 }
		

		 return rect;
	 }

	 //find the rect that is cloest to the GT rect
	 Point2f gtCenter=Point2f(gtRect.x+gtRect.width/2,gtRect.y+gtRect.width/2);
	 Test tmp;
	 int bestInd=-1;
	 float bestDis=1000000;
	 for(int i=0;i<result.size();i++)
	 {
		 Rect curRect=getRect(result[i]);
		 if(tmp.isCollide(curRect,gtRect))
		 {
			 Point2f curCenter=Point2f(curRect.x+curRect.width/2,curRect.y+curRect.width/2);
			 Point2f curDif=curCenter-gtCenter;
			 float curDifNorm=sqrtf(curDif.dot(curDif));
			 if(curDifNorm<bestDis)
			 {
				 bestDis=curDifNorm;
				 bestInd=i;
			 }
		 }
	 }
	 if(bestInd>=0)
	 {
		 rect=getEnlargedRect(result[bestInd]);
	 }

	 if(showFace&&bestInd>=0)
	 {
		 Mat tmp=cvarrToMat(img).clone();
		
		//  rectangle(tmp,getRect(result[bestInd]),Scalar(255),2);
		 rectangle(tmp,getEnlargedRect(result[bestInd]),Scalar(255),2);
		 rectangle(tmp,gtRect,Scalar(128),2);
		 imshow("Faces",tmp);
		 waitKey();
	 }
	 return rect;
}

Rect FaceDetector::getRect(Face &result)
{
	Rect rect;
	 rect.x=result.getX1();
	 rect.y=result.getY1();
	 rect.width=result.getWidth();
	 rect.height=result.getHeight();

	/* float curBottom=result.getY2();
	 rect.x+=(1-scale)/2.0f*rect.width;
	 rect.width*=scale;
	 rect.height*=scale;
	 rect.y=curBottom-rect.height;*/
	 return rect;
}

Rect FaceDetector::getEnlargedRect(Face &result)
{
	Rect rect=getRect(result);

	/*float orgWidth=(float)rect.width/scale;
	float newWidth=orgWidth*1.5f;
	float addedHeight=orgWidth*0.3f;
	Rect rectNew=rect;

	rectNew.width=rectNew.height=newWidth;

	float addedRange=(newWidth-rect.width)/2;
	rectNew.x-=addedRange;
	rectNew.y-=addedHeight;*/

	float orgWidth=(float)rect.width;
	float newWidth=orgWidth*1.4f;
	float addedHeight=orgWidth*0.1f;
	Rect rectNew=rect;

	rectNew.width=rectNew.height=newWidth;

	float addedRange=(newWidth-rect.width)/2;
	rectNew.x-=addedRange;
	rectNew.y-=addedHeight;

	return rectNew;
	
}



void FaceDetector::findFaceFull(IplImage *img,vector<Rect> & rects) //0.7753
{
	rects.clear();
	 vector<Face> result;  

	
	 result = libFace.detectFaces(img, cvSize(0,0));

	
	// result = libFace.detectFaces("D:\\Fuhao\\face dataset new\\faceRegression\\testset\\image_0001.png"); //result = libFace.detectFaces("123");
	 printf("faces: %d\n",result.size());
	 if(!result.empty())
	 {
		// Mat tmp=cvarrToMat(img).clone();
		 for(int i=0;i<result.size();i++)
		 {
			/* rectangle(tmp,Point(result[i].getX1(),
				 result[i].getY1() ),Point(result[i].getX2(),
				 result[i].getY2() ),Scalar(255),2);*/

			 Rect rect;
			 rect.x=result[i].getX1();
			 rect.y=result[i].getY1();
			 rect.width=result[i].getWidth();
			 rect.height=result[i].getHeight();

			 float curBottom=result[i].getY2();
			 rect.x+=(1-scale)/2.0f*rect.width;
			 rect.width*=scale;
			 rect.height*=scale;
			 rect.y=curBottom-rect.height;

			 rects.push_back(rect);

			 //rectangle(tmp,rect,Scalar(128),2);
		 }
		// imshow("Faces",tmp);
		// waitKey();

	 }

	
}