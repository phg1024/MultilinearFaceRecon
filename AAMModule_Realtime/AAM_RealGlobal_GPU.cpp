#include "AAM_RealGlobal_GPU.h"
//#include "numberDefine.h"


#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;

#include "mkl.h"
#include "mkl_lapacke.h"

#include "CodeTimer.h"
//int main()
//{
//	MatrixXd m(2,2);
//	m(0,0) = 3;
//	m(1,0) = 2.5;
//	m(0,1) = -1;
//	m(1,1) = m(1,0) + m(0,1);
//	std::cout << m << std::endl;
//}


Mat g_Hessian;
Mat g_inv_Hessian;
Adp_PCA_float *adpPcaGlobal;

MatrixXd g_hes,g_hes_inv;
VectorXd g_b;
void drawWordsOnImg(Mat &img,int x,int y,char *words,float number=-1)
{
	char name[500];
	sprintf(name,"%s%.5f",words,number);
	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	putText(img,name,Point(x,y),fontFace,0.6,Scalar(0,0,255));
}



extern "C" void solveAb(float *inputHessian, float *b,float *deltaX,int dim)
{
	for (int i=0;i<dim;i++)
	{
		for (int j=0;j<dim;j++)
		{
			g_hes(i,j)=inputHessian[i*dim+j];
		}
	}

	for (int i=0;i<dim;i++)
	{
		g_b(i)=b[i];
	}

	VectorXd x = g_hes.colPivHouseholderQr().solve(g_b);

	for (int i=0;i<dim;i++)
	{
		deltaX[i]=g_b(i);
	}
}

extern "C" void setNewMeanVec(float * _newMean)
{
	adpPcaGlobal->setNewMeanVec(_newMean);
}

extern "C" void updateModelCPU_thread()
{
	//adpPcaGlobal->updateModel();
	//cout<<"textureModel address: "<<adpPcaGlobal<<endl;
	cout<<"updating model in CPU\n";
	_beginthreadex(0,0,Adp_PCA_float::threadProc,(void*)adpPcaGlobal,0,0);
	//adpPcaGlobal->updateModel();
}

extern "C" void updateModelCPU(float *dataBlock,int sampleNum,float *newMeanVec)
{
	//cout<<adpPcaGlobal->dataForUse.cols()<<" "<<adpPcaGlobal->dataForUse.rows()<<endl;
	//GTB("updateModel");
	adpPcaGlobal->updateModel(dataBlock,sampleNum,true);
	adpPcaGlobal->getMeanAndModel(newMeanVec);
	//GTE("updateModel");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"update model time: "<<time<<" ms"<<endl;
}

extern "C" void invHessian(float *inputHessian, float *outputHessian,int dim)
{
	//Mat Hessian(dim,dim,CV_64FC1);
	
	int i,j;
	for (i=0;i<dim;i++)
	{
		for (j=i;j<dim;j++)
		{
			float curV=inputHessian[i*dim+j];
			g_Hessian.at<double>(i,j)=curV;
			if (i!=j)
			{
				g_Hessian.at<double>(j,i)=curV;
			}
		}
	}

	

	//Mat inv_H(dim,dim,CV_64FC1);
	g_inv_Hessian=g_Hessian.inv();
	//invert(g_Hessian,g_inv_Hessian);

	for (i=0;i<dim;i++)
	{
		for (j=0;j<dim;j++)
		{
			outputHessian[i*dim+j]=g_inv_Hessian.at<double>(i,j);
		}
	}

	//ofstream out("D:\\Fuhao\\cpu gpu validation\\h_new.txt");
	//for (i=0;i<dim;i++)
	//{
	//	for (j=0;j<dim;j++)
	//	{
	//		out<<g_Hessian.at<double>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();
	
	//for (int i=0;i<dim;i++)
	//{
	//	for (int j=0;j<dim;j++)
	//	{
	//		g_hes(i,j)=inputHessian[i*dim+j];
	//	}
	//}
	//g_hes_inv=g_hes.inverse();
	//for (int i=0;i<dim;i++)
	//{
	//	for (int j=0;j<dim;j++)
	//	{
	//		outputHessian[i*dim+j]=g_hes_inv(i,j);
	//	}
	//}

	//ofstream out("Hessian_CPU.txt",ios::out);
	//for(i=0;i<dim;i++)
	//{
	//	for (j=0;j<dim;j++)
	//	{
	//		out<<g_Hessian.at<double>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();

	//cout<<"CPU Hessian\n";
	//for (int i=0;i<50;i++)
	//{
	//	cout<<inputHessian[i]<<" ";
	//}
	//cout<<endl;

	//cout<<"CPU Inv_Hessian\n";
	//for (int i=0;i<50;i++)
	//{
	//	cout<<outputHessian[i]<<" ";
	//}
	//cout<<endl;
}

Mat g_inputImg;
Mat g_s_vec;
//Mat g_t_vec;
float *g_meanShape;
string dataDirGPU;
int globalCurrentNum;
string dataPureName;
extern "C" void checkIterationResult(float *parameters,int ptsNum,int s_dim,int t_dim,bool isgoon)
{
	int i,j;
	double *pts=new double[ptsNum*2];
	double *localPts=new double[ptsNum*2];

	int totalDim=s_dim+t_dim;
	float theta=parameters[totalDim];
	float k_scale=parameters[totalDim+1];
	float transform_x=parameters[totalDim+2];
	float transform_y=parameters[totalDim+3];

	for (i=0;i<ptsNum;i++)
	{
		localPts[i]=g_meanShape[i];
		localPts[ptsNum+i]=g_meanShape[ptsNum+i];
		//first, pca weights
		for (j=0;j<s_dim;j++)
		{
			//currentShape->ptsForMatlab[i]+=s_weight[j]*CV_MAT_ELEM(*s_vec,double,j,i);
			localPts[i]+=parameters[j]*g_s_vec.at<double>(j,i);
			localPts[ptsNum+i]+=parameters[j]*g_s_vec.at<double>(j,ptsNum+i);
		}
		//then, global transform
		pts[i]=k_scale*(cos(theta)*localPts[i]-sin(theta)*localPts[i+ptsNum])+transform_x;
		pts[ptsNum+i]=k_scale*(sin(theta)*localPts[i]+cos(theta)*localPts[i+ptsNum])+transform_y;
	}

	Mat cInputImage=g_inputImg.clone();

	Point center;
	for (i=0;i<ptsNum;i++)
	{
		center.x=pts[i];
		center.y=pts[ptsNum+i];
		circle(cInputImage,center,1,255);
	}
	
	namedWindow("0");
	imshow("0",cInputImage);
	if (isgoon)
	{
		waitKey(1);
	}
	else
		waitKey();

	if (0)
	{
	//	dataDirGPU="D:\\Fuhao\\Siggraph\\2013 data\\SDKcomparison\\Muscle 0\\kinect Studio Format\\seq_0\\";
	//dataDirGPU="D:\\Fuhao\\Siggraph\\2013 data\\SDKcomparison\\Muscle 0\\kinect Studio Format\\seq_1\\";
	string fullname;
			fullname=dataPureName+".jpg";
			imwrite(fullname.c_str(),cInputImage);
			//cout<<fullname<<endl;
		/*	sprintf(name, "%d_ori.jpg", currentFrame);
			fullname=dataDir+name;
			cvSaveImage(fullname.c_str(),Input);*/
			//Output the pts
			/*char pname[50];
			sprintf(pname, "color_%d.txt", 10000+globalCurrentNum);
			tmpName=pname;*/
			//fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
			fullname=dataPureName+".txt";
			ofstream out(fullname.c_str(),ios::out);
			out<<cInputImage.cols<<" "<<cInputImage.rows<<endl;
			out<<ptsNum<<" "<<ptsNum<<endl;
			for (int i=0;i<ptsNum;i++)
			{
				out<<pts[i]/(double)cInputImage.cols<<" "<<
					pts[i+ptsNum]/(double)cInputImage.rows<<endl;
			}
			out.close();
			//globalCurrentNum++;
	}
	
	cInputImage.release();
	delete []pts;
	delete []localPts;
}

void AAM_RealGlobal_GPU::setGlobalStartNum(int num)
{
	globalCurrentNum=num;
}

bool AAM_RealGlobal_GPU::getCurrentStatus()
{
	return adpPcaGlobal->readyToTransfer;
}

void AAM_RealGlobal_GPU::setCurrentStatus(bool s)
{
	adpPcaGlobal->readyToTransfer=s;
}

void AAM_RealGlobal_GPU::setSaveName(char* name)
{
	dataPureName=name;
}

void AAM_RealGlobal_GPU::prepareForTracking()
{
	shape_dim-=4;

	//get all needed data ready
	precompute();

	//preProcess_GPU();

	//now we are using combination framework, so use the combination preprocess routine
	preProcess_GPU_combination();

}

AAM_RealGlobal_GPU::AAM_RealGlobal_GPU(double _search_test,double _AAM_weight,double _priorweight,double _localPCAWeight,bool _isApt)
{
	
	face_cascade=(CvHaarClassifierCascade*)cvLoad("d:/OpenCV2.3/opencv/data/haarcascades/haarcascade_frontalface_alt.xml");
	//face_cascade=(CvHaarClassifierCascade*)cvLoad("d:/fuhao/OpenCV2.1/data/haarcascades/haarcascade_frontalface_alt.xml");
	faces_storage=cvCreateMemStorage(0);

	currentTemplate=NULL;
	currentShape=NULL;
	affineTable_strong=NULL;

	s_vec=t_vec=s_mean=t_mean=s_value=t_value=triangleList=NULL;
	listNum=NULL;
	triangleIndList=NULL;
	inputGradient_x=inputGradient_y=NULL;
	//leye_cascade.load("detection/haarcascade_mcs_lefteye.xml");
	//reye_cascade.load("detection/haarcascade_mcs_righteye.xml");
	//nose_cascade.load("detection/haarcascade_mcs_nose.xml");
	//mouth_cascade.load("detection/Boca.xml");
	warp=new PieceAffineWarpping();

	tran_x=tran_y=0;
	outputtime=false;
	currentLocalShape=NULL;
	smoothWeight=_search_test;
	stepLength=1.0;
	AAM_weight=_AAM_weight;
	resizeSize=1;
	lastErrorSum=-1;
	showSingleStep=false;
	initialTx=initialTy=0;
	initialScale=1;
	usingCUDA=false;

	usingGPU=false;
	cu_gradientX=cu_gradientY=NULL;

	fullSD_detection=full_Hessian_detection=fullSD_tran_detection=NULL;
	isSDDefined=false;
	SD_detection=NULL;
	trees=new RandTree();
	useDetectionResults=smoothWeight>0;
	smoothWeight_backup=smoothWeight;
	init();

	//initial trees
	//rt.setParameters(15,3,0,15,32,13);
	usingTrees=false;
	treeInitialization=false;


	priorWeight=_priorweight;
	priorWeight_backup=_priorweight;

	SD_prior=NULL;
	prob_mu=NULL;
	prob_conv=NULL;

	SD_local=NULL;


	shapeSample=new int *[40];
	for (int i=0;i<40;i++)
	{
		shapeSample[i]=new int [2];
	}

	conv_precalculated=NULL;

	localPCAWeight=_localPCAWeight;
	localPCAWeight_backup=_localPCAWeight;

	local_s_vec=NULL;
	local_s_mean=NULL;

	globalCurrentNum=0;

	isAdaptive=_isApt;
}

AAM_RealGlobal_GPU::~AAM_RealGlobal_GPU()
{
	endSection();
};

void AAM_RealGlobal_GPU::getAllNeededData(AAM_Train *trainedResult)
{
	//int shape_dim,texture_dim;
	//CvMat *s_vec,*t_vec,*s_mean,*t_mean;
	//CvMat *s_value,*t_value;
	//double texture_scale,shape_scale;
	//CvMat *triangleList;
	//void getJacobian(Shape *,CvMat *triangleList);
	//CascadeClassifier face_cascade,leye_cascade,reye_cascade,
	//	nose_cascade,mouth_cascade;
	//Shape **shapes;
	//Texture **textures;
	//CvMat *mask;//texture
	//int shapeWidth,shapeHeight;
	//Shape *meanShape;
	//Texture *meanTexture;
	//int **triangleIndList;
	//int *listNum;
	shape_dim=trainedResult->shape_dim;
	texture_dim=trainedResult->texture_dim;
	s_vec=trainedResult->s_vec;
	t_vec=trainedResult->t_vec;
	meanShape=trainedResult->meanShape;
	meanTexture=trainedResult->meantexure;
	s_mean=trainedResult->s_mean;
	t_mean=trainedResult->t_mean;
	s_value=trainedResult->s_value;
	t_value=trainedResult->t_value;
	texture_scale=trainedResult->texture_scale;
	shape_scale=trainedResult->shape_scale;
	triangleList=trainedResult->refShape->ref->TriangleIndex;
	//shapes=trainedResult->shape;
	//textures is useless,t_vec is what we need
	//textures=trainedResult->texture;

	triangleIndList=trainedResult->refShape->ref->triangleList;
	listNum=trainedResult->refShape->ref->listNum;
	isGlobaltransform=trainedResult->isGlobaltransform;

	
	pix_num=meanShape->pix_num;
	nband=meanTexture->nband;
	pts_Index=meanShape->pts_Index;
	inv_mask=meanShape->inv_mask;
	mask_withindex=meanShape->mask_withindex;
	affineTable=meanShape->affineTable;
//	affineTable_strong=meanShape->affineTable_strong;
//	meantexture_real=trainedResult->meantexure_real;
	

	//define the needed variables
	mask=meanShape->mask;
	shapeWidth=meanShape->width;
	shapeHeight=meanShape->height;
	width=meanShape->width;
	height=meanShape->height;
	WarpedInput=cvCreateImage(cvSize(width,height),meanTexture->depth,meanTexture->nband);
	errorImageMat=cvCreateMat(1,pix_num*nband,CV_64FC1);
	WarpedInput_mat=cvCreateMat(1,pix_num*nband,CV_64FC1);
	Template_mat=cvCreateMat(1,pix_num*nband,CV_64FC1);
	cvCopy(meanTexture->imgData,Template_mat);

	s_weight=new double[shape_dim];
	//last_Iteration_weight=new double[shape_dim+texture_dim+4];
	s_weight_vec=new double[shape_dim];
	t_weight=new double[texture_dim];
	
	textures=new Texture*[texture_dim];
	for (int i=0;i<texture_dim;i++)
	{
		textures[i]=new Texture();
		*textures[i]=(*meanTexture);
		for (int j=0;j<pix_num;j++)
		{
			CV_MAT_ELEM(*textures[i]->imgData,double,0,j)=
				CV_MAT_ELEM(*t_vec,double,i,j);
		}
	}

	
	//cout<<"assigned space\n";
//	cvConvert(Template,Template_mat);

	//add global transform vectors

}


void AAM_RealGlobal_GPU::searchVideo(string videoName)
{
	CvCapture* pCapture = NULL;
	IplImage* pFrame = NULL; 
	if( !(pCapture = cvCaptureFromFile(videoName.c_str())))
		return;
	
	//if (!isGlobaltransform)
	{
		shape_dim-=4;
	}

	//get all needed data ready
	precompute();

	if (usingGPU)
	{
		preProcess_GPU();
	}

	int f_num=cvGetCaptureProperty(pCapture,CV_CAP_PROP_FRAME_COUNT );
	IplImage **frames=new IplImage*[f_num];
	int i=0;
	while(pFrame = cvQueryFrame( pCapture ))
	{
		frames[i]=cvCreateImage(cvGetSize(pFrame),pFrame->depth,pFrame->nChannels);
		cvCopy(pFrame,frames[i]);
		i++;
	}
	DWORD dwStart, dwStop;  
	startNum=0;
	for (currentFrame=0;currentFrame<f_num;currentFrame++)
	{
		
			if (currentFrame>=startNum)
			{
				//dwStart=GetTickCount();
				search(frames[currentFrame]);
				//dwStop=GetTickCount();
			}
			//cout<<"frame: "<<currentFrame<<" time: "<<dwStop-dwStart<<"ms"<<endl;
			cout<<"frame: "<<currentFrame<<endl;
	}
	delete []frames;

	//cout<<i<<" "<<f_num<<endl;

	
	//for (int i=0;i<f_num;i++)
	//{
	//	frames[i] = cvRetrieveFrame( pCapture ,i);
	//	cvShowImage("frame",frames[i]);
	//	cvWaitKey();
	//}

	//LONG   dwStart,dwStop,d_totalstart; 
	//LONG   persecond; 

	//	QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);//询问系统一秒钟的频率 


	//cvNamedWindow("frame");

	//HessianStart=GetTickCount();

	//
	//for (int i=0;i<f_num;i++)
	//{
	//	
	//	cvShowImage("frame",frames[i]);
	//	//cvWaitKey(1);
	//	//cout<<frameNum<<endl;
	//}
	//cout<<"supposed time: "<<f_num/1<<endl;
	//cout<<"playing time: "<<(HessianStop-HessianStart)<<endl;

	//frameNum=0;
	//while(pFrame = cvQueryFrame( pCapture ))
	//{
	//	//if (frameNum>=0)
	//	//{
	//	//	search(pFrame);
	//	////	cvWaitKey(10);
	//	//}
	//	cvNamedWindow("frame");
	//	cvShowImage("frame",pFrame);
	//	cvWaitKey(1);
	//	//cout<<frameNum<<endl;
	//	frameNum++;
	//}
}

void AAM_RealGlobal_GPU::searchPics(string picListName)
{
	//read in the images
	dataDir=picListName.substr(0,picListName.find_last_of('\\')+1);
	ifstream in(picListName.c_str(),ios::in);
	int _imgnum;
	in>>_imgnum;

	IplImage **frames=new IplImage*[_imgnum];
	char imgName[500];
	in.getline(imgName,499,'\n');


	//track
	IplImage* pFrame = NULL; 
	shape_dim-=4;

	//get all needed data ready
	precompute();

	if (usingGPU)
	{
		preProcess_GPU();
	}
	
	//return;

	int i=0;
	DWORD dwStart, dwStop;  
	startNum=240-7;//video 4 276  427 661
	startNum=0;
	//startNum=215;
	//for(int i=0;i<_imgnum;i++)
	//{
	//	
	//	
	///*	namedWindow("0");
	//	imshow("0",imgList[i]);
	//	waitKey(20)*/;
	//	//cur++;
	//}
	//IplImage *tmp;
	Mat currentImage;
	IplImage *tmpImg=NULL;

	string labelName;

	
	for (currentFrame=0;currentFrame<_imgnum;currentFrame++)
	{
		in.getline(imgName,499,'\n');
		if (currentFrame>=startNum)
		{
			cout<<imgName<<endl;
			currentNamewithoutHouzhui=imgName;
			currentNamewithoutHouzhui=currentNamewithoutHouzhui.substr(0,currentNamewithoutHouzhui.size()-4);
			frames[currentFrame]=cvLoadImage(imgName);

			treeInitialization=false;
			//see if there is any tracking result
			if (1)
			{
				char ppname[500];
				//sprintf(ppname, "%d_optimized parameters.txt", currentFrame+1);
				string fullName=imgName;
				fullName=fullName.substr(0,fullName.find_last_of('.'));
				fullName+="_optimized parameters.txt";
				ifstream in(fullName.c_str(),ios::in);
				k_scale= 0.6166;
				theta=-0.0448 ;
				transform_x= 220.7249;
				transform_y= 121.2028;
				treeInitialization=true;
				if (in)
				{
					in>>k_scale>>theta>>transform_x>>transform_y;
				/*	while(in)
					{
						for (int i=0;i<shape_dim;i++)
						{
							in>>s_weight[i];
						}
						in>>k_scale>>theta>>transform_x>>transform_y;
					}*/
				/*	transform_x-=center[0]*k_scale;
					transform_y-=center[1]*k_scale;*/
					cout<<k_scale<<" "<<theta<<" "<<transform_x<<" "<<transform_y<<endl;
					//k_scale*=1.2;
					//in.close();
					treeInitialization=true;

					//using global transformation only
					if (1)
					{
						for (int i=0;i<shape_dim;i++)
						{
							s_weight[i]=0;
						}
					}
					//stepLength=0.5;
					//MaxIterNum=15;
				}
			}

			if(smoothWeight>0)
			{
				cout<<"loading probability map\n";
				labelName=imgName;
				labelName=labelName.substr(0,labelName.length()-3)+"txt";
				//trees->loadResult(labelName,frames[currentFrame]->width,frames[currentFrame]->height);
				//if(trees->numOfLabels>0)
				//	trees->getGradient();
				//trees->numOfLabels-=2;

				if (prob_conv==NULL)
				{
					prob_conv=new Mat[trees->labelNum-1];
					prob_mu=new double*[trees->labelNum-1];
					for (int i=0;i<trees->labelNum-1;i++)
					{
						prob_conv[i].create(2,2,CV_64FC1);
						prob_mu[i]=new double [2];
					}

					prob_conv_candidates=new Mat *[trees->labelNum-1];
					prob_mu_candidates=new double**[trees->labelNum-1];
					for (int i=0;i<trees->labelNum-1;i++)
					{
						prob_conv_candidates[i]=new Mat [100];
						prob_mu_candidates[i]=new double *[100];
						for(int j=0;j<100;j++)
						{
							prob_conv_candidates[i][j].create(2,2,CV_64FC1);
							prob_mu_candidates[i][j]=new double [2];
						}
					}
				}
			}
		

		/*	if (tmpImg==NULL)
			{
				tmpImg=cvCreateImage(cvSize(cvGetSize(frames[currentFrame]).width*resizeSize,
					cvGetSize(frames[currentFrame]).height*resizeSize),frames[currentFrame]->depth,
					frames[currentFrame]->nChannels);
			}
			cvResize(frames[currentFrame],tmpImg);*/

			cout<<meanShape->width<<" "<<meanShape->height<<endl;
			search(frames[currentFrame]);
			cvReleaseImage(&frames[currentFrame]);

			if (usingGPU)
			{
				break;
			}
			//dwStop=GetTickCount();
		}
		//cout<<"frame: "<<currentFrame<<" time: "<<dwStop-dwStart<<"ms"<<endl;
		cout<<"frame: "<<currentFrame<<endl;
	}


	/*for (int i=0;i<_imgnum;i++)
	{
		cvReleaseImage(&frames[i]);
	}*/
	delete []frames;
}
void AAM_RealGlobal_GPU::setResizeSize(int a)
{
	resizeSize=a;
}

void AAM_RealGlobal_GPU::detect( IplImage *img_in,double scale)
{
	//face_cascade=new CascadeClassifier("D:/OpenCV2.1/data/haarcascades/haarcascade_frontalface_alt.xml");
	//Mat img(img_in,0);
	int i = 0;
	double t = 0;
	
	const static Scalar colors[] =  { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255)} ;
	IplImage* gray = cvCreateImage( cvSize(img_in->width,img_in->height), 8, 1 );
	IplImage* smallimg = cvCreateImage( cvSize( cvRound (img_in->width/scale),
		cvRound (img_in->height/scale)),
		8, 1 );


	cvCvtColor( img_in, gray, CV_BGR2GRAY );
	cvResize( gray, smallimg, CV_INTER_LINEAR );
	cvEqualizeHist( smallimg, smallimg );
	Mat mask;
	if( face_cascade )
	{
		double t = (double)cvGetTickCount();
		CvSeq* faces = cvHaarDetectObjects( smallimg, face_cascade, faces_storage,
			1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
			cvSize(30, 30) );
		t = (double)cvGetTickCount() - t;
		//printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );
		for( i = 0; i < (faces ? faces->total : 0); i++ )
		{
			CvRect* r = (CvRect*)cvGetSeqElem( faces, i );
			faceCenter.x=(r->x+r->width/2)*scale;
			faceCenter.y=(r->y+r->height/2)*scale;
			face_scale = cvRound((r->width + r->height)*0.25*scale);
			
			//CvPoint pt2;
			//pt2.x=(r->x+r->width)*scale;
			//pt2.y=(r->y+r->height)*scale;
			//cvDrawCircle(img_in,faceCenter,3,cvScalar(255));
			//cvNamedWindow("0");
			//cvShowImage("0",img_in);
			//cvWaitKey();

		//	break;
			
		}
	}

	//if (abs(faceCenter.x-cvGetSize(img_in).width/2)>30)
	//{
	/*	faceCenter.x=296;
		faceCenter.y=201;*/
	//}
//	cvClearMemStorage( faces );
//	//t = (double)cvGetTickCount();
//	cvHaarDetectObjects( smallImg, face_cascade, faces,
//		1.1, 2, 0/*CV_HAAR_DO_CANNY_PRUNING*/,
//		cvSize(30, 30) );
//	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
//	{
//		faceCenter.x = cvRound((r->x + r->width*0.5)*scale);
//		faceCenter.y = cvRound((r->y + r->height*0.5)*scale);
//		face_scale = cvRound((r->width + r->height)*0.25*scale);
//	}
//	delete []face_cascade;
//	return;
////	t = (double)cvGetTickCount() - t;
////	printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
//	for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
//	{
//		Mat smallImgROI;
//		vector<Rect> leye,reye,nose,mouth;
//		Point center;
//		Scalar color = colors[i%8];
//		int radius;
//		center.x = cvRound((r->x + r->width*0.5)*scale);
//		center.y = cvRound((r->y + r->height*0.5)*scale);
//		radius = cvRound((r->width + r->height)*0.25*scale);
//		//circle( img, center, radius, color, 3, 8, 0 );
//		//continue;
//		//Rect()
//		//cvDrawRect(img,cvPoint(center.x-radius,center.Y-radius),cvPoint(center.x+radius,center.Y+radius),color,3);
//	//	if( face_cascade->empty() )
//	//		continue;
//		smallImgROI = smallImg(*r);
//		leye_cascade.detectMultiScale( smallImgROI, leye,
//			1.1, 2, 0
//			//|CV_HAAR_FIND_BIGGEST_OBJECT
//			//|CV_HAAR_DO_ROUGH_SEARCH
//			//|CV_HAAR_DO_CANNY_PRUNING
//			|CV_HAAR_SCALE_IMAGE
//			,
//			Size(3, 3) );
//		reye_cascade.detectMultiScale( smallImgROI, reye,
//			1.1, 2, 0
//			//|CV_HAAR_FIND_BIGGEST_OBJECT
//			//|CV_HAAR_DO_ROUGH_SEARCH
//			//|CV_HAAR_DO_CANNY_PRUNING
//			|CV_HAAR_SCALE_IMAGE
//			,
//			Size(3, 3) );
//		nose_cascade.detectMultiScale( smallImgROI, nose,
//			1.1, 2,
//			CV_HAAR_FIND_BIGGEST_OBJECT
//			//|CV_HAAR_DO_ROUGH_SEARCH
//			//|CV_HAAR_DO_CANNY_PRUNING
//			//|CV_HAAR_SCALE_IMAGE
//			,
//			Size(15, 15) );
//		mouth_cascade.detectMultiScale( smallImgROI, mouth,
//			1.1, 2, 
//			CV_HAAR_FIND_BIGGEST_OBJECT
//			//|CV_HAAR_DO_ROUGH_SEARCH
//			//|CV_HAAR_DO_CANNY_PRUNING
//			//|CV_HAAR_SCALE_IMAGE
//			,
//			Size(20, 15) );
//		for( vector<Rect>::const_iterator nr = leye.begin(); nr != leye.end(); nr++ )
//		{
//			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
//			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
//			radius = cvRound((nr->width + nr->height)*0.25*scale);
//			circle( img, center, radius, color, 3, 8, 0 );
//			//break;
//		}
//		for( vector<Rect>::const_iterator nr = reye.begin(); nr != reye.end(); nr++ )
//		{
//			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
//			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
//			radius = cvRound((nr->width + nr->height)*0.25*scale);
//			circle( img, center, radius, color, 3, 8, 0 );
//			//break;
//		}
//		for( vector<Rect>::const_iterator nr = nose.begin(); nr != nose.end(); nr++ )
//		{
//			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
//			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
//			radius = cvRound((nr->width + nr->height)*0.25*scale);
//			circle( img, center, radius, color, 3, 8, 0 );
//			//break;
//		}
//		for( vector<Rect>::const_iterator nr = mouth.begin(); nr != mouth.end(); nr++ )
//		{
//			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
//			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
//			radius = cvRound((nr->width + nr->height)*0.25*scale);
//			circle( img, center, radius, color, 3, 8, 0 );
//			//break;
//		}
//	}  
//	cv::imshow( "result", img );   
//	//cvWaitKey(300);
}

void AAM_RealGlobal_GPU::getCenter_Scale(IplImage *img)
{
	detect(img,1);
	//cvNamedWindow("1");
	//cvShowImage("1",img);
	s_scale=face_scale/width;
}

void AAM_RealGlobal_GPU::search(IplImage *img_in)
{



	IplImage* gray = cvCreateImage( cvSize(img_in->width,img_in->height), 8, 1 );
	cvCvtColor( img_in, gray, CV_BGR2GRAY );

	if (inputGradient_x==NULL)
	{
		mat_currentImage=cvCreateMat(cvGetSize(img_in).height,cvGetSize(img_in).width,CV_64FC1);
		m_currentImage=cvarrToMat(mat_currentImage);

		inputGradient_x=cvCreateMat(cvGetSize(img_in).height,cvGetSize(img_in).width,CV_64FC1);
		inputGradient_y=cvCreateMat(cvGetSize(img_in).height,cvGetSize(img_in).width,CV_64FC1);
		m_inputGradient_x=cvarrToMat(inputGradient_x);
		m_input_Gradient_y=cvarrToMat(inputGradient_y);
	}

	//get current gradient
	for (int i=0;i<mat_currentImage->rows;i++)
	{
		for (int j=0;j<mat_currentImage->cols;j++)
		{
			m_currentImage.at<double>(i,j)=cvGet2D(gray,i,j).val[0];
		}
	}

	gradient(mat_currentImage,inputGradient_x,inputGradient_y,NULL);

	//copy the gradient to the float array
	if (usingGPU)
	{
		if (cu_gradientX==NULL)
		{
			int MPN=MAX_PIXEL_NUM;
			cu_gradientX=new float[MPN];
			cu_gradientY=new float[MPN];
		}
		for (int i=0;i<m_currentImage.rows;i++)
		{
			for (int j=0;j<m_currentImage.cols;j++)
			{
				cu_gradientX[i*m_currentImage.cols+j]=m_inputGradient_x.at<double>(i,j);
				cu_gradientY[i*m_currentImage.cols+j]=m_input_Gradient_y.at<double>(i,j);
			}

		}
	}

	if (currentFrame==startNum)
	{
		getCenter_Scale(img_in);
	}
	

	if (usingGPU)
	{
		//if (showSingleStep)
		{
			g_inputImg=cvarrToMat(gray).clone();
			/*for (int i=0;i<meanShape->ptsNum*2;i++)
			{
				g_meanShape[i]=meanShape->ptsForMatlab[i];
			}
			g_s_vec=m_s_vec.clone();		*/	
		}
		iterate_GPU(gray);
	}
	else
	{
		iterate(gray);
	}
	
	
	cvReleaseImage(&gray);

	/*shape *shape=new shape();
	shape->sethostimage(img);
	texture *texture=new texture();*/
}

void AAM_RealGlobal_GPU::getJacobian(Shape *shape,CvMat *triangleList)
{
	//initialize full_Jacobian
	int dim=shape_dim+texture_dim;
	full_Jacobian=new double***[width];
	int ptsNum=shape->ptsNum;
	//2 rows for aw/ax and aw/ay.

	//initializ width*height*shape_dim
	for (int i=0;i<width;i++)
	{
		full_Jacobian[i]=new double**[height];
		for (int j=0;j<height;j++)
		{
			full_Jacobian[i][j]=new double *[2];
			for (int k=0;k<2;k++)
			{
				full_Jacobian[i][j][k]=new double[shape_dim];
			}
		}
	}

	//Jacobian=cvCreateMat(2,shape_dim,CV_64FC1);
	//Hessian=cvCreateMat(shape_dim,shape_dim,CV_64FC1);
	////initial Hessian
	//for(int i=0;i<shape_dim;i++)
	//{
	//	for (int j=0;j<shape_dim;j++)
	//	{
	//		CV_MAT_ELEM(*Hessian,double,i,j)=0;
	//	}
	//}

	//for (int i=0;i<Jacobian->cols;i++)
	//{
	//	CV_MAT_ELEM(*Jacobian,double,0,i)=0;
	//	CV_MAT_ELEM(*Jacobian,double,1,i)=0;
	//}

	////compute Jacobian using only feature points, seems wrong,TRY IT
	//for (int i=0;i<ptsNum;i++)
	//{
	//	for (int j=0;j<shape_dim;j++)
	//	{
	//		CV_MAT_ELEM(*Jacobian,double,0,j)+=CV_MAT_ELEM(*s_vec,double,j,i);
	//		CV_MAT_ELEM(*Jacobian,double,1,j)+=CV_MAT_ELEM(*s_vec,double,j,ptsNum+i);
	//		
	//	}
	//}


	//int x_location,y_location;

	double x0,x1,x2,x3,y0,y1,y2,y3;
	double alpha,beta,gamma;
	double fenmu;

	int triangleInd;
	int lastTriangleInd=0;
	Shape *dst=shape;

	//compute sd
	//compute hessian
	//CvMat *curhessian=cvCreateMat(shape_dim,shape_dim,CV_64FC1);
	//CvMat *newHessian=cvCreateMat(shape_dim,shape_dim,CV_64FC1);
	//double tmpGI[2];
	//CvMat *tmpSD=cvCreateMat(1,shape_dim,CV_64FC1);
	//double *oriSD=new double[shape_dim];
	//double **sum=new double *[pix_num];
	//double **subsum=new double*[texture_dim];
	//for (int i=0;i<texture_dim;i++)
	//{
	//	subsum[i]=new double[pix_num];
	//}
	//for (int i=0;i<pix_num;i++)
	//{
	//	sum[i]=new double [shape_dim];
	//}
	int tmpind;
	int totalInd=0;
	double ctex;//current texture value used in projected out 

	////intialize data
	//for (int i=0;i<shape_dim;i++)
	//{
	//	for (int j=0;j<shape_dim;j++)
	//	{
	//		CV_MAT_ELEM(*curhessian,double,i,j)=0;
	//		CV_MAT_ELEM(*newHessian,double,i,j)=0;
	//	}
	//}


	//double ***oriSD_ick=new double **[width];
	//for (int i=0;i<width;i++)
	//{
	//	oriSD_ick[i]=new double *[height];
	//	for (int j=0;j<height;j++)
	//	{
	//		oriSD_ick[i][j]=new double[shapeDim];
	//	}
	//}

	int tInd[3];
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			if (CV_MAT_ELEM(*mask,double,j,i)==0)
			{
				continue;
			}

			if (affineTable!=NULL)
			{
				triangleInd=affineTable[i][j]->triangleInd;
				alpha=affineTable[i][j]->alpha;
				beta=affineTable[i][j]->beta;
				gamma=affineTable[i][j]->gamma;
				tInd[0]=(int)CV_MAT_ELEM(*triangleList,double,triangleInd,0);
				tInd[1]=(int)CV_MAT_ELEM(*triangleList,double,triangleInd,1);
				tInd[2]=(int)CV_MAT_ELEM(*triangleList,double,triangleInd,2);
			}
			else
			{

			

				////set to 0, very important
				//for (int ii=0;ii<2;ii++)
				//{
				//	for (int jj=0;jj<shape_dim;jj++)
				//	{
				//		CV_MAT_ELEM(*Jacobian,double,ii,jj)=0;
				//	}
				//}

				triangleInd=-1;
				//判断是否在某个三角形内
				x0=i;
				y0=j;


				//caculate last triangle
				tInd[0]=(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,0);
				tInd[1]=(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,1);
				tInd[2]=(int)CV_MAT_ELEM(*triangleList,double,lastTriangleInd,2);
				x1=dst->pts[tInd[0]][0];
				x2=dst->pts[tInd[1]][0];
				x3=dst->pts[tInd[2]][0];

				y1=dst->pts[tInd[0]][1];
				y2=dst->pts[tInd[1]][1];
				y3=dst->pts[tInd[2]][1];

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
					fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
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
					for (int k=0;k<triangleList->rows;k++)
					{
						tInd[0]=(int)CV_MAT_ELEM(*triangleList,double,k,0);
						tInd[1]=(int)CV_MAT_ELEM(*triangleList,double,k,1);
						tInd[2]=(int)CV_MAT_ELEM(*triangleList,double,k,2);

						x1=dst->pts[tInd[0]][0];
						x2=dst->pts[tInd[1]][0];
						x3=dst->pts[tInd[2]][0];

						y1=dst->pts[tInd[0]][1];
						y2=dst->pts[tInd[1]][1];
						y3=dst->pts[tInd[2]][1];

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
						fenmu=(x2-x1)*(y3-y1)-(y2-y1)*(x3-x1);
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

			if(triangleInd!=-1)// 如果找到三角形，则进行插值
			{

				//for (int j=0;j<shape_dim;j++)
				//{
				//	CV_MAT_ELEM(*Jacobian,double,0,j)+=CV_MAT_ELEM(*s_vec,double,j,i);
				//	CV_MAT_ELEM(*Jacobian,double,1,j)+=CV_MAT_ELEM(*s_vec,double,j,ptsNum+i);

				//}
				//caculate the Jacobian
				for (int k=0;k<shape_dim;k++)
				{
					//		CV_MAT_ELEM(*Jacobian,double,0,j)+=CV_MAT_ELEM(*s_vec,double,j,i);
					//		CV_MAT_ELEM(*Jacobian,double,1,j)+=CV_MAT_ELEM(*s_vec,double,j,ptsNum+i);
					full_Jacobian[i][j][0][k]=alpha*CV_MAT_ELEM(*s_vec,double,k,tInd[0])+
						beta*CV_MAT_ELEM(*s_vec,double,k,tInd[1])+gamma*CV_MAT_ELEM(*s_vec,double,k,tInd[2]);
					full_Jacobian[i][j][1][k]=alpha*CV_MAT_ELEM(*s_vec,double,k,ptsNum+tInd[0])+
						beta*CV_MAT_ELEM(*s_vec,double,k,ptsNum+tInd[1])+gamma*CV_MAT_ELEM(*s_vec,double,k,ptsNum+tInd[2]);
			/*		CV_MAT_ELEM(*Jacobian,double,0,k)=alpha*CV_MAT_ELEM(*s_vec,double,k,tInd[0])+
						beta*CV_MAT_ELEM(*s_vec,double,k,tInd[1])+gamma*CV_MAT_ELEM(*s_vec,double,k,tInd[2]);
					CV_MAT_ELEM(*Jacobian,double,1,k)=alpha*CV_MAT_ELEM(*s_vec,double,k,ptsNum+tInd[0])+
						beta*CV_MAT_ELEM(*s_vec,double,k,ptsNum+tInd[1])+gamma*CV_MAT_ELEM(*s_vec,double,k,ptsNum+tInd[2]);*/
				}
				lastTriangleInd=triangleInd;
			}

			
	//	}
	//}
	//ofstream outJ("jacobian.txt",ios::out);
	//for (int k=0;k<shape_dim;k++)
	//{
	//	outJ<<CV_MAT_ELEM(*Jacobian,double,0,k)<<" "<<CV_MAT_ELEM(*Jacobian,double,1,k)<<endl;
	//}
	//outJ.close();



	////define jacobian,hessian,etc
	//ifstream in("hessian.txt",ios::in);
	//if(!in)
	//{

	//
	//for (int i=0;i<width;i++)
	//{
	//	cout<<i<<" ";
	//	for (int j=0;j<height;j++)
	//	{
	//		
	//		//if not in shape,let it be 0
	//		if (CV_MAT_ELEM(*mask,double,j,i)==0)
	//			continue;

			//tmpGI[0]=CV_MAT_ELEM(*gradient_Tx,double,j,i);
			//tmpGI[1]=CV_MAT_ELEM(*gradient_Ty,double,j,i);

			///*if(tmpGI[0]!=0||tmpGI[1]!=0)
			//	cout<<tmpGI[0]<<" "<<tmpGI[1]<<endl;*/
			//for (int k=0;k<shape_dim;k++)
			//{
			//	oriSD_ick[i][j][k]=tmpGI[0]*CV_MAT_ELEM(*Jacobian,double,0,k)+
			//		tmpGI[1]*CV_MAT_ELEM(*Jacobian,double,1,k);
			//}
		}
	}

	//hessian
	////ifstream in("hessian.txt",ios::in);
	//if(1)
	//{
	//	for (int m=0;m<texture_dim;m++)
	//	{
	//		//CvMat *img=textures[m]->getImageMat();
	//		for (int ii=0;ii<pix_num;ii++)
	//		{
	//			subsum[m][ii]=0;
	//		}
	//		//				subsum=0;
	//		for(int n=0;n<pix_num;n++)
	//		{

	//			ctex=CV_MAT_ELEM(*textures[m]->imgData,double,0,n);
	//			for (int pl=0;pl<shape_dim;pl++)
	//			{
	//				subsum[m][pl]+=ctex*oriSD_ick[inv_mask[n][0]][inv_mask[n][1]][pl];
	//			}		
	//		}
	//	}

	//	for (int i=0;i<pix_num;i++)
	//	{
	//		for (int j=0;j<shape_dim;j++)
	//		{
	//			sum[i][j]=0;
	//		}
	//		for (int m=0;m<texture_dim;m++)
	//		{
	//			ctex=CV_MAT_ELEM(*textures[m]->imgData,double,0,i);
	//			for (int pl=0;pl<shape_dim;pl++)
	//			{
	//				sum[i][pl]+=subsum[m][pl]*ctex;
	//			}
	//		}
	//	}
	//	//totalInd=0;
	//for (int i=0;i<width;i++)
	//{
	//	//cout<<i<<" ";
	//	for (int j=0;j<height;j++)
	//	{
	//		totalInd=CV_MAT_ELEM(*mask_withindex,double,j,i);
	//		//if not in shape,let it be 0
	//		if (totalInd==-1)
	//		{
	//			for (int k=0;k<shape_dim;k++)
	//				SD_ic[i][j][k]=0;//set to 0 if out of mesh
	//			continue;
	//		}
			/////////////////
	//		for (int ii=0;ii<shape_dim;ii++)
	//		{
	//			sum[ii]=0;
	//		}
	//		for (int m=0;m<texture_dim;m++)
	//		{
	//			//CvMat *img=textures[m]->getImageMat();
	//			//for (int ii=0;ii<pix_num;ii++)
	//			//{
	//			//	subsum[ii]=0;
	//			//}
	//			////				subsum=0;
	//			//for(int n=0;n<pix_num;n++)
	//			//{
	//		
	//			//	ctex=CV_MAT_ELEM(*textures[m]->imgData,double,0,n);
	//			//	for (int pl=0;pl<shape_dim;pl++)
	//			//	{
	//			//		subsum[pl]+=ctex*oriSD_ick[inv_mask[n][0]][inv_mask[n][1]][pl];
	//			//	}		
	//			//}


	//			//if (CV_MAT_ELEM(*mask,double,j,i)==1)
	//			{
	//				ctex=CV_MAT_ELEM(*textures[m]->imgData,double,0,totalInd);
	//				for (int pl=0;pl<shape_dim;pl++)
	//				{
	//					sum[pl]+=subsum[m][pl]*ctex;
	//				}
	//			}
	//	

	//		
	//			//for (int n=0;n<ptsNum;n++)
	//			//{
	//			//	for (int l=0;l<shape_dim;l++)
	//			//	{
	//			//		subsum[l]+=cvGet2D(img,shape->pts[n][1],shape->pts[n][0]).val[0]*oriSD[l];
	//			//	}
	//			//}
	///*			for (int l=0;l<shape_dim;l++)
	//			{
	//				sum[l]+=subsum[l]*cvGet2D(img,j,i).val[0];
	//			}*/
	//		}

	////		/////////////////
	//		for (int k=0;k<shape_dim;k++)
	//		{
	//			CV_MAT_ELEM(*tmpSD,double,0,k)=oriSD_ick[i][j][k]-sum[totalInd][k];
	//			//CV_MAT_ELEM(*tmpSD,double,0,k)=oriSD_ick[i][j][k];
	//			SD_ic[i][j][k]=CV_MAT_ELEM(*tmpSD,double,0,k);
	//			//SD_ic[i][j][k]=oriSD_ick[i][j][k];
	//		}

	//		cvMulTransposed(tmpSD,curhessian,1);
	//		cvAdd(Hessian,curhessian,newHessian);
	//		cvCopy(newHessian,Hessian);

	//		//totalInd++;
	//	}
	//}
	////ofstream out("hessian.txt",ios::out);
	//for (int i=0;i<shape_dim;i++)
	//{
	//	for (int j=0;j<shape_dim;j++)
	//	{
	//		cout<<CV_MAT_ELEM(*Hessian,double,i,j)<<" ";
	//	}
	//	cout<<endl;
	//}
	////cout.close();
	//}
	////else
	////{
	////	double tmp;
	////	for (int i=0;i<shape_dim;i++)
	////	{
	////		for (int j=0;j<shape_dim;j++)
	////		{
	////			in>>tmp;
	////			CV_MAT_ELEM(*Hessian,double,i,j)=tmp;
	////		}
	////	}
	////}
	//cvInv(Hessian,inv_Hessian);
	//cout<<"\nInvHessian\n";
	//for (int i=0;i<shape_dim;i++)
	//{
	//	for (int j=0;j<shape_dim;j++)
	//	{
	//		cout<<CV_MAT_ELEM(*inv_Hessian,double,i,j)<<" ";
	//	}
	//	cout<<endl;
	//}

	//for (int i=0;i<texture_dim;i++)
	//{
	//	delete []subsum[i];
	//}
	//delete []subsum;

	//for (int i=0;i<pix_num;i++)
	//{
	//	delete []sum[i];
	//}
	//delete []sum;



	//for (int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		delete []oriSD_ick[i][j];
	//	}
	//	delete []oriSD_ick[i];
	//}
	//delete []oriSD_ick;
}

void AAM_RealGlobal_GPU::precompute()
{

	//let's output all s_vec
	//double sum=0;
	//for(int i=0;i<shape_dim;i++)
	//{
	//	for(int j=0;j<shape_dim;j++)
	//	{
	//		sum=0;
	//		for (int k=0;k<meanShape->ptsNum*2;k++)
	//		{
	//			sum+=CV_MAT_ELEM(*s_vec,double,i,k)*CV_MAT_ELEM(*s_vec,double,j,k);
	//		}
	//		cout<<sum<<" ";

	//	}
	//	cout<<endl;
	//}
	//cout<<"endddddddddddddddddddddddddddddddddddd\n";
	cout<<"shape_dim :"<<shape_dim<<" texture_dim: "<<texture_dim<<endl;
	//set the dim of jacobian

	Texture *curr_texture=new Texture;//current texture in the iteration
	currentTemplate=new Texture;
	
	//Note: we need all the 
	setTemplate(meanTexture->getImageMat());
	//setTemplate(textures[1]->getImageMat());
	setShapeDim(shape_dim);
	int allDim=shape_dim+texture_dim+4;
	Hessian=cvCreateMat(allDim,allDim,CV_64FC1);
	inv_Hessian=cvCreateMat(allDim,allDim,CV_64FC1);
	curhessian=cvCreateMat(allDim,allDim,CV_64FC1);
	newHessian=cvCreateMat(allDim,allDim,CV_64FC1);
	tmpSD=cvCreateMat(1,allDim,CV_64FC1);


	gradient(Template,gradient_Tx,gradient_Ty,meanShape->marginMask);
	getJacobian(meanShape,triangleList);

	//IplImage *img=cvCreateImage(cvGetSize(meanTexture->getImage()),IPL_DEPTH_64F,meanTexture->getImage()->nChannels);
	////output jacobian
	//for(int i=0;i<width;i++)
	//{
	//	for (int j=0;j<height;j++)
	//	{
	//		cvSet2D(img,j,i,cvScalar(full_Jacobian[i][j][1][2]));
	//	}
	//}
	//cvNamedWindow("jacobian");
	//cvShowImage("jacobian",img);
	//cvWaitKey();

	//initialize
	g_ax=new CvMat *[texture_dim];
	g_ay=new CvMat *[texture_dim];

	for (int i=0;i<texture_dim;i++)
	{
		g_ax[i]=cvCreateMat(height,width,CV_64FC1);
		g_ay[i]=cvCreateMat(height,width,CV_64FC1);
	}

	//caculate the gradient
	//Texture *tmp=new Texture();
	//*tmp=(*meanTexture);
	for (int i=0;i<texture_dim;i++)
	{
		//cvGetRow(t_vec,tmp->imgData,i);
		gradient(textures[i]->getImageMat(),g_ax[i],g_ay[i],meanShape->marginMask);
	}

	//intialize the shape and texture weights
	for (int i=0;i<shape_dim;i++)
	{
		s_weight[i]=0;
	}
	//s_weight[0]=9;
	for (int i=0;i<texture_dim;i++)
	{
		t_weight[i]=0;
	}

//	if (smoothWeight>0)
	{
	/*	for (int i=0;i<texture_dim+shape_dim+4;i++)
		{
			last_Iteration_weight[i]=parametersLast[i];
		}*/
		ptsLast=new double *[meanShape->ptsNum];
		for (int i=0;i<meanShape->ptsNum;i++)
		{
			ptsLast[i]=new double [2];
		}
	}


	//initialize the sd image
	SD_ic=new double**[width];
	for (int i=0;i<width;i++)
	{
		SD_ic[i]=new double *[height];
		for (int j=0;j<height;j++)
		{
			//alpha, tho,theta,k,t
			SD_ic[i][j]=new double[allDim];
		}
	}

	//initialize the sd_smooth image
	SD_smooth=new double *[meanShape->ptsNum];
	for (int i=0;i<meanShape->ptsNum;i++)
	{
		SD_smooth[i]=new double [allDim];
		for (int j=shape_dim;j<shape_dim+texture_dim;j++)
		{
			SD_smooth[i][j]=0;
		}
	}




	//if (0)
	//{
	//	cur_XYShape=cvCreateMat(1,meanShape->ptsNum*2,CV_64FC1);
	//	cur_YXShape=cvCreateMat(1,meanShape->ptsNum*2,CV_64FC1);
	//	cvGetRow(s_vec,cur_XYShape,shape_dim-1);
	//	cvGetRow(s_vec,cur_YXShape,shape_dim-2);
	//	Cur_Global=cvCreateMat(3,3,CV_64FC1);
	//	Cur_Global_inv=cvCreateMat(3,3,CV_64FC1);
	//	cvSetZero(Cur_Global);
	//	CV_MAT_ELEM(*Cur_Global,double,2,2)=1;

	//	//set the gloabl vectors for base shapes
	//	int ptsnum=meanShape->ptsNum;
	//	s_vec_tran=cvCreateMat(shape_dim-4,meanShape->ptsNum*2,CV_64FC1);
	//	for (int i=0;i<shape_dim-4;i++)
	//	{
	//		for (int j=0;j<ptsnum*2;j++)
	//		{
	//			if (j<ptsnum)
	//			{
	//				CV_MAT_ELEM(*s_vec_tran,double,i,j)=-CV_MAT_ELEM(*s_vec,double,i,j+ptsnum);
	//			}
	//			else
	//			{
	//				CV_MAT_ELEM(*s_vec_tran,double,i,j)=CV_MAT_ELEM(*s_vec,double,i,j-ptsnum);
	//			}
	//			
	//		}
	//	}
	//	//then orthalized and normalized
	//	CvMat *cur_vec=cvCreateMat(1,ptsnum*2,CV_64FC1);
	//	CvMat *loop_vec=cvCreateMat(1,ptsnum*2,CV_64FC1);
	//	CvMat *res_vec=cvCreateMat(1,ptsnum*2,CV_64FC1);
	//	CvMat *loop_vec_tran=cvCreateMat(ptsnum*2,1,CV_64FC1);
	//	CvMat *pm=cvCreateMat(1,1,CV_64FC1);
	//	double pm_val;
	//	double ssnum;
	//	for (int i=0;i<shape_dim-4;i++)
	//	{
	//		cvGetRow(s_vec_tran,cur_vec,i);
	//		//first, all base vectors
	//		for (int j=0;j<shape_dim;j++)
	//		{
	//			cvGetRow(s_vec,loop_vec,j);
	//			cvTranspose(loop_vec,loop_vec_tran);
	//			cvMatMul(cur_vec,loop_vec_tran,pm);
	//			pm_val=CV_MAT_ELEM(*pm,double,0,0);
	//			for (int k=0;k<ptsnum*2;k++)
	//			{
	//				CV_MAT_ELEM(*s_vec_tran,double,i,k)-=pm_val*CV_MAT_ELEM(*s_vec,double,j,k);
	//			}
	//		}
	//		//then all current tran vectors
	//		for (int j=0;j<i;j++)
	//		{
	//			cvGetRow(s_vec_tran,loop_vec,j);
	//			cvTranspose(loop_vec,loop_vec_tran);
	//			cvMatMul(cur_vec,loop_vec_tran,pm);
	//			pm_val=CV_MAT_ELEM(*pm,double,0,0);
	//			for (int k=0;k<ptsnum*2;k++)
	//			{
	//				CV_MAT_ELEM(*s_vec_tran,double,i,k)-=pm_val*CV_MAT_ELEM(*s_vec_tran,double,j,k);
	//			}
	//		}

	//		//normalized
	//		ssnum=0;
	//		for (int j=0;j<ptsnum*2;j++)
	//		{
	//			ssnum+=CV_MAT_ELEM(*s_vec_tran,double,i,j)*CV_MAT_ELEM(*s_vec_tran,double,i,j);
	//		}
	//		ssnum=sqrt(ssnum);

	//		for (int j=0;j<ptsnum*2;j++)
	//		{
	//			CV_MAT_ELEM(*s_vec_tran,double,i,j)/=ssnum;
	//		}
	//	}
	//	///////////////////done////////////////////////////

	//}



	m_hessian=cvarrToMat(Hessian);
	m_inv_hessian.create(m_hessian.size(),m_hessian.type());
	m_mask=cvarrToMat(mask);
	m_mask_withindex=cvarrToMat(mask_withindex);
	m_gradient_Tx=cvarrToMat(gradient_Tx);
	m_gradient_Ty=cvarrToMat(gradient_Ty);
	m_triangleList=cvarrToMat(triangleList);
	m_s_vec=cvarrToMat(s_vec);
	m_t_vec=cvarrToMat(t_vec);

	m_g_ax=new Mat[texture_dim];
	m_g_ay=new Mat[texture_dim];
	for (int i=0;i<texture_dim;i++)
	{
		m_g_ax[i]=cvarrToMat(g_ax[i]);
		m_g_ay[i]=cvarrToMat(g_ay[i]);
	}
	m_errorImageMat=cvarrToMat(errorImageMat);

	////int dim=shape_dim+texture_dim;
	// fullSD.create(width*height,allDim,CV_64FC1);
	//// fullSD_gpu.create(width*height,dim,CV_64FC1);
	// fullSD_tran.create(allDim,width*height,CV_64FC1);
	//// fullSD_tran_GPU.create(dim,width*height,CV_64FC1);
	////gpu_Hessian.create(dim,dim,CV_64FC1);

	///gpu initialization
	//fullSD_gpu.create(pix_num,allDim,CV_64FC1);
	//fullSD_tran_GPU.create(allDim,pix_num,CV_64FC1);
	//gpu_Hessian.create(allDim,allDim,CV_64FC1);

	 fullSD.create(pix_num,allDim,CV_64FC1);
	 fullSD_tran.create(allDim,pix_num,CV_64FC1);
	full_Hessian.create(allDim,allDim,CV_64FC1);

	if (smoothWeight>0)
	{
		fullSD_smooth.create(meanShape->ptsNum,allDim,CV_64FC1);
		fullSD_tran_smooth.create(allDim,meanShape->ptsNum,CV_64FC1);
		full_Hessian_smooth.create(allDim,allDim,CV_64FC1);
	}

	//initialize the transform parameters
	theta=0;
	k_scale=1;
	transform_x=transform_y=0;

	warp_igx=cvCreateMat(height,width,CV_64FC1);
	warp_igy=cvCreateMat(height,width,CV_64FC1);

	m_warp_igx=cvarrToMat(warp_igx);
	m_warp_igy=cvarrToMat(warp_igy);

	//output triangleList for display
	string trName=dataDir+"triangleList.txt";
	ofstream out(trName.c_str(),ios::out);
	for (int k=0;k<triangleList->rows;k++)
	{
		out<<(int)CV_MAT_ELEM(*triangleList,double,k,0)<<" "<<
		(int)CV_MAT_ELEM(*triangleList,double,k,1)<<" "<<
		(int)CV_MAT_ELEM(*triangleList,double,k,2)<<endl;
	}

	//last parameters
	s_weight_last=new double[shape_dim];
	t_weight_last=new double[texture_dim];

	//parallel computing
	pNum=6;
	parallelStep=(double)pix_num/(double)pNum;
	p_fullSD=new Mat[pNum];
	p_fullSD_tran=new Mat[pNum];
	p_fullHessian=new Mat[pNum];
	for (int i=0;i<pNum;i++)
	{
		if (i<pNum-1)
		{
			p_fullSD[i].create(parallelStep,fullSD.cols,CV_64FC1);
			p_fullSD_tran[i].create(fullSD.cols,parallelStep,CV_64FC1);
		}
		else
		{
			p_fullSD[i].create(fullSD.rows-(pNum-1)*parallelStep,fullSD.cols,CV_64FC1);
			p_fullSD_tran[i].create(fullSD.cols,fullSD.rows-(pNum-1)*parallelStep,CV_64FC1);
		}
		p_fullHessian[i].create(fullSD.cols,fullSD.cols,CV_64FC1);

	}

	if (usingCUDA)
	{
		cuda_cols=shape_dim+texture_dim+4;
		cuda_rows=pix_num;
		cuda_row_pitch=cuda_rows;
		cuda_numF=cuda_cols;
		cuda_data_cpu.resize(MAX_COUNT);
	}

	int cind=0;
	indexTabel=new int *[3];
	for (int i=0;i<3;i++)
	{
		indexTabel[i]=new int[pix_num];
	}
	//indexTabel.create(3,pix_num*2,CV_64FC1);
	int triangleInd;
	for (int i=meanShape->minx;i<=meanShape->maxx;i++)
	{
		for (int j=meanShape->miny;j<=meanShape->maxy;j++)
		{
			triangleInd=affineTable[i][j]->triangleInd;
			if (triangleInd!=-1)
			{
				indexTabel[0][cind]=(int)m_triangleList.at<double>(triangleInd,0);
				indexTabel[1][cind]=(int)m_triangleList.at<double>(triangleInd,1);
				indexTabel[2][cind]=(int)m_triangleList.at<double>(triangleInd,2);
				cind++;
			}
		}
	}


	//set the initial mask to be all one
	imageMask=m_mask.clone();

	meanShapeCenter=meanShape->getcenter();
}

void AAM_RealGlobal_GPU::calculateData_onrun_AAM_combination(Mat &m_img,vector<int> &finalInd,bool isKeepPar)
{
	//need to be deleted if we desire speed
	/*if (showSingleStep)
	{
		g_inputImg=m_img.clone();
	}*/


	if (!isKeepPar)
	{
		for (int i=0;i<shape_dim;i++)
		{
			parameters[i]=s_weight[i];
		}
		for (int i=0;i<texture_dim;i++)
		{
			parameters[shape_dim+i]=0;
		}
		parameters[shape_dim+texture_dim]=initialTheta;
		parameters[shape_dim+texture_dim+1]=initialScale;
		parameters[shape_dim+texture_dim+2]=initialTx;
		parameters[shape_dim+texture_dim+3]=initialTy;
	}
	else
	{
		//cout<<"using previous info\n";

		for (int i=0;i<shape_dim;i++)
		{
			parameters[i]=s_weight[i];
		}
		/*for (int i=0;i<texture_dim;i++)
		{
			parameters[shape_dim+i]=0;
		}*/
		parameters[shape_dim+texture_dim]=initialTheta;
		parameters[shape_dim+texture_dim+1]=initialScale;
		parameters[shape_dim+texture_dim+2]=initialTx;
		parameters[shape_dim+texture_dim+3]=initialTy;

		parameters[shape_dim]=-1000000000;
		/*for (int i=0;i<texture_dim;i++)
		{
			parameters[shape_dim+i]=0;
		}*/
	}


	


	

	//no need to input current color image
	setData_onRun_AAM(parameters,m_img.cols,m_img.rows,finalInd);
}

bool AAM_RealGlobal_GPU::iterate_clean_CPU(Mat &m_img,RandTree *_trees,int type)
{
	trees=_trees;
	
	//setup the probability map


	if (prob_conv==NULL)
	{
		prob_conv=new Mat[trees->labelNum-1];
		prob_mu=new double*[trees->labelNum-1];
		for (int i=0;i<trees->labelNum-1;i++)
		{
			prob_conv[i].create(2,2,CV_64FC1);
			prob_mu[i]=new double [2];
		}

		prob_conv_candidates=new Mat *[trees->labelNum-1];
		prob_mu_candidates=new double**[trees->labelNum-1];
		for (int i=0;i<trees->labelNum-1;i++)
		{
			prob_conv_candidates[i]=new Mat [100];
			prob_mu_candidates[i]=new double *[100];
			for(int j=0;j<100;j++)
			{
				prob_conv_candidates[i][j].create(2,2,CV_64FC1);
				prob_mu_candidates[i][j]=new double [2];
			}
		}
	}


	if (inputGradient_x==NULL)
	{
		mat_currentImage=cvCreateMat(m_img.rows,m_img.cols,CV_64FC1);
		m_currentImage=cvarrToMat(mat_currentImage);

		inputGradient_x=cvCreateMat(m_img.rows,m_img.cols,CV_64FC1);
		inputGradient_y=cvCreateMat(m_img.rows,m_img.cols,CV_64FC1);
		m_inputGradient_x=cvarrToMat(inputGradient_x);
		m_input_Gradient_y=cvarrToMat(inputGradient_y);
	}

	//get current gradient
	for (int i=0;i<mat_currentImage->rows;i++)
	{
		for (int j=0;j<mat_currentImage->cols;j++)
		{
			m_currentImage.at<double>(i,j)=m_img.at<uchar>(i,j);
		}
	}

	gradient(mat_currentImage,inputGradient_x,inputGradient_y,NULL);



	return iterate_cpu(m_img,type);

}

void AAM_RealGlobal_GPU::iterate_clean(Mat &m_img)
{
	g_inputImg=m_img.clone();
	int i,j;
	//if(1)

		//double *p=meanShape->getcenter();


	//	transform_x=initialTx;
	//	transform_y=initialTy;

	//	k_scale=initialScale;

		
	/*	for (int i=0;i<shape_dim;i++)
		{
			parameters[i]=0;
		}*/
		for (i=0;i<texture_dim;i++)
		{
			parameters[shape_dim+i]=0;
		}
		parameters[shape_dim+texture_dim]=initialTheta;
		parameters[shape_dim+texture_dim+1]=initialScale;
		parameters[shape_dim+texture_dim+2]=initialTx;
		parameters[shape_dim+texture_dim+3]=initialTy;



		if (inputGradient_x==NULL)
		{
			mat_currentImage=cvCreateMat(m_img.rows,m_img.cols,CV_64FC1);
			m_currentImage=cvarrToMat(mat_currentImage);

			inputGradient_x=cvCreateMat(m_img.rows,m_img.cols,CV_64FC1);
			inputGradient_y=cvCreateMat(m_img.rows,m_img.cols,CV_64FC1);
			m_inputGradient_x=cvarrToMat(inputGradient_x);
			m_input_Gradient_y=cvarrToMat(inputGradient_y);
		}

		//get current gradient
		for (int i=0;i<mat_currentImage->rows;i++)
		{
			for (int j=0;j<mat_currentImage->cols;j++)
			{
				m_currentImage.at<double>(i,j)=m_img.at<uchar>(i,j);
			}
		}

		gradient(mat_currentImage,inputGradient_x,inputGradient_y,NULL);

		//copy the gradient to the float array
		if (usingGPU)
		{
			if (cu_gradientX==NULL)
			{
				int MPN=MAX_PIXEL_NUM;
				cu_gradientX=new float[MPN];
				cu_gradientY=new float[MPN];
			}
			for (int i=0;i<m_currentImage.rows;i++)
			{
				for (int j=0;j<m_currentImage.cols;j++)
				{
					cu_gradientX[i*m_currentImage.cols+j]=m_inputGradient_x.at<double>(i,j);
					cu_gradientY[i*m_currentImage.cols+j]=m_input_Gradient_y.at<double>(i,j);
				}

			}
		}


	//Mat m_img=cvarrToMat(Input);
	for (i=0;i<m_img.rows;i++)
	{
		for (j=0;j<m_img.cols;j++)
		{
			inputImg[i*m_img.cols+j]=m_img.at<uchar>(i,j);
		}
	}
	setData_onRun(parameters,inputImg,cu_gradientX,cu_gradientY,m_img.cols,m_img.rows);

	iterate_CUDA(m_img.cols,m_img.rows,smoothWeight,AAM_weight,currentFrame,startNum,parameters,meanShape->inv_mask);

}


void AAM_RealGlobal_GPU::iterate_GPU(IplImage *Input)
{
	//IplImage *warpedInput;//=cvCreateImage(cvGetSize(Template),Template->depth,Template->nChannels);

		//initialize
	//cout<<meanShape->width<<" "<<meanShape->height<<endl;
	int i,j;
	//if(1)
	if(currentFrame==startNum)
	{
		//currentShape->scale(0.98,1);
		//only do once to intialize face position
		double *p=meanShape->getcenter();
		//	cout<<p[0]<<" "<<p[1]<<endl;
		//	//offset=cvPoint(p[0],p[1])-faceCenter;	
		//	cvCopyImage(Input,imgcopy);
		//	cvNamedWindow("1");
		//	for (int i=0;i<ptsNum;i++)
		//	{
		//		cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
		//	}
		////	cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
		//	cvCircle(imgcopy,cvPoint((int)p[0],(int)p[1]),3,CV_RGB(0,0,255));
		//	cvShowImage("1",imgcopy);
		//	cvWaitKey(0);
		//currentShape->translate(p[0]-faceCenter.x,p[1]-faceCenter.y-15);
		//tran_x=p[0]-faceCenter.x;
		//tran_y=p[1]-faceCenter.y-30;

		//transform_x=faceCenter.x-p[0]+10;
		//transform_y=faceCenter.y-p[1]+55;

		transform_x=faceCenter.x-p[0]+initialTx;
		transform_y=faceCenter.y-p[1]+initialTy;

		k_scale=initialScale;

		
		for (int i=0;i<shape_dim;i++)
		{
			parameters[i]=s_weight[i];
		}
		for (i=0;i<texture_dim;i++)
		{
			parameters[shape_dim+i]=t_weight[i];
		}
		parameters[shape_dim+texture_dim]=theta;
		parameters[shape_dim+texture_dim+1]=k_scale;
		parameters[shape_dim+texture_dim+2]=transform_x;
		parameters[shape_dim+texture_dim+3]=transform_y;

	/*	transform_x=245;
		transform_y=340;*/
		//k_scale=1000;

	/*	transform_x=278;
		transform_y=166;*/
	}
	else
	{
	/*	for (int i=0;i<shape_dim;i++)
		{
			parameters[i]=s_weight[i];
		}
		for (i=0;i<texture_dim;i++)
		{
			parameters[shape_dim+i]=t_weight[i];
		}
		parameters[shape_dim+texture_dim]=theta;
		parameters[shape_dim+texture_dim+1]=k_scale;
		parameters[shape_dim+texture_dim+2]=transform_x;
		parameters[shape_dim+texture_dim+3]=transform_y;*/

		parameters[0]=-1000000000;
	}


	//denormalize
	//if(currentFrame==startNum)
	////if(0)
	//{
	//	char ppname[500];
	//	sprintf(ppname, "%d_parameters.txt", 10000+currentFrame);
	//	string tmpName=ppname;
	//	string fullName=dataDir+tmpName.substr(1,tmpName.length()-1);;
	//	ifstream in(fullName.c_str(),ios::in);
	//	if (in)
	//	{
	//		stepLength=0.5;
	//		//MaxIterNum=15;
	//	}
	//	while(in)
	//	{
	//		for (int i=0;i<shape_dim;i++)
	//		{
	//			in>>s_weight[i];
	//		}
	//		for (int i=0;i<texture_dim;i++)
	//		{
	//			in>>t_weight[i];
	//		}
	//		in>>theta>>k_scale>>transform_x>>transform_y;
	//	}
	//	in.close();

	//}
	//
	//set current parameters


	//for (int i=0;i<shape_dim;i++)
	//{
	//	cout<<parameters[i]<<" "<<s_weight[i]<<endl;
	//}


	Mat m_img=cvarrToMat(Input);
	for (i=0;i<m_img.rows;i++)
	{
		for (j=0;j<m_img.cols;j++)
		{
			inputImg[i*m_img.cols+j]=m_img.at<uchar>(i,j);
		}
	}
	setData_onRun(parameters,inputImg,cu_gradientX,cu_gradientY,m_img.cols,m_img.rows);

	/*cout<<"Mean Texture\n";
	for (i=0;i<100;i++)
	{
		cout<<meanTexture->m_imgData.at<double>(0,i)<<" ";
	}
	cout<<endl;*/

	//cout<<"eigen texture\n";
	//for (i=0;i<texture_dim;i++)
	//{
	//	cout<<m_t_vec.at<double>(0,i)<<" ";
	//}
	//cout<<endl;
	//cout<<meanTexture->m_imgData.cols<<endl;
	//call the iteration function
	iterate_CUDA(Input->width,Input->height,smoothWeight,AAM_weight,currentFrame,startNum,parameters,meanShape->inv_mask);

}

bool AAM_RealGlobal_GPU::iterate_cpu(Mat &img,int type)
{
	//IplImage *warpedInput;//=cvCreateImage(cvGetSize(Template),Template->depth,Template->nChannels);

	//denormalize
	
	IplImage *Input=&IplImage(img);

	//namedWindow("1");
	//imshow("1",cvarrToMat(Input));
	//waitKey();

	int dim=shape_dim+texture_dim+4;
	
	double threshold=0.00001;
	int MaxIterNum=20;

//	if (smoothWeight>0)
	{
		MaxIterNum=30;
	}
//	bool showSingleStep=true;

	
	
	//initialize
	//if (1)
	if(currentShape==NULL)
	{
		currentShape=new Shape(meanShape);
		currentLocalShape=new Shape(meanShape);
	}
	
	//always have smooth weight
	smoothWeight=smoothWeight_backup;
	if(1)
	{
		//currentShape->scale(0.98,1);
		//only do once to intialize face position

		//currentShape->getVertex(meanShape->ptsForMatlab,meanShape->ptsNum,meanShape->width,meanShape->height);

		for (int i=0;i<texture_dim;i++)
		{
			t_weight[i]=0;
		}
		transform_x=initialTx;
		transform_y=initialTy;
		theta=initialTheta;
		k_scale=initialScale;
		////if(!treeInitialization)
		//{
		//	for (int i=0;i<shape_dim;i++)
		//	{
		//		s_weight[i]=0;
		//	}

		//	//double *p=meanShape->getcenter();

		//	if (currentFrame==startNum)
		//	{
		//		transform_x=faceCenter.x+initialTx;
		//		transform_y=faceCenter.y+initialTy;
		//	}
		//
		//	theta=0;

		//	k_scale=initialScale;
		//}
		/*s_weight[0]=0.5;
		s_weight[1]=0.2;*/
	}
	

	currentTexture=new Texture();

	int ptsNum=meanShape->ptsNum;
	double *deltaP=new double[dim];
	double *sumSD=new double[dim];
	double *sumSmoothSD=new double[dim];
	double *sumDectionSD=new double[dim];
	double *deltaS_T=new double[ptsNum*2];

	//global adjustment
	double *deltaGlbal;
	if (isGlobaltransform)
	{
		deltaGlbal=new double[ptsNum*2];
	}
	double ssum=1;
	
	int times=0;//total running time

	int c_pixnum;//current pixel ind
	IplImage *imgcopy=cvCreateImage(cvGetSize(Input),Input->depth,Input->nChannels);



	double adjustGamma;

	//DWORD dwStart, dwStop;  
	//DWORD d_totalstart;

	Texture *meanfordisplay=new Texture;
	double lastError=999999;
	//if (lastErrorSum==-1)
	{
		lastErrorSum=999999;
	}
	
	//*meanfordisplay=(*meanTexture)	;

	DWORD HessianStart, HessianStop,d_totalstart;  
	//LONG   dwStart,dwStop,d_totalstart; 
	//LONG   persecond; 
	//if (outputtime)
	//{
	//		QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);//询问系统一秒钟的频率 
	//}

	double errorIm;
	double errorSum;
	double errorSum_detection;
	int tind;
	double alpha,beta,gamma;
	int triangleInd[3];
	double x1,x2,x3,y1,y2,y3;
	double cx1,cx2,cx3,cy1,cy2,cy3;
	double x0,y0;
	double fenmu;
	double tempPts[2];
	double tmpCurPts[2];
	int i,j,k;

	double tmpScale;
	int recaculatetime=0;
	//stepLength=1;
	//recaculate=false;
	increaseTime=0;
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	////下面是你要计算运行时间的程序代码 
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	//double   time=(t2-t1)/persecond; 

	Pts_Difference=1000000;


	//ofstream FullOut("FullOutput.txt",ios::out);
	//ofstream out_HHH("GPU_Hessian.txt",ios::out);
	//ofstream out_J("Jacobians_CPU.txt",ios::out);
	//ofstream out_E("E_C.txt",ios::out);
	ofstream outPutPts;
	char currentSaveName[500];
	char tmpSaveName[50];
	
	if (type==0) //has detection weight by defaught
	{
		smoothWeight=smoothWeight_backup;
	}
	else
		smoothWeight=0;
	
	priorWeight=priorWeight_backup;
	localPCAWeight=localPCAWeight_backup;
	//smoothWeight=0;
	cout<<AAM_weight<<" "<<smoothWeight<<" "<<priorWeight<<endl;

	cout<<pix_num<<endl;

	double totalweight,lastTotalWeight;
	lastTotalWeight=1000000;

	bool isAAM=false;
	int aamTimes=0;
	while(1)
	{
		if (outputtime)
		{
			HessianStart=GetTickCount();
			d_totalstart=HessianStart;
		}
		

		////test1: no update for texture---done,the code works
		////so the problem lies on texture updating
		//for (int i=0;i<texture_dim;i++)
		//{
		//	t_weight[i]=0;
		//}

		//get current shape
		//finally, we update the currentshape

		
		for (i=0;i<ptsNum;i++)
		{
			currentLocalShape->ptsForMatlab[i]=meanShape->ptsForMatlab[i]-meanShapeCenter[0];
			currentLocalShape->ptsForMatlab[ptsNum+i]=meanShape->ptsForMatlab[ptsNum+i]-meanShapeCenter[1];
			//first, pca weights
			for (j=0;j<shape_dim;j++)
			{
				//currentShape->ptsForMatlab[i]+=s_weight[j]*CV_MAT_ELEM(*s_vec,double,j,i);
				currentLocalShape->ptsForMatlab[i]+=s_weight[j]*m_s_vec.at<double>(j,i);
				currentLocalShape->ptsForMatlab[ptsNum+i]+=s_weight[j]*m_s_vec.at<double>(j,ptsNum+i);
			}
			//then, global transform
			currentShape->ptsForMatlab[i]=k_scale*(cos(theta)*currentLocalShape->ptsForMatlab[i]-sin(theta)*currentLocalShape->ptsForMatlab[i+ptsNum])+transform_x;
			currentShape->ptsForMatlab[ptsNum+i]=k_scale*(sin(theta)*currentLocalShape->ptsForMatlab[i]+cos(theta)*currentLocalShape->ptsForMatlab[i+ptsNum])+transform_y;

		/*	currentShape->ptsForMatlab[i]+=transform_x;
			currentShape->ptsForMatlab[i+ptsNum]+=transform_y;*/
		}
		
	/*	for (i=0;i<ptsNum;i++)
		{
			cout<<currentLocalShape->ptsForMatlab[i]<<" "<<currentLocalShape->ptsForMatlab[ptsNum+i]<<endl;
		}*/

		for (i=0;i<ptsNum;i++)
		{
			currentShape->pts[i][0]=currentShape->ptsForMatlab[i];
			currentShape->pts[i][1]=currentShape->ptsForMatlab[ptsNum+i];
			currentLocalShape->pts[i][0]=currentLocalShape->ptsForMatlab[i];
			currentLocalShape->pts[i][1]=currentLocalShape->ptsForMatlab[ptsNum+i];
			
		}

		//no update the w_dis
		//for (int ii=0;ii<trees->numOfLabels;ii++)
		//{
		//	int cindex=trees->interestPtsInd[ii];
		//	double tx,ty;
		//	tx=currentShape->pts[cindex][0];
		//	ty=currentShape->pts[cindex][1];
		//	for (int j=0;j<candidatePoints[ii].size();j++)
		//	{
		//		float currentDis=sqrt((tx-candidatePoints[ii][j].x)*
		//			(tx-candidatePoints[ii][j].x)+
		//			(ty-candidatePoints[ii][j].y)*
		//			(ty-candidatePoints[ii][j].y));
		//		float probAll=powf(2.7173f,-currentDis*currentDis/200);
		//		//probAll*=powf(e,-(maximumProb[finalInd[ii]]-1)*(maximumProb[finalInd[ii]]-1)/
		//		//(2*sigma1*sigma1));
		//		probForEachFeatureCandidates[ii][j]=probAll;

		//		/*	if (finalInd[ii]==2)
		//		{
		//		cout<<j<<" "<<currentDis<<" "<<probAll<<" "<<candidatePoints[cindex][j].x<<" "<<candidatePoints[cindex][j].y<<endl;

		//		}*/

		//	}
		//	candidateNum[ii]=candidatePoints[ii].size();
		//	//AAM_exp->probForEachFeature[ii]=1;
		//}

		if (0)
		{
			Mat curImg=cvarrToMat(Input).clone();
			cvtColor( curImg, curImg, CV_GRAY2RGB );
			
			Point c,c1;
			Scalar s,s1;
			s.val[0]=s.val[1]=0;
			s.val[2]=255;

			s1.val[0]=255;
			s1.val[1]=s1.val[2]=0;
			cvCopyImage(Input,imgcopy);
			cout<<"inlier num is "<<trees->numOfLabels<<endl;
			//for (int i=0;i<trees->numOfLabels;i++)
			//{
			///*	if (trees->interestPtsInd[i]!=27)
			//	{
			//		continue;
			//	}*/
			//	//invert(prob_conv[i],conv[i]);
			//	//currentDir=atan2(conv[i].at<double>(1,1),conv[i].at<double>(0,0));

			//	
			//	
			//	//for (int i=0;i<ptsNum;i++)
			//	//{
			//		//cvCircle(imgcopy,cvPoint(prob_mu[i][0],prob_mu[i][1]),1,CV_RGB(0,0,255));
			//		//c.x=prob_mu[i][0];c.y=prob_mu[i][1];

			//		////cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),1,CV_RGB(0,0,255));
			//		////c1.x=currentShape->pts[trees->interestPtsInd[i]][0];
			//		////c1.y=currentShape->pts[trees->interestPtsInd[i]][1];
			//		//circle(curImg,c,2,s);
			//		//circle(curImg,c1,3,s1);

			//	//}

			//	for (int j=0;j<candidatePoints[i].size();j++)
			//	{
			//		//c.x=prob_mu[i][0];c.y=prob_mu[i][1];

			//		//cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),1,CV_RGB(0,0,255));
			//		//c1.x=currentShape->pts[trees->interestPtsInd[i]][0];
			//		//c1.y=currentShape->pts[trees->interestPtsInd[i]][1];
			//		circle(curImg,candidatePoints[i][j],2,s);
			//		drawWordsOnImg(curImg,candidatePoints[i][j].x,candidatePoints[i][j].y,"",probForEachFeatureCandidates[i][j]);
			//	}
			//	
			//}
			for (int i=0;i<ptsNum;i++)
			{
				/*if (i!=28)
				{
					continue;
				}*/
				c.x=currentShape->pts[i][0];
				c.y=currentShape->pts[i][1];
				circle(curImg,c,2,s1);
			}

			cvNamedWindow("1");
			imshow("1",curImg);
			cvWaitKey();
		}
		
		//strcpy(currentSaveName,currentNamewithoutHouzhui.c_str());
		//sprintf(tmpSaveName, "_Iteration_%d.txt", times);
		//strcat(currentSaveName,tmpSaveName);
		////cout<<currentSaveName<<endl;
		//outPutPts.open(currentSaveName,ios::out);
		//for (i=0;i<ptsNum;i++)
		//{
		//	outPutPts<<currentShape->pts[i][0]<<" "<<currentShape->pts[i][1]<<endl;
		//}
		//outPutPts.close();

	/*	for (i=0;i<ptsNum;i++)
		{
			cout<<currentShape->pts[i][0]<<" "<<currentShape->pts[i][1]<<endl;
		}

		cvCopyImage(Input,imgcopy);
			cvNamedWindow("1");
			for (int i=0;i<ptsNum;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
					cvShowImage("1",imgcopy);
			cvWaitKey();
			}*/

		/*	for (int i=0;i<trees->numOfLabels;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[trees->interestPtsInd[i]][0],currentShape->pts[trees->interestPtsInd[i]][1]),1,CV_RGB(0,0,255));
			}*/

		/*	for (int i=0;i<trees->numOfLabels;i++)
			{
				for (int j=0;j<trees->candidates[i]->size();j++)
				{
					cvCircle(imgcopy,cvPoint(trees->candidates[i]->at(j).x,trees->candidates[i]->at(j).y),3,CV_RGB(0,0,255));
				}
			}*/
			//cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
//			cvCircle(imgcopy,cvPoint(p[0],p[1]),3,CV_RGB(0,0,255));
		
		//calculate conv and mu
		if (smoothWeight>0)
		//if(0)
		{
		//	cout<<"current shape pts 10:\n";
		//	cout<<currentShape->pts[10][0]<<" "<<currentShape->pts[10][1]<<endl;
			//for (i=0;i<trees->numOfLabels;i++)
			//{
			//	//cout<<"calculating conv and mu for feature "<<i<<" at "<<
			//	//currentShape->pts[trees->interestPtsInd[i]][0]<<" "<<currentShape->pts[trees->interestPtsInd[i]][1]<<endl;
			//
			//	//calculateMandC(trees->probabilityMap[i],currentShape->pts[trees->interestPtsInd[i]][0],
			//		//currentShape->pts[trees->interestPtsInd[i]][1],20,prob_mu[i],prob_conv[i]);
			//	
			//	calculateMandC_withGuidance(trees->probabilityMap[i],currentShape->pts[trees->interestPtsInd[i]][0],
			//		currentShape->pts[trees->interestPtsInd[i]][1],20,prob_mu[i],prob_conv[i],shapeSample[i],i);

			//	////if(i==0)
			//	//{
			//	//	int cx=currentShape->pts[trees->interestPtsInd[i]][0];
			//	//	int cy=currentShape->pts[trees->interestPtsInd[i]][1];
			//	//	cout<<prob_conv[i].at<double>(0,0)<<" "<<conv_precalculated[i].at<double>(cy*2,cx*2)<<endl;
			//	//	cout<<prob_conv[i].at<double>(0,1)<<" "<<conv_precalculated[i].at<double>(cy*2,cx*2+1)<<endl;
			//	//	cout<<prob_conv[i].at<double>(1,0)<<" "<<conv_precalculated[i].at<double>(cy*2+1,cx*2)<<endl;
			//	//	cout<<prob_conv[i].at<double>(1,1)<<" "<<conv_precalculated[i].at<double>(cy*2+1,cx*2+1)<<endl;
			//	//	cout<<"************************************************\n";
			//	//}

			//	//cout<<i<<endl;

			//	//int cx=currentShape->pts[trees->interestPtsInd[i]][0];
			//	//int cy=currentShape->pts[trees->interestPtsInd[i]][1];
			//	////cout<<trees->interestPtsInd[i]<<" "<<cx<<" "<<cy<<endl;
			//	//prob_conv[i].at<double>(0,0)=conv_precalculated[i].at<double>(cy*2,cx*2);
			//	//prob_conv[i].at<double>(0,1)=conv_precalculated[i].at<double>(cy*2,cx*2+1);
			//	//prob_conv[i].at<double>(1,0)=conv_precalculated[i].at<double>(cy*2+1,cx*2);
			//	//prob_conv[i].at<double>(1,1)=conv_precalculated[i].at<double>(cy*2+1,cx*2+1);

			//	//calculateMandC_autoSized(trees->probabilityMap[i],currentShape->pts[trees->interestPtsInd[i]][0],
			//		//currentShape->pts[trees->interestPtsInd[i]][1],60,prob_mu[i],prob_conv[i]);

			//	/*	p<<currentShape->pts[trees->interestPtsInd[i]][0]<<" "<<currentShape->pts[trees->interestPtsInd[i]][1]<<
			//	" "<<prob_mu[i][0]<<" "<<prob_mu[i][1]<<endl;*/


			//	/*cout<<"out calculated conv: "<<prob_conv[i].at<double>(0,0)<<" "<<
			//	prob_conv[i].at<double>(0,1)<<" "<<prob_conv[i].at<double>(1,0)<<" "<<prob_conv[i].at<double>(1,1)<<" "<<endl;*/
			//}

			////////////////////////updating the conv matrix/////////////////////////////////
			//for (i=0;i<trees->numOfLabels;i++)
			//{
			//	for (int j=0;j<candidatePoints[i].size();j++)
			//	{
			//		int tmp[2];
			//		tmp[0]=candidatePoints[i][j].x;
			//		tmp[1]=candidatePoints[i][j].y;
			//		
			//		//cout<<i<<" "<<tmp[0]<<" "<<tmp[1]<<endl;
			//	//	cout<<prob_conv_candidates[i][j].at<double>(0,0)<<endl;
			//		//cout<<prob_conv_candidates[i][j]<<endl;
			//		calculateMandC_withGuidance(trees->probabilityMap[i],currentShape->pts[trees->interestPtsInd[i]][0],
			//			currentShape->pts[trees->interestPtsInd[i]][1],20,prob_mu_candidates[i][j],prob_conv_candidates[i][j],tmp,i,j);
			//	}
			//}
			/////////////////////////////////////////////////////////////


			////show the direction
			//Mat conv[50];
			//double currentDir;
			//Mat curImg=cvarrToMat(Input).clone();
			//cvtColor( curImg, curImg, CV_GRAY2RGB );
			//
			//Point c,c1;
			//Scalar s,s1;
			//s.val[0]=s.val[1]=0;
			//s.val[2]=255;

			//s1.val[0]=255;
			//s1.val[1]=s1.val[2]=0;
			//cvCopyImage(Input,imgcopy);
			//for (int i=0;i<trees->numOfLabels;i++)
			//{
			///*	if (trees->interestPtsInd[i]!=27)
			//	{
			//		continue;
			//	}*/
			//	//invert(prob_conv[i],conv[i]);
			//	//currentDir=atan2(conv[i].at<double>(1,1),conv[i].at<double>(0,0));

			//	
			//	
			//	//for (int i=0;i<ptsNum;i++)
			//	//{
			//		//cvCircle(imgcopy,cvPoint(prob_mu[i][0],prob_mu[i][1]),1,CV_RGB(0,0,255));
			//		c.x=prob_mu[i][0];c.y=prob_mu[i][1];

			//		//cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),1,CV_RGB(0,0,255));
			//		//c1.x=currentShape->pts[trees->interestPtsInd[i]][0];
			//		//c1.y=currentShape->pts[trees->interestPtsInd[i]][1];
			//		circle(curImg,c,2,s);
			//		//circle(curImg,c1,3,s1);

			//	//}
			//	
			//}
			//cvNamedWindow("1");
			//imshow("1",curImg);
			//cvWaitKey();
		}
		
		

		

	/*	ofstream out("F:\\Projects\\Facial feature points detection\\Matlab code\\meanshape_me.txt",ios::out);
		for (i=0;i<ptsNum;i++)
		{
			out<<currentShape->pts[i][0]<<" "<<currentShape->pts[i][1]<<endl;
		}
		out.close();*/
		/*FullOut<<"----------------------current local shape-----------------------------\n";
		for (i=0;i<ptsNum;i++)
		{
			FullOut<<currentLocalShape->ptsForMatlab[i]<<" "<<currentLocalShape->ptsForMatlab[ptsNum+i]<<endl;
		}
		FullOut<<endl;
		
		FullOut<<"----------------------current shape-----------------------------\n";
		for (i=0;i<ptsNum;i++)
		{
			FullOut<<currentShape->ptsForMatlab[i]<<" "<<currentShape->ptsForMatlab[ptsNum+i]<<endl;
		}
		FullOut<<endl;*/

		//get new template
		getCurrentTexture(t_weight);

		if (outputtime)
		{
			HessianStop=GetTickCount();

			//cout<<"texture synthesizing  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}

		//see current shape	
		//if(currentFrame>=startNum)
		//{
		//	cvCopyImage(Input,imgcopy);
		//	cvNamedWindow("1");
		//	for (int i=0;i<ptsNum;i++)
		//	{
		//		cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
		//	}
		////	cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
		////	cvCircle(imgcopy,cvPoint((int)p[0],(int)p[1]),3,CV_RGB(0,0,255));
		//	cvShowImage("1",imgcopy);
		//	cvWaitKey(0);
		//}

		//cvNamedWindow("3");
		//*meanfordisplay=(*currentTemplate);
		//meanfordisplay->devide(texture_scale);
		//for (int i=0;i<meanfordisplay->pix_num;i++)
		//{
		//	CV_MAT_ELEM(*(meanfordisplay->imgDa ta),double,0,i)+=100;
		//}
		//IplImage *ccc=meanfordisplay->getImage();
		//for (int i=0;i<ptsNum;i++)
		//{
		//	cvCircle(ccc,cvPoint(meanShape->pts[i][0],meanShape->pts[i][1]),3,CV_RGB(0,0,255));
		//}
		//cvShowImage("3",ccc);
		//cvWaitKey();

		if (usingCUDA)
		{
			WarpedInput=warp->piecewiseAffineWarping_GPU(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->weightTabel,indexTabel);
		}
		else
		{
			WarpedInput=warp->piecewiseAffineWarping(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->affineTable);
		}
		

		/*cvNamedWindow("1");
		cvShowImage("1",WarpedInput);
		cvWaitKey(0);*/
	
		//WarpedInput=warp->piecewiseAffineWarping(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->weightTabel,indexTabel);
		//WarpedInput=warp->piecewiseAffineWarping(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->affineTable);
		currentTexture->getROI(WarpedInput,mask);


		
		if (outputtime)
		{
			HessianStop=GetTickCount();

		//	cout<<"warping  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}
	


		//compute error
		//first, normalize it
		
		

		//meanTexture->
		//meanTexture->devide(texture_scale);
		//for (int i=0;i<meanShape->pix_num;i++)
		//{
		//	CV_MAT_ELEM(*meanTexture->imgData,double,0,i)+=100.;
		//}
		//cvNamedWindow("1");
		//cvShowImage("1",meanTexture->getImage());
		//cvWaitKey(0);
	/*	cvNamedWindow("1");
		cvShowImage("1",currentTexture->getImage());
		cvWaitKey(0);*/
	/*	for (int lk=0;lk<100;lk++)
		{
			cout<<currentTexture->m_imgData.at<double>(0,lk)<<" ";
		}
		cout<<endl;*/


		tex_scale=currentTexture->normalize();
	//	cout<<"tex_scale: "<<tex_scale<<endl;
		/*cvNamedWindow("1");
		cvShowImage("1",currentTexture->getImage());
		cvWaitKey(0);*/
		tmpScale=currentTexture->pointMul(meanTexture);
		tex_scale/=tmpScale;
		currentTexture->devide(tmpScale);


		//FullOut<<"----------------------current texture-----------------------------\n";
		//for (i=0;i<100;i++)
		//{
		//	FullOut<<currentTexture->m_imgData.at<double>(0,i)<<" ";
		//}
		//FullOut<<endl;
	//	currentTexture->normalize();
	//	currentTexture->simple_normalize();

		

		//currentTexture->devide(texture_scale);
		//cvNamedWindow("1");
		//cvShowImage("1",currentTexture->getImage());
		//cvWaitKey(0);

		//cvConvert(currentTexture->getImage(),WarpedInput_mat);
		//get current texture

	

		//cvNamedWindow("Texture Updating...");
		//*meanfordisplay=(*currentTemplate);
		//meanfordisplay->devide(texture_scale);
		//for (int i=0;i<meanfordisplay->pix_num;i++)
		//{
		//	CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
		//}
		//cvShowImage("Texture Updating...",meanfordisplay->getImage());
		//cvWaitKey(0);
		
		//cvSub(currentTemplate->imgData,currentTexture->imgData,errorImageMat);
		m_errorImageMat=currentTemplate->m_imgData-currentTexture->m_imgData;

		double ava_tex_error=norm(m_errorImageMat)/m_errorImageMat.cols/tex_scale;
		//cout<<"average texture error: "<<ava_tex_error<<endl;
		if (ava_tex_error<0.01)
		{
		}

		
	/*	FullOut<<"----------------------errorImage-----------------------------\n";
		for (i=0;i<m_errorImageMat.cols;i++)
		{
			FullOut<<m_errorImageMat.at<double>(0,i)<<" ";
		}
		FullOut<<endl;*/
	
	/*	for (i=0;i<m_errorImageMat.cols;i++)
		{
			out_E<<m_errorImageMat.at<double>(0,i)<<" ";
		}
		out_E<<endl;
		out_E.close();*/

		if(outputtime)
		{
			HessianStop=GetTickCount();

			//cout<<"Error caculation  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}
	
		//cout<<"sim\n";
		getSD_sim();
		
		//for (i=0;i<width;i++)
		//{
		//	for (j=0;j<height;j++)
		//	{
		//		if (!m_mask.at<double>(j,i))
		//		{
		//			continue;
		//		}
		//		//cout<<i<<" "<<j<<endl;
		//		for (int io=0;io<shape_dim+texture_dim+4;io++)
		//		{
		//			out_J<<SD_ic[i][j][io]<<" ";
		//		}
		//		out_J<<endl;
		//	}
		//}
		//out_J.close();

		//FullOut<<"----------------------SD Image-----------------------------\n";
		//for (i=0;i<width;i++)
		//{
		//	for (j=0;j<height;j++)
		//	{
		//		if (!m_mask.at<double>(j,i))
		//		{
		//			continue;
		//		}
		//		//cout<<i<<" "<<j<<endl;
		//		for (int io=0;io<50;io++)
		//		{
		//			FullOut<<SD_ic[i][j][io]<<" ";
		//		}
		//		
		//		break;
		//	}
		//}
		//FullOut<<endl;



		if(outputtime)
		{
			HessianStop=GetTickCount();

			//cout<<"SD time: "<<(HessianStop-HessianStart)<<endl;
		}

		for (i=0;i<dim;i++)
		{
			sumSD[i]=deltaP[i]=0;
		}
		//cout<<"compute SD...\n";

		//compute SD
	//	c_pixnum=0;
		
		errorSum=0;
		for (i=0;i<width;i++)
		{
			for (j=0;j<height;j++)
			{
				if (!m_mask.at<double>(j,i))
				{
					continue;
				}
				//errorIm=CV_MAT_ELEM(*errorImageMat,double,0,(int)CV_MAT_ELEM(*mask_withindex,double,j,i));
				errorIm=m_errorImageMat.at<double>(0,(int)m_mask_withindex.at<double>(j,i));
				errorSum+=errorIm*errorIm;
				for (k=0;k<dim;k++)
				{
					sumSD[k]+=SD_ic[i][j][k]*errorIm;
			/*		if(i>0)
					cout<<SD_ic[i][j][k]<<" ";*/
				}
			//	c_pixnum++;
		/*		if(i>0)
				cout<<endl;*/
			}
		}
		//cout<<errorSum<<endl;
	/*	cout<<"sumSD\n";
		for (int i=0;i<dim;i++)
		{
			cout<<sumSD[i]<<" ";
		}
		cout<<endl;*/
		//if (errorSum<0.03&&(lastErrorSum-errorSum)<0.004&&lastErrorSum>errorSum)
		//{
		//	//goto save;
		//}

		/////////smooth adjustment////////////
		//if (lastError<errorSum)
		//{
		//	//stepLength/=2.0;
		//	//if (stepLength<0.0001)
		//	{
		//		stepLength=0.0001;
		//	}
		//}
		//if (lastErrorSum<errorSum&&(errorSum-lastErrorSum)/lastErrorSum>0.080)
		//{
		//	if (errorSum>0.1)
		//	{
		//		stepLength=1;
		//	}
		//	else if (errorSum>0.06)
		//	{
		//		stepLength=0.6;
		//	}
		//	else
		//		stepLength=0.1;
		//}

		//////////////////smooth version 1//////////////////////
	//	if (lastError<errorSum&&errorSum<0.1)
	//	{
	//		stepLength/=2.0;
	//		//recaculatetime++;
	//		if (stepLength<0.0000001)
	//		{
	//			goto save;
	//		}
	///*		for (i=0;i<shape_dim;i++)
	//		{
	//			s_weight[i]=s_weight_last[i];
	//		}
	//		for (i=0;i<texture_dim;i++)
	//		{
	//			t_weight[i]=t_weight_last[i];
	//		}
	//		theta=theta_last;
	//		k_scale=k_scale_last;
	//		transform_x=transform_x_last;
	//		transform_y=transform_y_last;*/
	//		times--;
	//		increaseTime=0;
	//		//lastError=errorSum;
	//		//recaculate=true;
	//		cout<<"Last error: "<<lastError<<" error: "<<errorSum<<" re-calculate! Step length from "<<stepLength*2<<" to "<<stepLength<<endl;
	//		//continue;
	//	}
	//	//else if (currentFrame>=3&&currentFrame<=480&&errorSum<0.2&&lastError<errorSum&&errorSum<0.15)
	//	//{
	//	//	stepLength/=2.0;
	//	//	//recaculatetime++;
	//	//	if (stepLength<0.0000001)
	//	//	{
	//	//		goto save;
	//	//	}
	//	//	times--;
	//	//	increaseTime=0;
	//	//	cout<<"Last error: "<<lastError<<" error: "<<errorSum<<" re-calculate! Step length from "<<stepLength*2<<" to "<<stepLength<<endl;
	//	//
	//	//}
	//	else if (errorSum>0.15)
	//	{
	//	/*	if (currentFrame>=403&&currentFrame<=480)
	//		{
	//			stepLength=0.2;
	//		}
	//		else*/
	//		stepLength=1;

	//	}
	//	else
	//	{
	//		increaseTime++;
	//		if (increaseTime>3)
	//		{
	//			stepLength*=1000;/*=1;*/;
	//			if (stepLength>1)
	//			{
	//				stepLength=1;
	//			}
	//	/*		if (currentFrame>=403&&currentFrame<=480)
	//			{
	//				if (stepLength>0.2)
	//				{
	//					stepLength=0.2;
	//				}
	//			}*/
	//			increaseTime=0;
	//		}
	//	}

		if(outputtime)
		{
			HessianStart=GetTickCount();
		}
		//HessianStart=GetTickCount();
		getHessian();
		//HessianStop=GetTickCount();

		//cout<<"Hessian  time: "<<(HessianStop-HessianStart)<<endl;
		if(outputtime)
		{
			HessianStop=GetTickCount();

			//cout<<"Hessian  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}
	
	/*	for (i=0;i<full_Hessian.cols;i++)
		{	
			{
				cout<<full_Hessian.at<double>(0,i)<<" ";
			}
		}
		cout<<endl;*/
		/*cout<<"----------------------Hessian-----------------------------\n";
		for (i=0;i<full_Hessian.cols;i++)
		{	
			{
				cout<<full_Hessian.at<double>(0,i)<<" ";
			}
		}
		cout<<endl;

		
		for (i=0;i<full_Hessian.rows;i++)
		{	
			for (j=0;j<full_Hessian.cols;j++)
			{
				out_HHH<<full_Hessian.at<double>(i,j)<<" ";
			}
			out_HHH<<endl;
		}
		out_HHH<<endl;
		out_HHH.close();

		FullOut<<"----------------------inv Hessian-----------------------------\n";
		for (i=0;i<m_inv_hessian.cols;i++)
		{	
			{
				FullOut<<m_inv_hessian.at<double>(0,i)<<" ";
			}
		}
		FullOut<<endl;*/

		////display
		//meanTexture->devide(texture_scale);
		//currentTexture->devide(texture_scale);
		//cvNamedWindow("1");
		//cvNamedWindow("2");
		//cvShowImage("1",currentTexture->getImage());
		//cvShowImage("2",meanTexture->getImage());
		//	cvWaitKey(0);






		/*else
		{
			stepLength*=2.0;
			if (stepLength>1)
			{
				stepLength=1;
			}
		}
		if (errorSum<0.05)
		{
			stepLength=0.1;
		}
		else
		{
			stepLength=1.0;
		}*/

		// int cind;
		//int currentPtsInd;
		//double currentSD[10][2];
		//errorSum_detection=0;
		//for (i=0;i<trees->numOfLabels;i++)
		//{
		//	currentPtsInd=trees->interestPtsInd[i];
		//	// cout<<currentPtsInd<<endl;

		//	currentSD[i][0]=currentSD[i][1]=0;
		//	for (j=0;j<trees->candidates[i]->size();j++)
		//	{
		//		currentSD[i][0]+=trees->candidates[i]->at(j).z*(currentShape->pts[currentPtsInd][0]-trees->candidates[i]->at(j).x);
		//		currentSD[i][1]+=trees->candidates[i]->at(j).z*(currentShape->pts[currentPtsInd][1]-trees->candidates[i]->at(j).y);
		//		errorSum_detection+=trees->candidates[i]->at(j).z*sqrt(((currentShape->pts[currentPtsInd][0]-trees->candidates[i]->at(j).x))*(currentShape->pts[currentPtsInd][0]-trees->candidates[i]->at(j).x)+
		//			((currentShape->pts[currentPtsInd][1]-trees->candidates[i]->at(j).y))*((currentShape->pts[currentPtsInd][1]-trees->candidates[i]->at(j).y)) );
		//		// cout<<i<<" "<<currentSD[i][0]<<" "<<currentSD[i][1]<<endl;
		//	}
		//}
		//cout<<"error dection: "<<errorSum_detection/(float)trees->numOfLabels<<endl;

		//if (times>1&&errorSum>10)
		//{
		//	//smoothWeight=0;
		//	//priorWeight=0;
		//	return false;
		//}

		/*if (smoothWeight>0&&times>15)
		{
			smoothWeight=0.0001;
		}*/


		/////////////////////////weight adjustment, seems no need this/////////////////////////////////////////////
		//if (((errorSum<0.15&&abs(lastError-errorSum)<0.01)||((errorSum<0.3)&&abs(lastError-errorSum)<0.001)||
		//	(lastError<errorSum&&errorSum<0.3))&&(times>0&&type==0)||(type==1&&times>10))
		//{
		//	//cout<<"****************set smoothweight to 0***********************\n";
		//	if (smoothWeight>0)
		//	{
		//		//smoothWeight=0.00005;
		//		smoothWeight=0.00005;
		//	}
		//	
		//	if (priorWeight>0&&(errorSum<0.15||times>10))
		//	{
		//	//	priorWeight=0.00001;
		//	}

		//	if (localPCAWeight>0&&(errorSum<0.15||times>10))
		//	{
		//		localPCAWeight=0.001;
		//	}
		//	//smoothWeight=priorWeight=localPCAWeight=0;
		//}
		////////////////////////////////////////////////////////////////////////

		/*if (times>10)
		{
			priorWeight
		}*/
		/*else if(useDetectionResults)
		{
			smoothWeight+=0.2;
			if (smoothWeight>1)
			{
				smoothWeight=1;
			}
		}*/
		//detection
		if (smoothWeight>0)
		{

			/* for (i=0;i<trees->numOfLabels;i++)
			 {
				 cout<<currentSD[i][0]<<" "<<currentSD[i][1]<<endl;
			 }*/
			
			double tmpError;
			int currentPtsInd;
		/*	for (i=0;i<dim;i++)
			{
				sumDectionSD[i]=0;
			}*/
			
			//output the positions
			for (j=0;j<trees->numOfLabels;j++)
			{
				currentPtsInd=trees->interestPtsInd[j];
	
				/*cout<<currentShape->pts[currentPtsInd][0]<<" "<<currentShape->pts[currentPtsInd][1]<<
					" "<<prob_mu[j][0]<<" "<<prob_mu[j][1]<<endl;*/
			}

		
			double tmp[2];
			for (i=0;i<dim;i++)
			{
				sumDectionSD[i]=0;
				//tmpError=trees->probabilityMap[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);
				//for (j=0;j<trees->numOfLabels;j++)
				//{
				//	currentPtsInd=trees->interestPtsInd[j];
				//	tmp[0]=SD_detection[j][0][i]*prob_conv[j].at<double>(0,0)+SD_detection[j][1][i]*prob_conv[j].at<double>(1,0);
				//	tmp[1]=SD_detection[j][0][i]*prob_conv[j].at<double>(0,1)+SD_detection[j][1][i]*prob_conv[j].at<double>(1,1);
				//	//cout<<tmp[0]<<" "<<tmp[1]<<endl;
				//	sumDectionSD[i]+=tmp[0]*(currentShape->pts[currentPtsInd][0]-prob_mu[j][0])+tmp[1]*(currentShape->pts[currentPtsInd][1]-prob_mu[j][1]);
				//}
				for (j=0;j<trees->numOfLabels;j++)
				{
					currentPtsInd=trees->interestPtsInd[j];
					for (int k=0;k<candidatePoints[j].size();k++)
					{
						tmp[0]=SD_detection[j][0][i]*prob_conv_candidates[j][k].at<double>(0,0)+SD_detection[j][1][i]*prob_conv_candidates[j][k].at<double>(1,0);
						tmp[1]=SD_detection[j][0][i]*prob_conv_candidates[j][k].at<double>(0,1)+SD_detection[j][1][i]*prob_conv_candidates[j][k].at<double>(1,1);
						//cout<<tmp[0]<<" "<<tmp[1]<<endl;
						sumDectionSD[i]+=tmp[0]*(currentShape->pts[currentPtsInd][0]-prob_mu_candidates[j][k][0])+
							tmp[1]*(currentShape->pts[currentPtsInd][1]-prob_mu_candidates[j][k][1]);
					}
					
					
				}
				//cout<<sumDectionSD[i]<<" ";
			}

			//for (i=0;i<dim;i++)
			//{
			//	sumDectionSD[i]=0;
			//	//tmpError=trees->probabilityMap[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);
			//	for (j=0;j<trees->numOfLabels;j++)
			//	{
			//		currentPtsInd=trees->interestPtsInd[j];
			//		tmp[0]=prob_conv[j].at<double>(0,0)*(currentShape->pts[currentPtsInd][0]-prob_mu[j][0])+
			//			prob_conv[j].at<double>(0,1)*(currentShape->pts[currentPtsInd][1]-prob_mu[j][1]);
			//		tmp[1]=prob_conv[j].at<double>(1,0)*(currentShape->pts[currentPtsInd][0]-prob_mu[j][0])+
			//			prob_conv[j].at<double>(1,1)*(currentShape->pts[currentPtsInd][1]-prob_mu[j][1]);
			//		//cout<<tmp[0]<<" "<<tmp[1]<<endl;
			//		sumDectionSD[i]+=SD_detection[j][0][i]*tmp[0]+SD_detection[j][1][i]*tmp[1];
			//	}
			//	//cout<<sumDectionSD[i]<<" ";
			//}

	//		cout<<"the RT jacobians\n";
	//		for (j=0;j<trees->numOfLabels;j++)
	//		{
	//			for (int i=0;i<dim;i++)
	//			{
	//				cout<<SD_detection[j][0][i]<<" "<<SD_detection[j][1][i]<<" ";
	//			}
	//			cout<<endl;	
	//		}

	//		cout<<"current E\n";
	//		for (j=0;j<trees->numOfLabels;j++)
	//		{
	//			currentPtsInd=trees->interestPtsInd[j];
	//			cout<<currentShape->pts[currentPtsInd][0]-prob_mu[j][0]<<" "<<currentShape->pts[currentPtsInd][1]-prob_mu[j][1]<<endl;
	//		}

		/*	cout<<"current conv\n";
			for (j=0;j<trees->numOfLabels;j++)
			{
				currentPtsInd=trees->interestPtsInd[j];
				cout<<prob_conv[j].at<double>(0,0)<<" "<<prob_conv[j].at<double>(0,1)<<" "<<
					prob_conv[j].at<double>(1,0)<<" "<<prob_conv[j].at<double>(1,1)<<endl;
			}*/

	//		
			/*	cout<<"conv*E\n";
				for (j=0;j<trees->numOfLabels;j++)
				{
					currentPtsInd=trees->interestPtsInd[j];
					cout<<prob_conv[j].at<double>(0,0)*(currentShape->pts[currentPtsInd][0]-prob_mu[j][0])+
						prob_conv[j].at<double>(0,1)*(currentShape->pts[currentPtsInd][1]-prob_mu[j][1])<<" "<<
						prob_conv[j].at<double>(1,0)*(currentShape->pts[currentPtsInd][0]-prob_mu[j][0])+
						prob_conv[j].at<double>(1,1)*(currentShape->pts[currentPtsInd][1]-prob_mu[j][1])<<endl;
				}*/
	///*
	//		cout<<"JE for AAM:\n";
	//		for (int i=0;i<dim;i++)
	//		{
	//			cout<<sumSD[i]*AAM_weight<<" ";
	//		}
	//		cout<<endl;*/

	//		cout<<"JE for detection:\n";
	//		for (int i=0;i<dim;i++)
	//		{
	//			cout<<sumDectionSD[i]*smoothWeight<<" ";
	//		}
	//		cout<<endl;
	///*		for (i=0;i<trees->numOfLabels;i++)
	//		{
	//			cout<<prob_mu[i][0]<<" "<<
	//				prob_mu[i][1]<<" ";
	//		}


	//		cout<<endl;
	//		for (i=0;i<trees->numOfLabels;i++)
	//		{
	//			cout<<currentShape->pts[trees->interestPtsInd[i]][0]-prob_mu[i][0]<<" "<<
	//				currentShape->pts[trees->interestPtsInd[i]][1]-prob_mu[i][1]<<" ";
	//		}
	//		cout<<endl;*/

	//		for (i=0;i<trees->numOfLabels;i++)
	//		{
	//			cout<<prob_conv[i].at<double>(1,0)<<" "<<prob_conv[i].at<double>(1,1)<<" ";
	//		}
	//		cout<<endl;

		}
	


		///////////////////need to commant out this////////////////////////
		stepLength=1.0;


	/*	if (priorWeight>0)
		{
			for (i=0;i<dim;i++)
			{
				cout<<s_weight[i]-priorMean[i]<<" ";
			}
			cout<<endl;
		}*/

		double tmpJE;
		for (i=0;i<dim;i++)
		{
			for (j=0;j<dim;j++)
			{
				tmpJE=AAM_weight*sumSD[j];
				if (smoothWeight>0)
				{
					tmpJE+=smoothWeight*sumDectionSD[j];
				}
				if (priorWeight>0)
				{
					tmpJE+=priorWeight*SD_prior[j];
				}

				if (localPCAWeight>0)
				{
					tmpJE+=localPCAWeight*SD_local[j];
				}
				deltaP[i]+=-stepLength*m_inv_hessian.at<double>(i,j)*tmpJE;
				
			}
			
		}

	/*	for (int i=0;i<shape_dim;i++)
		{
			cout<<deltaP[i]<<" ";
		}
		cout<<endl;*/


		if (outputtime)
		{
			HessianStop=GetTickCount();

			//cout<<"caculate deltaP time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}

		//cout<<"sweight before\n";
		//for (i=0;i<shape_dim;i++)
		//{
		//	cout<<s_weight[i]<<" ";
		//}
		////cout<<"tweight before\n";
		//for (i=0;i<texture_dim;i++)
		//{
		//	cout<<t_weight[i]<<" ";
		//}
		//cout<<theta<<" "<<k_scale<<" "<<transform_x<<" "<<transform_y<<endl;
		//cout<<endl;

		//cout<<"delta P:\n";
		//for (i=0;i<shape_dim+texture_dim+4;i++)
		//{
		//	cout<<deltaP[i]<<" ";
		//}
		//cout<<endl;


		//update the shape
		for (i=0;i<shape_dim;i++)
		{
			s_weight[i]+=deltaP[i];
		}

		//update t_weight
		{
			for (i=0;i<texture_dim;i++)
			{
				t_weight[i]+=deltaP[shape_dim+i];
			}
		}
		theta+=deltaP[shape_dim+texture_dim];
		k_scale+=deltaP[shape_dim+texture_dim+1];
		transform_x+=deltaP[shape_dim+texture_dim+2];
		transform_y+=deltaP[shape_dim+texture_dim+3];
		times++;
	
		//cout<<"theta scale tx ty: "<<theta<<" "<<k_scale<<" "<<transform_x<<" "<<transform_y<<endl;
	//	cout<<endl;

		//need to stop?
		ssum=0;
		for (i=0;i<shape_dim;i++)
		{
			ssum+=deltaP[i]*deltaP[i];
		}
		ssum=sqrt(ssum);

		//if (outputtime)
		//{
		//	cout<<times<<" Error:"<<errorSum<<"  steplength: "<<stepLength<<endl<<" Displayed sum:"<<ssum<<
		//		"last-current: "<<abs(errorSum-lastError)<<"  Ptsdifference:"<<Pts_Difference<<endl;
		//}
		/*if (times>30)
		{
			break;
		}*/
	//	if (errorSum<0.07||ssum<0.1||times>20||(currentFrame>startNum&&abs(errorSum-lastError)<0.00001))//||(errorSum<0.4&&ssum<3)||times>100)
		//	if(0)
	//if (errorSum<0.13||times>400||(currentFrame>startNum&&times>MaxIterNum)||(errorSum<0.23&&abs(errorSum-lastError)<0.001))
	//if(0)

		//////////////////////////check converge////////////////////////////////////////////
		//calculateTermValue(errorSum,totalweight);
		//if((abs(totalweight-lastTotalWeight)<0.0005&&!isAAM)||(isAAM&&(abs(errorSum-lastError)<0.00001||aamTimes>4))||(times>MaxIterNum&&!isAAM))
		/////////////////////////////////////////////////////////////////////////////////////////

	//if (times>MaxIterNum||((errorSum<0.1&&abs(errorSum-lastError)<0.00001)&&smoothWeight<0.05))	
	if ((!isAAM&&(times>MaxIterNum||(smoothWeight<0.05&&errorSum<0.1&&(abs(errorSum-lastError)<0.002)||Pts_Difference<0.05)))||
		(isAAM&&(abs(errorSum-lastError)<0.00001)||Pts_Difference<0.05))	
	{
		cout<<"smoothweight: "<<smoothWeight<<" "<<"ending times: "<<times<<"   Pts_Difference: "<<Pts_Difference<<endl;

		if (!isAAM||(isAAM&&aamTimes<5))
		{
			if(!isAAM)
			{
				cvCopyImage(Input,imgcopy);
				cvNamedWindow("AAM");
				for (int i=0;i<ptsNum;i++)
				{
					cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),1,CV_RGB(0,0,255));
				}
				cvShowImage("AAM",imgcopy);
				cvWaitKey();
			}
			

			smoothWeight=0.00005;
			priorWeight=0.0001;
			isAAM=true;
			//cout<<"AAM is on\n";
			//goonTimes=0;
			goto nosave;
		}
	//	else
	//	{



save:	
			//if (errorSum>0.15&recaculatetime<10)
			//{
			//	times=1;
			//	transform_y-=pow(1.0f,recaculatetime)*2.5*recaculatetime/k_scale;
			//	//k_scale*=1.1;
			//	recaculatetime++;
			//	MaxIterNum=60;
			//	goto nosave;
			//}
			//MaxIterNum=35;
			/*		else if (errorSum>0.08&&recaculatetime<20)
			{
			times=1;
			transform_y-=5.5;
			recaculatetime++;
			goto nosave;
			}*/


			//display the result
		{
			cvCopyImage(Input,imgcopy);
			cvNamedWindow("1");
			for (int i=0;i<ptsNum;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),1,CV_RGB(0,0,255));
			}
			cvShowImage("1",imgcopy);
			cvWaitKey(1);
		}
		//break;
		//save images
		string fullname;
		string tmpName;
		char name[500];
		//sprintf(name, "G:\\face database\\kinect data\\images for combination debug\\AAM with detection term\\%d.jpg", 49+currentFrame);
		if(type==0)
			sprintf(name, "AAM+detection/%s_%d.jpg", prefix,11361+currentFrame);
		else
			sprintf(name, "AAMPrior/%s_%d.jpg", prefix,11361+currentFrame);
		//tmpName=name;
		//fullname=dataDir+tmpName;
		cvSaveImage(name,imgcopy);

		//cout<<fullname<<endl;
		/*	sprintf(name, "%d_ori.jpg", currentFrame);
		fullname=dataDir+name;
		cvSaveImage(fullname.c_str(),Input);*/
		//Output the pts
		char pname[500];
		//sprintf(pname, "G:\\face database\\kinect data\\images for combination debug\\AAM with detection term\\%d.txt", 49+currentFrame);
		if (type==0)
			sprintf(pname, "AAM+detection/%s_%d.txt", prefix,11361+currentFrame);
		else
			sprintf(pname, "AAMPrior/%s_%d.txt", prefix,11361+currentFrame);

		//tmpName=pname;
		//fullname=dataDir+tmpName;
		ofstream out(pname,ios::out);
		out<<Input->width/resizeSize<<" "<<Input->height/resizeSize<<endl;
		out<<currentShape->ptsNum<<" "<<currentShape->ptsNum<<endl;
		for (int i=0;i<ptsNum;i++)
		{
			out<<currentShape->pts[i][0]/(double)Input->width<<" "<<
				currentShape->pts[i][1]/(double)Input->height<<endl;
		}
		out.close();

		sprintf(pname, "AAM+detection/%s_%d_para.txt",prefix, 11361+currentFrame);
		//tmpName=pname;
		//fullname=dataDir+tmpName;
		ofstream out1(pname,ios::out);
		for (int i=0;i<shape_dim;i++)
		{
			out1<<s_weight[i]<<" ";
		}
		for (int i=0;i<texture_dim;i++)
		{
			out1<<t_weight[i]<<" ";
		}
		out1<<theta<<" "<<k_scale<<" "<<transform_x<<" "<<transform_y<<endl;
		out1.close();

		sprintf(pname, "AAM+detection/%s_%d_term values.txt",prefix, 11361+currentFrame);
		ofstream out11(pname,ios::out);
		out11<<errorSum<<endl;
		out11.close();
		break;
		/*		sprintf(pname, "%d_iteration.txt", 10000+currentFrame);
		tmpName=pname;
		fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
		ofstream out_error(fullname.c_str(),ios::out);
		out_error<<times<<endl;
		out_error.close();

		sprintf(pname, "%d_parameters.txt", 10000+currentFrame);
		tmpName=pname;
		fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
		ofstream out1(fullname.c_str(),ios::out);
		for (int i=0;i<shape_dim;i++)
		{
		out1<<s_weight[i]<<" ";
		}
		for (int i=0;i<texture_dim;i++)
		{
		out1<<t_weight[i]<<" ";
		}
		out1<<theta<<" "<<k_scale<<" "<<transform_x<<" "<<transform_y<<endl;
		out1.close();*/

		for (int i=0;i<meanShape->ptsNum;i++)
		{
			ptsLast[i][0]=currentShape->pts[i][0];
			ptsLast[i][1]=currentShape->pts[i][1];
		}
		//	if (smoothWeight>0)
		//	{
		//	
		//		lastTemplate=currentTexture->m_imgData.clone();
		///*		for (int i=0;i<shape_dim;i++)
		//		{
		//			parametersLast[i]=s_weight[i];
		//		}
		//		for (int i=0;i<texture_dim;i++)
		//		{
		//			parametersLast[shape_dim+i]=t_weight[i];
		//		}
		//		parametersLast[shape_dim+texture_dim]=theta;
		//		parametersLast[shape_dim+texture_dim+1]=k_scale;
		//		parametersLast[shape_dim+texture_dim+2]=transform_x;
		//		parametersLast[shape_dim+texture_dim+3]=transform_y;*/
		//	}
		//lastErrorSum=errorSum;
		lastError=9999999;
		errorSum=10000000000;
		break;
		//}
		}
		else
		{
nosave:		//if(!recaculate)
			lastTotalWeight=totalweight;
			if (isAAM)
			{
				aamTimes++;
			}
				lastError=errorSum;
		//	else
			//	recaculate=false;
			for (int i=0;i<shape_dim;i++)
			{
				s_weight_last[i]=s_weight[i];
			}
			for (int i=0;i<texture_dim;i++)
			{
				t_weight_last[i]=t_weight[i];
			}
			theta_last=theta;
			k_scale_last=k_scale;
			transform_x_last=transform_x;
			transform_y_last=transform_y;
			//if (smoothWeight>0)
			{
				Pts_Difference=0;
				for (int i=0;i<meanShape->ptsNum;i++)
				{
					Pts_Difference+=sqrt((currentShape->pts[i][0]-ptsLast[i][0])*(currentShape->pts[i][0]-ptsLast[i][0])+
						(currentShape->pts[i][1]-ptsLast[i][1])*(currentShape->pts[i][1]-ptsLast[i][1]));
				}
				Pts_Difference/=(double)meanShape->ptsNum;
			}
			lastErrorSum=errorSum;

			for (int i=0;i<meanShape->ptsNum;i++)
			{
				ptsLast[i][0]=currentShape->pts[i][0];
				ptsLast[i][1]=currentShape->pts[i][1];
			}
			//cout<<times++<<" Error:"<<errorSum<<endl<<" Displayed sum:"<<ssum<<endl;
		}

	


	



		if (outputtime)
		{

			HessianStop=GetTickCount();

			//cout<<"total  time: "<<(double)(HessianStop-d_totalstart)<<endl;
		}

		if (smoothWeight>0)
		{
			smoothWeight-=0.1*smoothWeight_backup;
			if (smoothWeight<0)
			{
				smoothWeight=0.00005;
			}
		}
		/*if (priorWeight>0)
		{
			priorWeight-=priorWeight_backup*0.1;
			if (priorWeight<0)
			{
				priorWeight=0.0000001;
			}
		}
		if (localPCAWeight>0)
		{
			localPCAWeight-=localPCAWeight_backup*0.1;

			if (localPCAWeight<0)
			{
				localPCAWeight=0.0000001;
			}
		}*/
		//display
	//		if(times%20==0)
		if(showSingleStep)
		{
		/*	cout<<"shape weights:\n";
			for (int i=0;i<shape_dim;i++)
			{
				cout<<s_weight[i]<<" ";
			}
			cout<<endl;

			ofstream pts_out("F:\\Projects\\Facial feature points detection\\Matlab code\\pts.txt",ios::out);
			for (int i=0;i<ptsNum;i++)
			{
				pts_out<<meanShape->pts[i][0]<<" "<<meanShape->pts[i][1]<<endl;
			}
			pts_out.close();*/

			cout<<"-------------------pts difference: "<<Pts_Difference<<" times"<<times<<endl;
			cout<<"texture error"<<lastErrorSum<<" "<<errorSum<<endl;
			//calculate the four terms
			double AAMTermValue=0;
			double detectionTermValue=0;
			double localPriorTermValue=0;
			double localPCATermValue=0;

			AAMTermValue=AAM_weight*errorSum;

			double tmp[2];
			double tmpSUM;
			if (smoothWeight>0)
			{
				detectionTermValue=0;
				//for (int i=0;i<trees->numOfLabels;i++)
				//{
				//	tmp[0]=currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0];
				//	tmp[1]=currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1];
				//	tmpSUM=prob_conv[i].at<double>(0,0)*tmp[0]*tmp[0]+(prob_conv[i].at<double>(1,0)+prob_conv[i].at<double>(0,1))*tmp[0]*tmp[1]+
				//		prob_conv[i].at<double>(1,1)*tmp[1]*tmp[1];
				//	detectionTermValue+=probForEachFeature[i]*tmpSUM;
				//	/*detectionTermValue+=probForEachFeature[i]*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])+
				//		(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1])*(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1]);*/
				//}
				for (int i=0;i<trees->numOfLabels;i++)
				{
					for (int j=0;j<candidatePoints[i].size();j++)
					{
						tmp[0]=currentShape->pts[trees->interestPtsInd[i]][0]-candidatePoints[i][j].x;
						tmp[1]=currentShape->pts[trees->interestPtsInd[i]][1]-candidatePoints[i][j].y;
					
					
						tmpSUM=prob_conv_candidates[i][j].at<double>(0,0)*tmp[0]*tmp[0]+(prob_conv_candidates[i][j].at<double>(1,0)+prob_conv_candidates[i][j].at<double>(0,1))*tmp[0]*tmp[1]+
							prob_conv_candidates[i][j].at<double>(1,1)*tmp[1]*tmp[1];
						detectionTermValue+=probForEachFeatureCandidates[i][j]*tmpSUM;
					}
					
					/*detectionTermValue+=probForEachFeature[i]*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])+
						(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1])*(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1]);*/
				}
				detectionTermValue*=smoothWeight;
			}
			

			if (priorWeight>0)
			{
				localPriorTermValue=0;
				Mat tmpSWeight=Mat::zeros(shape_dim,1,CV_64FC1);
				for (int i=0;i<shape_dim;i++)
				{
					tmpSWeight.at<double>(i,0)=s_weight[i]-priorMean[i];
				}
				Mat sWeight_tran;
				transpose(tmpSWeight,sWeight_tran);
				Mat CValue=sWeight_tran*priorSigma(Range(0,shape_dim),Range(0,shape_dim))*tmpSWeight;
				localPriorTermValue=CValue.at<double>(0,0);
				localPriorTermValue*=priorWeight;
			}
		
			if (localPCAWeight>0)
			{
				Mat tmpLocalPCAWeight=Mat::zeros(local_shape_dim,1,CV_64FC1);
				for (int i=0;i<local_shape_dim;i++)
				{
					tmpLocalPCAWeight.at<double>(i,0)=m_local_s_mean.at<double>(0,i);
				}
				Mat localWeightTran;
				transpose(tmpLocalPCAWeight,localWeightTran);
				Mat c_localValute=localWeightTran*m_localHessian(Range(0,local_shape_dim),Range(0,local_shape_dim))*tmpLocalPCAWeight;
				localPCATermValue=c_localValute.at<double>(0,0);
				localPCATermValue*=localPCAWeight;
			}
			


			cout<<"term values:\n";
			cout<<AAM_weight<<" "<<smoothWeight<<" "<<priorWeight<<" "<<localPCAWeight<<endl;
			cout<<AAMTermValue<<" "<<detectionTermValue<<" "<<localPriorTermValue<<" "<<localPCATermValue<<" "<<AAMTermValue+detectionTermValue+localPriorTermValue+localPCATermValue<<endl;

			cvCopyImage(Input,imgcopy);
			cvNamedWindow("1");
			for (int i=0;i<ptsNum;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
			}

			for (int i=0;i<trees->numOfLabels;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[trees->interestPtsInd[i]][0],currentShape->pts[trees->interestPtsInd[i]][1]),1,CV_RGB(0,0,255));
			}

		/*	for (int i=0;i<trees->numOfLabels;i++)
			{
				for (int j=0;j<trees->candidates[i]->size();j++)
				{
					cvCircle(imgcopy,cvPoint(trees->candidates[i]->at(j).x,trees->candidates[i]->at(j).y),3,CV_RGB(0,0,255));
				}
			}*/
			//cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
//			cvCircle(imgcopy,cvPoint(p[0],p[1]),3,CV_RGB(0,0,255));
			cvShowImage("1",imgcopy);
			cvNamedWindow("2");
			*meanfordisplay=(*currentTexture);
			meanfordisplay->devide(texture_scale);
			for (int i=0;i<meanfordisplay->pix_num;i++)
			{
				CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
			}
			cvShowImage("2",meanfordisplay->getImage());
			//imwrite("F:\\Projects\\Facial feature points detection\\Matlab code\\input_warped.jpg",cvarrToMat(meanfordisplay->getImage()));
			cvNamedWindow("3");
			*meanfordisplay=(*currentTemplate);
			meanfordisplay->devide(texture_scale);
			for (int i=0;i<meanfordisplay->pix_num;i++)
			{
				CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
			}
			IplImage *ccc=meanfordisplay->getImage();
		/*	for (int i=0;i<ptsNum;i++)
			{
				cvCircle(ccc,cvPoint(meanShape->pts[i][0],meanShape->pts[i][1]),3,CV_RGB(0,0,255));
			}*/
			cvShowImage("3",ccc);
			//imwrite("F:\\Projects\\Facial feature points detection\\Matlab code\\synthesized.jpg",cvarrToMat(ccc));
			cvWaitKey();
		}

		//check if we need to update the texture
		//then we are done
	}

	delete currentTexture;
	return true;
}


void AAM_RealGlobal_GPU::iterate(IplImage *Input)
{
	//IplImage *warpedInput;//=cvCreateImage(cvGetSize(Template),Template->depth,Template->nChannels);

	//denormalize
	

	int dim=shape_dim+texture_dim+4;
	
	double threshold=0.00001;
	int MaxIterNum=20;

	if (smoothWeight_backup>0)
	{
		MaxIterNum=60;
	}
//	bool showSingleStep=true;

	
	
	//initialize
	//if (1)
	if(currentShape==NULL)
	{
		currentShape=new Shape(meanShape);
		currentLocalShape=new Shape(meanShape);
	}

	if(currentFrame>=startNum)
	{
		//currentShape->scale(0.98,1);
		//only do once to intialize face position

		//currentShape->getVertex(meanShape->ptsForMatlab,meanShape->ptsNum,meanShape->width,meanShape->height);

		for (int i=0;i<texture_dim;i++)
		{
			t_weight[i]=0;
		}

		if(!treeInitialization)
		{
			for (int i=0;i<shape_dim;i++)
			{
				s_weight[i]=0;
			}

			//double *p=meanShape->getcenter();

			if (currentFrame==startNum)
			{
				transform_x=faceCenter.x+initialTx;
				transform_y=faceCenter.y+initialTy;
			}
		
			theta=0;

			k_scale=initialScale;
		}
		/*s_weight[0]=0.5;
		s_weight[1]=0.2;*/
	}

	//see if there is any tracking result
	if (0)
	{
		char ppname[500];
		sprintf(ppname, "%d_optimized parameters.txt", currentFrame+1);
		string fullName=ppname;
		ifstream in(fullName.c_str(),ios::in);
		if (in)
		{
			while(in)
			{
				for (int i=0;i<shape_dim;i++)
				{
					in>>s_weight[i];
				}
				for (int i=0;i<texture_dim;i++)
				{
					in>>t_weight[i];
				}
				in>>theta>>k_scale>>transform_x>>transform_y;
			}
			in.close();

			//stepLength=0.5;
			//MaxIterNum=15;
		}
		
		//sprintf(ppname, "%d_optimized parameters.txt", 10000+currentFrame);
		//string tmpName=ppname;
		//string fullName=dataDir+tmpName.substr(1,tmpName.length()-1);;
		//ifstream in(fullName.c_str(),ios::in);
		//if (in)
		//{
		//	stepLength=0.5;
		//	//MaxIterNum=15;
		//}
		//while(in)
		//{
		//	for (int i=0;i<shape_dim;i++)
		//	{
		//		in>>s_weight[i];
		//	}
		//	for (int i=0;i<texture_dim;i++)
		//	{
		//		in>>t_weight[i];
		//	}
		//	in>>theta>>k_scale>>transform_x>>transform_y;
		//}
		//in.close();

	}
	

	currentTexture=new Texture();

	int ptsNum=meanShape->ptsNum;
	double *deltaP=new double[dim];
	double *sumSD=new double[dim];
	double *sumSmoothSD=new double[dim];
	double *sumDectionSD=new double[dim];
	double *deltaS_T=new double[ptsNum*2];

	//global adjustment
	double *deltaGlbal;
	if (isGlobaltransform)
	{
		deltaGlbal=new double[ptsNum*2];
	}
	double ssum=1;
	
	int times=0;//total running time

	int c_pixnum;//current pixel ind
	IplImage *imgcopy=cvCreateImage(cvGetSize(Input),Input->depth,Input->nChannels);



	double adjustGamma;

	//DWORD dwStart, dwStop;  
	//DWORD d_totalstart;

	Texture *meanfordisplay=new Texture;
	double lastError=999999;
	if (lastErrorSum==-1)
	{
		lastErrorSum=999999;
	}
	
	//*meanfordisplay=(*meanTexture)	;

	DWORD HessianStart, HessianStop,d_totalstart;  
	//LONG   dwStart,dwStop,d_totalstart; 
	//LONG   persecond; 
	//if (outputtime)
	//{
	//		QueryPerformanceFrequency((LARGE_INTEGER   *)&persecond);//询问系统一秒钟的频率 
	//}

	double errorIm;
	double errorSum;
	double errorSum_detection;
	int tind;
	double alpha,beta,gamma;
	int triangleInd[3];
	double x1,x2,x3,y1,y2,y3;
	double cx1,cx2,cx3,cy1,cy2,cy3;
	double x0,y0;
	double fenmu;
	double tempPts[2];
	double tmpCurPts[2];
	int i,j,k;

	double tmpScale;
	int recaculatetime=0;
	//stepLength=1;
	//recaculate=false;
	increaseTime=0;
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t1); 
	////下面是你要计算运行时间的程序代码 
	//QueryPerformanceCounter((LARGE_INTEGER   *)&t2); 
	//double   time=(t2-t1)/persecond; 

	Pts_Difference=1000000;


	//ofstream FullOut("FullOutput.txt",ios::out);
	//ofstream out_HHH("GPU_Hessian.txt",ios::out);
	//ofstream out_J("Jacobians_CPU.txt",ios::out);
	//ofstream out_E("E_C.txt",ios::out);
	ofstream outPutPts;
	char currentSaveName[500];
	char tmpSaveName[50];

	while(1)
	{
		if (outputtime)
		{
			HessianStart=GetTickCount();
			d_totalstart=HessianStart;
		}
		

		////test1: no update for texture---done,the code works
		////so the problem lies on texture updating
		//for (int i=0;i<texture_dim;i++)
		//{
		//	t_weight[i]=0;
		//}

		//get current shape
		//finally, we update the currentshape

		
		for (i=0;i<ptsNum;i++)
		{
			currentLocalShape->ptsForMatlab[i]=meanShape->ptsForMatlab[i]-meanShapeCenter[0];
			currentLocalShape->ptsForMatlab[ptsNum+i]=meanShape->ptsForMatlab[ptsNum+i]-meanShapeCenter[1];
			//first, pca weights
			for (j=0;j<shape_dim;j++)
			{
				//currentShape->ptsForMatlab[i]+=s_weight[j]*CV_MAT_ELEM(*s_vec,double,j,i);
				currentLocalShape->ptsForMatlab[i]+=s_weight[j]*m_s_vec.at<double>(j,i);
				currentLocalShape->ptsForMatlab[ptsNum+i]+=s_weight[j]*m_s_vec.at<double>(j,ptsNum+i);
			}
			//then, global transform
			currentShape->ptsForMatlab[i]=k_scale*(cos(theta)*currentLocalShape->ptsForMatlab[i]-sin(theta)*currentLocalShape->ptsForMatlab[i+ptsNum])+transform_x;
			currentShape->ptsForMatlab[ptsNum+i]=k_scale*(sin(theta)*currentLocalShape->ptsForMatlab[i]+cos(theta)*currentLocalShape->ptsForMatlab[i+ptsNum])+transform_y;

		/*	currentShape->ptsForMatlab[i]+=transform_x;
			currentShape->ptsForMatlab[i+ptsNum]+=transform_y;*/
		}
		
	/*	for (i=0;i<ptsNum;i++)
		{
			cout<<currentShape->ptsForMatlab[i]<<" "<<currentShape->ptsForMatlab[ptsNum+i]<<endl;
		}*/

		for (i=0;i<ptsNum;i++)
		{
			currentShape->pts[i][0]=currentShape->ptsForMatlab[i];
			currentShape->pts[i][1]=currentShape->ptsForMatlab[ptsNum+i];
		}
		
		strcpy(currentSaveName,currentNamewithoutHouzhui.c_str());
		sprintf(tmpSaveName, "_Iteration_%d.txt", times);
		strcat(currentSaveName,tmpSaveName);
		//cout<<currentSaveName<<endl;
		outPutPts.open(currentSaveName,ios::out);
		for (i=0;i<ptsNum;i++)
		{
			outPutPts<<currentShape->pts[i][0]<<" "<<currentShape->pts[i][1]<<endl;
		}
		outPutPts.close();

	/*	for (i=0;i<ptsNum;i++)
		{
			cout<<currentShape->pts[i][0]<<" "<<currentShape->pts[i][1]<<endl;
		}

		cvCopyImage(Input,imgcopy);
			cvNamedWindow("1");
			for (int i=0;i<ptsNum;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
					cvShowImage("1",imgcopy);
			cvWaitKey();
			}*/

		/*	for (int i=0;i<trees->numOfLabels;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[trees->interestPtsInd[i]][0],currentShape->pts[trees->interestPtsInd[i]][1]),1,CV_RGB(0,0,255));
			}*/

		/*	for (int i=0;i<trees->numOfLabels;i++)
			{
				for (int j=0;j<trees->candidates[i]->size();j++)
				{
					cvCircle(imgcopy,cvPoint(trees->candidates[i]->at(j).x,trees->candidates[i]->at(j).y),3,CV_RGB(0,0,255));
				}
			}*/
			//cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
//			cvCircle(imgcopy,cvPoint(p[0],p[1]),3,CV_RGB(0,0,255));
		
		//calculate conv and mu
		for (i=0;i<trees->numOfLabels;i++)
		{
			//cout<<"calculating conv and mu for feature "<<i<<" at "<<
				//currentShape->pts[trees->interestPtsInd[i]][0]<<" "<<currentShape->pts[trees->interestPtsInd[i]][1]<<endl;
			calculateMandC(trees->probabilityMap[i],currentShape->pts[trees->interestPtsInd[i]][0],
				currentShape->pts[trees->interestPtsInd[i]][1],16,prob_mu[i],prob_conv[i]);

		/*	cout<<currentShape->pts[trees->interestPtsInd[i]][0]<<" "<<currentShape->pts[trees->interestPtsInd[i]][1]<<
				" "<<prob_mu[i][0]<<" "<<prob_mu[i][1]<<endl;*/


			/*cout<<"out calculated conv: "<<prob_conv[i].at<double>(0,0)<<" "<<
				prob_conv[i].at<double>(0,1)<<" "<<prob_conv[i].at<double>(1,0)<<" "<<prob_conv[i].at<double>(1,1)<<" "<<endl;*/
		}
	/*	ofstream out("F:\\Projects\\Facial feature points detection\\Matlab code\\meanshape_me.txt",ios::out);
		for (i=0;i<ptsNum;i++)
		{
			out<<currentShape->pts[i][0]<<" "<<currentShape->pts[i][1]<<endl;
		}
		out.close();*/
		/*FullOut<<"----------------------current local shape-----------------------------\n";
		for (i=0;i<ptsNum;i++)
		{
			FullOut<<currentLocalShape->ptsForMatlab[i]<<" "<<currentLocalShape->ptsForMatlab[ptsNum+i]<<endl;
		}
		FullOut<<endl;
		
		FullOut<<"----------------------current shape-----------------------------\n";
		for (i=0;i<ptsNum;i++)
		{
			FullOut<<currentShape->ptsForMatlab[i]<<" "<<currentShape->ptsForMatlab[ptsNum+i]<<endl;
		}
		FullOut<<endl;*/

		//get new template
		getCurrentTexture(t_weight);

		if (outputtime)
		{
			HessianStop=GetTickCount();

			cout<<"texture synthesizing  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}

		//see current shape	
		//if(currentFrame>=startNum)
		//{
		//	cvCopyImage(Input,imgcopy);
		//	cvNamedWindow("1");
		//	for (int i=0;i<ptsNum;i++)
		//	{
		//		cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
		//	}
		////	cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
		////	cvCircle(imgcopy,cvPoint((int)p[0],(int)p[1]),3,CV_RGB(0,0,255));
		//	cvShowImage("1",imgcopy);
		//	cvWaitKey(0);
		//}

		//cvNamedWindow("3");
		//*meanfordisplay=(*currentTemplate);
		//meanfordisplay->devide(texture_scale);
		//for (int i=0;i<meanfordisplay->pix_num;i++)
		//{
		//	CV_MAT_ELEM(*(meanfordisplay->imgDa ta),double,0,i)+=100;
		//}
		//IplImage *ccc=meanfordisplay->getImage();
		//for (int i=0;i<ptsNum;i++)
		//{
		//	cvCircle(ccc,cvPoint(meanShape->pts[i][0],meanShape->pts[i][1]),3,CV_RGB(0,0,255));
		//}
		//cvShowImage("3",ccc);
		//cvWaitKey();

		if (usingCUDA)
		{
			WarpedInput=warp->piecewiseAffineWarping_GPU(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->weightTabel,indexTabel);
		}
		else
		{
			WarpedInput=warp->piecewiseAffineWarping(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->affineTable);
		}
		

		/*cvNamedWindow("1");
		cvShowImage("1",WarpedInput);
		cvWaitKey(0);*/
	
		//WarpedInput=warp->piecewiseAffineWarping(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->weightTabel,indexTabel);
		//WarpedInput=warp->piecewiseAffineWarping(Input,WarpedInput,currentShape,meanShape,triangleList,true,meanShape->affineTable);
		currentTexture->getROI(WarpedInput,mask);


		
		if (outputtime)
		{
			HessianStop=GetTickCount();

			cout<<"warping  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}
	


		//compute error
		//first, normalize it
		
		

		//meanTexture->
		//meanTexture->devide(texture_scale);
		//for (int i=0;i<meanShape->pix_num;i++)
		//{
		//	CV_MAT_ELEM(*meanTexture->imgData,double,0,i)+=100.;
		//}
		//cvNamedWindow("1");
		//cvShowImage("1",meanTexture->getImage());
		//cvWaitKey(0);
	/*	cvNamedWindow("1");
		cvShowImage("1",currentTexture->getImage());
		cvWaitKey(0);*/
	/*	for (int lk=0;lk<100;lk++)
		{
			cout<<currentTexture->m_imgData.at<double>(0,lk)<<" ";
		}
		cout<<endl;*/


		tex_scale=currentTexture->normalize();
	//	cout<<"tex_scale: "<<tex_scale<<endl;
		/*cvNamedWindow("1");
		cvShowImage("1",currentTexture->getImage());
		cvWaitKey(0);*/
		tmpScale=currentTexture->pointMul(meanTexture);
		tex_scale/=tmpScale;
		currentTexture->devide(tmpScale);


		//FullOut<<"----------------------current texture-----------------------------\n";
		//for (i=0;i<100;i++)
		//{
		//	FullOut<<currentTexture->m_imgData.at<double>(0,i)<<" ";
		//}
		//FullOut<<endl;
	//	currentTexture->normalize();
	//	currentTexture->simple_normalize();

		

		//currentTexture->devide(texture_scale);
		//cvNamedWindow("1");
		//cvShowImage("1",currentTexture->getImage());
		//cvWaitKey(0);

		//cvConvert(currentTexture->getImage(),WarpedInput_mat);
		//get current texture

	

		//cvNamedWindow("Texture Updating...");
		//*meanfordisplay=(*currentTemplate);
		//meanfordisplay->devide(texture_scale);
		//for (int i=0;i<meanfordisplay->pix_num;i++)
		//{
		//	CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
		//}
		//cvShowImage("Texture Updating...",meanfordisplay->getImage());
		//cvWaitKey(0);
		
		//cvSub(currentTemplate->imgData,currentTexture->imgData,errorImageMat);
		m_errorImageMat=currentTemplate->m_imgData-currentTexture->m_imgData;

		double ava_tex_error=norm(m_errorImageMat)/m_errorImageMat.cols/tex_scale;
		cout<<"average texture error: "<<ava_tex_error<<endl;
		if (ava_tex_error<0.01)
		{
		}

		
	/*	FullOut<<"----------------------errorImage-----------------------------\n";
		for (i=0;i<m_errorImageMat.cols;i++)
		{
			FullOut<<m_errorImageMat.at<double>(0,i)<<" ";
		}
		FullOut<<endl;*/
	
	/*	for (i=0;i<m_errorImageMat.cols;i++)
		{
			out_E<<m_errorImageMat.at<double>(0,i)<<" ";
		}
		out_E<<endl;
		out_E.close();*/

		if(outputtime)
		{
			HessianStop=GetTickCount();

			cout<<"Error caculation  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}
	
		//cout<<"sim\n";
		getSD_sim();
		
		//for (i=0;i<width;i++)
		//{
		//	for (j=0;j<height;j++)
		//	{
		//		if (!m_mask.at<double>(j,i))
		//		{
		//			continue;
		//		}
		//		//cout<<i<<" "<<j<<endl;
		//		for (int io=0;io<shape_dim+texture_dim+4;io++)
		//		{
		//			out_J<<SD_ic[i][j][io]<<" ";
		//		}
		//		out_J<<endl;
		//	}
		//}
		//out_J.close();

		//FullOut<<"----------------------SD Image-----------------------------\n";
		//for (i=0;i<width;i++)
		//{
		//	for (j=0;j<height;j++)
		//	{
		//		if (!m_mask.at<double>(j,i))
		//		{
		//			continue;
		//		}
		//		//cout<<i<<" "<<j<<endl;
		//		for (int io=0;io<50;io++)
		//		{
		//			FullOut<<SD_ic[i][j][io]<<" ";
		//		}
		//		
		//		break;
		//	}
		//}
		//FullOut<<endl;



		if(outputtime)
		{
			HessianStop=GetTickCount();

			cout<<"SD time: "<<(HessianStop-HessianStart)<<endl;
		}

		for (i=0;i<dim;i++)
		{
			sumSD[i]=deltaP[i]=0;
		}
		//cout<<"compute SD...\n";

		//compute SD
	//	c_pixnum=0;
		
		errorSum=0;
		for (i=0;i<width;i++)
		{
			for (j=0;j<height;j++)
			{
				if (!m_mask.at<double>(j,i))
				{
					continue;
				}
				//errorIm=CV_MAT_ELEM(*errorImageMat,double,0,(int)CV_MAT_ELEM(*mask_withindex,double,j,i));
				errorIm=m_errorImageMat.at<double>(0,(int)m_mask_withindex.at<double>(j,i));
				errorSum+=errorIm*errorIm;
				for (k=0;k<dim;k++)
				{
					sumSD[k]+=SD_ic[i][j][k]*errorIm;
			/*		if(i>0)
					cout<<SD_ic[i][j][k]<<" ";*/
				}
			//	c_pixnum++;
		/*		if(i>0)
				cout<<endl;*/
			}
		}

	/*	cout<<"sumSD\n";
		for (int i=0;i<dim;i++)
		{
			cout<<sumSD[i]<<" ";
		}
		cout<<endl;*/
		//if (errorSum<0.03&&(lastErrorSum-errorSum)<0.004&&lastErrorSum>errorSum)
		//{
		//	//goto save;
		//}

		/////////smooth adjustment////////////
		//if (lastError<errorSum)
		//{
		//	//stepLength/=2.0;
		//	//if (stepLength<0.0001)
		//	{
		//		stepLength=0.0001;
		//	}
		//}
		//if (lastErrorSum<errorSum&&(errorSum-lastErrorSum)/lastErrorSum>0.080)
		//{
		//	if (errorSum>0.1)
		//	{
		//		stepLength=1;
		//	}
		//	else if (errorSum>0.06)
		//	{
		//		stepLength=0.6;
		//	}
		//	else
		//		stepLength=0.1;
		//}

		//////////////////smooth version 1//////////////////////
	//	if (lastError<errorSum&&errorSum<0.1)
	//	{
	//		stepLength/=2.0;
	//		//recaculatetime++;
	//		if (stepLength<0.0000001)
	//		{
	//			goto save;
	//		}
	///*		for (i=0;i<shape_dim;i++)
	//		{
	//			s_weight[i]=s_weight_last[i];
	//		}
	//		for (i=0;i<texture_dim;i++)
	//		{
	//			t_weight[i]=t_weight_last[i];
	//		}
	//		theta=theta_last;
	//		k_scale=k_scale_last;
	//		transform_x=transform_x_last;
	//		transform_y=transform_y_last;*/
	//		times--;
	//		increaseTime=0;
	//		//lastError=errorSum;
	//		//recaculate=true;
	//		cout<<"Last error: "<<lastError<<" error: "<<errorSum<<" re-calculate! Step length from "<<stepLength*2<<" to "<<stepLength<<endl;
	//		//continue;
	//	}
	//	//else if (currentFrame>=3&&currentFrame<=480&&errorSum<0.2&&lastError<errorSum&&errorSum<0.15)
	//	//{
	//	//	stepLength/=2.0;
	//	//	//recaculatetime++;
	//	//	if (stepLength<0.0000001)
	//	//	{
	//	//		goto save;
	//	//	}
	//	//	times--;
	//	//	increaseTime=0;
	//	//	cout<<"Last error: "<<lastError<<" error: "<<errorSum<<" re-calculate! Step length from "<<stepLength*2<<" to "<<stepLength<<endl;
	//	//
	//	//}
	//	else if (errorSum>0.15)
	//	{
	//	/*	if (currentFrame>=403&&currentFrame<=480)
	//		{
	//			stepLength=0.2;
	//		}
	//		else*/
	//		stepLength=1;

	//	}
	//	else
	//	{
	//		increaseTime++;
	//		if (increaseTime>3)
	//		{
	//			stepLength*=1000;/*=1;*/;
	//			if (stepLength>1)
	//			{
	//				stepLength=1;
	//			}
	//	/*		if (currentFrame>=403&&currentFrame<=480)
	//			{
	//				if (stepLength>0.2)
	//				{
	//					stepLength=0.2;
	//				}
	//			}*/
	//			increaseTime=0;
	//		}
	//	}

		if(outputtime)
		{
			HessianStart=GetTickCount();
		}
		//HessianStart=GetTickCount();
		getHessian();
		//HessianStop=GetTickCount();

		//cout<<"Hessian  time: "<<(HessianStop-HessianStart)<<endl;
		if(outputtime)
		{
			HessianStop=GetTickCount();

			cout<<"Hessian  time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}
	
		/*cout<<"----------------------Hessian-----------------------------\n";
		for (i=0;i<full_Hessian.cols;i++)
		{	
			{
				cout<<full_Hessian.at<double>(0,i)<<" ";
			}
		}
		cout<<endl;

		
		for (i=0;i<full_Hessian.rows;i++)
		{	
			for (j=0;j<full_Hessian.cols;j++)
			{
				out_HHH<<full_Hessian.at<double>(i,j)<<" ";
			}
			out_HHH<<endl;
		}
		out_HHH<<endl;
		out_HHH.close();

		FullOut<<"----------------------inv Hessian-----------------------------\n";
		for (i=0;i<m_inv_hessian.cols;i++)
		{	
			{
				FullOut<<m_inv_hessian.at<double>(0,i)<<" ";
			}
		}
		FullOut<<endl;*/

		////display
		//meanTexture->devide(texture_scale);
		//currentTexture->devide(texture_scale);
		//cvNamedWindow("1");
		//cvNamedWindow("2");
		//cvShowImage("1",currentTexture->getImage());
		//cvShowImage("2",meanTexture->getImage());
		//	cvWaitKey(0);






		/*else
		{
			stepLength*=2.0;
			if (stepLength>1)
			{
				stepLength=1;
			}
		}
		if (errorSum<0.05)
		{
			stepLength=0.1;
		}
		else
		{
			stepLength=1.0;
		}*/

		// int cind;
		//int currentPtsInd;
		//double currentSD[10][2];
		//errorSum_detection=0;
		//for (i=0;i<trees->numOfLabels;i++)
		//{
		//	currentPtsInd=trees->interestPtsInd[i];
		//	// cout<<currentPtsInd<<endl;

		//	currentSD[i][0]=currentSD[i][1]=0;
		//	for (j=0;j<trees->candidates[i]->size();j++)
		//	{
		//		currentSD[i][0]+=trees->candidates[i]->at(j).z*(currentShape->pts[currentPtsInd][0]-trees->candidates[i]->at(j).x);
		//		currentSD[i][1]+=trees->candidates[i]->at(j).z*(currentShape->pts[currentPtsInd][1]-trees->candidates[i]->at(j).y);
		//		errorSum_detection+=trees->candidates[i]->at(j).z*sqrt(((currentShape->pts[currentPtsInd][0]-trees->candidates[i]->at(j).x))*(currentShape->pts[currentPtsInd][0]-trees->candidates[i]->at(j).x)+
		//			((currentShape->pts[currentPtsInd][1]-trees->candidates[i]->at(j).y))*((currentShape->pts[currentPtsInd][1]-trees->candidates[i]->at(j).y)) );
		//		// cout<<i<<" "<<currentSD[i][0]<<" "<<currentSD[i][1]<<endl;
		//	}
		//}
		//cout<<"error dection: "<<errorSum_detection/(float)trees->numOfLabels<<endl;
		if (errorSum<0.1)
		{
			cout<<"set smoothweight to 0\n";
			smoothWeight=0;
		}
		else if(useDetectionResults)
		{
			smoothWeight+=0.2;
			if (smoothWeight>1)
			{
				smoothWeight=1;
			}
		}
		//detection
		if (smoothWeight>0)
		{

			/* for (i=0;i<trees->numOfLabels;i++)
			 {
				 cout<<currentSD[i][0]<<" "<<currentSD[i][1]<<endl;
			 }*/
			
			double tmpError;
			int currentPtsInd;
		/*	for (i=0;i<dim;i++)
			{
				sumDectionSD[i]=0;
			}*/
			
			//output the positions
			for (j=0;j<trees->numOfLabels;j++)
			{
				currentPtsInd=trees->interestPtsInd[j];
	
				cout<<currentShape->pts[currentPtsInd][0]<<" "<<currentShape->pts[currentPtsInd][1]<<
					" "<<prob_mu[j][0]<<" "<<prob_mu[j][1]<<endl;
			}

			double tmp[2];
			for (i=0;i<dim;i++)
			{
				sumDectionSD[i]=0;
				//tmpError=trees->probabilityMap[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);
				for (j=0;j<trees->numOfLabels;j++)
				{
					currentPtsInd=trees->interestPtsInd[j];
					tmp[0]=SD_detection[j][0][i]*prob_conv[j].at<double>(0,0)+SD_detection[j][1][i]*prob_conv[j].at<double>(1,0);
					tmp[1]=SD_detection[j][0][i]*prob_conv[j].at<double>(0,1)+SD_detection[j][1][i]*prob_conv[j].at<double>(1,1);
					//cout<<tmp[0]<<" "<<tmp[1]<<endl;
					sumDectionSD[i]+=tmp[0]*(currentShape->pts[currentPtsInd][0]-prob_mu[j][0])+tmp[1]*(currentShape->pts[currentPtsInd][1]-prob_mu[j][1]);
				}
				//cout<<sumDectionSD[i]<<" ";
			}
			//cout<<endl;

			//for (i=0;i<trees->numOfLabels;i++)
			//{
			//	currentPtsInd=trees->interestPtsInd[i];
			//	tmpError=sqrt(trees->gradientMapX[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0])*
			//		trees->gradientMapX[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0])+
			//		trees->gradientMapY[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0])*
			//		trees->gradientMapY[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]));
			//	//tmpError=trees->probabilityMap[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);
			//	for (j=0;j<dim;j++)
			//	{
			//		sumDectionSD[j]+=SD_detection[i][j]*tmpError;
			//	}
			//}

			//for (i=0;i<dim;i++)
			//{
			//	sumDectionSD[i]=0;
			//	for (j=0;j<trees->numOfLabels;j++)
			//	{
			//		sumDectionSD[i]+=SD_detection[j][i][0]*;
			//	}
			//	//cout<<sumSmoothSD[i]<<" ";
			//}

	
			/*for (j=0;j<meanShape->ptsNum;j++)
			{
				cind=m_mask_withindex.at<double>(meanShape->pts[j][1],meanShape->pts[j][0]);
				cout<<(lastTemplate.at<double>(0,cind)-
					currentTexture->m_imgData.at<double>(0,cind))<<endl;
			}	*/
		//	cout<<endl;
		}
		//smooth sd
		//if (smoothWeight>0&&currentFrame>startNum)
		//{
		//	 int cind;
		//	for (i=0;i<dim;i++)
		//	{
		//		sumSmoothSD[i]=0;
		//		for (j=0;j<meanShape->ptsNum;j++)
		//		{
		//			cind=m_mask_withindex.at<double>(meanShape->pts[j][1],meanShape->pts[j][0]);
		//			sumSmoothSD[i]+=SD_smooth[j][i]*(lastTemplate.at<double>(0,cind)-
		//				currentTexture->m_imgData.at<double>(0,cind));
		//		}
		//		//cout<<sumSmoothSD[i]<<" ";
		//	}
		//	/*for (j=0;j<meanShape->ptsNum;j++)
		//	{
		//		cind=m_mask_withindex.at<double>(meanShape->pts[j][1],meanShape->pts[j][0]);
		//		cout<<(lastTemplate.at<double>(0,cind)-
		//			currentTexture->m_imgData.at<double>(0,cind))<<endl;
		//	}	*/
		////	cout<<endl;
		//}

	//	cout<<"compute deltaP...\n";
		//compute deltaP
	//	adjustGamma=(currentTexture->pointMul(currentTexture))/currentTexture->pointMul(currentTemplate);


		///////////////////need to commant out this////////////////////////
		stepLength=1.0;




		double tmpJE;
		for (i=0;i<dim;i++)
		{
			for (j=0;j<dim;j++)
			{
				tmpJE=AAM_weight*sumSD[j];
				if (smoothWeight>0)
				{
					tmpJE+=smoothWeight*sumDectionSD[j];
				}
				if (priorWeight>0)
				{
					tmpJE+=priorWeight*SD_prior[j];
				}
				deltaP[i]+=-stepLength*m_inv_hessian.at<double>(i,j)*tmpJE;

				//if (smoothWeight>0&&i>=shape_dim+texture_dim)
				//{
				//	//deltaP[i]+=-stepLength*m_inv_hessian.at<double>(i,j)*sumSD[j];
				//	deltaP[i]+=-stepLength*m_inv_hessian.at<double>(i,j)*(AAM_weight*sumSD[j]+smoothWeight*sumDectionSD[j]);
				///*	if (m_inv_hessian.at<double>(i,j)!=0)
				//	{
				//		cout<<m_inv_hessian.at<double>(i,j)<<" "<<sumDectionSD[j]<<endl;
				//	}*/
				//	
				//}
				//else
				//{
				//	deltaP[i]+=-stepLength*m_inv_hessian.at<double>(i,j)*sumSD[j];
				//}
				
			}
		//	cout<<deltaP<<" ";
		}

	/*	cout<<"deltaP\n";
		for (i=texture_dim+shape_dim;i<texture_dim+shape_dim+4;i++)
		{
			cout<<deltaP[i]<<" ";
		}
		cout<<endl;*/
	/*	if (times==0)
		{
			transform_x+=deltaP[texture_dim+shape_dim+2];
			transform_y+=deltaP[texture_dim+shape_dim+3];
			smoothWeight=0;
			times++;
			continue;
		}*/

	/*	FullOut<<"----------------------Hessian-----------------------------\n";
		for (i=0;i<dim;i++)
		{	
			{
				FullOut<<deltaP[i]<<" ";
			}
		}
		FullOut<<endl;*/


		if (outputtime)
		{
			HessianStop=GetTickCount();

			cout<<"caculate deltaP time: "<<(HessianStop-HessianStart)<<endl;
			HessianStart=GetTickCount();
		}

		//update the shape
		for (i=0;i<shape_dim;i++)
		{
			s_weight[i]+=deltaP[i];
		}

		//update t_weight
		{
			for (i=0;i<texture_dim;i++)
			{
				t_weight[i]+=deltaP[shape_dim+i];
			}
		}
		theta+=deltaP[shape_dim+texture_dim];
		k_scale+=deltaP[shape_dim+texture_dim+1];
		transform_x+=deltaP[shape_dim+texture_dim+2];
		transform_y+=deltaP[shape_dim+texture_dim+3];
		times++;
	
		cout<<"theta scale tx ty: "<<theta<<" "<<k_scale<<" "<<transform_x<<" "<<transform_y<<endl;
	//	cout<<endl;

		//need to stop?
		ssum=0;
		for (i=0;i<shape_dim;i++)
		{
			ssum+=deltaP[i]*deltaP[i];
		}
		ssum=sqrt(ssum);

		//if (outputtime)
		{
			cout<<times<<" Error:"<<errorSum<<"  steplength: "<<stepLength<<endl<<" Displayed sum:"<<ssum<<
				"last-current: "<<abs(errorSum-lastError)<<"  Ptsdifference:"<<Pts_Difference<<endl;
		}
		if (times>30)
		{
			break;
		}
	//	if (errorSum<0.07||ssum<0.1||times>20||(currentFrame>startNum&&abs(errorSum-lastError)<0.00001))//||(errorSum<0.4&&ssum<3)||times>100)
			if(0)
	//if (errorSum<0.13||times>400||(currentFrame>startNum&&times>MaxIterNum)||(errorSum<0.23&&abs(errorSum-lastError)<0.001))
	//if (times>30||(currentFrame>startNum&&times>MaxIterNum)||(errorSum<0.2&&abs(Pts_Difference)<0.0001))	
	{
		cout<<"ending times: "<<times<<"   Pts_Difference: "<<Pts_Difference<<endl;
save:	
		 //if (errorSum>0.15&recaculatetime<10)
			//{
			//	times=1;
			//	transform_y-=pow(1.0f,recaculatetime)*2.5*recaculatetime/k_scale;
			//	//k_scale*=1.1;
			//	recaculatetime++;
			//	MaxIterNum=60;
			//	goto nosave;
			//}
			//MaxIterNum=35;
	/*		else if (errorSum>0.08&&recaculatetime<20)
			{
				times=1;
				transform_y-=5.5;
				recaculatetime++;
				goto nosave;
			}*/
		
		
			//display the result
			{
				cvCopyImage(Input,imgcopy);
				cvNamedWindow("1");
				for (int i=0;i<ptsNum;i++)
				{
					cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),1,CV_RGB(0,0,255));
				}
				cvShowImage("1",imgcopy);
				cvWaitKey(1);
			}

			//save images
			string fullname;
			string tmpName;
			char name[50];
			sprintf(name, "%d.jpg", 10000+currentFrame);
			tmpName=name;
			fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
			cvSaveImage(fullname.c_str(),imgcopy);
			//cout<<fullname<<endl;
		/*	sprintf(name, "%d_ori.jpg", currentFrame);
			fullname=dataDir+name;
			cvSaveImage(fullname.c_str(),Input);*/
			//Output the pts
			char pname[50];
			sprintf(pname, "%d.txt", 10000+currentFrame);
			tmpName=pname;
			fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
			ofstream out(fullname.c_str(),ios::out);
			out<<Input->width/resizeSize<<" "<<Input->height/resizeSize<<endl;
			out<<currentShape->ptsNum<<" "<<currentShape->ptsNum<<endl;
			for (int i=0;i<ptsNum;i++)
			{
				out<<currentShape->pts[i][0]/(double)Input->width<<" "<<
					currentShape->pts[i][1]/(double)Input->height<<endl;
			}
			sprintf(pname, "%d_iteration.txt", 10000+currentFrame);
			tmpName=pname;
			fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
			ofstream out_error(fullname.c_str(),ios::out);
			out_error<<times<<endl;
			out_error.close();

			sprintf(pname, "%d_parameters.txt", 10000+currentFrame);
			tmpName=pname;
			fullname=dataDir+tmpName.substr(1,tmpName.length()-1);
			ofstream out1(fullname.c_str(),ios::out);
			for (int i=0;i<shape_dim;i++)
			{
				out1<<s_weight[i]<<" ";
			}
			for (int i=0;i<texture_dim;i++)
			{
				out1<<t_weight[i]<<" ";
			}
			out1<<theta<<" "<<k_scale<<" "<<transform_x<<" "<<transform_y<<endl;
			out1.close();

			for (int i=0;i<meanShape->ptsNum;i++)
			{
				ptsLast[i][0]=currentShape->pts[i][0];
				ptsLast[i][1]=currentShape->pts[i][1];
			}
			if (smoothWeight>0)
			{
			
				lastTemplate=currentTexture->m_imgData.clone();
		/*		for (int i=0;i<shape_dim;i++)
				{
					parametersLast[i]=s_weight[i];
				}
				for (int i=0;i<texture_dim;i++)
				{
					parametersLast[shape_dim+i]=t_weight[i];
				}
				parametersLast[shape_dim+texture_dim]=theta;
				parametersLast[shape_dim+texture_dim+1]=k_scale;
				parametersLast[shape_dim+texture_dim+2]=transform_x;
				parametersLast[shape_dim+texture_dim+3]=transform_y;*/
			}
			lastErrorSum=errorSum;
			break;
		}
		else
		{
nosave:		//if(!recaculate)
			
				lastError=errorSum;
		//	else
			//	recaculate=false;
			for (int i=0;i<shape_dim;i++)
			{
				s_weight_last[i]=s_weight[i];
			}
			for (int i=0;i<texture_dim;i++)
			{
				t_weight_last[i]=t_weight[i];
			}
			theta_last=theta;
			k_scale_last=k_scale;
			transform_x_last=transform_x;
			transform_y_last=transform_y;

		
			//if (smoothWeight>0)
			{
				Pts_Difference=0;
				for (int i=0;i<meanShape->ptsNum;i++)
				{
					Pts_Difference+=sqrt((currentShape->pts[i][0]-ptsLast[i][0])*(currentShape->pts[i][0]-ptsLast[i][0])+
						(currentShape->pts[i][1]-ptsLast[i][1])*(currentShape->pts[i][1]-ptsLast[i][1]));
				}
				Pts_Difference/=(double)meanShape->ptsNum;
			}
			for (int i=0;i<meanShape->ptsNum;i++)
			{
				ptsLast[i][0]=currentShape->pts[i][0];
				ptsLast[i][1]=currentShape->pts[i][1];
			}

		
			//cout<<times++<<" Error:"<<errorSum<<endl<<" Displayed sum:"<<ssum<<endl;
		}

	


	



		if (outputtime)
		{

			HessianStop=GetTickCount();

			cout<<"total  time: "<<(double)(HessianStop-d_totalstart)<<endl;
		}

		
		//display
	//		if(times%20==0)
		if(showSingleStep&&currentFrame>=startNum)
		{
		/*	cout<<"shape weights:\n";
			for (int i=0;i<shape_dim;i++)
			{
				cout<<s_weight[i]<<" ";
			}
			cout<<endl;

			ofstream pts_out("F:\\Projects\\Facial feature points detection\\Matlab code\\pts.txt",ios::out);
			for (int i=0;i<ptsNum;i++)
			{
				pts_out<<meanShape->pts[i][0]<<" "<<meanShape->pts[i][1]<<endl;
			}
			pts_out.close();*/

			cvCopyImage(Input,imgcopy);
			cvNamedWindow("1");
			for (int i=0;i<ptsNum;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
			}

			for (int i=0;i<trees->numOfLabels;i++)
			{
				cvCircle(imgcopy,cvPoint(currentShape->pts[trees->interestPtsInd[i]][0],currentShape->pts[trees->interestPtsInd[i]][1]),1,CV_RGB(0,0,255));
			}

		/*	for (int i=0;i<trees->numOfLabels;i++)
			{
				for (int j=0;j<trees->candidates[i]->size();j++)
				{
					cvCircle(imgcopy,cvPoint(trees->candidates[i]->at(j).x,trees->candidates[i]->at(j).y),3,CV_RGB(0,0,255));
				}
			}*/
			//cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
//			cvCircle(imgcopy,cvPoint(p[0],p[1]),3,CV_RGB(0,0,255));
			cvShowImage("1",imgcopy);
			cvNamedWindow("2");
			*meanfordisplay=(*currentTexture);
			meanfordisplay->devide(texture_scale);
			for (int i=0;i<meanfordisplay->pix_num;i++)
			{
				CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
			}
			cvShowImage("2",meanfordisplay->getImage());
			//imwrite("F:\\Projects\\Facial feature points detection\\Matlab code\\input_warped.jpg",cvarrToMat(meanfordisplay->getImage()));
			cvNamedWindow("3");
			*meanfordisplay=(*currentTemplate);
			meanfordisplay->devide(texture_scale);
			for (int i=0;i<meanfordisplay->pix_num;i++)
			{
				CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
			}
			IplImage *ccc=meanfordisplay->getImage();
		/*	for (int i=0;i<ptsNum;i++)
			{
				cvCircle(ccc,cvPoint(meanShape->pts[i][0],meanShape->pts[i][1]),3,CV_RGB(0,0,255));
			}*/
			cvShowImage("3",ccc);
			//imwrite("F:\\Projects\\Facial feature points detection\\Matlab code\\synthesized.jpg",cvarrToMat(ccc));
			cvWaitKey();
		}

		//check if we need to update the texture
		//then we are done
	}

	delete currentTexture;

}

//void AAM_RealGlobal_GPU::iterate_GPU(IplImage *Input)
//{
//	//IplImage *warpedInput;//=cvCreateImage(cvGetSize(Template),Template->depth,Template->nChannels);
//
//		//initialize
//	//cout<<meanShape->width<<" "<<meanShape->height<<endl;
//	int i,j;
//	//if(currentFrame==startNum)
//	if(1)
//	{
//		//currentShape->scale(0.98,1);
//		//only do once to intialize face position
//		double *p=meanShape->getcenter();
//		//	cout<<p[0]<<" "<<p[1]<<endl;
//		//	//offset=cvPoint(p[0],p[1])-faceCenter;	
//		//	cvCopyImage(Input,imgcopy);
//		//	cvNamedWindow("1");
//		//	for (int i=0;i<ptsNum;i++)
//		//	{
//		//		cvCircle(imgcopy,cvPoint(currentShape->pts[i][0],currentShape->pts[i][1]),3,CV_RGB(0,0,255));
//		//	}
//		////	cvCircle(imgcopy,faceCenter,3,CV_RGB(0,0,255));
//		//	cvCircle(imgcopy,cvPoint((int)p[0],(int)p[1]),3,CV_RGB(0,0,255));
//		//	cvShowImage("1",imgcopy);
//		//	cvWaitKey(0);
//		//currentShape->translate(p[0]-faceCenter.x,p[1]-faceCenter.y-15);
//		//tran_x=p[0]-faceCenter.x;
//		//tran_y=p[1]-faceCenter.y-30;
//
//		//transform_x=faceCenter.x-p[0]+10;
//		//transform_y=faceCenter.y-p[1]+55;
//
//		transform_x=faceCenter.x-p[0]+initialTx;
//		transform_y=faceCenter.y-p[1]+initialTy;
//
//		k_scale=initialScale;
//
//		
//		for (int i=0;i<shape_dim;i++)
//		{
//			parameters[i]=s_weight[i];
//		}
//		for (i=0;i<texture_dim;i++)
//		{
//			parameters[shape_dim+i]=t_weight[i];
//		}
//		parameters[shape_dim+texture_dim]=theta;
//		parameters[shape_dim+texture_dim+1]=k_scale;
//		parameters[shape_dim+texture_dim+2]=transform_x;
//		parameters[shape_dim+texture_dim+3]=transform_y;
//
//	/*	transform_x=245;
//		transform_y=340;*/
//		//k_scale=1000;
//
//	/*	transform_x=278;
//		transform_y=166;*/
//	}
//	else
//	{
//		parameters[0]=-1000000000;
//	}
//
//
//	//denormalize
//	//if(currentFrame==startNum)
//	////if(0)
//	//{
//	//	char ppname[500];
//	//	sprintf(ppname, "%d_parameters.txt", 10000+currentFrame);
//	//	string tmpName=ppname;
//	//	string fullName=dataDir+tmpName.substr(1,tmpName.length()-1);;
//	//	ifstream in(fullName.c_str(),ios::in);
//	//	if (in)
//	//	{
//	//		stepLength=0.5;
//	//		//MaxIterNum=15;
//	//	}
//	//	while(in)
//	//	{
//	//		for (int i=0;i<shape_dim;i++)
//	//		{
//	//			in>>s_weight[i];
//	//		}
//	//		for (int i=0;i<texture_dim;i++)
//	//		{
//	//			in>>t_weight[i];
//	//		}
//	//		in>>theta>>k_scale>>transform_x>>transform_y;
//	//	}
//	//	in.close();
//
//	//}
//	//
//	//set current parameters
//
//
//	//for (int i=0;i<shape_dim;i++)
//	//{
//	//	cout<<parameters[i]<<" "<<s_weight[i]<<endl;
//	//}
//
//	if (usingTrees)
//	{
//		Mat m_img=cvarrToMat(Input);
//		for (i=0;i<m_img.rows;i++)
//		{
//			for (j=0;j<m_img.cols;j++)
//			{
//				inputImg[i*m_img.cols+j]=m_img.at<uchar>(i,j);
//			}
//		}
//		float * GPU_Img=setData_onRun(parameters,inputImg,cu_gradientX,cu_gradientY,m_img.cols,m_img.rows);
//		rt.predict_rt_GPU_AAM(Input,GPU_Img); //use randomized trees to get results
//		return;
//	}
//
//
//
//	//using AAM with robust error functions to get the final result
//
//
//
//	//setData_onrun_withAAM()
//	//rt.predict_rt_GPU(Input);
//	
//
//	/*cout<<"Mean Texture\n";
//	for (i=0;i<100;i++)
//	{
//		cout<<meanTexture->m_imgData.at<double>(0,i)<<" ";
//	}
//	cout<<endl;*/
//
//	//cout<<"eigen texture\n";
//	//for (i=0;i<texture_dim;i++)
//	//{
//	//	cout<<m_t_vec.at<double>(0,i)<<" ";
//	//}
//	//cout<<endl;
//	//cout<<meanTexture->m_imgData.cols<<endl;
//	//call the iteration function
//	iterate_CUDA(Input->width,Input->height,smoothWeight,AAM_weight,currentFrame,startNum,parameters,meanShape->inv_mask);
//
//}

//output: template_mat
void AAM_RealGlobal_GPU::getCurrentTexture(double *tw)
{
	//we use the real meantexture
	*currentTemplate=*meanTexture;

	//Texture *meanfordisplay=new Texture;;

	//if (outputtime)
	//{
	//}
	//cout<<"texture weights\n";

	
	//for (int i=0;i<texture_dim;i++)
	//{
	//	cout<<tw[i]<<" ";
	//	sum=0;
	//	for (int j=0;j<pix_num;j++)
	//	{
	//		CV_MAT_ELEM(*currentTemplate->imgData,double,0,j)+=
	//			CV_MAT_ELEM(*textures[i]->imgData,double,0,j)*tw[i];
	//	}
	///*	cvNamedWindow("Texture Updating...");
	//	*meanfordisplay=(*currentTemplate);
	//	meanfordisplay->devide(texture_scale);
	//	for (int i=0;i<meanfordisplay->pix_num;i++)
	//	{
	//		CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
	//	}
	//	cvShowImage("Texture Updating...",meanfordisplay->getImage());
	//	cvWaitKey(0);*/
	//	//(*currentTemplate)=(*currentTemplate)+(*textures[i])*(tw[i]);
	//}

	double sum;
	//#pragma omp parallel for reduction(+: sum)  
	for (int i=0;i<pix_num;i++)
	{
		//cout<<tw[i]<<" ";
		sum=0;
		for (int j=0;j<texture_dim;j++)
		{
		/*	sum+=
				CV_MAT_ELEM(*textures[j]->imgData,double,0,i)*tw[j];*/
			sum+=
				textures[j]->m_imgData.at<double>(0,i)*tw[j];
		}
		//CV_MAT_ELEM(*currentTemplate->imgData,double,0,i)+=sum;
		currentTemplate->m_imgData.at<double>(0,i)+=sum;
	/*	cvNamedWindow("Texture Updating...");
		*meanfordisplay=(*currentTemplate);
		meanfordisplay->devide(texture_scale);
		for (int i=0;i<meanfordisplay->pix_num;i++)
		{
			CV_MAT_ELEM(*(meanfordisplay->imgData),double,0,i)+=100;
		}
		cvShowImage("Texture Updating...",meanfordisplay->getImage());
		cvWaitKey(0);*/
		//(*currentTemplate)=(*currentTemplate)+(*textures[i])*(tw[i]);
	}

	//for (int jj=0;jj<100;jj++)
	//{
	//	cout<<currentTemplate->m_imgData.at<double>(0,jj)<<" ";
	//}
	//cout<<endl;
	//float ssnum=0;
	//for (int jj=0;jj<pix_num;jj++)
	//{
	//	ssnum+=currentTemplate->m_imgData.at<double>(0,jj);
	//}
	//cout<<ssnum/pix_num<<endl;

	//normalize
	currentTemplate->normalize();
	//cout<<cscale/currentTemplate->pointMul(meanTexture)<<endl;

	currentTemplate->devide(currentTemplate->pointMul(meanTexture));
	//currentTemplate->simple_normalize();
	//cout<<endl;

	return;
	//
	int cind;
	double ctex;
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			if (CV_MAT_ELEM(*mask,double,j,i)==0)
			{
				continue;
			}
			else
			{
				cind=CV_MAT_ELEM(*mask_withindex,double,j,i);
				ctex=CV_MAT_ELEM(*meanTexture->imgData,double,0,cind);
				for(int k=0;k<texture_dim;k++)
				{
					ctex+=CV_MAT_ELEM(*textures[k]->imgData,double,0,cind);
				}
				CV_MAT_ELEM(*Template_mat,double,0,cind)=ctex;
			}
		}
	}
}

void AAM_RealGlobal_GPU::getSD_sim()
{
	int sd_dim=texture_dim+shape_dim+4;
	double cgradient[2];
	int cind;

	//need to caculate current gradient
	//gradient(currentTexture->getImageMat(),warp_igx,warp_igy,meanShape->marginMask);
	//m_warp_igx=cvarrToMat(warp_igx);
	//m_warp_igy=cvarrToMat(warp_igy);
	//namedWindow("1");
	//imshow("1",m_warp_igx+100);


	//gradient caculate by warping
	//warp the gradient image

	//Mat m_gx,m_gy;
	//m_gx=cvarrToMat(inputGradient_x);
	//m_gy=cvarrToMat(inputGradient_y);
	//cout<<"gradient X\n";
	//for (int kk=0;kk<50;kk++)
	//{
	//	cout<<m_gx.at<double>(0,kk)<<" ";
	//}
	//cout<<endl;
	//cout<<"gradient Y\n";
	//for (int kk=0;kk<50;kk++)
	//{
	//	cout<<m_gy.at<double>(0,kk)<<" ";
	//}
	//cout<<endl;

	warp_igx=warp->piecewiseAffineWarping(inputGradient_x,warp_igx,currentShape,meanShape,triangleList,true,meanShape->weightTabel,indexTabel);
	warp_igy=warp->piecewiseAffineWarping(inputGradient_y,warp_igy,currentShape,meanShape,triangleList,true,meanShape->weightTabel,indexTabel);

	/*warp_igx=warp->piecewiseAffineWarping(inputGradient_x,warp_igx,currentShape,meanShape,triangleList,true,meanShape->affineTable);
	warp_igy=warp->piecewiseAffineWarping(inputGradient_y,warp_igy,currentShape,meanShape,triangleList,true,meanShape->affineTable);*/
	m_warp_igx=cvarrToMat(warp_igx);
	m_warp_igy=cvarrToMat(warp_igy);



	//cout<<"gradient before scale X\n";
	//for (int kk=0;kk<50;kk++)
	//{
	//	cout<<m_warp_igx.at<double>(meanShape->inv_mask[kk][1],meanShape->inv_mask[kk][0])<<" ";
	//}
	//cout<<endl;
	//cout<<"gradient before scale Y\n";
	//for (int kk=0;kk<50;kk++)
	//{
	//	cout<<m_warp_igy.at<double>(meanShape->inv_mask[kk][1],meanShape->inv_mask[kk][0])<<" ";
	//}
	//cout<<endl;


	m_warp_igx*=tex_scale;
	m_warp_igy*=tex_scale;

	/*ofstream out_G("gradientX_CPU.txt",ios::out);

	for (int i=0;i<m_warp_igx.rows;i++)
	{
		for (int j=0;j<m_warp_igx.cols;j++)
		{
			out_G<<m_warp_igx.at<double>(i,j)<<" ";
		}
		out_G<<endl;
	}
	out_G.close();*/

	//cout<<"gradient X\n";
	//for (int kk=0;kk<50;kk++)
	//{
	//	cout<<m_warp_igx.at<double>(meanShape->inv_mask[kk][1],meanShape->inv_mask[kk][0])<<" ";
	//}
	//cout<<endl;
	//cout<<"gradient Y\n";
	//for (int kk=0;kk<50;kk++)
	//{
	//	cout<<m_warp_igy.at<double>(meanShape->inv_mask[kk][1],meanShape->inv_mask[kk][0])<<" ";
	//}
	//cout<<endl;
	//cout<<"tex_scale: "<<tex_scale<<endl;
	//namedWindow("2");
	//imshow("2",m_warp_igx);
	//waitKey();


	int i,j,k;

	double alpha,beta,gamma;
	double fenmu;
	int tInd[3];

	int triangleInd;

	double sumtx,sumty;

	int ptsNum=currentShape->ptsNum;

	//#pragma omp parallel for
	//for (int m=0;m<width*height;m++)
	//{
	//	i=m/height;
	//	j=(m%height);
	//	{
	double costheta=cos(theta);
	double sintheta=sin(theta);
	//return;
	for (i=0;i<width;i++)
	{
		for (j=0;j<height;j++)
		{
			//if (CV_MAT_ELEM(*mask,double,j,i)==0)
			if (m_mask.at<double>(j,i)==0)
			{
				for (k=0;k<sd_dim;k++)
				{
					SD_ic[i][j][k]=0;
				}
				continue;
			}
			else if (imageMask.at<double>(j,i)==0)
			{
				for (k=0;k<sd_dim;k++)
				{
					SD_ic[i][j][k]=0;
				}
				continue;
			}
			/*cgradient[0]=CV_MAT_ELEM(*gradient_Tx,double,j,i);
			cgradient[1]=CV_MAT_ELEM(*gradient_Ty,double,j,i);*/

			if (affineTable!=NULL)
			{
				triangleInd=affineTable[i][j]->triangleInd;
				alpha=affineTable[i][j]->alpha;
				beta=affineTable[i][j]->beta;
				gamma=affineTable[i][j]->gamma;
			//	tInd[0]=(int)CV_MAT_ELEM(*triangleList,double,triangleInd,0);
			//	tInd[1]=(int)CV_MAT_ELEM(*triangleList,double,triangleInd,1);
			//	tInd[2]=(int)CV_MAT_ELEM(*triangleList,double,triangleInd,2);

				tInd[0]=(int)m_triangleList.at<double>(triangleInd,0);
				tInd[1]=(int)m_triangleList.at<double>(triangleInd,1);
				tInd[2]=(int)m_triangleList.at<double>(triangleInd,2);
			}
			//get the three triangleindex and weight, and then sum them
			//if(triangleInd!=-1)// 如果找到三角形，则进行插值
			{
				sumtx=alpha*currentLocalShape->ptsForMatlab[tInd[0]]+
					beta*currentLocalShape->ptsForMatlab[tInd[1]]+gamma*currentLocalShape->ptsForMatlab[tInd[2]];
				sumty=alpha*currentLocalShape->ptsForMatlab[tInd[0]+ptsNum]+
					beta*currentLocalShape->ptsForMatlab[tInd[1]+ptsNum]+gamma*currentLocalShape->ptsForMatlab[tInd[2]+ptsNum];
			}
			//get the three triangleindex and weight, and then sum them
			//for alpha, theta,k,and t
			cgradient[0]=-(m_warp_igx.at<double>(j,i)*costheta+
				m_warp_igy.at<double>(j,i)*sintheta);
			cgradient[1]=-(m_warp_igx.at<double>(j,i)*(-sintheta)+
				m_warp_igy.at<double>(j,i)*costheta);

			for (int k=0;k<shape_dim;k++)
			{
				SD_ic[i][j][k]=k_scale*(cgradient[0]*full_Jacobian[i][j][0][k]+
				 cgradient[1]*full_Jacobian[i][j][1][k]);
			}
			//k
			SD_ic[i][j][shape_dim+texture_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			//rho
			//cind=CV_MAT_ELEM(*mask_withindex,double,j,i);
			cind=m_mask_withindex.at<double>(j,i);
			//#pragma omp parallel for
			for (int k=shape_dim+0;k<shape_dim+texture_dim;k++)
			{
				//SD_ic[i][j][k]=0;
				//SD_ic[i][j][k]=CV_MAT_ELEM(*textures[k-shape_dim]->imgData,double,0,cind);
				SD_ic[i][j][k]=textures[k-shape_dim]->m_imgData.at<double>(0,cind);
			}



			//theta
			cgradient[0]=-(m_warp_igx.at<double>(j,i)*(-sintheta)+
				m_warp_igy.at<double>(j,i)*costheta);
			cgradient[1]=-(m_warp_igx.at<double>(j,i)*(-costheta)+
				m_warp_igy.at<double>(j,i)*(-sintheta));
			SD_ic[i][j][shape_dim+texture_dim]=k_scale*(cgradient[0]*sumtx+cgradient[1]*sumty);


			//t
			SD_ic[i][j][shape_dim+texture_dim+2]=-m_warp_igx.at<double>(j,i);
			SD_ic[i][j][shape_dim+texture_dim+3]=-m_warp_igy.at<double>(j,i);

		}
	}

	//detection constraints
	if (smoothWeight>0)
	{
		if (SD_detection==NULL)
		{
			//detection
			int j;
	/*		SD_detection=new double *[trees->numOfLabels];
			for (int i=0;i<trees->numOfLabels;i++)
			{
				SD_detection[i]=new double [shape_dim+texture_dim+4];
				for (int j=0;j<shape_dim+texture_dim+4;j++)
				{
					SD_detection[i][j]=0;
				}
			}*/

			SD_detection=new double **[trees->labelNum-1];
			for (int i=0;i<trees->labelNum-1;i++)
			{
				SD_detection[i]=new double *[2];//[shape_dim+texture_dim+4];
				for (int j=0;j<2;j++)
				{
					SD_detection[i][j]=new double[shape_dim+texture_dim+4];
					for (int k=0;k<shape_dim+texture_dim+4;k++)
					{
						SD_detection[i][j][k]=0;
					}
				}
				
			}

		}
		//shape
		int startInd=0;
		int currentPtsInd;
		double tmp[2];
		//double currentSD[640*480][2];
		for (i=0;i<trees->numOfLabels;i++)
		{
			currentPtsInd=trees->interestPtsInd[i];

			//cout<<currentPtsInd<<endl;
	
			/*currentSD[i][0]=currentSD[i][1]=0;
			for (j=0;j<trees->candidates[i]->size();j++)
			{
				currentSD[i][0]+=trees->candidates[i][j].z*(currentShape->pts[currentPtsInd][0]-trees->candidates[i][j].x);
				currentSD[i][1]+=trees->candidates[i][j].z*(currentShape->pts[currentPtsInd][1]-trees->candidates[i][j].y);
			}*/
	
			
			for (j=0;j<shape_dim;j++)
			{
				//cout<<k_scale<<" "<<costheta<<" "<<sintheta<<" "<<
				//	m_s_vec.at<double>(j,currentPtsInd)<<" "<<m_s_vec.at<double>(j,meanShape->ptsNum+currentPtsInd)<<endl;
				tmp[0]=k_scale*(costheta*m_s_vec.at<double>(j,currentPtsInd)-sintheta*m_s_vec.at<double>(j,meanShape->ptsNum+currentPtsInd));
				tmp[1]=k_scale*(sintheta*m_s_vec.at<double>(j,currentPtsInd)+costheta*m_s_vec.at<double>(j,meanShape->ptsNum+currentPtsInd));
				/*SD_detection[i][j]=tmp[0]*trees->gradientMapX[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0])+
					tmp[1]*trees->gradientMapY[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);*/
				SD_detection[i][0][j]=tmp[0];
				SD_detection[i][1][j]=tmp[1];
			}
		

			//theta
			tmp[0]=k_scale*(-sintheta*currentLocalShape->pts[currentPtsInd][0]-costheta*currentLocalShape->pts[currentPtsInd][1]);
			tmp[1]=k_scale*(costheta*currentLocalShape->pts[currentPtsInd][0]-sintheta*currentLocalShape->pts[currentPtsInd][1]);
		/*	SD_detection[i][shape_dim+texture_dim]=tmp[0]*trees->gradientMapX[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0])+
				tmp[1]*trees->gradientMapY[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);*/
			SD_detection[i][0][shape_dim+texture_dim]=tmp[0];
			SD_detection[i][1][shape_dim+texture_dim]=tmp[1];

			//k
			tmp[0]=(costheta*currentLocalShape->pts[currentPtsInd][0]-sintheta*currentLocalShape->pts[currentPtsInd][1]);
			tmp[1]=(sintheta*currentLocalShape->pts[currentPtsInd][0]+costheta*currentLocalShape->pts[currentPtsInd][1]);
			/*SD_detection[i][shape_dim+texture_dim+1]=tmp[0]*trees->gradientMapX[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0])+
				tmp[1]*trees->gradientMapY[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);*/
			SD_detection[i][0][shape_dim+texture_dim+1]=tmp[0];
			SD_detection[i][1][shape_dim+texture_dim+1]=tmp[1];

			//cout<<"current index: "<<currentPtsInd<<" "<<currentLocalShape->pts[currentPtsInd][0]<<" "<<currentLocalShape->pts[currentPtsInd][1]<<endl;
			//all the texture part is 0
			//for (j=shape_dim+1;j<shape_dim+texture_dim;j++)
			//{
			//	/*SD_detection[i][j]=0;*/
			//	SD_detection[i][0][j]=0;
			//	SD_detection[i][1][j]=0;
			//}


			////theta
			//SD_detection[i][shape_dim+texture_dim][0]=0;
			//SD_detection[i][shape_dim+texture_dim][1]=0;
			////SD_detection[i][shape_dim+texture_dim]=tmp[0]*currentSD[0]+tmp[1]*currentSD[1];

			////k
			//SD_detection[i][shape_dim+texture_dim+1][0]=0;
			//SD_detection[i][shape_dim+texture_dim+1][1]=0;

			/*SD_detection[i][shape_dim+texture_dim+2]=trees->gradientMapX[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);
	
			SD_detection[i][shape_dim+texture_dim+3]=trees->gradientMapY[i].at<double>(currentShape->pts[currentPtsInd][1],currentShape->pts[currentPtsInd][0]);*/

			SD_detection[i][0][shape_dim+texture_dim+2]=1;
			SD_detection[i][1][shape_dim+texture_dim+2]=0;

			SD_detection[i][0][shape_dim+texture_dim+3]=0;
			SD_detection[i][1][shape_dim+texture_dim+3]=1;

		}


		//output
		/*cout<<"SD detection\n";
		for (i=0;i<trees->numOfLabels;i++)
		{
			if (1)
			{
				for (j=0;j<shape_dim+texture_dim+4;j++)
				{
					cout<<SD_detection[i][0][j]<<" ";
				}
				cout<<endl;

				for (j=0;j<shape_dim+texture_dim+4;j++)
				{
					cout<<SD_detection[i][1][j]<<" ";
				}
				cout<<endl;
			}

		}*/


		//double tmp[2];
		//double t_gx,t_gy;
		//for (i=0;i<meanShape->ptsNum;i++)
		//{
		//	t_gx=m_warp_igx.at<double>(meanShape->pts[i][1],meanShape->pts[i][0]);
		//	t_gy=m_warp_igy.at<double>(meanShape->pts[i][1],meanShape->pts[i][0]);
		//	for (j=0;j<shape_dim;j++)
		//	{
		//		SD_smooth[i][j]=-k_scale*(t_gx*((costheta*m_s_vec.at<double>(j,i)-
		//			sintheta*m_s_vec.at<double>(j,meanShape->ptsNum+i)))+t_gy*
		//		(sintheta*m_s_vec.at<double>(j,i)+costheta*m_s_vec.at<double>(j,meanShape->ptsNum+i)));
		//	}
		//	for (j=shape_dim;j<shape_dim+texture_dim;j++)
		//	{
		//		SD_smooth[i][j]=0;
		//	}

		//

		//	//theta
		//	tmp[0]=k_scale*(-sintheta*currentLocalShape->pts[i][0]-costheta*
		//		currentLocalShape->pts[i][1]);
		//	tmp[1]=k_scale*(costheta*currentLocalShape->pts[i][0]-sintheta*
		//		currentLocalShape->pts[i][1]);
		//	SD_smooth[i][shape_dim+texture_dim]=-(t_gx*tmp[0]+t_gy*tmp[1]);

		//	//k
		//	tmp[0]=costheta*currentLocalShape->pts[i][0]-sintheta*
		//		currentLocalShape->pts[i][1];
		//	tmp[1]=sintheta*currentLocalShape->pts[i][0]+costheta*
		//		currentLocalShape->pts[i][1];
		//	SD_smooth[i][shape_dim+texture_dim+1]=-(t_gx*tmp[0]+t_gy*tmp[1]);

		//	//t
		//	SD_smooth[i][shape_dim+texture_dim+2]=-t_gx;
		//	SD_smooth[i][shape_dim+texture_dim+3]=-t_gy;
		//}
	}

	//for (i=0;i<width;i++)
	//{
	//	for (j=0;j<height;j++)
	//	{
	//		//if (CV_MAT_ELEM(*mask,double,j,i)==0)
	//		if (m_mask.at<double>(j,i)!=0)
	//		{
	//			for (int k=0;k<shapeDim;k++)
	//			{
	//				cout<<SD_ic[i][j][k]<<" ";
	//			}
	//			cout<<endl;			
	//			break;
	//		}
	//	}
	//}
	if (priorWeight>0)
	{
		if (SD_prior==NULL)
		{
			SD_prior=new double[shape_dim+texture_dim+4];
		}
		for (int i=0;i<shape_dim+texture_dim+4;i++)
		{
			SD_prior[i]=0;
		}
		for(int i=0;i<priorSigma.rows;i++)
		{
			for (int j=0;j<shape_dim;j++)
			{
				SD_prior[i]+=priorSigma.at<double>(i,j)*(s_weight[j]-priorMean[j]);//SD done
			}
		}

		/*ofstream out("F:\\Projects\\Facial feature points detection\\Matlab code\\s and prior.txt",ios::out);
		for (int i=0;i<shape_dim;i++)
		{
			out<<s_weight[i]<<" "<<priorMean[i]<<endl;
		}
		out.close();*/
		//ofstream out("D:\\Fuhao\\face dataset\\train_all_final\\priorMean.txt",ios::out);
		//for (int i=0;i<shape_dim;i++)
		//{
		//	out<<priorMean[i]<<endl;
		//}
		//out.close();

		////ofstream out1("F:\\Projects\\Facial feature points detection\\Matlab code\\priorSigma.txt",ios::out);
		//ofstream out1("D:\\Fuhao\\face dataset\\train_all_final\\inv_Sigma.txt",ios::out);
		//for (int i=0;i<priorSigma.rows;i++)
		//{
		//	for (int j=0;j<priorSigma.cols;j++)
		//	{
		//		out1<<priorSigma.at<double>(i,j)<<" ";
		//	}
		//	out1<<endl;
		//}
		//out1.close();




		/*cout<<"SD prior\n";
		for (int i=0;i<shape_dim+texture_dim+4;i++)
		{
			cout<<SD_prior[i]<<" ";
		}*/
	}

	if (localPCAWeight>0)
	{
		if (SD_local==NULL)
		{
			SD_local=new double[shape_dim+texture_dim+4];
		}
		for (int i=0;i<shape_dim+texture_dim+4;i++)
		{
			if (i<shape_dim)
			{
				SD_local[i]=0;
				for (int j=0;j<shape_dim;j++)
				{
					SD_local[i]+=m_localHessian.at<double>(i,j)*(s_weight[j]-m_local_s_mean.at<double>(0,j));
				}
			}
			else
			{
				SD_local[i]=0;
			}
		}
	}

	//optical flow
	//if (smoothWeight>0&&currentFrame>startNum)
	//{
	//	double tmp[2];
	//	double t_gx,t_gy;
	//	for (i=0;i<meanShape->ptsNum;i++)
	//	{
	//		t_gx=m_warp_igx.at<double>(meanShape->pts[i][1],meanShape->pts[i][0]);
	//		t_gy=m_warp_igy.at<double>(meanShape->pts[i][1],meanShape->pts[i][0]);
	//		for (j=0;j<shape_dim;j++)
	//		{
	//			SD_smooth[i][j]=-k_scale*(t_gx*((costheta*m_s_vec.at<double>(j,i)-
	//				sintheta*m_s_vec.at<double>(j,meanShape->ptsNum+i)))+t_gy*
	//			(sintheta*m_s_vec.at<double>(j,i)+costheta*m_s_vec.at<double>(j,meanShape->ptsNum+i)));
	//		}
	//		for (j=shape_dim;j<shape_dim+texture_dim;j++)
	//		{
	//			SD_smooth[i][j]=0;
	//		}

	//	

	//		//theta
	//		tmp[0]=k_scale*(-sintheta*currentLocalShape->pts[i][0]-costheta*
	//			currentLocalShape->pts[i][1]);
	//		tmp[1]=k_scale*(costheta*currentLocalShape->pts[i][0]-sintheta*
	//			currentLocalShape->pts[i][1]);
	//		SD_smooth[i][shape_dim+texture_dim]=-(t_gx*tmp[0]+t_gy*tmp[1]);

	//		//k
	//		tmp[0]=costheta*currentLocalShape->pts[i][0]-sintheta*
	//			currentLocalShape->pts[i][1];
	//		tmp[1]=sintheta*currentLocalShape->pts[i][0]+costheta*
	//			currentLocalShape->pts[i][1];
	//		SD_smooth[i][shape_dim+texture_dim+1]=-(t_gx*tmp[0]+t_gy*tmp[1]);

	//		//t
	//		SD_smooth[i][shape_dim+texture_dim+2]=-t_gx;
	//		SD_smooth[i][shape_dim+texture_dim+3]=-t_gy;
	//	}

	///*	for (int jj=0;jj<ptsNum;jj++)
	//	{
	//		for (int ii=shape_dim+texture_dim;ii<shape_dim+texture_dim+4;ii++)
	//		{
	//			cout<<SD_ic[(int)meanShape->pts[jj][0]][(int)meanShape->pts[jj][1]]<<" ";
	//		}
	//		cout<<endl;
	//	}
	//	for (int jj=0;jj<ptsNum;jj++)
	//	{
	//		for (int ii=shape_dim+texture_dim;ii<shape_dim+texture_dim+4;ii++)
	//		{
	//			cout<<SD_smooth[jj][ii]<<" ";
	//		}
	//		cout<<endl;
	//	}*/
	//}

	
	

	////set all the numbers to zero

	//#pragma omp parallel for
	//for (int m=0;m<width*height;m++)
	//{
	//	i=m/height;
	//	j=(m%height);
	//	{
	//		if (m_mask.at<double>(j,i)==0)
	//		{
	//			for (int k=0;k<sd_dim;k++)
	//			{
	//				SD_ic[i][j][k]=0;
	//			}
	//		}
	//	}
	//}

	////#pragma omp parallel for
	//for (int m=0;m<width*height;m++)
	//{
	//	i=m/height;
	//	j=(m%height);
	//	{

	////for (int i=0;i<width;i++)
	////{
	////	for (int j=0;j<height;j++)
	////	{
	//		//if (CV_MAT_ELEM(*mask,double,j,i)==0)
	//		if (m_mask.at<double>(j,i)!=0)
	//		{
	//			/*cgradient[0]=CV_MAT_ELEM(*gradient_Tx,double,j,i);
	//			cgradient[1]=CV_MAT_ELEM(*gradient_Ty,double,j,i);*/

	//			cgradient[0]=m_gradient_Tx.at<double>(j,i);
	//			cgradient[1]=m_gradient_Ty.at<double>(j,i);
	//			for (int k=0;k<texture_dim;k++)
	//			{
	//				{
	//					/*cgradient[0]+=t_weight[k]*CV_MAT_ELEM(*g_ax[k],double,j,i);
	//					cgradient[1]+=t_weight[k]*CV_MAT_ELEM(*g_ay[k],double,j,i);*/
	//					cgradient[0]+=t_weight[k]*m_g_ax[k].at<double>(j,i);
	//					cgradient[1]+=t_weight[k]*m_g_ay[k].at<double>(j,i);
	//				}

	//			}
	//			for (int k=0;k<shape_dim;k++)
	//			{
	//				SD_ic[i][j][k]=cgradient[0]*full_Jacobian[i][j][0][k]+
	//					cgradient[1]*full_Jacobian[i][j][1][k];
	//			}
	//			//cind=CV_MAT_ELEM(*mask_withindex,double,j,i);
	//			cind=m_mask_withindex.at<double>(j,i);
	//			//#pragma omp parallel for
	//			for (int k=shape_dim+0;k<shape_dim+texture_dim;k++)
	//			{
	//				//SD_ic[i][j][k]=0;
	//				//SD_ic[i][j][k]=CV_MAT_ELEM(*textures[k-shape_dim]->imgData,double,0,cind);
	//				SD_ic[i][j][k]=textures[k-shape_dim]->m_imgData.at<double>(0,cind);
	//			}
	//	
	//		}

	//	}
	//}


}

//hessian
void AAM_RealGlobal_GPU::getHessian()
{
	//DWORD dwStart, dwStop;  
	int dim=shape_dim+texture_dim+4;


//	cvSetZero(Hessian);
//	
////	dwStart=GetTickCount();
//
//	int i,j;
//	#pragma omp parallel for
//	for (int m=0;m<width*height;m++)
//	{
//		i=m/height;
//		j=(m%height);
//	//	cout<<i<<" "<<j<<endl;
//		if(CV_MAT_ELEM(*mask,double,j,i)==1)
//		{
//			//for (int k=0;k<dim;k++)
//			//{
//			//	CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//			//}
//			for (int k=0;k<dim;k++)
//			{
//				//CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//				for (int l=0;l<dim;l++)
//				{
//					CV_MAT_ELEM(*Hessian,double,k,l)+=SD_ic[i][j][k]*SD_ic[i][j][l];
//				}
//
//			}
//		}
//	}
//
//	m_hessian=0;
//
//	int i,j;
//	#pragma omp parallel for
//	for (int m=0;m<width*height;m++)
//	{
//		i=m/height;
//		j=(m%height);
//		//	cout<<i<<" "<<j<<endl;
//		if(m_mask.at<double>(j,i)==1)
//		{
//			//for (int k=0;k<dim;k++)
//			//{
//			//	CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//			//}
//			for (int k=0;k<dim;k++)
//			{
//				//CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//				for (int l=0;l<dim;l++)
//				{
//					m_hessian.at<double>(k,l)+=SD_ic[i][j][k]*SD_ic[i][j][l];
//				}
//
//			}
//		}
//	}
//
//
//	//for (int i=0;i<width;i++)
//	//{
//	//	for (int j=0;j<height;j++)
//	//	{
//	//		if(CV_MAT_ELEM(*mask,double,j,i)==1)
//	//		{
//	//			//for (int k=0;k<dim;k++)
//	//			//{
//	//			//	CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//	//			//}
//	//			for (int k=0;k<dim;k++)
//	//			{
//	//				//CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//	//				for (int l=0;l<dim;l++)
//	//				{
//	//					CV_MAT_ELEM(*Hessian,double,k,l)+=SD_ic[i][j][k]*SD_ic[i][j][l];
//	//				}
//	//				
//	//			}
//	//		}
//	//
//	//	}
//	//}
//
//	//double sum;
//
//	//			//for (int k=0;k<dim;k++)
//	//			//{
//	//			//	CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//	//			//}
//	////#pragma omp parallel for reduction(+: sum)  
//	//for (int k=0;k<dim;k++)
//	//{
//	//	//CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//	//	for (int l=0;l<dim;l++)
//	//	{
//	//		sum=0;
//	//		for (int i=0;i<width;i++)
//	//		{
//	//			for (int j=0;j<height;j++)
//	//			{
//	//				if(CV_MAT_ELEM(*mask,double,j,i)==1)
//	//				{
//	//					sum+=SD_ic[i][j][k]*SD_ic[i][j][l];
//	//				}
//
//	//			}
//	//		}
//	//		CV_MAT_ELEM(*Hessian,double,k,l)=sum;
//
//	//	}
//	//}
//
//	//for (int i=0;i<width;i++)
//	//{
//	//	for (int j=0;j<height;j++)
//	//	{
//	//		if(CV_MAT_ELEM(*mask,double,j,i)==1)
//	//		{
//	//			//for (int k=0;k<dim;k++)
//	//			//{
//	//			//	CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//	//			//}
//	//			for (int k=0;k<dim;k++)
//	//			{
//	//				CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
//	//			}
//
//	//			cvMulTransposed(tmpSD,curhessian,1);
//	//			cvAdd(Hessian,curhessian,newHessian);
//	//			cvCopy(newHessian,Hessian);	
//	//		}
//	//	
//	//	}
//	//}
//	//dwStop=GetTickCount();
//
////	cout<<"gethessian  time: "<<(HessianStop-HessianStart)<<endl;
////	dwStart=GetTickCount();
//	cvInv(Hessian,inv_Hessian);

int i,j,k;

///////////////////////////direct version, slowest/////////////////////////////////////
	//m_hessian=0;	
	//#pragma omp parallel for
	//for (int m=0;m<width*height;m++)
	//{
	////	#pragma omp critical
	//	{
	//		i=m/height;
	//		j=(m%height);
	//	}
	//
	//	//	cout<<i<<" "<<j<<endl;
	//	if(m_mask.at<double>(j,i)==1)
	//	{
	//		//for (int k=0;k<dim;k++)
	//		//{
	//		//	CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
	//		//}
	//		for (int k=0;k<dim;k++)
	//		{
	//			//CV_MAT_ELEM(*tmpSD,double,0,k)=SD_ic[i][j][k];
	//			for (int l=0;l<dim;l++)
	//			{
	//				m_hessian.at<double>(k,l)+=SD_ic[i][j][k]*SD_ic[i][j][l];
	//			}

	//		}
	//	}
	//}
	//invert(m_hessian,m_inv_hessian);


	/////////////////////////matrix version/////////////////////////////////
	//Mat  fullSD(1,dim,CV_32FC1);
	//

	//	//#pragma omp parallel for
	//	for (int m=0;m<width*height;m++)
	//	{
	//		//#pragma omp critical
	//		{
	//			i=m/height;
	//			j=(m%height);
	//			for (k=0;k<dim;k++)
	//			{
	//		
	//				fullSD.at<float>(0,k)=SD_ic[i][j][k];
	//			}
	//		}
	//	}
	//





	/////////////CPU version, faster than gpu...//////////////////
	

	//	#pragma omp parallel for 
	//	for (int m=0;m<width*height;m++)
	//	{
	//		//#pragma omp critical
	//		{
	//			i=m/height;
	//			j=(m%height);
	//			for (k=0;k<dim;k++)
	//			{
	//				 fullSD.at<double>(m,k)=SD_ic[i][j][k];
	//			}
	//		}
	//	}
	//	transpose(fullSD,fullSD_tran);
	//full_Hessian=fullSD_tran*fullSD;
	//invert(full_Hessian,m_inv_hessian);

/////////////CPU version, with eliminating 0...fastest//////////////////
//	//DWORD dwStart, dwStop;  
//	int ind=0;
//	//dwStart=GetTickCount();
//	//#pragma omp parallel for 
//	for (int m=0;m<width*height;m++)
//	{
//		//#pragma omp critical
//		{
//			i=m/height;
//			j=(m%height);
//			if(m_mask.at<double>(j,i)==1)
//			{
//				for (k=0;k<dim;k++)
//				{
//					fullSD.at<double>(ind,k)=SD_ic[i][j][k];
//				}
//				ind++;
//			}
//		
//		}
//	}
////	dwStop=GetTickCount();
//	//cout<<"set value time: "<<dwStop-dwStart<<endl;
//
//	//dwStart=GetTickCount();
//	//most time consuming part
//	//fullSD=fullSD.colRange(0,ind);
//	//transpose(fullSD,fullSD_tran);
//	//full_Hessian=fullSD_tran.colRange(0,ind)*fullSD.rowRange(0,ind);
//	//dwStop=GetTickCount();
//	//cout<<"jt*j time: "<<dwStop-dwStart<<endl;
//
//	transpose(fullSD,fullSD_tran);
//	full_Hessian=fullSD_tran*fullSD;
//
//	//add the smooth term
//	if (smoothWeight>0&&currentFrame>startNum)
//	{
//		int i,j;
//		for (i=0;i<meanShape->ptsNum;i++)
//		{
//			for (j=0;j<dim;j++)
//			{
//				fullSD_smooth.at<double>(i,j)=SD_smooth[i][j];
//				//fullSD_smooth.at<double>(meanShape->ptsNum+i,j)=SD_smooth[i][j][1];
//			}
//		}
//		transpose(fullSD_smooth,fullSD_tran_smooth);
//		full_Hessian_smooth=fullSD_tran_smooth*fullSD_smooth;
//		//full_Hessian+=smoothWeight*full_Hessian_smooth;
//		full_Hessian=AAM_weight*full_Hessian+smoothWeight*full_Hessian_smooth;
//	}
//
//	invert(full_Hessian,m_inv_hessian);
//
//
//	fullSD_tran.release();
//	full_Hessian.release();
//////////////////////////////////////////////////////////////

////////////////////CUDA version//////////////////////////////
	if (usingCUDA)
	{
		int ind=0;
		//dwStart=GetTickCount();
		//#pragma omp parallel for 
		for (int m=0;m<width*height;m++)
		{
			//#pragma omp critical
			{
				i=m/height;
				j=(m%height);
	/*			if (i==180&&j==91)
				{
					cout<<i<<" "<<j<<endl;
				}*/
				
				if(m_mask.at<double>(j,i)==1&&imageMask.at<double>(j,i)==1)
				{
					for (k=0;k<dim;k++)
					{
						//fullSD.at<double>(ind,k)=SD_ic[i][j][k];
						cuda_data_cpu[ind]=SD_ic[i][j][k];
						ind++;
					}
					
				}

			}
		}
		//cout<<"ind num: "<<ind<<endl;
		aa(cuda_rows,cuda_cols,cuda_row_pitch,cuda_numF,cuda_data_cpu);//,meanShape->pix_num*(dim));
		for (int i=0;i<dim;i++)
		{
			for (int j=0;j<dim;j++)
			{
				full_Hessian.at<double>(i,j)=cuda_data_cpu[i*dim+j];
			}
		}
		
	/*	cout<<"Output Hessian GPU\n";
		ofstream out1("Hessian_GPU.txt",ios::out);
		for (int m=0;m<full_Hessian.rows;m++)
		{
			for (int n=0;n<full_Hessian.cols;n++)
			{
				out1<<full_Hessian.at<double>(m,n)<<" ";
			}
			out1<<endl;
		}*/

		//ind=0;
		////dwStart=GetTickCount();
		////#pragma omp parallel for 
		//for (int m=0;m<width*height;m++)
		//{
		//	//#pragma omp critical
		//	{
		//		i=m/height;
		//		j=(m%height);
		//		if(m_mask.at<double>(j,i)==1)
		//		{
		//			for (k=0;k<dim;k++)
		//			{
		//				fullSD.at<double>(ind,k)=SD_ic[i][j][k];
		//			}
		//			ind++;
		//		}

		//	}
		//}


		//transpose(fullSD,fullSD_tran);
		//full_Hessian=fullSD_tran*fullSD;
		//ofstream out("Hessian_CPU.txt",ios::out);
		//for (int m=0;m<full_Hessian.rows;m++)
		//{
		//	for (int n=0;n<full_Hessian.cols;n++)
		//	{
		//		out<<full_Hessian.at<double>(m,n)<<" ";
		//	}
		//	out<<endl;
		//}
	}
//////////////////////////////////////////////////////////////
	else
	{
		////////////GPU version//////////
		//#pragma omp parallel for 
		int ind=0;
		//dwStart=GetTickCount();
		//#pragma omp parallel for 

		for (int m=0;m<width*height;m++)
		{
			//#pragma omp critical
			{
				i=m/height;
				j=(m%height);
				if(m_mask.at<double>(j,i)==1)
				{
					for (k=0;k<dim;k++)
					{
						fullSD.at<double>(ind,k)=SD_ic[i][j][k];
					}
					ind++;
				}

			}
		}
		cout<<"total dim: "<<ind*dim<<endl;

		transpose(fullSD,fullSD_tran);
		full_Hessian=fullSD_tran*fullSD;
		/*ofstream out("Hessian_CPU.txt",ios::out);
		for (int m=0;m<full_Hessian.rows;m++)
		{
			for (int n=0;n<full_Hessian.cols;n++)
			{
				out<<full_Hessian.at<double>(m,n)<<" ";
			}
			out<<endl;
		}*/

		//parral CPU version
		/*for (int i=0;i<pNum-1;i++)
		{
		p_fullSD[i]=fullSD.rowRange(Range(i*parallelStep,(i+1)*parallelStep-1));
		transpose(p_fullSD[i],p_fullSD_tran[i]);
		}
		p_fullSD[pNum-1]=fullSD.rowRange(Range((pNum-1)*parallelStep,fullSD.rows-1));
		transpose(p_fullSD[pNum-1],p_fullSD_tran[pNum-1]);


		#pragma omp parallel for 
		for (int i=0;i<pNum;i++)
		{
		p_fullHessian[i]=p_fullSD_tran[i]*p_fullSD[i];
		}
		full_Hessian=0;
		for (int i=0;i<pNum;i++)
		{
		full_Hessian+=p_fullHessian[i];
		}

		for (int i=0;i<pNum;i++)
		{
		p_fullSD[i].release();
		p_fullSD_tran[i].release();
		p_fullHessian[i].release();
		}*/
		///////////////////////////////////////////////////////////////////

		/////////////////GPU//////////////////////

		//fullSD_gpu=fullSD;
		//transpose(fullSD_gpu,fullSD_tran_GPU);
		//gpu_Hessian=fullSD_tran_GPU*fullSD_gpu;
		//full_Hessian=Mat(gpu_Hessian);
		//////////////////////////////////////////

		/*ofstream out1("Hessian_GPU.txt",ios::out);
		for (int m=0;m<full_Hessian.rows;m++)
		{
		for (int n=0;n<full_Hessian.cols;n++)
		{
		out1<<full_Hessian.at<double>(m,n)<<" ";
		}
		out1<<endl;
		}*/
	}

	/*cout<<"AAM hessian\n";
	for (int i=0;i<full_Hessian.rows;i++)
	{
		for (int j=0;j<full_Hessian.cols;j++)
		{
			cout<<full_Hessian.at<double>(i,j)<<" ";
		}
		break;
	}*/

 	full_Hessian*=AAM_weight;
	//ofstream out("F:\\Projects\\Facial feature points detection\\Matlab code\\Hessian.txt",ios::out);
	//for (int i=0;i<full_Hessian.rows;i++)
	//{
	//	for (int j=0;j<full_Hessian.cols;j++)
	//	{
	//		out<<full_Hessian.at<double>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();

	//add the smooth term
	if (smoothWeight>0)
	{
		//if (!isSDDefined)
		//{
		//	//cout<<"trees defined:"<<trees->numOfLabels<<" "<<dim<<endl;
		//	fullSD_detection.create(trees->numOfLabels,dim,CV_64FC1);
		//	fullSD_tran_detection.create(dim,trees->numOfLabels,CV_64FC1);
		//	full_Hessian_detection.create(dim,dim,CV_64FC1);
		//	isSDDefined=true;
		//}
		//int i,j;
		//for (i=0;i<trees->numOfLabels;i++)
		//{
		//	for (j=0;j<dim;j++)
		//	{
		//		fullSD_detection.at<double>(i,j)=SD_detection[i][0][j];
		//	}
		//}
		//transpose(fullSD_detection,fullSD_tran_detection);
		//full_Hessian_detection=fullSD_tran_detection*fullSD_detection;

		if (!isSDDefined)
		{
			//cout<<"trees defined:"<<trees->numOfLabels<<" "<<dim<<endl;
			fullSD_detection.create(2,dim,CV_64FC1);
			fullSD_tran_detection.create(dim,2,CV_64FC1);
			full_Hessian_detection.create(dim,dim,CV_64FC1);
			isSDDefined=true;
		}
		full_Hessian_detection=0;

		//Mat tmp=Mat::zeros(dim,trees->numOfLabels*2,CV_64FC1);
		////Mat tmpConv=Mat::zeros(trees->numOfLabels*2,trees->numOfLabels*2,CV_64FC1);
		//Mat tmpJ=Mat::zeros(dim,trees->numOfLabels*2,CV_64FC1);
		int i,j;
		for (i=0;i<trees->numOfLabels;i++)
		{
			for (j=0;j<dim;j++)
			{
				fullSD_detection.at<double>(0,j)=SD_detection[i][0][j];
				fullSD_detection.at<double>(1,j)=SD_detection[i][1][j];
			}
			transpose(fullSD_detection,fullSD_tran_detection);
			
			//full_Hessian_detection+=fullSD_tran_detection*prob_conv[i]*fullSD_detection;
			for (j=0;j<candidatePoints[i].size();j++)
			{
				full_Hessian_detection+=fullSD_tran_detection*prob_conv_candidates[i][j]*fullSD_detection;
			}
			

			//Mat lala=tmp(Range(i*2,i*2+2),Range(i*2,i*2+2));
			//cout<<lala.rows<<" "<<lala.cols<<endl;
			//for (j=0;j<dim;j++)
			//{
			//	tmpJ.at<double>(j,2*i)=SD_detection[i][0][j];
			//	tmpJ.at<double>(j,2*i+1)=SD_detection[i][1][j];
			//	//fullSD_detection.at<double>(0,j)=
			//	//fullSD_detection.at<double>(1,j)=SD_detection[i][1][j];
			//}
			//tmp.colRange(Range(i*2,i*2+2))+=fullSD_tran_detection*prob_conv[i];
			//tmpConv(Range(i*2,i*2+2),Range(i*2,i*2+2))+=prob_conv[i];
		}

		/*cout<<"Conv\n";
		for (int i=0;i<tmp.rows;i++)
		{
			for (int j=0;j<tmp.cols;j++)
			{
				cout<<tmpConv.at<double>(i,j)<<" ";
			}
			cout<<endl;
		}
*/
		//Mat result=tmpJ*tmpConv;

	/*	cout<<"J\n";
		for (int i=0;i<tmpJ.rows;i++)
		{
			if (i<shape_dim)
			{
				continue;
			}
			for (int j=0;j<tmpJ.cols;j++)
			{
				cout<<tmpJ.at<double>(i,j)<<" ";
			}
			cout<<endl;

		}*/

		//cout<<"JConv _intotal\n";
		//for (int i=0;i<result.rows;i++)
		//{
		////	cout<<endl;
		//	for (int j=0;j<result.cols;j++)
		//	{
		//		cout<<result.at<double>(i,j)<<" ";
		//	}
		//	cout<<endl;
		////	break;
		//}

		//cout<<"JConv _intotal_conv\n";
		//for (int i=0;i<result.cols;i++)
		//{
		//	//	cout<<endl;
		//	for (int j=0;j<result.rows;j++)
		//	{
		//		cout<<result.at<double>(j,i)<<" ";
		//	}
		//	cout<<endl;
		////	break;
		//}

		//cout<<"JConv\n";
		//for (int i=0;i<tmp.rows;i++)
		//{
		//	cout<<endl;
		//	for (int j=0;j<tmp.cols;j++)
		//	{
		//		cout<<tmp.at<double>(i,j)<<" ";
		//	}
		//	cout<<endl;
		//	//break;
		//}

		//cout<<"RT hessian\n";
		//for (int i=0;i<full_Hessian.rows;i++)
		//{
		//	for (int j=0;j<full_Hessian.cols;j++)
		//	{
		//		cout<<full_Hessian_detection.at<double>(i,j)<<" ";
		//	}
		//	cout<<endl;
		//}
		
		


	/*	cout<<"Hessian\n";
		for (i=0;i<full_Hessian_detection.rows;i++)
		{
			for (j=0;j<full_Hessian_detection.cols;j++)
			{
				cout<<full_Hessian_detection.at<double>(i,j)<<" ";
			}
			cout<<endl;
		}*/
		//full_Hessian+=smoothWeight*full_Hessian_smooth;
		full_Hessian+=smoothWeight*full_Hessian_detection;

		
	/*	cout<<"Hessian\n";
		for (i=0;i<full_Hessian.rows;i++)
		{
			for (j=0;j<full_Hessian.cols;j++)
			{
				cout<<full_Hessian.at<double>(i,j)<<" ";
			}
			cout<<endl;
		}*/
	}
	
	if (priorWeight>0)
	{
		full_Hessian+=priorWeight*priorSigma;
	}

	if (localPCAWeight>0)
	{
		full_Hessian+=localPCAWeight*m_localHessian;
	}

	//ofstream out1("F:\\Projects\\Facial feature points detection\\Matlab code\\Hessian_withDet.txt",ios::out);
	//for (int i=0;i<full_Hessian.rows;i++)
	//{
	//	for (int j=0;j<full_Hessian.cols;j++)
	//	{
	//		out1<<full_Hessian.at<double>(i,j)<<" ";
	//	}
	//	out1<<endl;
	//}
	//out1.close();

	////add the smooth term
	//if (smoothWeight>0&&currentFrame>startNum)
	//{
	//	int i,j;
	//	for (i=0;i<meanShape->ptsNum;i++)
	//	{
	//		for (j=0;j<dim;j++)
	//		{
	//			fullSD_smooth.at<double>(i,j)=SD_smooth[i][j];
	//			//fullSD_smooth.at<double>(meanShape->ptsNum+i,j)=SD_smooth[i][j][1];
	//		}
	//	}
	//	transpose(fullSD_smooth,fullSD_tran_smooth);
	//	full_Hessian_smooth=fullSD_tran_smooth*fullSD_smooth;
	//	//full_Hessian+=smoothWeight*full_Hessian_smooth;
	//	full_Hessian=AAM_weight*full_Hessian+smoothWeight*full_Hessian_smooth;
	//}

	invert(full_Hessian,m_inv_hessian);
	//invert(Mat(gpu_Hessian),m_inv_hessian);
	fullSD_tran.release();
	/////////////////////////////////////////////

//	cout<<"Hessian inverse  time: "<<(HessianStop-HessianStart)<<endl;

	//cout<<"hessian\n";
	//for (int i=0;i<dim;i++)
	//{
	//	for (int j=0;j<dim;j++)
	//	{
	//		cout<<CV_MAT_ELEM(*Hessian,double,i,j)<<" ";
	//	}
	//	cout<<endl;
	//}
	//for (int i=0;i<dim;i++)
	//{
	//	for (int j=0;j<dim;j++)
	//	{
	//		cout<<CV_MAT_ELEM(*inv_Hessian,double,i,j)<<" ";
	//	}
	//	cout<<endl;
	//}
	
}

void AAM_RealGlobal_GPU::loadResult(string name)
{
	SL_Basis SL_engine;
	ifstream in(name.c_str(),ios::in);
	in>>shape_dim>>texture_dim;
	s_vec=SL_engine.loadMatrix(in,s_vec);
	t_vec=SL_engine.loadMatrix(in,t_vec);
	meanShape=new Shape();
	meanShape->load(in);
	meanTexture=new Texture();
	meanTexture->load(in);
	s_mean=SL_engine.loadMatrix(in,s_mean);
	t_mean=SL_engine.loadMatrix(in,t_mean);
	s_value=SL_engine.loadMatrix(in,s_value);
	t_value=SL_engine.loadMatrix(in,t_value);
	in>>texture_scale>>shape_scale;
	triangleList=SL_engine.loadMatrix(in,triangleList);
	triangleIndList=SL_engine.loadMatrix(in,triangleIndList);
	listNum=SL_engine.loadMatrix(in,listNum);
	in>>isGlobaltransform;

	//	meanShape->getMask(triangleList);

}

void AAM_RealGlobal_GPU::getAllNeededData(string name)
{
	cout<<"loading model...\n";
	dataDir=name.substr(0,name.find_last_of('\\'));
	loadResult(name);
	pix_num=meanShape->pix_num;
	nband=meanTexture->nband;
	pts_Index=meanShape->pts_Index;
	inv_mask=meanShape->inv_mask;
	mask_withindex=meanShape->mask_withindex;
	affineTable=meanShape->affineTable;
	//	affineTable_strong=meanShape->affineTable_strong;
	//	meantexture_real=trainedResult->meantexure_real;


	//define the needed variables
	mask=meanShape->mask;
	shapeWidth=meanShape->width;
	shapeHeight=meanShape->height;
	width=meanShape->width;
	height=meanShape->height;
	WarpedInput=cvCreateImage(cvSize(width,height),meanTexture->depth,meanTexture->nband);
	errorImageMat=cvCreateMat(1,pix_num*nband,CV_64FC1);
	WarpedInput_mat=cvCreateMat(1,pix_num*nband,CV_64FC1);
	Template_mat=cvCreateMat(1,pix_num*nband,CV_64FC1);
	cvCopy(meanTexture->imgData,Template_mat);

	s_weight=new double[shape_dim];
	//last_Iteration_weight=new double[shape_dim+texture_dim+4];
	s_weight_vec=new double[shape_dim];
	t_weight=new double[texture_dim];

	textures=new Texture*[texture_dim];
	for (int i=0;i<texture_dim;i++)
	{
		textures[i]=new Texture();
		*textures[i]=(*meanTexture);
		for (int j=0;j<pix_num;j++)
		{
			CV_MAT_ELEM(*textures[i]->imgData,double,0,j)=
				CV_MAT_ELEM(*t_vec,double,i,j);
		}
	}

	//set up the incremental PCA model
	textureEigenValues.create(texture_dim,1,CV_64FC1);
	string lastName=name.substr(name.find_last_of('\\')+1,name.length()-name.find_last_of('\\')-1);
	lastName=lastName.substr(lastName.find_first_of('_'),lastName.length()-lastName.find_first_of('_'));
	lastName=dataDir+"\\textureEigenValue"+lastName;

	ifstream in2(lastName.c_str(),ios::in);
	if (in2)
	{
		for (int i=0;i<texture_dim;i++)
		{
			in2>>textureEigenValues.at<double>(i,0);
		}
	}

	string listName=dataDir+"\\ptsList.txt";
	ifstream inn(listName.c_str(),ios::in);
	inn>>subjNum;
	inn.close();
	textureModel=new Adp_PCA_float(t_vec->cols,texture_dim,false,BlockNumGlobal);
	textureModel->setModel(cvarrToMat(t_vec),textureEigenValues,cvarrToMat(t_mean),subjNum,texture_dim);

	adpPcaGlobal=textureModel;


	//get the prior Sigma
	//shape_dim=shape_dim-4 due to the abundent global transformations
	string sigmaName=dataDir+"\\inv_Sigma.txt";
	priorSigma=Mat::zeros(shape_dim+texture_dim+4-4,shape_dim+texture_dim+4-4,CV_64FC1);


	double readin_tmp;
	ifstream sigmaIn(sigmaName.c_str(),ios::in);
	if (sigmaIn)
	{
		for (int i=0;i<shape_dim-4;i++)
		{
			for (int j=0;j<shape_dim-4;j++)
			{
				/*sigmaIn>>readin_tmp;
				priorSigma.at<double>(i,j)=readin_tmp;*/

				sigmaIn>>priorSigma.at<double>(i,j);
			}
		}
	}

	//for (int i=0;i<shape_dim;i++)
	//{
	//	for (int j=0;j<shape_dim;j++)
	//	{
	//		cout<<priorSigma.at<double>(i,j)<<" ";
	//	}
	//	cout<<endl;
	//}

	
	string pmName=dataDir+"\\priorMean.txt";
	priorMean=new double [shape_dim+texture_dim+4-4];
	ifstream priorMeanIn(pmName.c_str(),ios::in);
	if(priorMeanIn)
	{
		for (int i=0;i<shape_dim-4;i++)
		{
			priorMeanIn>>priorMean[i];
		}
	}

	m_localHessian=Mat::zeros(shape_dim+texture_dim,shape_dim+texture_dim,CV_64FC1);

	g_hes.resize(shape_dim+texture_dim,shape_dim+texture_dim);
	g_hes_inv.resize(shape_dim+texture_dim,shape_dim+texture_dim);
	g_b.resize(shape_dim+texture_dim);
	//if (smoothWeight>0)
	//{
	//	parametersLast=new double[shape_dim+texture_dim+4];
	//	for (int i=0;i<shape_dim+texture_dim+4;i++)
	//	{
	//		parametersLast[i]=0;
	//	}
	//	parametersLast[shape_dim+texture_dim]=0;
	//	parametersLast[shape_dim+texture_dim+1]=1;
	//	parametersLast[shape_dim+texture_dim+2]=0;
	//	parametersLast[shape_dim+texture_dim+3]=0;
	////	parametersLast[shape_dim+texture_dim]=
	//}
	cout << "done." << endl;
}

void AAM_RealGlobal_GPU::setSmoothnessWeight(double _smoothWeight)
{
	smoothWeight=_smoothWeight;
	smoothWeight_backup=_smoothWeight;
}

void AAM_RealGlobal_GPU::setHostData(float*data ,Mat &m,int style=0)
{
	int i,j;
	int realRows;
	if (style==0)
	{
		realRows=m.rows;
	}
	else
	{
		realRows=style;
	}
	for (i=0;i<realRows;i++)
	{
		for (j=0;j<m.cols;j++)
		{
			data[i*m.cols+j]=m.at<double>(i,j);
		}
	}
}

void AAM_RealGlobal_GPU::preProcess_GPU()
{
	int localMCN=MAX_COUNT_NUM;
	int localMPN=MAX_PIXEL_NUM;
	int localMPD=MAX_POINT_DIM;

	float *host_shapeJacobians=new float[localMPN*(shape_dim)*2];
	float *host_s_vec=new float[localMCN];
	float *host_t_vec=new float[texture_dim*localMPN];
	float *host_s_mean=new float[localMPD];
	float *host_t_mean=new float[localMPN];
	float *host_warpTabel=new float[localMPN*3];
	float *host_triangle_indexTabel=new float[localMPN*3];
	float *host_MaskTabel=new float[meanShape->width*meanShape->height];

	setHostData(host_s_vec,m_s_vec,shape_dim);
	setHostData(host_t_vec,m_t_vec,texture_dim);
	setHostData(host_MaskTabel,cvarrToMat(meanShape->mask_withindex));
	
	int i,j;

	//for (i=0;i<m_s_vec.rows;i++)
	//{
	//	for (j=0;j<m_s_vec.cols;j++)
	//	{
	//		cout<<host_s_vec[i*m_s_vec.cols+j]<<" "<<m_s_vec.at<double>(i,j)<<endl;
	//	}
	//}





	for (i=0;i<meanShape->ptsNum*2;i++)
	{
		host_s_mean[i]=meanShape->ptsForMatlab[i];
	}
	setHostData(host_t_mean,cvarrToMat(meanTexture->imgData));
	
	
	int tHeight=meanShape->height;
	int tWidth=meanShape->width;
	int cNum;
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			cNum=i*tWidth+j;
			host_warpTabel[3*cNum+0]=0;
			host_warpTabel[3*cNum+1]=0;
			host_warpTabel[3*cNum+2]=0;
			host_triangle_indexTabel[3*cNum+0]=-1;
			host_triangle_indexTabel[3*cNum+1]=-1;
			host_triangle_indexTabel[3*cNum+2]=-1;
		}
	}

	for (i=0;i<pix_num;i++)
	{
		cNum=meanShape->inv_mask[i][1]*tWidth+meanShape->inv_mask[i][0];
		//cout<<cNum<<endl;
		for (j=0;j<3;j++)
		{
			host_warpTabel[cNum*3+j]=meanShape->weightTabel[i][j];
			host_triangle_indexTabel[cNum*3+j]=indexTabel[j][i];
		}
	}

	//jacobian
	int k;
	//int allDim=shape_dim+texture_dim+4;
	
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			if (CV_MAT_ELEM(*mask,double,i,j)==0)
			{
				for (k=0;k<shape_dim;k++)
				{
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k]=0;
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k+shape_dim]=0;
				}
			}
			else
			{
				for (k=0;k<shape_dim;k++)
				{
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k]=full_Jacobian[j][i][0][k];
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k+shape_dim]=full_Jacobian[j][i][1][k];
				}
				
			}
			
		}
	}


	//foward index tabel
	/*float *host_fowardIndexTabel=new float[meanShape->width*meanShape->height];
	Mat m_mask_withIndex=cvarrToMat(meanShape->mask_withindex);
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			host_fowardIndexTabel[i*tWidth+j]=m_mask_withIndex.at<double>(i,j);
		}
	}*/
	

	setData_Preprocess(host_s_vec,host_t_vec,host_s_mean,host_t_mean,host_warpTabel,
		host_triangle_indexTabel,shape_dim,texture_dim,meanShape->ptsNum,pix_num,
		meanShape->width,meanShape->height,host_shapeJacobians,host_MaskTabel,host_MaskTabel,showSingleStep);
	
		
	//delete
	delete []host_s_vec;
	delete []host_t_vec;
	delete []host_s_mean;
	delete []host_t_mean;
	delete []host_warpTabel;
	delete []host_triangle_indexTabel;
	delete []host_shapeJacobians;
	delete []host_MaskTabel;
	//delete []host_fowardIndexTabel;

	int MPD=MAX_POINT_DIM;
	int MPN=MAX_PIXEL_NUM;
	parameters=new float[MPD];
	inputImg=new float[MPN];

	g_Hessian.create(shape_dim+texture_dim+4,shape_dim+texture_dim+4,CV_64FC1);
	g_inv_Hessian.create(shape_dim+texture_dim+4,shape_dim+texture_dim+4,CV_64FC1);


	if (usingGPU)
	{
		g_meanShape=new float[meanShape->ptsNum*2];
		for (int i=0;i<meanShape->ptsNum*2;i++)
		{
			g_meanShape[i]=meanShape->ptsForMatlab[i];
		}
		g_s_vec=m_s_vec.clone();			
	}
	/*cu_gradientX=new float[meanShape->width*meanShape->height];
	cu_gradientY=new float[meanShape->width*meanShape->height];*/
}


void AAM_RealGlobal_GPU::preProcess_GPU_combination()
{
	int localMCN=MAX_COUNT_NUM;
	int localMPN=MAX_PIXEL_NUM;
	int localMPD=MAX_POINT_DIM;

	float *host_shapeJacobians=new float[localMPN*(shape_dim)*2];
	float *host_s_vec=new float[localMCN];
	float *host_t_vec=new float[texture_dim*localMPN];
	float *host_s_mean=new float[localMPD];
	float *host_t_mean=new float[localMPN];
	float *host_warpTabel=new float[localMPN*3];
	float *host_triangle_indexTabel=new float[localMPN*3];
	float *host_MaskTabel=new float[meanShape->width*meanShape->height];

	setHostData(host_s_vec,m_s_vec,shape_dim);
	setHostData(host_t_vec,m_t_vec,texture_dim);
	setHostData(host_MaskTabel,cvarrToMat(meanShape->mask_withindex));
	
	int i,j;

	//for (i=0;i<m_s_vec.rows;i++)
	//{
	//	for (j=0;j<m_s_vec.cols;j++)
	//	{
	//		cout<<host_s_vec[i*m_s_vec.cols+j]<<" "<<m_s_vec.at<double>(i,j)<<endl;
	//	}
	//}





	//for (i=0;i<meanShape->ptsNum*2;i++)
	//{
	//	host_s_mean[i]=meanShape->ptsForMatlab[i];
	//}
	for (int i=0;i<meanShape->ptsNum;i++)
	{
		host_s_mean[i]=meanShape->ptsForMatlab[i]-meanShapeCenter[0];
		host_s_mean[i+meanShape->ptsNum]=meanShape->ptsForMatlab[i+meanShape->ptsNum]-meanShapeCenter[1];
	}
	setHostData(host_t_mean,cvarrToMat(meanTexture->imgData));
	
	
	int tHeight=meanShape->height;
	int tWidth=meanShape->width;
	int cNum;
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			cNum=i*tWidth+j;
			host_warpTabel[3*cNum+0]=0;
			host_warpTabel[3*cNum+1]=0;
			host_warpTabel[3*cNum+2]=0;
			host_triangle_indexTabel[3*cNum+0]=-1;
			host_triangle_indexTabel[3*cNum+1]=-1;
			host_triangle_indexTabel[3*cNum+2]=-1;
		}
	}

	for (i=0;i<pix_num;i++)
	{
		cNum=meanShape->inv_mask[i][1]*tWidth+meanShape->inv_mask[i][0];
		//cout<<cNum<<endl;
		for (j=0;j<3;j++)
		{
			host_warpTabel[cNum*3+j]=meanShape->weightTabel[i][j];
			host_triangle_indexTabel[cNum*3+j]=indexTabel[j][i];
		}
	}

	//jacobian
	int k;
	//int allDim=shape_dim+texture_dim+4;
	
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			if (CV_MAT_ELEM(*mask,double,i,j)==0)
			{
				for (k=0;k<shape_dim;k++)
				{
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k]=0;
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k+shape_dim]=0;
				}
			}
			else
			{
				for (k=0;k<shape_dim;k++)
				{
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k]=full_Jacobian[j][i][0][k];
					host_shapeJacobians[(i*tWidth+j)*shape_dim*2+k+shape_dim]=full_Jacobian[j][i][1][k];
				}
				
			}
			
		}
	}


	//foward index tabel
	/*float *host_fowardIndexTabel=new float[meanShape->width*meanShape->height];
	Mat m_mask_withIndex=cvarrToMat(meanShape->mask_withindex);
	for (i=0;i<tHeight;i++)
	{
		for (j=0;j<tWidth;j++)
		{
			host_fowardIndexTabel[i*tWidth+j]=m_mask_withIndex.at<double>(i,j);
		}
	}*/
	

	setData_AAM_combination(host_s_vec,host_t_vec,host_s_mean,host_t_mean,host_warpTabel,
		host_triangle_indexTabel,shape_dim,texture_dim,meanShape->ptsNum,pix_num,
		meanShape->width,meanShape->height,host_shapeJacobians,host_MaskTabel,host_MaskTabel,showSingleStep,
		isAdaptive,textureModel->dataForUse.data());
	
		
	//delete
	delete []host_s_vec;
	delete []host_t_vec;
	delete []host_s_mean;
	delete []host_t_mean;
	delete []host_warpTabel;
	delete []host_triangle_indexTabel;
	delete []host_shapeJacobians;
	delete []host_MaskTabel;
	//delete []host_fowardIndexTabel;

	int MPD=MAX_POINT_DIM;
	int MPN=MAX_PIXEL_NUM;
	parameters=new float[MPD];
	inputImg=new float[MPN];

	g_Hessian.create(shape_dim+texture_dim+4,shape_dim+texture_dim+4,CV_64FC1);
	g_inv_Hessian.create(shape_dim+texture_dim+4,shape_dim+texture_dim+4,CV_64FC1);


	///need to be deleted if we desire speed
	if (1)
	{
		g_meanShape=new float[meanShape->ptsNum*2];
		for (int i=0;i<meanShape->ptsNum*2;i++)
		{
			g_meanShape[i]=meanShape->ptsForMatlab[i];
		}

		for (int i=0;i<meanShape->ptsNum;i++)
		{
			g_meanShape[i]=meanShape->ptsForMatlab[i]-meanShapeCenter[0];
			g_meanShape[i+meanShape->ptsNum]=meanShape->ptsForMatlab[i+meanShape->ptsNum]-meanShapeCenter[1];
		}
		g_s_vec=m_s_vec.clone();			
	}
	/*cu_gradientX=new float[meanShape->width*meanShape->height];
	cu_gradientY=new float[meanShape->width*meanShape->height];*/
}

void AAM_RealGlobal_GPU::calculateMandC_autoSized(Mat &prob_map,double dx, double dy, int gridSize,double *mu,Mat &conv)
{
	int i,j;
	int step=gridSize/2;
	int x=dx;int y=dy;//currently, set x=ind(dx) directly
	//search for the best window size. using the eigen value is not a good measure. we now seek for modes in global
	int bestSize;float bestR=1000;

	float lastDx,lastDy;

	bestSize=20;
	lastDx=10000;
	lastDy=10000;
	for(int currentsize=10;currentsize<70;currentsize+=10)
	{
		gridSize=currentsize;
		step=gridSize/2;
		vector<float> x_list,y_list; 
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				//only calculate the position with large value
				if(prob_map.at<double>(i,j)>0.3)
				{
					x_list.push_back(j);
					y_list.push_back(i);
				}
			}
		}

		//if no probability, the enlarge the size without doubt
		if (x_list.size()==0)
		{
			continue;
		}
		
		//calculate std
		Mat cord_vec=Mat::zeros(x_list.size(),2,CV_64FC1);
		for (int i=0;i<cord_vec.rows;i++)
		{
			cord_vec.at<double>(i,0)=x_list[i];
			cord_vec.at<double>(i,1)=y_list[i];
		}
		Scalar meanX,meanY;
		Scalar stddevX,stddevY;
		meanStdDev(cord_vec.col(0),meanX,stddevX);
		meanStdDev(cord_vec.col(1),meanY,stddevY);

		//double stdXV=0;
		//for (int i=0;i<cord_vec.rows;i++)
		//{
		//	stdXV+=(cord_vec.at<double>(i,0)-meanX.val[0])*(cord_vec.at<double>(i,0)-meanX.val[0]);
		//}
		//cout<<"my calculation: "<<sqrtf(stdXV/cord_vec.rows)<<" "<<sqrtf(stdXV)/cord_vec.rows<<
		//	" "<<stddevX.val[0]<<endl;
		
		//meanX.val[0];
		//meanY.val[0];
		//stddevX.val[0]/=(cord_vec.rows);
		//stddevY.val[0]/=(cord_vec.rows);
		//stddevX.val[0]/=sqrtf(cord_vec.rows);
		//stddevY.val[0]/=sqrtf(cord_vec.rows);
	
		if (stddevX.val[0]<lastDx*1.5f&&stddevY.val[0]<lastDy*1.5f)

		//	stddevX.val[0]/=sqrtf(cord_vec.rows);
		//	stddevY.val[0]/=sqrtf(cord_vec.rows);
	//	if (stddevX.val[0]<lastDx&&stddevY.val[0]<lastDy)
		{
			lastDx=stddevX.val[0];
			lastDy=stddevY.val[0];
			bestSize=currentsize;
		}
		//else
			//break;
		//cout<<currentsize<<" "<<meanX.val[0]<<" "<<stddevX.val[0]<<" "<<
		//	meanY.val[0]<<" "<<stddevY.val[0]<<endl;

		//namedWindow("Prob Map");
		//Mat tmp=prob_map.clone()*0;
		//tmp(Range(y-step,y+step),Range(x-step,x+step))+=prob_map(Range(y-step,y+step),Range(x-step,x+step));

		////////////////////////draw the circle/////////////////////////////
		//Point center;

		//center.x=mu[0];
		//center.y=mu[1];
		//circle(tmp,center,2,255);
		//imshow("Prob Map",tmp);
		//waitKey();


		////calculate alpha
		//double **alpha=new double *[gridSize+1];
		//for (i=0;i<gridSize+1;i++)
		//{
		//	alpha[i]=new double[gridSize+1];
		//}

		//if (sumofProb>0)
		//{
		//	for (i=y-step;i<=y+step;i++)	//go over the local window
		//	{
		//		for (j=x-step;j<=x+step;j++)
		//		{
		//			alpha[i-y+step][j-x+step]=prob_map.at<double>(i,j)/sumofProb;
		//		}
		//	}
		//}
		//else
		//{
		//	for (i=y-step;i<=y+step;i++)	//go over the local window
		//	{
		//		for (j=x-step;j<=x+step;j++)
		//		{
		//			alpha[i-y+step][j-x+step]=1.0f/(float)((gridSize+1)*(gridSize+1));
		//		}
		//	}
		//}

		////then calculate mu
		//double mux,muy;
		////	mux=x-step+sumx;
		////	muy=y-step+sumy;
		////cout<<mux<<" "<<muy<<endl;
		//mux=muy=0;
		//for (i=y-step;i<=y+step;i++)
		//{
		//	for (j=x-step;j<=x+step;j++)
		//	{
		//		mux+=alpha[i-y+step][j-x+step]*j;
		//		muy+=alpha[i-y+step][j-x+step]*i;
		//		//cout<<mux<<" "<<muy<<endl;
		//	}
		//	/*	if (i%20==0)
		//	{
		//	cout<<x<<" "<<y<<" "<<mux<<" "<<muy<<endl;
		//	}*/

		//}
		//mu[0]=mux;mu[1]=muy;

		////ofstream out("G:\\face database\\kinect data\\exp results\\currentProb.txt",ios::out);
		////for (i=y-step;i<=y+step;i++)	//go over the local window
		////{
		////	for (j=x-step;j<=x+step;j++)
		////	{
		////		out<<prob_map.at<double>(i,j)<<" ";
		////	}
		////	out<<endl;
		////}
		////out.close();

		////cout<<mux<<" "<<muy<<endl;


		//conv*=0;
		//if (sumofProb>0)
		//{
		//	for (i=y-step;i<=y+step;i++)	//go over the local window
		//	{
		//		for (j=x-step;j<=x+step;j++)
		//		{
		//			conv.at<double>(0,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(j-mux);
		//			conv.at<double>(0,1)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
		//			conv.at<double>(1,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
		//			conv.at<double>(1,1)+=alpha[i-y+step][j-x+step]*(i-muy)*(i-muy);
		//		}
		//	}
		//}
		//else
		//{
		//	conv.at<double>(0,0)=1;
		//	conv.at<double>(0,1)=0;
		//	conv.at<double>(1,0)=0;
		//	conv.at<double>(1,1)=1;
		//}
	


		////calculate (r+1)^2/r
		//double r=(conv.at<double>(0,0)+conv.at<double>(1,1))*(conv.at<double>(0,0)+conv.at<double>(1,1))/
		//	(conv.at<double>(0,0)*conv.at<double>(1,1)-conv.at<double>(0,1)*conv.at<double>(1,0));
		//cout<<"window size: "<<gridSize<<" ratio is: "<<r<<" mux"<<mux<<" muy"<<muy<<endl;

		//namedWindow("Prob Map");
		//Mat tmp=prob_map.clone()*0;
		//tmp(Range(y-step,y+step),Range(x-step,x+step))+=prob_map(Range(y-step,y+step),Range(x-step,x+step));

		////////////////////////draw the circle/////////////////////////////
		//Point center;

		//center.x=mu[0];
		//center.y=mu[1];
		//circle(tmp,center,2,255);
		//imshow("Prob Map",tmp);
		//waitKey();

		//if (r<bestR)
		//{
		//	bestR=r;
		//	bestSize=currentsize;
		//	cout<<"updated!! window size: "<<gridSize<<" ratio is: "<<r<<endl;
		//}
	}

	//cout<<"bestSize: "<<bestSize<<endl;

	//namedWindow("Prob Map");
	//Mat tmp=prob_map.clone()*0;
	//tmp(Range(y-bestSize/2,y+bestSize/2),Range(x-bestSize/2,x+bestSize/2))+=
	//	prob_map(Range(y-bestSize/2,y+bestSize/2),Range(x-bestSize/2,x+bestSize/2));

	////////////////////////draw the circle/////////////////////////////
	//Point center;

	//center.x=mu[0];
	//center.y=mu[1];
	//circle(tmp,center,2,255);
	//imshow("Prob Map",tmp);
	//waitKey();
	
	calculateMandC(prob_map,dx,dy,bestSize,mu,conv);
}

//gridSize: the size of the local window, firstly set to around 32
//output: mu and convaience matrix
void AAM_RealGlobal_GPU::calculateMandC(Mat &prob_map,double dx, double dy, int gridSize,double *mu,Mat &conv)
{
	int i,j;
	int step=gridSize/2;

	int x=dx;int y=dy;//currently, set x=ind(dx) directly
	//cout<<x<<" "<<y<<endl;

	/////////////////check the prob_map and position////////////////////
	/*namedWindow("Prob Map");
	Mat tmp=prob_map.clone()*0;
	tmp(Range(y-step,y+step),Range(x-step,x+step))+=prob_map(Range(y-step,y+step),Range(x-step,x+step));*/

	////////////////////////draw the circle/////////////////////////////
	//Point center;
	//
	//	center.x=x;
	//	center.y=y;
	//	circle(tmp,center,2,255);
	//	imshow("Prob Map",tmp);
	//waitKey();
	////////////////////////////////////////////////////////////////////





	double sumofProb=0;
	for (i=y-step;i<=y+step;i++)	//go over the local window
	{
		for (j=x-step;j<=x+step;j++)
		{
			sumofProb+=prob_map.at<double>(i,j);
		}
	}

	//calculate alpha
	double **alpha=new double *[gridSize+1];
	for (i=0;i<gridSize+1;i++)
	{
		alpha[i]=new double[gridSize+1];
	}

	if (sumofProb>0)
	{
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				alpha[i-y+step][j-x+step]=prob_map.at<double>(i,j)/sumofProb;
			}
		}
	}
	else
	{
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				alpha[i-y+step][j-x+step]=1.0f/(float)((gridSize+1)*(gridSize+1));
			}
		}
	}
	

	//double sumx=0;
	//double sumy=0;
	//for (i=0;i<2*step+1;i++)
	//{
	//	for (j=0;j<2*step+1;j++)
	//	{
	//	//	cout<<i<<" "<<j<<endl;
	//		sumx+=alpha[i][j]*j;
	//		sumy+=alpha[i][j]*i;
	//	}
	//	//cout<<x<<" "<<y<<" "<<mux<<" "<<muy<<endl;
	//}
	//cout<<"total sumx "<<sumx<<" total sumy"<<sumy<<endl;

	//then calculate mu
	double mux,muy;
//	mux=x-step+sumx;
//	muy=y-step+sumy;
	//cout<<mux<<" "<<muy<<endl;
	mux=muy=0;
	for (i=y-step;i<=y+step;i++)
	{
		for (j=x-step;j<=x+step;j++)
		{
			mux+=alpha[i-y+step][j-x+step]*j;
			muy+=alpha[i-y+step][j-x+step]*i;
			//cout<<mux<<" "<<muy<<endl;
		}
	/*	if (i%20==0)
		{
			cout<<x<<" "<<y<<" "<<mux<<" "<<muy<<endl;
		}*/
		
	}
	mu[0]=mux;mu[1]=muy;

	//ofstream out("G:\\face database\\kinect data\\exp results\\currentProb.txt",ios::out);
	//for (i=y-step;i<=y+step;i++)	//go over the local window
	//{
	//	for (j=x-step;j<=x+step;j++)
	//	{
	//		out<<prob_map.at<double>(i,j)<<" ";
	//	}
	//	out<<endl;
	//}
	//out.close();

	//cout<<mux<<" "<<muy<<endl;


	conv*=0;
	if (sumofProb>0)
	{
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				conv.at<double>(0,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(j-mux);
				conv.at<double>(0,1)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
				conv.at<double>(1,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
				conv.at<double>(1,1)+=alpha[i-y+step][j-x+step]*(i-muy)*(i-muy);
			}
		}
	}
	else
	{
		conv.at<double>(0,0)=1;
		conv.at<double>(0,1)=0;
		conv.at<double>(1,0)=0;
		conv.at<double>(1,1)=1;
	}
	
	
	invert(conv,conv);
	

	//cout<<"in calculated conv: "<<conv.at<double>(0,0)<<" "<<
	//	conv.at<double>(0,1)<<" "<<conv.at<double>(1,0)<<" "<<conv.at<double>(1,1)<<" "<<endl;
	

	//for (i=0;i<gridSize+1;i++)
	//{
	//	delete []alpha[i];
	//}
//	delete []alpha;

	/*int intX,intY;
	double ratioX,ratioY,tpx1,tpx2;

	intX=(int)x;
	intY=(int)y;
	ratioX=(x-intX);
	ratioY=y-intY;
	tpx1=(1-ratioX)*prob_map.at<double>(intY,intX)+ratioX*
		prob_map.at<double>(intY,intX+1);
	tpx2=(1-ratioX)*prob_map.at<double>(intY+1,intX)+ratioX*
		prob_map.at<double>(intY+1,intX+1);
	
	double alpha =(1-ratioY)*tpx1+ratioY*tpx2;*/
}

void AAM_RealGlobal_GPU::calculateMandC_withGuidance(Mat &prob_map,double dx, double dy, int gridSize,double *mu,Mat &conv,int *shapeG,int featureID,int candidateID)
{
	int i,j;
	int step=gridSize/2;

	int x=dx;int y=dy;//currently, set x=ind(dx) directly
	//cout<<x<<" "<<y<<endl;

	/////////////////check the prob_map and position////////////////////
	/*namedWindow("Prob Map");
	Mat tmp=prob_map.clone()*0;
	tmp(Range(y-step,y+step),Range(x-step,x+step))+=prob_map(Range(y-step,y+step),Range(x-step,x+step));*/

	////////////////////////draw the circle/////////////////////////////
	//Point center;
	//
	//	center.x=x;
	//	center.y=y;
	//	circle(tmp,center,2,255);
	//	imshow("Prob Map",tmp);
	//waitKey();
	////////////////////////////////////////////////////////////////////


	//calculate mu
	double mux,muy;

	////////////////////////calculatue mean every time/////////////////////////////////
	double sumofProb=0;
	for (i=y-step;i<=y+step;i++)	//go over the local window
	{
		for (j=x-step;j<=x+step;j++)
		{
			//cout<<i<<" "<<j<<endl;
			sumofProb+=prob_map.at<double>(i,j);
		}
	}

	//calculate alpha
	double **alpha=new double *[gridSize+1];
	for (i=0;i<gridSize+1;i++)
	{
		alpha[i]=new double[gridSize+1];
	}

	if (sumofProb>0)
	{
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				alpha[i-y+step][j-x+step]=prob_map.at<double>(i,j)/sumofProb;
			}
		}
	}
	else
	{
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				alpha[i-y+step][j-x+step]=1.0f/(float)((gridSize+1)*(gridSize+1));
			}
		}
	}
//	mux=x-step+sumx;
//	muy=y-step+sumy;
	//cout<<mux<<" "<<muy<<endl;
	mux=muy=0;
	for (i=y-step;i<=y+step;i++)
	{
		for (j=x-step;j<=x+step;j++)
		{
			mux+=alpha[i-y+step][j-x+step]*j;
			muy+=alpha[i-y+step][j-x+step]*i;
			//cout<<mux<<" "<<muy<<endl;
		}

		
	}

	//if(1)
	if (abs(mux-shapeG[0])<5&&abs(muy-shapeG[1])<5)
	{
		mu[0]=mux;mu[1]=muy;
	}
	else
	{
		mu[0]=shapeG[0];mu[1]=shapeG[1];
		mux=mu[0];muy=mu[1];
	}
	
	/////////////////////////////////////////////////////////////////////////////////////////

	//no calculation at all
	mux=shapeG[0];muy=shapeG[1];
	mu[0]=shapeG[0];mu[1]=shapeG[1];

	//cout<<mux<<" "<<muy<<" "<<prob_map.at<double>(muy,mux)<<endl;

	conv*=0;
	if (sumofProb>0)
	{
		for (i=y-step;i<=y+step;i++)	//go over the local window
		{
			for (j=x-step;j<=x+step;j++)
			{
				conv.at<double>(0,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(j-mux);
				conv.at<double>(0,1)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
				conv.at<double>(1,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
				conv.at<double>(1,1)+=alpha[i-y+step][j-x+step]*(i-muy)*(i-muy);
			}
		}
	}
	else
	{
		conv.at<double>(0,0)=1;
		conv.at<double>(0,1)=0;
		conv.at<double>(1,0)=0;
		conv.at<double>(1,1)=1;
	}
	
	//cout<<"********************************************";
	//cout<<sumofProb<<" "<<mux<<" "<<muy<<endl;
	//cout<<conv.at<double>(0,0)<<" "<<conv.at<double>(0,1)<<" "<<
	//	conv.at<double>(1,0)<<" "<<conv.at<double>(1,1)<<endl;

	invert(conv,conv);

	//no calculation, just identity
	/*conv.at<double>(0,0)=1;
	conv.at<double>(0,1)=0;
	conv.at<double>(1,0)=0;
	conv.at<double>(1,1)=1;*/

	//conv*=probForEachFeature[featureID];
	conv*=probForEachFeatureCandidates[featureID][candidateID];

	//cout<<featureID<<" "<<probForEachFeature[featureID]<<" "<<prob_map.at<double>(muy,mux)<<endl;
}

void AAM_RealGlobal_GPU::calculateMandC_preCalculate(Mat &prob_map,int gridSize,int lableId,int largeWindowSize)
{
	int i,j;
	int step=gridSize/2;

	//suppose we tolerate 1/3 of the width
	//int largeWindowSize=prob_map.cols/3;
	//calculate conv for all the pixles around current location

	int mux=prob_mu[lableId][0];int muy=prob_mu[lableId][1];//currently, set x=ind(dx) directly


	Mat localConv=Mat::zeros(2,2,CV_64FC1);
	for (int cx=mux-largeWindowSize/2;cx<=mux+largeWindowSize/2;cx++)
	{
		for (int cy=muy-largeWindowSize/2;cy<=muy+largeWindowSize;cy++)
		{
			if (prob_map.at<double>(cy,cx)<0.3)
			{
				continue;
			}
			//get the sum of the prob 
			double sumofProb=0;
			for (int i=cy-step;i<=cy+step;i++)	//go over the local window
			{
				for (int j=cx-step;j<=cx+step;j++)
				{
					sumofProb+=prob_map.at<double>(i,j);
				}
			}
			if (sumofProb==0)
			{
					localConv.at<double>(0,0)=1;
					localConv.at<double>(0,1)=0;
					localConv.at<double>(1,0)=0;
					localConv.at<double>(1,1)=1;
			}
			else
			{
				localConv*=0;
				for (int i=cy-step;i<=cy+step;i++)	//go over the local window
				{
					for (int j=cx-step;j<=cx+step;j++)
					{
						localConv.at<double>(0,0)+=prob_map.at<double>(i,j)/sumofProb*(j-mux)*(j-mux);
						localConv.at<double>(0,1)+=prob_map.at<double>(i,j)/sumofProb*(j-mux)*(i-muy);
						localConv.at<double>(1,0)+=prob_map.at<double>(i,j)/sumofProb*(j-mux)*(i-muy);
						localConv.at<double>(1,1)+=prob_map.at<double>(i,j)/sumofProb*(i-muy)*(i-muy);
					}
				}
				invert(localConv,localConv);
			}
			//cout<<cy<<" "<<cx<<" "<<conv_precalculated[lableId].rows<<" "<<conv_precalculated[lableId].cols<<endl;
			conv_precalculated[lableId].at<double>(cy*2,cx*2)=localConv.at<double>(0,0);
			conv_precalculated[lableId].at<double>(cy*2,cx*2+1)=localConv.at<double>(0,1);
			conv_precalculated[lableId].at<double>(cy*2+1,cx*2)=localConv.at<double>(1,0);
			conv_precalculated[lableId].at<double>(cy*2+1,cx*2+1)=localConv.at<double>(1,1);
		}
	}
	//cout<<x<<" "<<y<<endl;

	/////////////////check the prob_map and position////////////////////
	/*namedWindow("Prob Map");
	Mat tmp=prob_map.clone()*0;
	tmp(Range(y-step,y+step),Range(x-step,x+step))+=prob_map(Range(y-step,y+step),Range(x-step,x+step));*/

	////////////////////////draw the circle/////////////////////////////
	//Point center;
	//
	//	center.x=x;
	//	center.y=y;
	//	circle(tmp,center,2,255);
	//	imshow("Prob Map",tmp);
	//waitKey();
	////////////////////////////////////////////////////////////////////



	///////////////////////////////////////////////////////////////////////////////////////////

	//mu[0]=shapeG[0];mu[1]=shapeG[1];


	//conv*=0;
	////if (sumofProb>0)
	////{
	////	for (i=y-step;i<=y+step;i++)	//go over the local window
	////	{
	////		for (j=x-step;j<=x+step;j++)
	////		{
	////			conv.at<double>(0,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(j-mux);
	////			conv.at<double>(0,1)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
	////			conv.at<double>(1,0)+=alpha[i-y+step][j-x+step]*(j-mux)*(i-muy);
	////			conv.at<double>(1,1)+=alpha[i-y+step][j-x+step]*(i-muy)*(i-muy);
	////		}
	////	}
	////}
	////else
	//{
	//	conv.at<double>(0,0)=1;
	//	conv.at<double>(0,1)=0;
	//	conv.at<double>(1,0)=0;
	//	conv.at<double>(1,1)=1;
	//}
	//
	//
	//invert(conv,conv);
}

void AAM_RealGlobal_GPU::calculateTermValue(double errorSum,double &totalweight)
{
		double AAMTermValue=0;
			double detectionTermValue=0;
			double localPriorTermValue=0;
			

			AAMTermValue=AAM_weight*errorSum;

			double tmp[2];
			double tmpSUM;
			if (smoothWeight>0)
			{
				detectionTermValue=0;
				//for (int i=0;i<trees->numOfLabels;i++)
				//{
				//	tmp[0]=currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0];
				//	tmp[1]=currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1];
				//	tmpSUM=prob_conv[i].at<double>(0,0)*tmp[0]*tmp[0]+(prob_conv[i].at<double>(1,0)+prob_conv[i].at<double>(0,1))*tmp[0]*tmp[1]+
				//		prob_conv[i].at<double>(1,1)*tmp[1]*tmp[1];
				//	detectionTermValue+=probForEachFeature[i]*tmpSUM;
				//	/*detectionTermValue+=probForEachFeature[i]*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])+
				//		(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1])*(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1]);*/
				//}
				for (int i=0;i<trees->numOfLabels;i++)
				{
					for (int j=0;j<candidatePoints[i].size();j++)
					{
						tmp[0]=currentShape->pts[trees->interestPtsInd[i]][0]-candidatePoints[i][j].x;
						tmp[1]=currentShape->pts[trees->interestPtsInd[i]][1]-candidatePoints[i][j].y;
					
					
						tmpSUM=prob_conv_candidates[i][j].at<double>(0,0)*tmp[0]*tmp[0]+(prob_conv_candidates[i][j].at<double>(1,0)+prob_conv_candidates[i][j].at<double>(0,1))*tmp[0]*tmp[1]+
							prob_conv_candidates[i][j].at<double>(1,1)*tmp[1]*tmp[1];
						detectionTermValue+=probForEachFeatureCandidates[i][j]*tmpSUM;
					}
					
					/*detectionTermValue+=probForEachFeature[i]*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])*(currentShape->pts[trees->interestPtsInd[i]][0]-shapeSample[i][0])+
						(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1])*(currentShape->pts[trees->interestPtsInd[i]][1]-shapeSample[i][1]);*/
				}
				detectionTermValue*=smoothWeight;
			}
			

			if (priorWeight>0)
			{
				localPriorTermValue=0;
				Mat tmpSWeight=Mat::zeros(shape_dim,1,CV_64FC1);
				for (int i=0;i<shape_dim;i++)
				{
					tmpSWeight.at<double>(i,0)=s_weight[i]-priorMean[i];
				}
				Mat sWeight_tran;
				transpose(tmpSWeight,sWeight_tran);
				Mat CValue=sWeight_tran*priorSigma(Range(0,shape_dim),Range(0,shape_dim))*tmpSWeight;
				localPriorTermValue=CValue.at<double>(0,0);
				localPriorTermValue*=priorWeight;
			}
			totalweight=AAMTermValue+detectionTermValue+localPriorTermValue;
}