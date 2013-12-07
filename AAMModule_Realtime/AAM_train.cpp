#include "AAM_train.h"

AAM_Train::AAM_Train(string name)
{
	getTrainingdata(name);
	meanShape=new Shape();
	isGlobaltransform=false;

	SL_engine=new SL_Basis;
	//set the dirName
	int ind=name.find_last_of('\\');
	if (ind<0)
	{
		ind=name.find_last_of('/');
	}
	dirName=name.substr(0,ind+1);
}

void AAM_Train::getTrainingdata(string dirName)
{
	cout<<dirName<<endl;
	fstream in(dirName.c_str(),ios::in);
	in>>shapeNum;
	//shapeNum=40;
//	cout<<shapeNum<<endl;
	shape=new Shape*[shapeNum];
	char cname[500];
//	in.getline(cname,499);
	string name;
	for (int i=0;i<shapeNum;i++)
	{
	//	name="";
		shape[i]=new Shape;
		in.getline(cname,499);
		name=cname;
		if(name.length()<3)
		{
			i--;
			continue;
		}
		shape[i]->getVertex(name);
	}
	
}
double AAM_Train::normalize(double *shape,int ptsnum)
{
	double sum[2]={0,0};
	for (int i=0;i<ptsnum;i++)
	{
		sum[0]+=shape[i]*shape[i];
		sum[1]+=shape[ptsnum+i]*shape[ptsnum+i];
	}
	double ssrtsum=sqrt(sum[0]+sum[1]);

	for (int i=0;i<ptsnum;i++)
	{
		shape[i]/=ssrtsum;
		shape[ptsnum+i]/=ssrtsum;
	}
	return ssrtsum;
}
//#define  BUFSIZE 1000
void AAM_Train::getMeanShape()
{
	double threshold=0.002;
	//transplate all the shapes to its gradivity
	for (int i=0;i<shapeNum;i++)
	{
		shape[i]->centerPts(2);
	}

	//normalize shape[0],save as ref
	int refInd=4;

	//namedWindow("1");
	//imshow("1",cvarrToMat(shape[refInd]->hostImage));
	//waitKey();
	//shape[refInd]->normalize(1);
	//if we want to control the complexity, do it here
	double *refshape=new double[shape[refInd]->ptsNum*2];
	for (int i=0;i<shape[refInd]->ptsNum*2;i++)
	{
		refshape[i]=shape[refInd]->ptsForMatlab[i]*1.0;
	}
	shape_scale=normalize(refshape,shape[refInd]->ptsNum);


	//align the shape
	int width=shape[refInd]->ptsNum;
	int height=2;
	int arraysize=width*2*sizeof(double);
	mxArray *referShape = NULL,*curMatMean=NULL,*newMatMean=NULL, *inputShape=NULL,*result = NULL;
	if (!(ep = engOpen("\0"))) {
		fprintf(stderr, "\nCan't start MATLAB engine\n");
		return;
	}
	referShape=mxCreateDoubleMatrix(width,height,mxREAL);
	curMatMean=mxCreateDoubleMatrix(width,height,mxREAL);
	newMatMean=mxCreateDoubleMatrix(width,height,mxREAL);
	inputShape=mxCreateDoubleMatrix(width,height,mxREAL);
	result=mxCreateDoubleMatrix(width,height,mxREAL);
	double *currentMean=new double[width*2];
	double *newMean=new double[width*2];

	//set the default refer shape
	memcpy((void *)mxGetPr(referShape), (void *)refshape, arraysize);
	
	//initialize currentMean to the refer shape
	for (int i=0;i<width*2;i++)
	{
		currentMean[i]=refshape[i];
	}

	ofstream out3("F:/imgdata/AAM training data/train/refshape.txt",ios::out);
	//for (int i=0;i<shapeNum;i++)
	{
		for(int j=0;j<width;j++)
		{
			out3<<refshape[j]<<" "<<refshape[width+j]<<endl;
		}
	}
	out3.close();
	
	//char buffer[BUFSIZE+1];
	engPutVariable(ep, "referShape", referShape);
	engPutVariable(ep, "result", result);
//	engEvalString(ep,"save('F:/imgdata/AAM training data/train/refershape.mat','referShape');");
	
//	engPutVariable(ep, "result", newMatMean);
	while(1)
	{
		//set current mean
		memcpy((void *)mxGetPr(curMatMean), (void *)currentMean, arraysize);
		engPutVariable(ep, "curMatMean", curMatMean);
		//allign all the shapes
		for (int i=0;i<shapeNum;i++)
		{
			memcpy((void *)mxGetPr(inputShape), (void *)shape[i]->ptsForMatlab, arraysize);
			engPutVariable(ep, "inputShape", inputShape);
			engEvalString(ep, "[m,result]=procrustes(curMatMean,inputShape);");
			result = engGetVariable(ep,"result");
//			delete []shape[i]->ptsForMatlab;
			shape[i]->ptsForMatlab=mxGetPr(result);	

			engEvalString(ep,"save('F:/imgdata/AAM training data/train/result.mat','result','curMatMean','inputShape');");
			//shape[i]->scale(shape[i]->scaleParameter,1);//reconver the scale
		}
		//caculate newMean and allign it to ref and normalize
		for (int i=0;i<width*2;i++)
		{
			newMean[i]=0;
		}
		for (int i=0;i<width*2;i++)
		{
			for (int j=0;j<shapeNum;j++)
			{
				newMean[i]+=shape[j]->ptsForMatlab[i];
			}
			newMean[i]/=shapeNum;
		}

		


		memcpy((void *)mxGetPr(newMatMean), (void *)newMean, arraysize);
		engPutVariable(ep, "newMatMean", newMatMean);
		engEvalString(ep, "[m,result]=procrustes(referShape,newMatMean);");
		//engEvalString(ep,"save('F:/imgdata/AAM training data/train/result.mat','result','referShape','newMatMean');");
		//delete []newMean;
		newMean=mxGetPr( engGetVariable(ep,"result"));	
		//normalize newMean
		normalize(newMean,width);

		ofstream out1("F:/imgdata/AAM training data/train/meanshape_ori.txt",ios::out);
		//for (int i=0;i<shapeNum;i++)
		{
			for(int j=0;j<width;j++)
			{
				out1<<newMean[j]<<" "<<newMean[width+j]<<endl;
			}
		}
		out1.close();

		//caculate the diff of means, if smaller than threshold, stop
		double differ=0,n2_newmean=0;
		for (int i=0;i<width;i++)
		{
			differ+=sqrt((newMean[i]-refshape[i])*(newMean[i]-refshape[i])+
				(newMean[width+i]-refshape[width+i])*(newMean[width+i]-refshape[width+i]));
		//	n2_newmean+=sqrt(newMean[i]*newMean[i]+newMean[width+i]*newMean[width+i]);
		}
		cout<<"current difference: "<<differ/shape_scale<<endl;
		if (differ/shape_scale<threshold)
		{

			//ofstream out("meanshape.txt",ios::out);
			//for (int i=0;i<meanShape->ptsNum;i++)
			//{
			//	out<<meanShape->pts[i][0]<<" "<<meanShape->pts[i][1]<<endl;
			//}
			//out.close();
			//save the new mean
			meanShape->getVertex(newMean,width,1,1);
			meanShape->scale(shape_scale,1);
			break;
		}

		//oldmean=newmean
		for (int i=0;i<width*2;i++)
		{
			currentMean[i]=newMean[i];
		}

	}
	//rescale all the shapes

	engClose(ep);

	//for (int i=0;i<shapeNum;i++)
	//{
	//	shape[i]->scale(scaleParameter,2);//scale to normal size,only for traning data
	//}

	//tangent space if needed

	ofstream out("F:/imgdata/AAM training data/train/allignedshape.txt",ios::out);
	for (int i=0;i<shapeNum;i++)
	{
		for(int j=0;j<width;j++)
		{
			out<<shape[i]->ptsForMatlab[j]<<" "<<shape[i]->ptsForMatlab[width+j]<<endl;
		}
	}
	out.close();

	ofstream out1("F:/imgdata/AAM training data/train/meanshape.txt",ios::out);
	//for (int i=0;i<shapeNum;i++)
	{
		for(int j=0;j<width;j++)
		{
			out1<<meanShape->ptsForMatlab[j]<<" "<<meanShape->ptsForMatlab[width+j]<<endl;
		}
	}
	out1.close();
	
	
	//delete []currentMean;
	//delete []newMean;
	//delete []refshape;
}

void AAM_Train::getTexture()
{
	texture=new Texture *[shapeNum];
	for (int i=0;i<shapeNum;i++)
	{
		texture[i]=new Texture();
	}

	//setup the reference face
	refShape=new AAM_Common();
	meanShape->translate(meanShape->minx-5,meanShape->miny-5); //to be in positive position in image
	meanShape->width=cvRound(meanShape->maxx-meanShape->minx+10);
	meanShape->height=cvRound(meanShape->maxy-meanShape->miny+10);
	refShape->ref->run_triangulation(meanShape);
	//refShape->ref->Interpolation



	IplImage *dstImg=cvCreateImage(cvSize(meanShape->width,meanShape->height),
		shape[0]->hostImage->depth,shape[0]->hostImage->nChannels);

	meanShape->setHostImage(dstImg);
	meanShape->getMask(refShape->ref->TriangleIndex);
	meanShape->getPtsIndex();

	meanShape->getMargin();
	meanShape->getTabel(refShape->ref->TriangleIndex);
	//meanShape->getTabel_strong(refShape->ref->TriangleIndex);
	//texture already read, warp them to reference
	for (int i=0;i<shapeNum;i++)
	{
		cout<<i<<endl;
		dstImg=refShape->piecewiseAffineWarping(shape[i]->hostImage,dstImg,shape[i],meanShape,
			refShape->ref->TriangleIndex,false,meanShape->affineTable);
		texture[i]->getROI(dstImg,meanShape->mask);
		
		//cvNamedWindow("Warped");
		//cvShowImage("Warped",dstImg);
		////cvWaitKey();

		//cvNamedWindow("Ori");
		//cvShowImage("Ori",shape[i]->hostImage);
		//cvWaitKey();
	}
}

void AAM_Train::getMeanTexture()
{


	//texture[0]->normalize();

	Texture meanT_curr,meanT_last,meanT_tmp;
	meanT_curr=(*texture[0]);
	texture_scale=meanT_curr.normalize();
	//cout<<meanShape->pix_num<<endl;
	double err=100000;
	double threshold =0.000001;

	while (err>threshold)
	{
		meanT_last=meanT_curr;
		for (int i=0;i<shapeNum;i++)
		{
			texture[i]->normalize();
			texture[i]->devide(texture[i]->pointMul(&meanT_curr));
		}

		
		meanT_curr.setZero();
		for (int i=0;i<shapeNum;i++)
		{
			meanT_curr.texture_add(texture[i]);
		}
		meanT_curr.devide(shapeNum);
		meanT_curr.normalize();
		meanT_tmp=meanT_curr;
		meanT_tmp.texture_deduce(&meanT_last);
		err=sqrt(meanT_tmp.pointMul(&meanT_tmp))/texture_scale;
		cout<<"current texture error: "<<err<<endl;
	}

	//try to get the meantexture without normalizetion
	//meanT_curr.setZero();
	//for (int i=0;i<shapeNum;i++)
	//{
	//	meanT_curr.texture_add(texture[i]);
	//}
	//meanT_curr.devide(shapeNum);

	meantexure=new Texture();
	(*meantexure)=meanT_curr;


}

void AAM_Train::shape_pca()
{
	int dimention=2;
	double resudial=0.98;
	int ptsNum=shape[0]->ptsNum*dimention;
	CvMat *pData=cvCreateMat(shapeNum,ptsNum,CV_64FC1);
	for (int i=0;i<shapeNum;i++)
	{
		for (int j=0;j<ptsNum;j++)
		{
			//CV_MAT_ELEM(*pData,double,i,j)=shape[i]->ptsForMatlab[j];

			//here,we keep the shape in the same scale with the meanshape
			CV_MAT_ELEM(*pData,double,i,j)=shape[i]->ptsForMatlab[j];
		}
		
	}
	s_mean = cvCreateMat(1, ptsNum, CV_64FC1);
	s_value = cvCreateMat(1, min(shapeNum,ptsNum), CV_64FC1);
	CvMat *s_PCAvec = cvCreateMat( min(shapeNum,ptsNum), ptsNum, CV_64FC1); 
	cvCalcPCA( pData, s_mean, s_value, s_PCAvec, CV_PCA_DATA_AS_ROW );

	double sumEigVal=0;
	for (int i=0;i<s_value->cols;i++)
	{
		sumEigVal+=CV_MAT_ELEM(*s_value,double,0,i);
	}

	double sumCur=0;
	for (int i=0;i<s_value->cols;i++)
	{
		sumCur+=CV_MAT_ELEM(*s_value,double,0,i);
		if (sumCur/sumEigVal>=resudial)
		{
			shape_dim=i+1;
			break;
		}
	}

	//if consider global transform, we will add another 4 shape vector and orthamized?
	if (isGlobaltransform)
	{
		s_vec=cvCreateMat(shape_dim+4,ptsNum,CV_64FC1);
		for (int i=0;i<shape_dim;i++)
		{
			for (int j=0;j<ptsNum;j++)
			{
				CV_MAT_ELEM(*s_vec,double,i,j)=CV_MAT_ELEM(*s_PCAvec,double,i,j);
			}	
		}
		//add the four shape vectors
		for (int j=0;j<ptsNum;j++)
		{
			CV_MAT_ELEM(*s_vec,double,shape_dim+3,j)=meanShape->ptsForMatlab[j];
			if (j<ptsNum/2)
			{
				CV_MAT_ELEM(*s_vec,double,shape_dim+2,j)=-meanShape->ptsForMatlab[ptsNum/2+j];
				CV_MAT_ELEM(*s_vec,double,shape_dim+1,j)=1;
				CV_MAT_ELEM(*s_vec,double,shape_dim+0,j)=0;
			}
			else
			{
				CV_MAT_ELEM(*s_vec,double,shape_dim+2,j)=meanShape->ptsForMatlab[j-ptsNum/2];
				CV_MAT_ELEM(*s_vec,double,shape_dim+1,j)=0;
				CV_MAT_ELEM(*s_vec,double,shape_dim+0,j)=1;
			}
			
		}
		
		//center and normalization
		//only need center shape_dim and shape_dim+1
		double c[2];
		for (int i=shape_dim+2;i<shape_dim+4;i++)
		{
			c[0]=c[1]=0;
			for (int j=0;j<ptsNum;j++)
			{
				if (j<ptsNum/2)
				{
					c[0]+=CV_MAT_ELEM(*s_vec,double,i,j);
				}
				else
				{
					c[1]+=CV_MAT_ELEM(*s_vec,double,i,j);
				}
				
			}
			c[0]/=(ptsNum/2);
			c[1]/=(ptsNum/2);
			for (int j=0;j<ptsNum;j++)
			{
				if (j<ptsNum/2)
				{
					CV_MAT_ELEM(*s_vec,double,i,j)-=c[0];
				}
				else
				{
					CV_MAT_ELEM(*s_vec,double,i,j)-=c[1];
				}
			}
		}
			double ssnum;
		for (int i=shape_dim;i<shape_dim+4;i++)
		{
			ssnum=0;
			for (int j=0;j<ptsNum;j++)
			{
				ssnum+=CV_MAT_ELEM(*s_vec,double,i,j)*CV_MAT_ELEM(*s_vec,double,i,j);
			}
			ssnum=sqrt(ssnum);

			for (int j=0;j<ptsNum;j++)
			{
				CV_MAT_ELEM(*s_vec,double,i,j)/=ssnum;
			}
		}

		//then orthilization
		CvMat *cur_vec=cvCreateMat(1,ptsNum,CV_64FC1);
		CvMat *loop_vec=cvCreateMat(1,ptsNum,CV_64FC1);
		CvMat *res_vec=cvCreateMat(1,ptsNum,CV_64FC1);
		CvMat *loop_vec_tran=cvCreateMat(ptsNum,1,CV_64FC1);
		CvMat *pm=cvCreateMat(1,1,CV_64FC1);
		double pm_val;
		for (int i=shape_dim+2;i<shape_dim+4;i++)
		{
			cvGetRow(s_vec,cur_vec,i);
			for (int j=0;j<i;j++)
			{
				cvGetRow(s_vec,loop_vec,j);
				cvTranspose(loop_vec,loop_vec_tran);
				cvMatMul(cur_vec,loop_vec_tran,pm);
				pm_val=CV_MAT_ELEM(*pm,double,0,0);
				for (int k=0;k<ptsNum;k++)
				{
					CV_MAT_ELEM(*s_vec,double,i,k)-=pm_val*CV_MAT_ELEM(*s_vec,double,j,k);
				}
			}
		}

		//finally, normalize
		for (int i=shape_dim;i<shape_dim+4;i++)
		{
			ssnum=0;
			for (int j=0;j<ptsNum;j++)
			{
				ssnum+=CV_MAT_ELEM(*s_vec,double,i,j)*CV_MAT_ELEM(*s_vec,double,i,j);
			}
			ssnum=sqrt(ssnum);

			for (int j=0;j<ptsNum;j++)
			{
				CV_MAT_ELEM(*s_vec,double,i,j)/=ssnum;
			}
		}


		shape_dim+=4;

		
	}
	else
	{
		s_vec=cvCreateMat(shape_dim,ptsNum,CV_64FC1);
		for (int i=0;i<shape_dim;i++)
		{
			for (int j=0;j<ptsNum;j++)
			{
				CV_MAT_ELEM(*s_vec,double,i,j)=CV_MAT_ELEM(*s_PCAvec,double,i,j);
			}

		}
	}
	cvReleaseMat(&s_PCAvec);
	cvReleaseMat(&pData);
	////s_vec scale
	ofstream out("shapes.txt",ios::out);
	for (int i=0;i<s_vec->rows-4;i++)
	{
		out<<CV_MAT_ELEM(*s_value,double,0,i)<<" ";
	}
	out<<endl;
	for (int i=0;i<s_vec->rows;i++)
	{
		for (int j=0;j<s_vec->cols;j++)
		{
			out<<CV_MAT_ELEM(*s_vec,double,i,j)<<" ";
		}
		out<<endl;
	}
	out.close();
	
}

void AAM_Train::texture_pca()
{
	int dimention=2;
	double resudial=0.98;
	int ptsNum=texture[0]->imgData->cols;
	CvMat *pData=cvCreateMat(shapeNum,ptsNum,CV_64FC1);
	for (int i=0;i<shapeNum;i++)
	{
		for (int j=0;j<ptsNum;j++)
		{
			CV_MAT_ELEM(*pData,double,i,j)=
				CV_MAT_ELEM(*(texture[i]->imgData),double,0,j);
		}

	}
	t_mean = cvCreateMat(1, ptsNum, CV_64FC1);
	t_value = cvCreateMat(1, min(shapeNum,ptsNum), CV_64FC1);
	t_vec = cvCreateMat( min(shapeNum,ptsNum), ptsNum, CV_64FC1); 
	cvCalcPCA( pData, t_mean, t_value, t_vec, CV_PCA_DATA_AS_ROW );

	double sumEigVal=0;
	for (int i=0;i<t_value->cols;i++)
	{
		double tmp=CV_MAT_ELEM(*t_value,double,0,i);
		sumEigVal+=CV_MAT_ELEM(*t_value,double,0,i);
	}

	double sumCur=0;
	for (int i=0;i<t_value->cols;i++)
	{
		sumCur+=CV_MAT_ELEM(*t_value,double,0,i);
		if (sumCur/sumEigVal>=resudial)
		{
			texture_dim=i+1;
			break;
		}
	}

	//get the meanvalue
	meantexure_real=new Texture;
	*meantexure_real=(*meantexure);
	cvCopy(t_mean,meantexure_real->imgData);
	
}

void AAM_Train::setGlobal(bool a)
{
	isGlobaltransform=a;
}


void AAM_Train::saveResult()
{
	cout<<"saving..."<<endl;
	string saveName=dirName+"trainedResault.txt";
	ofstream out(saveName.c_str(),ios::out);
	out<<shape_dim<<" "<<texture_dim<<endl;
	SL_engine->saveMatrix(out,s_vec);
	SL_engine->saveMatrix(out,t_vec);

	meanShape->save(out);
	meantexure->save(out);
	SL_engine->saveMatrix(out,s_mean);
	SL_engine->saveMatrix(out,t_mean);
	SL_engine->saveMatrix(out,s_value);
	SL_engine->saveMatrix(out,t_value);
	out<<texture_scale<<" "<<shape_scale<<endl;
	SL_engine->saveMatrix(out,refShape->ref->TriangleIndex);
	
	SL_engine->saveMatrix(out,refShape->ref->triangleList,meanShape->ptsNum,10);
	SL_engine->saveMatrix(out,refShape->ref->listNum,meanShape->ptsNum);
	out<<isGlobaltransform<<endl;

	
	//SL_engine->saveMatrix(out,)

}


/*oid AAM_Train::saveMatrix(ofstream &out,CvMat *mat)
{
	for (int i=0;i<mat->rows;i++)
	{
		for (int j=0;j<mat->cols;j++)
		{
			out<<CV_MAT_ELEM(*mat,double,i,j)<<" ";
		}
		out<<endl;
	}
}*/