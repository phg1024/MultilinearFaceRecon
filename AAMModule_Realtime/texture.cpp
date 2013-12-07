#include "texture.h"
#include "saveandread.h"

Texture::Texture(IplImage *input,CvMat *mask)
{
	NormailizedImg=NULL;
	img=NULL;
	mask=NULL;
	imgData=NULL;
	//m_imgData=NULL;
	getROI(input,mask);
}

Texture::Texture()
{
	NormailizedImg=NULL;
	img=NULL;
	mask=NULL;
	imgData=NULL;
	//m_imgData=NULL;
}

void Texture::setImg(IplImage *input)
{
	//shape needed
	if (img!=NULL)
	{
		cvReleaseImage(&img);
		//m_imgData->release();
	}
	if (mask!=NULL)
	{
		cvReleaseMat(&mask);
	}
	
	col=width=cvGetSize(input).width;
	row=height=cvGetSize(input).height;
	nband=input->nChannels;
	img=cvCreateImage(cvGetSize(input),input->depth,input->nChannels);
	cvCopyImage(input,img);
	depth=input->depth;
	nchanels=input->nChannels;
	mask=cvCreateMat(height,width,CV_64FC1);

	m_mask=cvarrToMat(mask);

	//cvarrToMat()
//	cvGet2D(img,height-1,width-1);
//	cout<<width<<" "<<height<<" "<<row*col<<endl;

}

void Texture::getROI(IplImage *input,CvMat *mask_in)
{
	setImg(input);
	
	cvCopy(mask_in,mask);
	m_mask=cvarrToMat(mask);
	pix_num=0;
	for (int i=0;i<width;i++)
	{		
		for (int j=0;j<height;j++)
		{
			//if (CV_MAT_ELEM(*mask,double,j,i)==1)
			if (m_mask.at<double>(j,i)==1)
			{
				pix_num++;
			}
		}
	}
	if(imgData==NULL)
	{
		imgData=cvCreateMat(1,pix_num*(input->nChannels),CV_64FC1);
		m_imgData=cvarrToMat(imgData);
	}
	
	int cnum=0;
	for (int i=0;i<width;i++)
	{		
		for (int j=0;j<height;j++)
		{
			if (m_mask.at<double>(j,i)==1)
			{
				for (int k=0;k<input->nChannels;k++)
				{
					//CV_MAT_ELEM(*imgData,double,0,cnum*(input->nChannels)+k)=cvGet2D(img,j,i).val[k];
					m_imgData.at<double>(0,cnum*(input->nChannels)+k)=cvGet2D(img,j,i).val[k];
				}
				cnum++;
			}

		}
	}
//	m_imgData=cvarrToMat(imgData);
}

double Texture::pointMul(Texture *in2)
{
	double sum=0;
	for (int i=0;i<imgData->cols;i++)
	{
	/*	sum+=CV_MAT_ELEM(*imgData,double,0,i)*
			CV_MAT_ELEM(*(in2->imgData),double,0,i);*/
		sum+=m_imgData.at<double>(0,i)*in2->m_imgData.at<double>(0,i);
	}
	return sum;
}

void Texture::texture_add(Texture *in2)
{
	//double sum=0;
	for (int i=0;i<imgData->cols;i++)
	{
	/*	CV_MAT_ELEM(*imgData,double,0,i)+=
			CV_MAT_ELEM(*(in2->imgData),double,0,i);*/
		m_imgData.at<double>(0,i)+=in2->m_imgData.at<double>(0,i);
	}
//	return sum;
}

void Texture::texture_deduce(Texture *in2)
{
	//double sum=0;
	for (int i=0;i<imgData->cols;i++)
	{
	/*	CV_MAT_ELEM(*imgData,double,0,i)-=
			CV_MAT_ELEM(*(in2->imgData),double,0,i);*/
		m_imgData.at<double>(0,i)-=in2->m_imgData.at<double>(0,i);
	}
	//	return sum;
}

void Texture::devide(double num)
{
	for (int i=0;i<imgData->cols;i++)
	{
		m_imgData.at<double>(0,i)/=num;
	}

	if (NormailizedImg!=NULL)
	{
		cvReleaseImage(&(NormailizedImg));
		NormailizedImg=NULL;
	}
}

double Texture::normalize()
{
	double m_value=cvMean(imgData);
	//cout<<"m_value: "<<m_value<<endl;
	//Scalar s=cv::mean(m_imgData);
	////double m_value=cv::mean(m_imgData).val[0];
	//double m_value=s.val[0];
	double sqrsum=0;
	for (int i=0;i<imgData->cols;i++)
	{
		m_imgData.at<double>(0,i)-=m_value;
		sqrsum+=m_imgData.at<double>(0,i)*m_imgData.at<double>(0,i);
	}
	double sqrtNum=1.0f/sqrt(sqrsum);
	
	for (int i=0;i<imgData->cols;i++)
	{
		m_imgData.at<double>(0,i)*=sqrtNum;
	}

	if (NormailizedImg!=NULL)
	{
		cvReleaseImage(&(NormailizedImg));
		NormailizedImg=NULL;
	}
	return sqrtNum;
}

double Texture::simple_normalize()
{
//	double m_value=cvMean(imgData);
	double sqrsum=0;
	for (int i=0;i<imgData->cols;i++)
	{
	//	CV_MAT_ELEM(*imgData,double,0,i)-=m_value;
		sqrsum+=m_imgData.at<double>(0,i)*m_imgData.at<double>(0,i);
	}
	double sqrtNum=1.0f/sqrt(sqrsum);

	for (int i=0;i<imgData->cols;i++)
	{
		m_imgData.at<double>(0,i)*=sqrtNum;
	}

	if (NormailizedImg!=NULL)
	{
		cvReleaseImage(&(NormailizedImg));
		NormailizedImg=NULL;
	}
	return sqrtNum;
}

void Texture::setZero()
{
	//for (int i=0;i<imgData->cols;i++)
	//{
	//	CV_MAT_ELEM(*(imgData),double,0,i)=0;
	//}
	m_imgData=0;
}

Texture & operator -(Texture &in1,Texture &in2)
{

	//for (int i=0;i<in1.imgData->cols;i++)
	//{
	//	in1.m_imgData.at<double>(0,i)=in1.m_imgData.at<double>(0,i)-
	//		in2.m_imgData.at<double>(0,i);
	//}
	in1.m_imgData=in1.m_imgData-in2.m_imgData;
	return in1;
}

Texture& Texture::operator =(Texture &in2)
{
	setImg(in2.img);
	if(imgData==NULL)
	imgData=cvCreateMat(in2.imgData->rows,in2.imgData->cols,in2.imgData->type);
	cvCopy(in2.mask,mask);
	cvCopy(in2.imgData,imgData);
	pix_num=in2.pix_num;

	m_imgData=cvarrToMat(imgData);
	m_mask=cvarrToMat(mask);

	if (NormailizedImg!=NULL)
	{
		cvReleaseImage(&(NormailizedImg));
		NormailizedImg=in2.NormailizedImg;
	}
	return *this;
}

double operator *(Texture &in1,Texture &in)
{
	double sum=0;
	for (int i=0;i<in1.imgData->cols;i++)
	{
		sum+=in1.m_imgData.at<double>(0,i)*
			in.m_imgData.at<double>(0,i);
	}
	return sum;
}

Texture& operator /(Texture& in,double scale)
{

	for (int i=0;i<in.imgData->cols;i++)
	{
		in.m_imgData.at<double>(0,i)/=scale;
	}
	if (in.NormailizedImg!=NULL)
	{
		cvReleaseImage(&(in.NormailizedImg));
		in.NormailizedImg=NULL;
	}
	return in;
}

Texture& operator +(Texture &in1,Texture &in)
{
	//double sum=0;
	for (int i=0;i<in1.imgData->cols;i++)
	{
		in1.m_imgData.at<double>(0,i)=in1.m_imgData.at<double>(0,i)+
			in.m_imgData.at<double>(0,i);
	}
	if (in.NormailizedImg!=NULL)
	{
		cvReleaseImage(&(in.NormailizedImg));
		in.NormailizedImg=NULL;
	}
	return in1;
}

IplImage *Texture::getImage()
{
	if (NormailizedImg!=NULL)
	{
		return NormailizedImg;
	}
	NormailizedImg=cvCreateImage(cvSize(width,height),depth,nchanels);
	int pixelnum=0;
	CvScalar s;
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
			//if (CV_MAT_ELEM(*mask,double,j,i))
			if (m_mask.at<double>(j,i))
			{
				for (int k=0;k<nchanels;k++)
				{
					//s.val[k]=CV_MAT_ELEM(*imgData,double,0,pixelnum*nchanels+k);
					s.val[k]=m_imgData.at<double>(0,pixelnum*nchanels+k);
				}
				cvSet2D(NormailizedImg,j,i,s);
				pixelnum++;
			}
			else
			{
				s.val[0]=s.val[1]=s.val[2]=0;
				cvSet2D(NormailizedImg,j,i,s);
			}
		}
	}
	//cvNamedWindow("1");
	//cvShowImage("1",NormailizedImg);
	//cvWaitKey(0);
	return NormailizedImg;
}

CvMat *Texture::getImageMat()
{
	CvMat *imgMat=cvCreateMat(height,width,CV_64FC1);
	Mat m_imgMat=cvarrToMat(imgMat);
	//NormailizedImg=cvCreateImage(cvSize(width,height),depth,nchanels);
	m_imgMat=0;
	int pixelnum=0;
	//CvScalar s;
	for (int i=0;i<width;i++)
	{
		for (int j=0;j<height;j++)
		{
	/*		CV_MAT_ELEM(*imgMat,double,j,i)=0;
			if(CV_MAT_ELEM(*mask,double,j,i))
			{
				CV_MAT_ELEM(*imgMat,double,j,i)=CV_MAT_ELEM(*imgData,double,0,pixelnum);
				pixelnum++;
			}*/
			if(m_mask.at<double>(j,i))
			{
				m_imgMat.at<double>(j,i)=m_imgData.at<double>(0,pixelnum);
				pixelnum++;
			}
		}
	}
	//cvNamedWindow("1");
	//cvShowImage("1",NormailizedImg);
	//cvWaitKey(0);
//	m_imgMat.convertTo(imgMat)
	return imgMat;
}

Texture& operator *(Texture &in1,double w)
{
	//double sum=0;
	for (int i=0;i<in1.imgData->cols;i++)
	{
		//CV_MAT_ELEM(*(in1.imgData),double,0,i)*=w;
		in1.m_imgData.at<double>(0,i)*=w;
	}
	return in1;
}

void Texture::save(ofstream &out)
{
	//imagedata
	//width,height
	//mask
	SL_Basis slEngine;
	slEngine.saveMatrix(out,imgData);
	slEngine.saveMatrix(out,img);
	out<<width<<" "<<height<<" "<<pix_num<<endl;
	out<<nband<<" "<<nchanels<<" "<<depth<<" "<<row<<" "<<col<<endl;
	slEngine.saveMatrix(out,mask);

}

void Texture::load(ifstream &in)
{
	SL_Basis slEngine;
	imgData=slEngine.loadMatrix(in,imgData);
	img=slEngine.loadMatrix(in,img);
	in>>width>>height>>pix_num;
	in>>nband>>nchanels>>depth>>row>>col;
	mask=slEngine.loadMatrix(in,mask);
	m_imgData=cvarrToMat(imgData);
	m_mask=cvarrToMat(mask);

}