#include "saveandread.h"

void SL_Basis::saveMatrix(ofstream &out,CvMat *mat)
{
	out<<mat->rows<<" "<<mat->cols<<endl;
	for (int i=0;i<mat->rows;i++)
	{
		for (int j=0;j<mat->cols;j++)
		{
			out<<CV_MAT_ELEM(*mat,double,i,j)<<" ";
		}
		out<<endl;
	}
}

CvMat* SL_Basis::loadMatrix(ifstream &in,CvMat *mat)
{
	int rows,cols;
	in>>rows>>cols;
	if (mat!=NULL)
	{
		cvReleaseMat(&mat);
		
	}
	mat=cvCreateMat(rows,cols,CV_64FC1);
	for (int i=0;i<mat->rows;i++)
	{
		for (int j=0;j<mat->cols;j++)
		{
			in>>CV_MAT_ELEM(*mat,double,i,j);
		}
	}
	return mat;
}

void SL_Basis::saveMatrix(ofstream &out,IplImage *img)
{
	out<<img->width<<" "<<img->height<<endl;
	out<<img->depth<<" "<<img->nChannels<<endl;
	for (int i=0;i<img->width;i++)
	{
		for (int j=0;j<img->height;j++)
		{
			for (int k=0;k<img->nChannels;k++)
			{
				out<<cvGet2D(img,j,i).val[k]<<" ";
			}
		}
		out<<endl;
	}
}

IplImage* SL_Basis::loadMatrix(ifstream &in,IplImage *img)
{
	int width,height,depth,nchannels;
	in>>width>>height>>depth>>nchannels;
	if (img!=NULL)
	{
		cvReleaseImage(&img);
		
	}
	img=cvCreateImage(cvSize(width,height),depth,nchannels);

	CvScalar tmp;
	for (int i=0;i<img->width;i++)
	{
		for (int j=0;j<img->height;j++)
		{
			for (int k=0;k<img->nChannels;k++)
			{
				in>>tmp.val[k];
			}
			cvSet2D(img,j,i,tmp);
		}
	}
	return img;
}

void SL_Basis::saveMatrix(ofstream& out,double *data,int length)
{
	out<<length<<endl;
	for(int i=0;i<length;i++)
		out<<data[i]<<" ";
	out<<endl;
}

double* SL_Basis::loadMatrix(ifstream &in,double *data)
{
	int length;
	in>>length;
	if (data!=NULL)
	{
		delete []data;
		
	}
	data=new double[length];
	for (int i=0;i<length;i++)
	{
		in>>data[i];
	}
	return data;
}

void SL_Basis::saveMatrix(ofstream& out,int *data,int length)
{
	out<<length<<endl;
	for(int i=0;i<length;i++)
		out<<data[i]<<" ";
	out<<endl;
}

int* SL_Basis::loadMatrix(ifstream &in,int *data)
{
	int length;
	in>>length;
	if (data!=NULL)
	{
		delete []data;
		
	}
	data=new int[length];
	for (int i=0;i<length;i++)
	{
		in>>data[i];
	}
	return data;
}

void SL_Basis::saveMatrix(ofstream& out,int **data,int s1,int s2)
{
	out<<s1<<" "<<s2<<endl;
	for (int i=0;i<s1;i++)
	{
		for (int j=0;j<s2;j++)
		{
			out<<data[i][j]<<" ";
		}
		out<<endl;
	}
}

int ** SL_Basis::loadMatrix(ifstream &in,int **data)
{
	int s1,s2;
	in>>s1>>s2;
	data=new int *[s1];
	for (int i=0;i<s1;i++)
	{
		data[i]=new int [s2];
		for (int j=0;j<s2;j++)
		{
			in>>data[i][j];
		}
	}
	return data;
}

void SL_Basis::saveMatrix(ofstream& out,double **data,int s1,int s2)
{
	out<<s1<<" "<<s2<<endl;
	for (int i=0;i<s1;i++)
	{
		for (int j=0;j<s2;j++)
		{
			out<<data[i][j]<<" ";
		}
		out<<endl;
	}
}

double ** SL_Basis::loadMatrix(ifstream &in,double **data)
{
	int s1,s2;
	in>>s1>>s2;
	data=new double *[s1];
	for (int i=0;i<s1;i++)
	{
		data[i]=new double [s2];
		for (int j=0;j<s2;j++)
		{
			in>>data[i][j];
		}
	}
	return data;
}