#include "aptPCA_float.h"
#include "CodeTimer.h"

Adp_PCA_float::Adp_PCA_float(int _dim,int _maximumDim,bool _outputData,int _blockNum)
{
	dim=_dim;
	maximumDim=_maximumDim;
	meanA=MatrixXf::Zero(dim,1);
	
	n=0;

	outputData=_outputData;

	blockNum=_blockNum;
	dataForUse=MatrixXf::Zero(dim,_blockNum);
}

void Adp_PCA_float::setModel(Mat &model,Mat &eigenVec,Mat &m,int dataNum,int modelDim)
{
	n=dataNum;
	meanA=MatrixXf::Zero(m.cols,1);
	for (int i=0;i<m.cols;i++)
	{
		meanA(i,0)=m.at<double>(0,i);
	}

	eigenVector=MatrixXf::Zero(model.cols,modelDim);
	for (int i=0;i<modelDim;i++)
	{
		for (int j=0;j<model.cols;j++)
		{
			eigenVector(j,i)=model.at<double>(i,j);
		}
	}

	eigenValueMat=MatrixXf::Zero(modelDim,modelDim);
	for (int i=0;i<modelDim;i++)
	{
		eigenValueMat(i,i)=eigenVec.at<double>(i,0);
	}

	
	//double check
	//cout<<model.rows<<" "<<model.cols<<" "<<m.cols<<" "<<eigenVec.rows<<endl;

	//cout<<eigenVector.rows()<<" "<<eigenVector.cols()<<" "<<meanA.rows()<<" "<<meanA.cols()<<endl;
	//cout<<eigenValueMat<<endl;
}

void Adp_PCA_float::setModel(float *model,float *eigenVec,float *m,int dataNum,int modelDim)
{
	n=dataNum;
	meanA=MatrixXf::Zero(dim,1);
	for (int i=0;i<dim;i++)
	{
		meanA(i,0)=m[i];
	}

	eigenVector=MatrixXf::Zero(dim,modelDim);
	for (int i=0;i<modelDim;i++)
	{
		for (int j=0;j<dim;j++)
		{
			eigenVector(j,i)=model[i*dim+j];
		}
	}

	eigenValueMat=MatrixXf::Zero(modelDim,modelDim);
	for (int i=0;i<modelDim;i++)
	{
		eigenValueMat(i,i)=eigenVec[i];
	}

	//double check
	//cout<<model.rows<<" "<<model.cols<<" "<<m.cols<<" "<<eigenVec.rows<<endl;

	//cout<<eigenVector.rows()<<" "<<eigenVector.cols()<<" "<<meanA.rows()<<" "<<meanA.cols()<<endl;
	//cout<<eigenValueMat<<endl;
}

void Adp_PCA_float::initialModel(MatrixXf &data)
{
	n=data.cols();

	meanA=data.col(0)*0;
	for (int i=0;i<n;i++)
	{
		meanA+=data.col(i);
	}

	
	meanA/=n;

	




	if (n>1)
	{
		for (int i=0;i<n;i++)
		{
			data.col(i)=data.col(i)-meanA;
		}
		//cout<<data.topLeftCorner(20,5)<<endl;
		ofstream out("mean.txt",ios::out);
		for (int i=0;i<meanA.rows();i++)
		{
			for (int j=0;j<meanA.cols();j++)
			{
				out<<meanA(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();

		ofstream out1("data.txt",ios::out);
		for (int i=0;i<data.rows();i++)
		{
			for (int j=0;j<data.cols();j++)
			{
				out1<<data(i,j)<<" ";
			}
			out1<<endl;
		}
		out1.close();



		//cout<<data.topLeftCorner(20,5)<<endl;
		
		JacobiSVD<MatrixXf> svd(data, ComputeThinU | ComputeThinV);
		//JacobiSVD<MatrixXf> svd(data, ComputeFullU | ComputeFullV);
		eigenVector=svd.matrixU();
		MatrixXf eigenValueMat_tmp=svd.singularValues();

	
		int curRemainDim;
		if (maximumDim==-1)
		{
			double sumE=svd.singularValues().sum();
			double threshold=sumE*0.9;
			double curValue=0;
			for (int i=0;i<eigenValueMat.rows();i++)
			{
				curValue+=eigenValueMat(i,i);
				if (curValue>threshold)
				{
					curRemainDim=i;
					break;
				}
			}
		}
		else
		{
			curRemainDim=std::min<int>(maximumDim,n);
			curRemainDim=std::min<int>(curRemainDim,eigenValueMat_tmp.rows());
		}
		eigenValueMat=eigenValueMat_tmp.topLeftCorner(curRemainDim,1).asDiagonal();
	}
}

void Adp_PCA_float::normalizeMean()
{
	double m=meanA.array().mean();

	double s=0;
	for (int i=0;i<meanA.rows();i++)
	{
		meanA(i,0)-=m;
		s+=meanA(i,0)*meanA(i,0);
	}
	if (s>0)
	{
		meanA/=sqrt(s);
	}

}

void Adp_PCA_float::updateModel(float *data,int _sampleNum,bool isNormalize/* =false */)
{
	updateModel(dataForUse,isNormalize);
}

void Adp_PCA_float::getMeanAndModel(float *meanVec)
{
	for (int i=0;i<dim;i++)
	{
		meanVec[i]=meanA(i,0);
	}
	for (int i=0;i<maximumDim;i++)
	{
		for (int j=0;j<dim;j++)
		{
			meanVec[i*dim+j+dim]=eigenVector(j,i);
		}
	}
}

void Adp_PCA_float::updateModel(MatrixXf &B,bool isNormalize)
{
	if (n==0)
	{
		initialModel(B);
		return;
	}
	
	//float f=0.95;
	//get mean of B:dim*m
	int m=B.cols();
	MatrixXf meanB=B.col(0)*0;
	for (int i=0;i<m;i++)
	{
		meanB+=B.col(i);
	}
	meanB/=m;

	

	//compute b_ast amd r
	


	MatrixXf B_Hat(dim,m+1);
	for (int i=0;i<m;i++)
	{
		B_Hat.col(i)=B.col(i)-meanB;
		//B_Hat.col(i)=B.col(i)-meanA;
	}

//	//check if the data is almost identical
	float curResValue=B_Hat.topLeftCorner(dim,m).cwiseAbs().maxCoeff();
//	cout<<"cur maximum value: "<<curResValue<<endl;
	if(curResValue<0.03)
	{
		//cout<<"no change: "<<curResValue<<endl;
		return;
	}




	//B_Hat.col(m)=sqrtf((float)(m*n)/(float)(m+n))*(meanB-meanA);
	B_Hat.col(m)=(meanB-meanA);
	B_Hat.col(m)*=sqrtf((float)(m*n)/(float)(m+n));

	MatrixXf U_t=eigenVector.transpose();
	MatrixXf proj=U_t*B_Hat;
	MatrixXf residual=B_Hat-eigenVector*proj;

	float curRes=residual.topLeftCorner(residual.rows(),residual.cols()-1).norm();
	if (curRes<0.3)
	{
		//cout<<"low res "<<curRes<<endl;
		return;
	}
	

//	cout<<residual.rows()<<" "<<residual.cols()<<endl;
	//GTB("1");

	HouseholderQR<MatrixXf> qr(residual);
	MatrixXf thinQ(MatrixXf::Identity(residual.rows(),residual.cols()));
	MatrixXf B_bst=qr.householderQ()*thinQ;

	//GTE("1");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"used time per iteration: "<<time<<endl;

	MatrixXf R(MatrixXf::Zero(eigenValueMat.rows()+m+1,eigenValueMat.rows()+m+1));

	/*MatrixXf tt=U_t*B_Hat;
	cout<<m<<" "<<n<<endl;
	cout<<tt.rows()<<" "<<tt.cols()<<endl;
	cout<<B_bst.rows()<<" "<<B_bst.cols()<<endl;
	MatrixXf ttt=B_bst.transpose()*residual;
	cout<<ttt.rows()<<" "<<ttt.cols()<<endl;*/

	R.topLeftCorner(eigenValueMat.rows(),eigenValueMat.rows())=eigenValueMat;
	//R.topLeftCorner(eigenValueMat.rows(),eigenValueMat.rows())=eigenValueMat*f;
	R.topRightCorner(eigenValueMat.rows(),m+1)=U_t*B_Hat;
	R.bottomRightCorner(m+1,m+1)=B_bst.transpose()*residual;
	//return; 6ms
	

	JacobiSVD<MatrixXf> svd(R, ComputeThinU | ComputeThinV);

	MatrixXf u_ast=svd.matrixU();
	MatrixXf eigenValueMat_tmp=svd.singularValues();
	//return; 11ms

	MatrixXf adjU(eigenVector.rows(),eigenVector.cols()+B_bst.cols());
	adjU.topLeftCorner(eigenVector.rows(),eigenVector.cols())=eigenVector;
	adjU.topRightCorner(B_bst.rows(),B_bst.cols())=B_bst;

	//cout<<adjU.rows()<<" "<<adjU.cols()<<endl;
	//cout<<eigenVector.rows()<<" "<<eigenVector.cols()<<endl;
	//cout<<u_ast.rows()<<" "<<u_ast.cols()<<endl;

	//update mean
	//meanA=meanA*n/(n+m)+meanB*m/(n+m);
	meanA=(meanA*n+meanB*m)/(float)(n+m);
	//meanA=(meanA*n*f+meanB*m)/((float)n*f+(float)m);
	//cout<<n<<endl;
	if (isNormalize)
	{
		normalizeMean();
	}

	n+=m;
	//decide the remain dim
	int curRemainDim;
	if (maximumDim==-1)
	{
		double sumE=svd.singularValues().sum();
		double threshold=sumE*0.9;
		double curValue=0;
		for (int i=0;i<eigenValueMat.rows();i++)
		{
			curValue+=eigenValueMat(i,i);
			if (curValue>threshold)
			{
				curRemainDim=i;
				break;
			}
		}
	}
	else
	{
		curRemainDim=std::min<int>(maximumDim,n);
		curRemainDim=std::min<int>(curRemainDim,eigenValueMat_tmp.rows());
	}
	//return; //13ms

	eigenVector=adjU*u_ast.topLeftCorner(u_ast.rows(),curRemainDim);
	//cout<<eigenValueMat.rows()<<" "<<eigenValueMat.cols()<<endl;

	//MatrixXf tttt=eigenValueMat_tmp.topLeftCorner(curRemainDim,1).asDiagonal();
	//cout<<tttt<<endl;
	eigenValueMat=eigenValueMat_tmp.topLeftCorner(curRemainDim,1).asDiagonal();


	//meanA=(meanA*n);
	//cout<<eigenValueMat.diagonal()<<endl;
	//if (eigenVector.cols()>maximumDim)
	//{
	//	eigenVector=eigenVector.topLeftCorner(eigenVector.rows(),maximumDim);
	//	eigenValueMat=eigenValueMat.topLeftCorner(eigenVector.rows(),maximumDim);
	//}

	
	
	//R(Range(0,eigenValueMat.rows),Range(eigenValueMat.cols,R.cols))+=U_t*B_Hat;
	//R(Range(eigenValueMat.rows,R.rows),Range(0,eigenValueMat.cols))+=B_bst*residual;

	
	if(outputData)
	{
		ofstream out("meanCPU.txt",ios::out);
		for (int i=0;i<meanA.rows();i++)
		{
			out<<meanA(i,0)<<" ";
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("ed_cpu.txt",ios::out);
		for (int i=0;i<B_Hat.rows();i++)
		{
			
			for (int j=0;j<B_Hat.cols();j++)
			{
				out<<B_Hat(i,j)<<" ";
			}
			out<<endl;
			
			
		}
	
		out.close();
	}
	if(outputData)
	{
		ofstream out("projCPU.txt",ios::out);
		for (int i=0;i<proj.rows();i++)
		{
			for (int j=0;j<proj.cols();j++)
			{
				out<<proj(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("adjU_CPU.txt",ios::out);
		for (int i=0;i<adjU.rows();i++)
		{
			for (int j=0;j<adjU.cols();j++)
			{
				out<<adjU(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("res_CPU.txt",ios::out);
		for (int i=0;i<residual.rows();i++)
		{
			for (int j=0;j<residual.cols();j++)
			{
				out<<residual(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}

	if(outputData)
	{
		ofstream out("Q_CPU.txt",ios::out);
		for (int i=0;i<B_bst.rows();i++)
		{
			for (int j=0;j<B_bst.cols();j++)
			{
				out<<B_bst(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("bigR_CPU.txt",ios::out);
		for (int i=0;i<R.rows();i++)
		{
			for (int j=0;j<R.cols();j++)
			{
				out<<R(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("u_CPU.txt",ios::out);
		for (int i=0;i<u_ast.rows();i++)
		{
			for (int j=0;j<u_ast.cols();j++)
			{
				out<<u_ast(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("s_CPU.txt",ios::out);
		for (int i=0;i<eigenValueMat_tmp.rows();i++)
		{
			for (int j=0;j<eigenValueMat_tmp.cols();j++)
			{
				out<<eigenValueMat_tmp(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("un_CPU.txt",ios::out);
		for (int i=0;i<eigenVector.rows();i++)
		{
			for (int j=0;j<eigenVector.cols();j++)
			{
				out<<eigenVector(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
	if(outputData)
	{
		ofstream out("sm_CPU.txt",ios::out);
		for (int i=0;i<eigenValueMat.rows();i++)
		{
			for (int j=0;j<eigenValueMat.cols();j++)
			{
				out<<eigenValueMat(i,j)<<" ";
			}
			out<<endl;
		}
		out.close();
	}
}

void Adp_PCA_float::checkReconError(MatrixXf &data)
{
	for (int i=0;i<data.cols();i++)
	{
		data.col(i)-=meanA;
	}

	MatrixXf diff=data-eigenVector*(eigenVector.transpose()*data);
	//MatrixXf absDiff=diff.array().abs().mean();
	double err=diff.array().abs().mean();

	MatrixXf squareDif(diff.rows(),diff.cols());
	for (int i=0;i<diff.rows();i++)
	{
		for (int j=0;j<diff.cols();j++)
		{
			squareDif(i,j)=diff(i,j)*diff(i,j);
		}
	}
	double squareErr=sqrt(squareDif.array().mean());
	cout<<"error: "<<err<<endl;
	cout<<"rms error: "<<squareErr<<endl;
}