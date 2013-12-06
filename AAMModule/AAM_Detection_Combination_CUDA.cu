#include <string>
#include <fstream>
#include "CUDA_basic.h"
#include <math.h>

//#include <cutil.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <helper_math.h>


#include "AAM_Detection_Combination_CUDA.h"
//#include "MM.h"
  #include <thrust/transform_reduce.h>
#include "CodeTimer.h"
#include "mkl.h"
#include "mkl_lapacke.h"
using namespace std;

cudaStream_t stream1, stream2, stream3, stream4;

void solveAxB(float *A,float *b,int dim)
{
	LAPACKE_spotrf(LAPACK_COL_MAJOR, 'U', dim, A, dim);
	int nrhs = 1;
	LAPACKE_spotrs(LAPACK_COL_MAJOR, 'U', dim, nrhs, A, dim, b, dim);

}

AAM_Detection_Combination_CUDA trackEngine;

#define DeviceVectorInit(x, len) \
	x.resize(len); \
	x##raw_gpu_ = raw_pointer_cast(&x[0]);

//randomized trees
texture <float,2> currentColorImg_combination,currentDepthImg_combination;
int currentID_combination;
cudaChannelFormatDesc desc_combination = cudaCreateChannelDesc<float>();

//result in texture
texture <float,2> detectionResult;

texture <float,2> texture_s_vec;

texture <float,2> texture_preConv;

const int blocksPerGrid=32;
const int threadsPerBlock_global=256;

const int maximumPtsNumPitch=32;

template <typename T>
struct abs_diff : public thrust::binary_function<T,T,T>
{
	__host__ __device__
		T operator()(const T& a, const T& b)
	{
		return (a-b)*(a-b);
	}
};

extern "C" void setupUsedIndex(int *host_Index,int ptsNum)
{
	CUDA_CALL(cudaMalloc((void **)&trackEngine.cu_usedIndex,ptsNum*sizeof(int)));
	CUDA_CALL(cudaMemcpy(trackEngine.cu_usedIndex,host_Index,ptsNum*sizeof(int),cudaMemcpyHostToDevice));
}

extern "C" void setupWeight(double AAMWeight,double RTweight,double priorWeight,double localWeight)
{
	trackEngine.AAMWeight=AAMWeight;
	trackEngine.RTWeight=RTweight;
	trackEngine.RTWeight_backup=RTweight;
	trackEngine.priorWeight=priorWeight;
	trackEngine.localWeight=localWeight;
	trackEngine.priorWeight_backup=priorWeight;
}

extern "C" void setData_onrun_shared(float *colorImg,float *depthImg,int width,int height)
{
	AAM_Detection_Combination_CUDA *data=&trackEngine;

	/*for (int i=0;i<height*width;i++)
	{
		data->host_colorImage[i]=colorImg[i];
		data->host_depthImage[i]=depthImg[i];
	}*/
	//CUDA_CALL(cudaMemcpyAsync(data->cu_colorImage,data->host_colorImage,width*height*sizeof(float),cudaMemcpyHostToDevice,stream1));
	//CUDA_CALL(cudaMemcpyAsync(data->cu_depthImage,data->host_depthImage,width*height*sizeof(float),cudaMemcpyHostToDevice,stream2));
	//cudaDeviceSynchronize ();

	//CUDA_CALL(cudaMemcpy(data->cu_colorImage,data->host_colorImage,width*height*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(data->cu_depthImage,data->host_depthImage,width*height*sizeof(float),cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMemcpy(data->cu_colorImage,colorImg,width*height*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(data->cu_depthImage,depthImg,width*height*sizeof(float),cudaMemcpyHostToDevice));


	CUDA_CALL(cudaBindTexture2D( NULL, currentColorImg_combination,
		data->cu_colorImage,
		desc_combination,  width,height,
		sizeof(float) * width));
	CUDA_CALL(cudaBindTexture2D( NULL, currentDepthImg_combination,
		data->cu_depthImage,
		desc_combination,  width,height,
		sizeof(float) * width));

	
	////if (!data->hasImage)
	//{
	//	CUDA_CALL(cudaBindTexture2D( NULL, currentImg,
	//		data->cu_currentImage,
	//		desc,  width,height,
	//		sizeof(float) * width));
	//	//data->hasImage=true;
	//}
}

void convertTrees2Array_combination(Node *root_CPU, Node_GPU **root_GPU,int labelNum)
{
	//if (root_CPU->label!=-1)
	//{
	//	cout<<currentID<<" "<<root_CPU->label<<endl;
	//}
	//cout<<currentID<<" ";
	int rootID=currentID_combination;
	root_GPU[rootID]->parameters[0]=root_CPU->pos1[0];
	root_GPU[rootID]->parameters[1]=root_CPU->pos1[1];
	root_GPU[rootID]->parameters[2]=root_CPU->pos2[0];
	root_GPU[rootID]->parameters[3]=root_CPU->pos2[1];
	root_GPU[rootID]->parameters[4]=root_CPU->label;
	root_GPU[rootID]->parameters[5]=root_CPU->nLevel;
	root_GPU[rootID]->parameters[8]=root_CPU->num_all;
	root_GPU[rootID]->parameters[9]=root_CPU->threshold;
	if (root_CPU->l_child==NULL)
	{
		root_GPU[rootID]->parameters[6]=-1;
	}
	if (root_CPU->r_child==NULL)
	{
		root_GPU[rootID]->parameters[7]=-1;
	}

	if (root_CPU->l_child==NULL&&root_CPU->r_child==NULL)//root
	{
		//root_GPU[currentID]->num_of_each_class=new float[labelNum];
		//cout<<"leafe "<<currentID<<endl;
		//int ok=false;
		for (int i=0;i<labelNum;i++)
		{
			root_GPU[rootID]->parameters[10+i]=root_CPU->num_of_each_class[i];
		/*	if (root_CPU->num_of_each_class[i]>0)
			{
				ok=true;
			}*/
		}
	
		//if (!ok)
		//	/*cout<<rootID<<endl;*/
		//{
		//	for (int i=0;i<labelNum;i++)
		//	{
		//		cout<<root_CPU->num_of_each_class[i]<<" ";
		//	}	
		//	cout<<endl;
		//}
		//tree is clear!
	
	}
	if (root_CPU->l_child!=NULL)
	{
		currentID_combination++;
		root_GPU[rootID]->parameters[6]=currentID_combination;
		convertTrees2Array_combination(root_CPU->l_child, root_GPU,labelNum);
	}
	if (root_CPU->r_child!=NULL)
	{
		currentID_combination++;
		root_GPU[rootID]->parameters[7]=currentID_combination;
		convertTrees2Array_combination(root_CPU->r_child, root_GPU,labelNum);
	}
}

thrust::device_vector<float> probMap;

extern "C" void setData_RandomizedTrees_combination(int _max_depth,int _min_sample_count,double _regression_accuracy, int _max_num_of_trees_in_the_forest,int _windowSize, int labelNum, Node **trees_cpu,int treeNum,bool withDepth,int type)
{
	

	RandmizedTree_CUDA *data;
	
	if (type==0)
	{
		data=trackEngine.randomizedTrees_Color;
	}
	else
	{
		data=trackEngine.randomizedTrees_Depth;
	}
	data->max_depth=_max_depth;
	data->min_sample_count=_min_sample_count;
	data->regression_accuracy=_regression_accuracy;
	data->max_num_of_trees_in_the_forest=_max_num_of_trees_in_the_forest;
	data->windowSize=_windowSize;
	data->labelNum=labelNum;

	cout<<"begin feeding trees\n";
	//conversion the tree structure into array
	data->host_trees=new Node_GPU **[data->max_num_of_trees_in_the_forest];
	
	data->MaxNumber=(1-pow((float)2,_max_depth)/(-1));
	int MaxNumber=data->MaxNumber;
	for (int i=0;i<treeNum;i++)
	{
		data->host_trees[i]=new Node_GPU *[MaxNumber];
		for (int j=0;j<MaxNumber;j++)
		{
			data->host_trees[i][j]=new Node_GPU();
			data->host_trees[i][j]->parameters=new float[10+MAX_LABEL_NUMBER];
			data->host_trees[i][j]->parameters[6]=data->host_trees[i][j]->parameters[7]=-2;
		data->host_trees[i][j]->parameters[4]=-1;
		data->host_trees[i][j]->parameters[8]=0;
		data->host_trees[i][j]->parameters[0]=data->host_trees[i][j]->parameters[1]=data->host_trees[i][j]->parameters[2]=data->host_trees[i][j]->parameters[3]=0;
		data->host_trees[i][j]->parameters[9]=0;
		}
		currentID_combination=0;
		convertTrees2Array_combination(trees_cpu[i],data->host_trees[i],labelNum);
		cout<<i<<" "<<currentID_combination<<endl;
	}
	//cout<<"left and right node index: "<<data->host_trees[0][0]->parameters[6]<<" "<<data->host_trees[0][0]->parameters[7]<<endl;

	//ofstream out("tree 0.txt",ios::out);
	//outputTrees(data->host_trees[0],0,out);
	//out.close();

	cout<<"copying trees to GPU\n";
	//CUDA_CALL(cudaMalloc((void **)&data->leafnode,1*sizeof(LeafNode_GPU)));

	//CUDA_CALL(cudaMalloc((void **)&data->trees,data->max_num_of_trees_in_the_forest*sizeof(Node_GPU*)));
	////CUDA_CALL(cudaMemcpy(data->trees,data->host_trees,data->max_num_of_trees_in_the_forest*sizeof(Node_GPU **),cudaMemcpyHostToDevice));
	////data->trees=new Node_GPU **[data->max_num_of_trees_in_the_forest];
	//for (int i=0;i<treeNum;i++)
	//{
	//	CUDA_CALL(cudaMalloc((void **)&data->trees[i],MaxNumber*sizeof(Node_GPU)));
	////	CUDA_CALL(cudaMemcpy(data->trees[i],data->host_trees[i],MaxNumber*sizeof(Node_GPU *),cudaMemcpyHostToDevice));
	//	cout<<i<<endl;
	//	for (int j=0;j<MaxNumber;j++)
	//	{
	//	//	cout<<j<<" ";
	//		CUDA_CALL(cudaMalloc((void **)&data->trees[i][j].cu_parameters,10*sizeof(int)));
	//		CUDA_CALL(cudaMemcpy(data->trees[i][j].cu_parameters,data->host_trees[i][j]->parameters,10*sizeof(int),cudaMemcpyHostToDevice));
	//		if (data->host_trees[i][j]->parameters[6]==-1&&data->host_trees[i][j]->parameters[7]==-1)
	//		{
	//			CUDA_CALL(cudaMalloc((void **)&data->trees[i][j].num_of_each_class,MAX_LABEL_NUMBER*sizeof(float)));
	//			CUDA_CALL(cudaMemcpy(data->trees[i][j].num_of_each_class,data->host_trees[i][j]->num_of_each_class,
	//				MAX_LABEL_NUMBER*sizeof(float),cudaMemcpyHostToDevice));
	//		}
	//	}
	//}	

	////////////////////////////using global memory////////////////////////////////////////////////////
	//float *host_vectorTrees=new float[(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum];

	////root_GPU[currentID]->parameters[0]=root_CPU->pos1[0];
	////root_GPU[currentID]->parameters[1]=root_CPU->pos1[1];
	////root_GPU[currentID]->parameters[2]=root_CPU->pos2[0];
	////root_GPU[currentID]->parameters[3]=root_CPU->pos2[1];
	////root_GPU[currentID]->parameters[4]=root_CPU->label;
	////root_GPU[currentID]->parameters[5]=root_CPU->nLevel;
	////root_GPU[currentID]->parameters[8]=root_CPU->num_all;
	////root_GPU[currentID]->parameters[9]=root_CPU->threshold;

	////cout<<"tree num: "<<treeNum<<endl;


	//
	//cout<<MaxNumber<<endl;
	//cout<<"assigning values\n";
	//for (int i=0;i<treeNum;i++)
	//{
	//	cout<<i<<endl;
	//	for (int j=0;j<MaxNumber;j++)
	//	{
	//		//cout<<i<<" "<<j<<endl;
	//	/*	for (int k=0;k<)
	//		{
	//		}*/
	//		for (int k=0;k<10+MAX_LABEL_NUMBER;k++)
	//		{
	//			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+k]=data->host_trees[i][j]->parameters[k];
	//		}

	///*		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+0]=data->host_trees[i][j].pos1[0];
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+1]=data->host_trees[i][j].pos1[1];
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+2]=data->host_trees[i][j].pos2[0];
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+3]=data->host_trees[i][j].pos2[1];
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+4]=data->host_trees[i][j].label;
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+5]=data->host_trees[i][j].nLevel;
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+8]=data->host_trees[i][j].num_all;
	//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+9]=data->host_trees[i][j].threshold;*/
	//		
	//		

	//		//if (trees_cpu[i][j].l_child==NULL)
	//		//{
	//		//	host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+6]=-1;
	//		//}
	//		//if (trees_cpu[i][j].r_child==NULL)
	//		//{
	//		//	host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+7]=-1;
	//		//}

	//		//for (int i=0;i<MAX_LABEL_NUMBER;i++)
	//		//{
	//		//	host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+10+i]=0;
	//		//}

	//		//if ((trees_cpu[i][j].l_child==NULL)&&trees_cpu[i][j].r_child==NULL)//root
	//		//{
	//		//	for (int i=0;i<labelNum;i++)
	//		//	{
	//		//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+10+i]=trees_cpu[i][j].num_of_each_class[i];
	//		//	}
	//		//}
	//	}
	//}

	//cout<<"copying values\n";

	////using global memory
	//CUDA_CALL(cudaMalloc((void **)&data->cu_vectorTrees,(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float)));
	//CUDA_CALL(cudaMemcpy(data->cu_vectorTrees,host_vectorTrees,(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float),cudaMemcpyHostToDevice));

	//delete []host_vectorTrees;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//treeNum=2;
	///////////////////////////////////////using texture memory//////////////////////////////////////////////////////////////////////
	float *host_vectorTrees=new float[(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum];
	cout<<MaxNumber<<endl;
	cout<<"assigning values\n";
	for (int i=0;i<treeNum;i++)
	{
		cout<<i<<endl;
		for (int j=0;j<MaxNumber;j++)
		{
			for (int k=0;k<10+MAX_LABEL_NUMBER;k++)
			{
				host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+k]=data->host_trees[i][j]->parameters[k];
			}
		}
	}

	CUDA_CALL(cudaMalloc((void **)&data->cu_vectorTrees,(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float)));
	CUDA_CALL(cudaMemcpy(data->cu_vectorTrees,host_vectorTrees,(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float),cudaMemcpyHostToDevice));

	cout<<"width "<< (10+MAX_LABEL_NUMBER)<<" heigth:"<<MaxNumber*treeNum<<" maxNumber:"<<MaxNumber<<endl;

	//CUDA_CALL(cudaBindTexture2D( NULL, trees_device,
	//	RandomizedTreeEngine.cu_vectorTrees,
	//	desc,  (10+MAX_LABEL_NUMBER),MaxNumber*treeNum,
	//	sizeof(float) * (10+MAX_LABEL_NUMBER)));
	//	trees_device.filterMode=cudaFilterModePoint;
			
	
	//CUDA_CALL(cudaBindTexture2D( NULL, trees_device_1D,
	//	RandomizedTreeEngine.cu_vectorTrees,
	//(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float)));
	
	//CUDA_CALL(cudaBindTexture2D( NULL, currentDepthImg,
	//	data->cu_depthImage,
	//	desc,  width,height,
	//	sizeof(float) * width));
	
	delete []host_vectorTrees;
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////
	for (int i=0;i<treeNum;i++)
	{
		for (int j=0;j<MaxNumber;j++)
		{
			delete data->host_trees[i][j];
		}
		delete data->host_trees[i];
	}
	delete []data->host_trees;

	/*cout<<"tree 0 on GPU\n";
	outputTree<<<(data->MaxNumber)/256+1,256>>>(data->cu_vectorTrees,data->MaxNumber,0,data->labelNum);*/
	//CUDA_CALL( cudaMalloc(&data->trees, data->max_num_of_trees_in_the_forest * sizeof(Node_GPU**)) );
	//for (int i=0;i<treeNum;i++)
	//{
	//	data->trees[i]=new Node_GPU *[MaxNumber];
	//	for (int j=0;j<MaxNumber;j++)
	//	{
	//		data->trees[i][j]=new Node_GPU();
	//	}
	//	convertTrees2Array(trees_cpu[i],data->trees[i],0,labelNum);
	//}

	//test
	//Node_GPU*** testTree=new Node_GPU**[data->max_num_of_trees_in_the_forest];
	//CUDA_CALL(cudaMemcpy(testTree,data->trees,data->max_num_of_trees_in_the_forest*sizeof(Node_GPU **),cudaMemcpyDeviceToHost));
	//for (int i=0;i<treeNum;i++)
	//{
	//	for (int j=0;j<MaxNumber;j++)
	//	{
	//		if (data->host_trees[i][j]->l_child_ind==-1&&data->host_trees[i][j]->r_child_ind==-1)
	//		{
	//			continue;
	//		}
	//		//for (int k=0;k<labelNum;k++)
	//		if (j==0)
	//		{
	//			{
	//				cout<<"groundtruth: "<<data->host_trees[i][j]->l_child_ind<<endl;
	//			}
	//			//for (int k=0;k<labelNum;k++)
	//			{
	//				cout<<"current: "<<testTree[i][j]->l_child_ind<<endl;
	//			}
	//		}
	//		
	//		
	//	}
	//	
	//}

	/*Node_GPU *curent=testTree[0][0];
	while(1)
	{
		cout<<curent->l_child_ind<<" "<<curent->r_child_ind<<endl;
		if (curent->l_child_ind==-1&&curent->r_child_ind==-1)
		{
			cout<<curent->label<<endl;
			break;
		}
		else if(1)
		{
			if (curent->l_child_ind<0)
			{
				break;
			}
			curent=testTree[0][curent->l_child_ind];
		}
		else
		{
			if (curent->r_child_ind<0)
			{
				break;
			}
			curent=testTree[0][curent->r_child_ind];
		}
	}
	cout<<"test done!\n";*/
	//data->trees=new Node_GPU[(1-pow(2,_max_depth)/(-1))];
	
	//we do not need allocate memory for rt now. It is in combination, and shared.
	//if(!withDepth)
	//{
	//	CUDA_CALL( cudaMalloc(&data->cu_currentImage, MPN * sizeof(float)) );
	//}
	//else
	//{
	//	CUDA_CALL( cudaMalloc(&data->cu_colorImage, MPN * sizeof(float)) );
	//	CUDA_CALL( cudaMalloc(&data->cu_depthImage, MPN * sizeof(float)) );
	//}

	

	CUDA_CALL( cudaMalloc(&data->cu_LabelResult, MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );


	//CUDA_CALL( cudaMalloc(&data->cu_LabelResultEachTree, treeNum*MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );
	/*CUDA_CALL(cudaBindTexture( NULL, detectionResult1D,
	treeNum*MPN*(1+MAX_LABEL_NUMBER) * sizeof(float));*/
	
	
//	cout<<"labelResult GPU set"<<MPN*(1+MAX_LABEL_NUMBER)<<endl;
////	maximumWidth=640;
	
	
	
//	if (type==1)
//	{
		CUDA_CALL( cudaMalloc(&data->cu_LabelFullResult, MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );
	//}
	
	
//
	//CUDA_CALL(cudaBindTexture2D( NULL, currentDepthImg,
	//	data->cu_depthImage,
	//	desc,  width,height,
	//	sizeof(float) * width));

	if (type==0)
	{
		/*probMap.resize(MPN*(1+MAX_LABEL_NUMBER));
		data->cu_LabelFullResult=thrust::raw_pointer_cast(&(probMap[0]));*/

		//CUDA_CALL( cudaMalloc(&data->cu_LabelFullResult, MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );

		CUDA_CALL(cudaBindTexture2D( NULL, detectionResult,
			data->cu_LabelFullResult,
			desc_combination, 640 ,(1+MAX_LABEL_NUMBER)*480,
			sizeof(float) * 640));
		//currentImg.filterMode=cudaFilterModePoint;
		currentColorImg_combination.filterMode=cudaFilterModePoint;
		currentDepthImg_combination.filterMode=cudaFilterModePoint;
		
	}
	



	
	//data->host_LabelResult=new LabelResult[MPN];

	CUDA_CALL( cudaMalloc(&data->cu_maxIndPerBlock, blocksPerGrid*2*MAX_LABEL_NUMBER * sizeof(float)) );
	data->maxIndPerBlock=new float[blocksPerGrid*2*MAX_LABEL_NUMBER];

	showCUDAMemoryUsage();
	//CUDA_CALL( cudaMalloc(&data->cu_maxValuePerBlock, blocksPerGrid * sizeof(float)) );
}

__device__ void getProb_depth_depth_combination(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber,float cT=1,float sT=0)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label;float threshold;;
	int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	int startInd;
	//int currentInd=treeInd;
	while(1)
	//for(int i=0;i<2;i++)
	{
		//printf("%d\n",startInd);
		startInd=initialInd+currentInd*(10+MAX_LABEL_NUMBER);
		 l_child_ind=trees[startInd+6];
		 r_child_ind=trees[startInd+7];
		 pos1[0]=trees[startInd+0];
		 pos1[1]=trees[startInd+1];
		  pos2[0]=trees[startInd+2];
		   pos2[1]=trees[startInd+3];
		   label=trees[startInd+4];
		   threshold=trees[startInd+9];

		
		//printf("after getting values %d %d\n",l_child_ind,r_child_ind);
		if (l_child_ind==-1&&r_child_ind==-1)//leaf
		//if (1)//leaf
		{			
	/*		printf("leafNode \n");
			for (int i=0;i<6;i++)
			{
				printf("%f ",trees[startInd+10+i]);
			}
			printf("\n");*/
			ind[0]=currentInd;
			ind[1]=label;
			break;
		}

		float curDepth=1.0f/tex2D(currentDepthImg_combination,pos[0],pos[1]);

		float realPos1[2];
		realPos1[0]=cT*(float)pos1[0]-sT*(float)pos1[1];
		realPos1[1]=sT*(float)pos1[0]+cT*(float)pos1[1];

		float realPos2[2];
		realPos2[0]=cT*(float)pos2[0]-sT*(float)pos2[1];
		realPos2[1]=sT*(float)pos2[0]+cT*(float)pos2[1];

		//according to the train style
		//if (trainStyle==0)
		{
			if (tex2D(currentDepthImg_combination, realPos1[0]*curDepth+pos[0],realPos1[1]*curDepth+pos[1])>
				tex2D(currentDepthImg_combination, realPos2[0]*curDepth+pos[0], realPos2[1]*curDepth+pos[1])+threshold)
			{

				if (l_child_ind<0)
				{
					break;
				}
				//	printf("going to left %d\n",l_child_ind);
				currentInd=l_child_ind;
				//current=&(root[l_child_ind]);
			}
			else
			{

				if (r_child_ind<0)//leaf
				{
					break;
				}
				//	printf("going to right %d\n",r_child_ind);
				currentInd=r_child_ind;
				//current=&(root[r_child_ind]);
			}
		}
	}
}

__device__ void getProb_depth_color_combination(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber,float cT=1,float sT=0)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label;float threshold;;
	int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	int startInd;
	//int currentInd=treeInd;
	while(1)
	//for(int i=0;i<2;i++)
	{
		//printf("%d\n",startInd);
		startInd=initialInd+currentInd*(10+MAX_LABEL_NUMBER);
		 l_child_ind=trees[startInd+6];
		 r_child_ind=trees[startInd+7];
		 pos1[0]=trees[startInd+0];
		 pos1[1]=trees[startInd+1];
		  pos2[0]=trees[startInd+2];
		   pos2[1]=trees[startInd+3];
		   label=trees[startInd+4];
		   threshold=trees[startInd+9];

		
		//printf("after getting values %d %d\n",l_child_ind,r_child_ind);
		if (l_child_ind==-1&&r_child_ind==-1)//leaf
		//if (1)//leaf
		{			
	/*		printf("leafNode \n");
			for (int i=0;i<6;i++)
			{
				printf("%f ",trees[startInd+10+i]);
			}
			printf("\n");*/
			ind[0]=currentInd;
			ind[1]=label;
			break;
		}

		float curDepth=1.0f/tex2D(currentDepthImg_combination,pos[0],pos[1]);

		float realPos1[2];
		realPos1[0]=cT*(float)pos1[0]-sT*(float)pos1[1];
		realPos1[1]=sT*(float)pos1[0]+cT*(float)pos1[1];

		float realPos2[2];
		realPos2[0]=cT*(float)pos2[0]-sT*(float)pos2[1];
		realPos2[1]=sT*(float)pos2[0]+cT*(float)pos2[1];

		//according to the train style
		/*if (mat.at<float>((double)current->pos1[1]*curDepth+pos.y,(double)current->pos1[0]*curDepth+pos.x)>
			mat.at<float>((double)current->pos2[1]*curDepth+pos.y,(double)current->pos2[0]*curDepth+pos.x)+current->threshold)*/
		{
			if (tex2D(currentColorImg_combination,realPos1[0]*curDepth+pos[0],realPos1[1]*curDepth+pos[1])>
				tex2D(currentColorImg_combination,realPos2[0]*curDepth+pos[0],realPos2[1]*curDepth+pos[1])+threshold)
			{

				if (l_child_ind<0)
				{
					break;
				}
				//	printf("going to left %d\n",l_child_ind);
				currentInd=l_child_ind;
				//current=&(root[l_child_ind]);
			}
			else
			{

				if (r_child_ind<0)//leaf
				{
					break;
				}
				//	printf("going to right %d\n",r_child_ind);
				currentInd=r_child_ind;
				//current=&(root[r_child_ind]);
			}
		}	
	}
}

__device__ void getProb_depth_depth_color_combination(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label;float threshold;;
	int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	int startInd;
	//int currentInd=treeInd;
	while(1)
	//for(int i=0;i<2;i++)
	{
		//printf("%d\n",startInd);
		startInd=initialInd+currentInd*(10+MAX_LABEL_NUMBER);
		 l_child_ind=trees[startInd+6];
		 r_child_ind=trees[startInd+7];
		 pos1[0]=trees[startInd+0];
		 pos1[1]=trees[startInd+1];
		  pos2[0]=trees[startInd+2];
		   pos2[1]=trees[startInd+3];
		   label=trees[startInd+4];
		   threshold=trees[startInd+9];

		
		//printf("after getting values %d %d\n",l_child_ind,r_child_ind);
		if (l_child_ind==-1&&r_child_ind==-1)//leaf
		//if (1)//leaf
		{			
	/*		printf("leafNode \n");
			for (int i=0;i<6;i++)
			{
				printf("%f ",trees[startInd+10+i]);
			}
			printf("\n");*/
			ind[0]=currentInd;
			//ind[1]=label;
			break;
		}

		float curDepth=1.0f/tex2D(currentDepthImg_combination,pos[0],pos[1]);
		{
			if (threshold>=0)
			{
				if (tex2D(currentDepthImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])==0||
					tex2D(currentDepthImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])==0)
				{
					//ind[0]=-1;
					ind[1]=-1;
					//break;
				}
				if (tex2D(currentDepthImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentDepthImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
				{

					if (l_child_ind<0)
					{
						break;
					}
					//	printf("going to left %d\n",l_child_ind);
					currentInd=l_child_ind;
					//current=&(root[l_child_ind]);
				}
				else
				{

					if (r_child_ind<0)//leaf
					{
						break;
					}
					//	printf("going to right %d\n",r_child_ind);
					currentInd=r_child_ind;
					//current=&(root[r_child_ind]);
				}
			}
			else
			{
				if (tex2D(currentColorImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentColorImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+(-threshold-1))
				{

					if (l_child_ind<0)
					{
						break;
					}
					//	printf("going to left %d\n",l_child_ind);
					currentInd=l_child_ind;
					//current=&(root[l_child_ind]);
				}
				else
				{

					if (r_child_ind<0)//leaf
					{
						break;
					}
					//	printf("going to right %d\n",r_child_ind);
					currentInd=r_child_ind;
					//current=&(root[r_child_ind]);
				}
			}
		
		}
	
	}
}

__device__ void getProb_depth_combination(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber,int trainStyle)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label;
	float threshold;
	int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	int startInd;
	//int currentInd=treeInd;
	while(1)
	//for(int i=0;i<2;i++)
	{
		//printf("%d\n",startInd);
		startInd=initialInd+currentInd*(10+MAX_LABEL_NUMBER);
		 l_child_ind=trees[startInd+6];
		 r_child_ind=trees[startInd+7];
		 pos1[0]=trees[startInd+0];
		 pos1[1]=trees[startInd+1];
		  pos2[0]=trees[startInd+2];
		   pos2[1]=trees[startInd+3];
		   label=trees[startInd+4];
		   threshold=trees[startInd+9];

		
		//printf("after getting values %d %d\n",l_child_ind,r_child_ind);
		if (l_child_ind==-1&&r_child_ind==-1)//leaf
		//if (1)//leaf
		{			
	/*		printf("leafNode \n");
			for (int i=0;i<6;i++)
			{
				printf("%f ",trees[startInd+10+i]);
			}
			printf("\n");*/
			ind[0]=currentInd;
			ind[1]=label;
			break;
		}

		float curDepth=1.0f/tex2D(currentDepthImg_combination,pos[0],pos[1]);

		//according to the train style
		if (trainStyle==0)
		{
			if (tex2D(currentDepthImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentDepthImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
			{

				if (l_child_ind<0)
				{
					break;
				}
				//	printf("going to left %d\n",l_child_ind);
				currentInd=l_child_ind;
				//current=&(root[l_child_ind]);
			}
			else
			{

				if (r_child_ind<0)//leaf
				{
					break;
				}
				//	printf("going to right %d\n",r_child_ind);
				currentInd=r_child_ind;
				//current=&(root[r_child_ind]);
			}
		}
		else if (trainStyle==1)
		{
			if (tex2D(currentColorImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentColorImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
			{

				if (l_child_ind<0)
				{
					break;
				}
				//	printf("going to left %d\n",l_child_ind);
				currentInd=l_child_ind;
				//current=&(root[l_child_ind]);
			}
			else
			{

				if (r_child_ind<0)//leaf
				{
					break;
				}
				//	printf("going to right %d\n",r_child_ind);
				currentInd=r_child_ind;
				//current=&(root[r_child_ind]);
			}
		}
		else
		{
			if (threshold>=0)
			{
				if (tex2D(currentDepthImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentDepthImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
				{

					if (l_child_ind<0)
					{
						break;
					}
					//	printf("going to left %d\n",l_child_ind);
					currentInd=l_child_ind;
					//current=&(root[l_child_ind]);
				}
				else
				{

					if (r_child_ind<0)//leaf
					{
						break;
					}
					//	printf("going to right %d\n",r_child_ind);
					currentInd=r_child_ind;
					//current=&(root[r_child_ind]);
				}
			}
			else
			{
				if (tex2D(currentColorImg_combination,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentColorImg_combination,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1]))
				{

					if (l_child_ind<0)
					{
						break;
					}
					//	printf("going to left %d\n",l_child_ind);
					currentInd=l_child_ind;
					//current=&(root[l_child_ind]);
				}
				else
				{

					if (r_child_ind<0)//leaf
					{
						break;
					}
					//	printf("going to right %d\n",r_child_ind);
					currentInd=r_child_ind;
					//current=&(root[r_child_ind]);
				}
			}
		
		}
	
	}
}

const int perBlockForRT=512;
__global__ void predict_prob_withDepth_color_combination_eachTree(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,int sx=0,int ex=1000,int sy=0,int ey=1000)
{

	int offset=blockIdx.x;
	int treeID=threadIdx.x;
	int step=perBlockForRT;

	if (offset>=step||treeID>treeNum)
	{
		return;
	}

	
	int totalNum=width*height;

	__shared__ float label_prob_all[MAX_LABEL_NUMBER];
	
	

	while(offset<width*height)
	{
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;
		int totalNumPerImg=width*height;

		//0.18
		if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
			(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		{
			if (treeID==0)
			{
				for (int i=0;i<labelNum;i++)
				{
					result[offset+i*totalNumPerImg]=0;
				}
			}	
			offset+=step;
			continue;
		}
		
		{
			for (int i=0;i<labelNum;i++)
			{
				label_prob_all[i]=0;
			}
			__syncthreads();

			//1.7

			int ind[2]={0,0};
			int startInd;


			getProb_depth_color_combination(trees,treeID,pos,ind,0,MaxNumber);
			startInd=treeID*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);

			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{
				for (int j=0;j<labelNum;j++)
				{
					//label_prob_all[j]+=trees[startInd+10+j];
					atomicAdd(label_prob_all+j,trees[startInd+10+j]);
				}
			}			

			__syncthreads();
			//0.1
			if (treeID==0)
			{
				for (int i=0;i<labelNum;i++)
				{
					result[offset+totalNumPerImg*i]=label_prob_all[i]/(float)treeNum;
					//result[offset+totalNumPerImg*i]=1;
					//	tex2D(detectionResult2D,i*height+pos[0],pos[1])=label_prob_all[i]/(float)treeNum;
				}
			}

		}
	
		
		offset+=step;
		__syncthreads();
	}
	
}

__global__ void predict_prob_withDepth_color_depth_combination(float *result,int labelNum,int treeNum_color,int width,int height,int windowSize,float *trees_color,int MaxNumber,int treeNum_depth,float *trees_depth,int sx=0,int ex=1000,int sy=0,int ey=1000)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>=width*height)
	{
		return;
	}

	int pos[2];
	pos[0]=offset%width;
	pos[1]=offset/width;
	int totalNumPerImg=width*height;

	//0.18
	if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
		(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		//if(trackEngine.cu_depthImage[offset]==0)
	{
		//result[offset+MAX_LABEL_NUMBER*totalNumPerImg]=labelNum-1;
		//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
		for (int i=0;i<labelNum;i++)
		{
			result[offset+i*totalNumPerImg]=0;
		}
		return;
	}

	//1.7
	double label_prob_color[MAX_LABEL_NUMBER],label_prob_depth[MAX_LABEL_NUMBER];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_color[i]=0;
		label_prob_depth[i]=0;
	}
	int ind_color[2]={0,0};
	int ind_depth[2]={0,0};
	int startInd_color,startInd_depth;

	//return;
	//about 210ms
	for (int i=0;i<treeNum_color;i++)
	{

		getProb_depth_color_combination(trees_color,i,pos,ind_color,0,MaxNumber);
		startInd_color=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind_color[0]*(10+MAX_LABEL_NUMBER);

		if (trees_color[startInd_color+6]==-1&&trees_color[startInd_color+7]==-1) //reach a leaf
		{

			for (int j=0;j<labelNum;j++)
			{
				label_prob_color[j]+=trees_color[startInd_color+10+j];
			}
		}			
	}

	for (int i=0;i<treeNum_depth;i++)
	{

		getProb_depth_depth_combination(trees_depth,i,pos,ind_depth,0,MaxNumber);
		startInd_depth=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind_depth[0]*(10+MAX_LABEL_NUMBER);

		if (trees_depth[startInd_depth+6]==-1&&trees_depth[startInd_depth+7]==-1) //reach a leaf
		{

			for (int j=0;j<labelNum;j++)
			{
				label_prob_depth[j]+=trees_depth[startInd_depth+10+j];
			}
		}			
	}

	//0.1

		result[offset+totalNumPerImg*0]=label_prob_color[0]/(float)treeNum_color;
		result[offset+totalNumPerImg*1]=label_prob_color[1]/(float)treeNum_color;
		result[offset+totalNumPerImg*2]=label_prob_color[2]/(float)treeNum_color;
		result[offset+totalNumPerImg*3]=label_prob_color[3]/(float)treeNum_color;

	for (int i=4;i<labelNum;i++)
	{
			result[offset+totalNumPerImg*i]=(label_prob_color[i]/(float)treeNum_color+label_prob_depth[i]/(float)treeNum_depth)*0.5;		
		//	tex2D(detectionResult2D,i*height+pos[0],pos[1])=label_prob_all[i]/(float)treeNum;
	}
}


__global__ void predict_prob_withDepth_color_combination(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,int sx=0,int ex=1000,int sy=0,int ey=1000,float cT=1,float sT=0)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>=width*height)
	{
		return;
	}

	int pos[2];
	pos[0]=offset%width;
	pos[1]=offset/width;
	int totalNumPerImg=width*height;

	//0.18
	if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
		(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		//if(trackEngine.cu_depthImage[offset]==0)
	{
		//result[offset+MAX_LABEL_NUMBER*totalNumPerImg]=labelNum-1;
		//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
		for (int i=0;i<labelNum;i++)
		{
			result[offset+i*totalNumPerImg]=0;
		}
		return;
	}

	//1.7
	double label_prob_all[MAX_LABEL_NUMBER];
	for (int i=0;i<labelNum;i++)
	{
		label_prob_all[i]=0;
	}
	int ind[2]={0,0};
	int startInd;

	//return;
	//about 210ms
	for (int i=0;i<treeNum;i++)
	{

		getProb_depth_color_combination(trees,i,pos,ind,0,MaxNumber,cT,sT);
		startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);

		if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
		{

			for (int j=0;j<labelNum;j++)
			{
				label_prob_all[j]+=trees[startInd+10+j];
			}
		}			
	}

	//0.1
	for (int i=0;i<labelNum;i++)
	{
		result[offset+totalNumPerImg*i]=label_prob_all[i]/(float)treeNum;
		//	tex2D(detectionResult2D,i*height+pos[0],pos[1])=label_prob_all[i]/(float)treeNum;
	}
}

__global__ void colorDepthCombination(float *depthResult,float *colorResult,float *result,int num_depthTrees,	int num_colorTrees,int labelNum, int width,int height)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<width*height)
	{
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;
		if(tex2D(currentDepthImg_combination,pos[0],pos[1])==0)
			return;

		int totalNumPerImg=width*height;

		int totalTreeNum=num_colorTrees+num_depthTrees;
		for (int i=0;i<labelNum;i++)
		{
			float colorVal=colorResult[offset+totalNumPerImg*i];
			float depthVal=depthResult[offset+totalNumPerImg*i];
			//result[offset+totalNumPerImg*i]=(num_colorTrees*colorVal+num_depthTrees*depthVal)/totalTreeNum;
			if (i<4)
			{
				continue;
			}
			result[offset+totalNumPerImg*i]=(colorVal+depthVal)/2;

			//result[offset+totalNumPerImg*i]=depthVal;
		}
	}
}

__global__ void predict_prob_withDepth_depth_combination_withCombination_EachTree(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,float *colorResult,int sx=0,int ex=1000,int sy=0,int ey=1000)
{
	int offset=blockIdx.x;
	int treeID=threadIdx.x;
	int step=perBlockForRT;

	if (offset>=step||treeID>treeNum)
	{	
		return;
	}

	
	int totalNum=width*height;

	__shared__ float label_prob_all[MAX_LABEL_NUMBER];



	while(offset<width*height)
	{

		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		int totalNumPerImg=width*height;

		if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
			(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		{
			offset+=step;continue;
		}
		{
			for (int i=0;i<labelNum;i++)
			{
				label_prob_all[i]=0;
			}
			__syncthreads();

			int ind[2]={0,0};
			int startInd;
			//int cur_row,cur_col;
			//about 210ms

			getProb_depth_depth_combination(trees,treeID,pos,ind,0,MaxNumber);

			startInd=treeID*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);

			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{

				for (int j=0;j<labelNum;j++)
				{
					//label_prob_all[j]+=trees[startInd+10+j];
					atomicAdd(label_prob_all+j,trees[startInd+10+j]);
				}
			}
			__syncthreads();

			if (treeID==0)
			{
				float tmp_depth,tmp_color;

			/*	for (int i=0;i<labelNum;i++)
				{
					tmp_depth=label_prob_all[i]/(float)treeNum;
				
					result[offset+totalNumPerImg*i]=tmp_depth;
				}*/

				for (int i=4;i<labelNum;i++)
				{
					tmp_depth=label_prob_all[i]/(float)treeNum;
					tmp_color=colorResult[offset+totalNumPerImg*i];
					result[offset+totalNumPerImg*i]=(tmp_color+tmp_depth)*0.5;
				}
			}
		}
		
		offset+=step;
		__syncthreads();
	}
}



__global__ void predict_prob_withDepth_depth_combination(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,int sx=0,int ex=1000,int sy=0,int ey=1000)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>=width*height)
	{
		return;
	}
	if (offset<width*height)
	{
	//	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	/*	if (offset!=100*width+100)
		{
			return;
		}*/
	//	printf("%d\n",offset);
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		int totalNumPerImg=width*height;

		if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
			(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		{

			int currentInd=offset*(1+MAX_LABEL_NUMBER);
			//if (label_prob_all[maxInd]>threshold)
			{
				result[offset+MAX_LABEL_NUMBER*totalNumPerImg]=labelNum-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (int i=0;i<labelNum;i++)
				{
					result[offset+i*totalNumPerImg]=0;
				}
				//result[currentInd+0]=labelNum-1;
				////result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				//for (int i=0;i<labelNum;i++)
				//{
				//	result[currentInd+1+i]=0;
				//}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}
			return;
		}


		double label_prob_all[MAX_LABEL_NUMBER];
		for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}
		int ind[2]={0,0};
		int startInd;
		//int cur_row,cur_col;
		//about 210ms
		for (int i=0;i<treeNum;i++)
		{

			getProb_depth_depth_combination(trees,i,pos,ind,0,MaxNumber);
			//startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
			//
			//if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			//{
			//	
			//	for (j=0;j<labelNum;j++)
			//	{
			//		label_prob_all[j]+=trees[startInd+10+j];
			//	}
			//}
		
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);

			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{

				for (int j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
			}
		}
		
		//return;
		////find the most frequent label
		//float maxInd=0;
		//double maxNum=-1;
		//for (i=0;i<labelNum;i++)
		//{
		//	if (label_prob_all[i]>maxNum)
		//	{
		//		maxNum=label_prob_all[i];
		//		maxInd=i;
		//	}
		//}

		////about 20ms
		//int currentInd=offset*(1+MAX_LABEL_NUMBER);
		////if (label_prob_all[maxInd]>threshold)
		//{
		//	result[currentInd+0]=maxInd;
		//	//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
		//	for (i=0;i<labelNum;i++)
		//	{
		//		result[currentInd+1+i]=label_prob_all[i]/(float)treeNum;
		//	}
		//	//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
		//}
		for (int i=0;i<labelNum;i++)
		{
			result[offset+totalNumPerImg*i]=label_prob_all[i]/(float)treeNum;
		}
	
	}
}


__global__ void predict_prob_withDepth_depth_combination_withCombination(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,float *colorResult,int sx=0,int ex=1000,int sy=0,int ey=1000,float cT=1,float sT=0)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>=width*height)
	{
		return;
	}
	if (offset<width*height)
	{
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		int totalNumPerImg=width*height;

		if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
			(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		{

			return;
		}


		double label_prob_all[MAX_LABEL_NUMBER];
		for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}
		int ind[2]={0,0};
		int startInd;
		//int cur_row,cur_col;
		//about 210ms
		for (int i=0;i<treeNum;i++)
		{

			getProb_depth_depth_combination(trees,i,pos,ind,0,MaxNumber,cT,sT);
		
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);

			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{

				for (int j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
			}
		}
		float tmp_depth,tmp_color;
		
	/*	for (int i=0;i<labelNum;i++)
		{
			 tmp_depth=label_prob_all[i]/(float)treeNum;
			tmp_color=colorResult[offset+totalNumPerImg*i];
			result[offset+totalNumPerImg*i]=tmp_color;
		}*/

		int eyeNum=4;
		for (int i=0;i<eyeNum;i++)
		{
			tmp_depth=label_prob_all[i]/(float)treeNum;
			tmp_color=colorResult[offset+totalNumPerImg*i];
			result[offset+totalNumPerImg*i]=tmp_color;
			//result[offset+totalNumPerImg*i]=(tmp_depth*tmp_color);
			//	result[offset+totalNumPerImg*i]=(tmp_depth+tmp_depth)*0.5;
		}

		for (int i=eyeNum;i<labelNum;i++)
		{
			/*if (i>=12&&i<=17)
			{
				continue;
			}*/
			tmp_depth=label_prob_all[i]/(float)treeNum;
			tmp_color=colorResult[offset+totalNumPerImg*i];
			result[offset+totalNumPerImg*i]=(tmp_depth+tmp_color)*0.5;
		//result[offset+totalNumPerImg*i]=(tmp_depth*tmp_color);
			//	result[offset+totalNumPerImg*i]=(tmp_depth+tmp_depth)*0.5;
		}
	
	}
}


__global__ void predict_prob_withDepth_depth_combination_withLastFrame(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,float *colorResult,
	float *lastShape,int *usedIndex,int ptsNum,int sx=0,int ex=1000,int sy=0,int ey=1000)
{
	int tID=threadIdx.x;
	__shared__ float currentShape[44];//21*2
	if (tID<labelNum-1)
	{
		currentShape[tID]=lastShape[usedIndex[tID]];
		currentShape[tID+labelNum]=lastShape[usedIndex[tID]+ptsNum];
	/*	currentShape[tID]=lastShape[tID];
		currentShape[tID+labelNum]=lastShape[tID+ptsNum];*/
	}
	__syncthreads();

	int offset=tID+blockIdx.x*blockDim.x;
	if (offset>=width*height)
	{
		return;
	}
	if (offset<width*height)
	{
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		int totalNumPerImg=width*height;

		if(((pos[0]<=sx||pos[0]>=ex||pos[1]<=sy||pos[1]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)||
			(pos[0]<=windowSize||pos[0]>=width-windowSize||	pos[1]<=windowSize||pos[1]>=height-windowSize))
		{

			return;
		}


		double label_prob_all[MAX_LABEL_NUMBER];
		for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}
		int ind[2]={0,0};
		int startInd;
		//int cur_row,cur_col;
		//about 210ms
		for (int i=0;i<treeNum;i++)
		{

			getProb_depth_depth_combination(trees,i,pos,ind,0,MaxNumber);
		
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);

			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{

				for (int j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
			}
		}


		//get the posterior prob

		float curPrior[22];
		for (int i=0;i<labelNum-1;i++)
		{
			float cDx=pos[0]-currentShape[i];
			float cDy=pos[1]-currentShape[i+labelNum];
			float curDis=cDx*cDx+cDy*cDy;
			curPrior[i]=powf(2.7173f,-curDis/3200);//(2*sigma*sigma),sigma=3
		}

		float tmp_depth,tmp_color;
		//powf(2.7173f,-curDis/(2*sigma*sigma))

		for (int i=0;i<11;i++)
		{
			tmp_depth=label_prob_all[i]/(float)treeNum;
			tmp_color=colorResult[offset+totalNumPerImg*i];
			//result[offset+totalNumPerImg*i]=(tmp_depth+tmp_color)*0.5;
			result[offset+totalNumPerImg*i]=(tmp_depth*tmp_color)*curPrior[i];
			//	result[offset+totalNumPerImg*i]=(tmp_depth+tmp_depth)*0.5;
		}

		for (int i=11;i<labelNum-1;i++)
		{
			tmp_depth=label_prob_all[i]/(float)treeNum;
			tmp_color=colorResult[offset+totalNumPerImg*i];
			result[offset+totalNumPerImg*i]=(tmp_depth+tmp_color)*0.5*curPrior[i];
		//result[offset+totalNumPerImg*i]=(tmp_depth*tmp_color);
			//	result[offset+totalNumPerImg*i]=(tmp_depth+tmp_depth)*0.5;
		}

	
	
	}
}



//suppose the maximum class number is 10
__global__ void predict_prob_withDepth_combination(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,int trainStyle)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<width*height)
	{
	//	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	/*	if (offset!=100*width+100)
		{
			return;
		}*/
	//	printf("%d\n",offset);
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		if(tex2D(currentDepthImg_combination,pos[0],pos[1])==0)
		{

			int currentInd=offset*(1+MAX_LABEL_NUMBER);
			//if (label_prob_all[maxInd]>threshold)
			{
				result[currentInd+0]=labelNum-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (int i=0;i<labelNum;i++)
				{
					result[currentInd+1+i]=0;
				}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}
			return;
		}

		int i,j;

		if (pos[0]<=windowSize||pos[0]>=width-windowSize||
			pos[1]<=windowSize||pos[1]>=height-windowSize)
		{
			int currentInd=offset*(1+MAX_LABEL_NUMBER);
			//if (label_prob_all[maxInd]>threshold)
			{
				result[currentInd+0]=-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (i=0;i<labelNum;i++)
				{
					result[currentInd+1+i]=0;
				}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}

			return;
		}

		double label_prob_all[MAX_LABEL_NUMBER];
		for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}
		int ind[2]={0,0};
		int startInd;

		//about 210ms
		for (i=0;i<treeNum;i++)
		{

			getProb_depth_combination(trees,i,pos,ind,0,MaxNumber,trainStyle);
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
			
			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{
				
				for (j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
			}			
		}
		
	//	return;
		////find the most frequent label
		float maxInd=0;
		double maxNum=-1;
		for (i=0;i<labelNum;i++)
		{
			if (label_prob_all[i]>maxNum)
			{
				maxNum=label_prob_all[i];
				maxInd=i;
			}
		}

		//about 20ms
		int currentInd=offset*(1+MAX_LABEL_NUMBER);
		//if (label_prob_all[maxInd]>threshold)
		{
			result[currentInd+0]=maxInd;
			//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
			for (i=0;i<labelNum;i++)
			{
				result[currentInd+1+i]=label_prob_all[i]/(float)treeNum;
			}
			//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
		}
		
	}
}

__global__ void predict_prob_withDepth_depth_color_ImageResult_combination(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<width*height)
	{
	//	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	/*	if (offset!=100*width+100)
		{
			return;
		}*/
	//	printf("%d\n",offset);
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		int totalNumPerImg=width*height;

		if(tex2D(currentDepthImg_combination,pos[0],pos[1])==0)
		{
			//if (label_prob_all[maxInd]>threshold)
			{
				result[offset+MAX_LABEL_NUMBER*totalNumPerImg]=labelNum-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (int i=0;i<labelNum;i++)
				{
					result[offset+i*totalNumPerImg]=0;
				}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}
			return;
		}

		int i,j;

		if (pos[0]<=windowSize||pos[0]>=width-windowSize||
			pos[1]<=windowSize||pos[1]>=height-windowSize)
		{
			//int currentInd=offset*(1+MAX_LABEL_NUMBER);
			//if (label_prob_all[maxInd]>threshold)
			{
				result[offset+MAX_LABEL_NUMBER*totalNumPerImg]=-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (i=0;i<labelNum;i++)
				{
					result[offset+i*totalNumPerImg]=0;
				}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}

			return;
		}

		double label_prob_all[MAX_LABEL_NUMBER];
		for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}
		int ind[2]={0,0};
		int startInd;

		//about 210ms
		for (i=0;i<treeNum;i++)
		{

			getProb_depth_depth_color_combination(trees,i,pos,ind,0,MaxNumber);
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
			
			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{
				
				for (j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
			}			
		}
		
		//return;
		////find the most frequent label
		float maxInd=0;
		double maxNum=-1;
		for (i=0;i<labelNum;i++)
		{
			if (label_prob_all[i]>maxNum)
			{
				maxNum=label_prob_all[i];
				maxInd=i;
			}
		}

		//about 20ms, very time consuming
		//int currentInd=offset*(1+MAX_LABEL_NUMBER);
		//if (label_prob_all[maxInd]>threshold)
		//{
			result[offset+MAX_LABEL_NUMBER*totalNumPerImg]=maxInd;
			
		//	tex2D(detectionResult2D,labelNum*height+pos[0],pos[1])=maxInd;
			//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
			for (i=0;i<labelNum;i++)
			{
				result[offset+totalNumPerImg*i]=label_prob_all[i]/(float)treeNum;
			//	tex2D(detectionResult2D,i*height+pos[0],pos[1])=label_prob_all[i]/(float)treeNum;
			}
			//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
		//}
		
	}
}

__global__ void predict_prob_withDepth_depth_color_combination(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<width*height)
	{
	//	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	/*	if (offset!=100*width+100)
		{
			return;
		}*/
	//	printf("%d\n",offset);
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;

		if(tex2D(currentDepthImg_combination,pos[0],pos[1])==0)
		{

			int currentInd=offset*(1+MAX_LABEL_NUMBER);
			//if (label_prob_all[maxInd]>threshold)
			{
				result[currentInd+0]=labelNum-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (int i=0;i<labelNum;i++)
				{
					result[currentInd+1+i]=0;
				}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}
			return;
		}

		int i,j;

		if (pos[0]<=windowSize||pos[0]>=width-windowSize||
			pos[1]<=windowSize||pos[1]>=height-windowSize)
		{
			int currentInd=offset*(1+MAX_LABEL_NUMBER);
			//if (label_prob_all[maxInd]>threshold)
			{
				result[currentInd+0]=-1;
				//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
				for (i=0;i<labelNum;i++)
				{
					result[currentInd+1+i]=0;
				}
				//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
			}

			return;
		}

		double label_prob_all[MAX_LABEL_NUMBER];
		for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}
		int ind[2]={0,0};
		int startInd;

		//about 210ms
		for (i=0;i<treeNum;i++)
		{

			getProb_depth_depth_color_combination(trees,i,pos,ind,0,MaxNumber);

			/*if (ind[0]==-1)
			{
				continue;
			}*/
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
			
			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{
				
				for (j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
			}			
		}
		
		//return;
		////find the most frequent label
		float maxInd=0;
		double maxNum=-1;
		for (i=0;i<labelNum;i++)
		{
			if (label_prob_all[i]>maxNum)
			{
				maxNum=label_prob_all[i];
				maxInd=i;
			}
		}

		//about 20ms, very time consuming
		int currentInd=offset*(1+MAX_LABEL_NUMBER);
		//if (label_prob_all[maxInd]>threshold)
		//{
			result[currentInd]=maxInd;
			
		//	tex2D(detectionResult2D,labelNum*height+pos[0],pos[1])=maxInd;
			//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
			for (i=0;i<labelNum;i++)
			{
				result[currentInd+1+i]=label_prob_all[i]/(float)treeNum;
			//	tex2D(detectionResult2D,i*height+pos[0],pos[1])=label_prob_all[i]/(float)treeNum;
			}
			//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
		//}
		
	}
}

__global__ void findMaxValue_full(int width,int height,float *result)
{
	__shared__ float maxIndex[threadsPerBlock_global];
	__shared__ float maxValue[threadsPerBlock_global];
	
	int labelId=blockIdx.y;

	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int cacheIndex=threadIdx.x;

	int pos[2];

	int minInd; float max_value;
	float currentValue;
	max_value=0;
	minInd=0;

	int gridSize=blockDim.x*gridDim.x;
	while (offset<width*height)
	{
		//tex2D(currentDepthImg_combination,pos[0],pos[1])
		pos[0]=offset%width;
		pos[1]=offset/width;

		
		currentValue=tex2D(detectionResult,pos[0],labelId*height+pos[1]);


		//currentValue=detectionResult[offset+width*height*2];
		//currentValue=detectionResult[offset*(1+MAX_LABEL_NUMBER)];
		if (currentValue>max_value)
		{
			max_value=currentValue;
			minInd=offset;
		}
		offset+=gridSize;
	}
	maxIndex[cacheIndex]=minInd;
	maxValue[cacheIndex]=max_value;

	__syncthreads();
	

	int i=blockDim.x/2;
	while (i!=0)
	{
		if (cacheIndex<i)
		{
			if (maxValue[cacheIndex]<maxValue[cacheIndex+i])
			{
				maxValue[cacheIndex]=maxValue[cacheIndex+i];
				maxIndex[cacheIndex]=maxIndex[cacheIndex+i];
			}
		}
		__syncthreads();
		i/=2;
	}

	if (cacheIndex==0)
	{
		result[blockIdx.x+labelId*blocksPerGrid*2]=maxIndex[0];
		/*pos[0]=(int)maxIndex[0]%width;
		pos[1]=(int)maxIndex[0]/width;
		result[blockIdx.x+blocksPerGrid]=tex2D(detectionResult,pos[0],labelId*height+pos[1]);*/
		result[blockIdx.x+blocksPerGrid+labelId*blocksPerGrid*2]=maxValue[0];
	}
}


__global__ void findMaxValue(int width,int height,int labelId,float *result)
{
	__shared__ float maxIndex[threadsPerBlock_global];
	__shared__ float maxValue[threadsPerBlock_global];
	

	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int cacheIndex=threadIdx.x;

	int pos[2];

	int minInd; float max_value;
	float currentValue;
	max_value=0;
	minInd=0;
	while (offset<width*height)
	{
		//tex2D(currentDepthImg_combination,pos[0],pos[1])
		pos[0]=offset%width;
		pos[1]=offset/width;

		
		currentValue=tex2D(detectionResult,pos[0],labelId*height+pos[1]);


		//currentValue=detectionResult[offset+width*height*2];
		//currentValue=detectionResult[offset*(1+MAX_LABEL_NUMBER)];
		if (currentValue>max_value)
		{
			max_value=currentValue;
			minInd=offset;
		}
		offset+=blockDim.x*gridDim.x;
	}
	maxIndex[cacheIndex]=minInd;
	maxValue[cacheIndex]=max_value;

	__syncthreads();
	

	int i=blockDim.x/2;
	while (i!=0)
	{
		if (cacheIndex<i)
		{
			if (maxValue[cacheIndex]<maxValue[cacheIndex+i])
			{
				maxValue[cacheIndex]=maxValue[cacheIndex+i];
				maxIndex[cacheIndex]=maxIndex[cacheIndex+i];
			}
		}
		__syncthreads();
		i/=2;
	}

	if (cacheIndex==0)
	{
		result[blockIdx.x]=maxIndex[0];
		/*pos[0]=(int)maxIndex[0]%width;
		pos[1]=(int)maxIndex[0]/width;
		result[blockIdx.x+blocksPerGrid]=tex2D(detectionResult,pos[0],labelId*height+pos[1]);*/
		result[blockIdx.x+blocksPerGrid]=maxValue[0];
	}


}

__global__ void findMaxValue(int width,int height,int labelId,float *result,int sx,int ex,int sy,int ey)
{
	__shared__ float maxIndex[threadsPerBlock_global];
	__shared__ float maxValue[threadsPerBlock_global];
	

	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int cacheIndex=threadIdx.x;

	int pos[2];

	int minInd; float max_value;
	float currentValue;
	max_value=0;
	minInd=0;
	while (offset<width*height)
	{
		//tex2D(currentDepthImg_combination,pos[0],pos[1])
		pos[0]=offset%width;
		pos[1]=offset/width;
		
		if (pos[0]<sx||pos[0]>ex||pos[1]<sy||pos[1]>ey)
		{
			offset+=blockDim.x*gridDim.x;
			continue;
		}

	/*	if (pos[0]<3||pos[0]>width-4||pos[1]<3||pos[1]>height-4)
		{
			currentValue=0;
		}
		else
		{
			currentValue=tex2D(detectionResult,pos[0],labelId*height+pos[1]-2);
			currentValue+=tex2D(detectionResult,pos[0]-1,labelId*height+pos[1]-2);
			currentValue+=tex2D(detectionResult,pos[0]-2,labelId*height+pos[1]-2);
			currentValue+=tex2D(detectionResult,pos[0]+1,labelId*height+pos[1]-2);
			currentValue+=tex2D(detectionResult,pos[0]+2,labelId*height+pos[1]-2);

			currentValue+=tex2D(detectionResult,pos[0],labelId*height+pos[1]-1);
			currentValue+=tex2D(detectionResult,pos[0]-1,labelId*height+pos[1]-1);
			currentValue+=tex2D(detectionResult,pos[0]-2,labelId*height+pos[1]-1);
			currentValue+=tex2D(detectionResult,pos[0]+1,labelId*height+pos[1]-1);
			currentValue+=tex2D(detectionResult,pos[0]+2,labelId*height+pos[1]-1);

			currentValue+=tex2D(detectionResult,pos[0],labelId*height+pos[1]);
			currentValue+=tex2D(detectionResult,pos[0]-1,labelId*height+pos[1]);
			currentValue+=tex2D(detectionResult,pos[0]-2,labelId*height+pos[1]);
			currentValue+=tex2D(detectionResult,pos[0]+1,labelId*height+pos[1]);
			currentValue+=tex2D(detectionResult,pos[0]+2,labelId*height+pos[1]);

			currentValue+=tex2D(detectionResult,pos[0],labelId*height+pos[1]+1);
			currentValue+=tex2D(detectionResult,pos[0]-1,labelId*height+pos[1]+1);
			currentValue+=tex2D(detectionResult,pos[0]-2,labelId*height+pos[1]+1);
			currentValue+=tex2D(detectionResult,pos[0]+1,labelId*height+pos[1]+1);
			currentValue+=tex2D(detectionResult,pos[0]+2,labelId*height+pos[1]+1);

			currentValue+=tex2D(detectionResult,pos[0],labelId*height+pos[1]+2);
			currentValue+=tex2D(detectionResult,pos[0]-1,labelId*height+pos[1]+2);
			currentValue+=tex2D(detectionResult,pos[0]-2,labelId*height+pos[1]+2);
			currentValue+=tex2D(detectionResult,pos[0]+1,labelId*height+pos[1]+2);
			currentValue+=tex2D(detectionResult,pos[0]+2,labelId*height+pos[1]+2);
		}*/
	
		currentValue=tex2D(detectionResult,pos[0],labelId*height+pos[1]);


		//currentValue=detectionResult[offset+width*height*2];
		//currentValue=detectionResult[offset*(1+MAX_LABEL_NUMBER)];
		if (currentValue>max_value)
		{
			max_value=currentValue;
			minInd=offset;
		}
		offset+=blockDim.x*gridDim.x;
	}
	maxIndex[cacheIndex]=minInd;
	maxValue[cacheIndex]=max_value;

	__syncthreads();

	

	int i=blockDim.x/2;
	while (i!=0)
	{
		if (cacheIndex<i)
		{
			if (maxValue[cacheIndex]<maxValue[cacheIndex+i])
			{
				maxValue[cacheIndex]=maxValue[cacheIndex+i];
				maxIndex[cacheIndex]=maxIndex[cacheIndex+i];
			}
		}
		__syncthreads();
		i/=2;
	}

	if (cacheIndex==0)
	{
		result[blockIdx.x]=maxIndex[0];
		/*pos[0]=(int)maxIndex[0]%width;
		pos[1]=(int)maxIndex[0]/width;
		result[blockIdx.x+blocksPerGrid]=tex2D(detectionResult,pos[0],labelId*height+pos[1]);*/
		result[blockIdx.x+blocksPerGrid]=maxValue[0];
	}
}

__global__ void findLocalMaxima(int width,int height,float *detectionResult,int sx,int ex,int sy,int ey,float *modes)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>width*height)
	{
		return;
	}
	int pos[2];
	pos[0]=offset%width;
	pos[1]=offset/width;

	if ((pos[0]<=sx||pos[0]>=ex||pos[0]<=sy||pos[0]>=ey)||tex2D(currentDepthImg_combination,pos[0],pos[1])==0)
	{
		return;
	}

	//test if it is local maxima

}


const int meanShiftGridSize=51;
const int meanShiftGridArea=meanShiftGridSize*meanShiftGridSize;
const int meanShiftThreadNum=512;
__global__ void cu_meanshift(int featureNum,int height,float *featureLocations)
{
	int offset=blockIdx.x;
	if (offset>=featureNum)
	{
		return;
	}

	

	float curP[2];
	curP[0]=featureLocations[offset];
	curP[1]=featureLocations[offset+featureNum];

	int center[2];
	center[0]=curP[0];
	center[1]=curP[1];
	__shared__ float imgData[51][51];
	if (threadIdx.x==0)
	{
		for (int i=-25;i<=25;i++)
		{
			for (int j=-25;j<=25;j++)
			{
				imgData[i+25][j+25]=tex2D(detectionResult,curP[0]+i,offset*height+curP[1]+j);
			}
		}
	}
	__syncthreads();
	
	return;
	bool isstop=false;
	float tmpP[2]; float sumProb;float curDis;float ttProb;

	int localWindowSize=20;
	float sigma=3;
	float curProb;
	int times=0;
	while(!isstop)
	{
		tmpP[0]=tmpP[1]=0;
		sumProb=0;
		for (int m=curP[1]-localWindowSize/2;m<=curP[1]+localWindowSize/2;m++)
		{
			for (int n=curP[0]-localWindowSize/2;n<=curP[0]+localWindowSize/2;n++)
			{
				//curProb=tex2D(detectionResult,n,offset*height+m);
				curProb=imgData[n-center[0]+25][m-center[1]+25];
				if (curProb>0)
				{
					curDis=((float)m-curP[1])*((float)m-curP[1])+
						((float)n-curP[0])*((float)n-curP[0]);
					ttProb=powf(2.7173f,-curDis/(2*sigma*sigma))*curProb;
					tmpP[0]+=ttProb*(float)n;
					tmpP[1]+=ttProb*(float)m;
					sumProb+=ttProb;
				}

			}
		}
		if (sumProb==0)
		{
			return;
		}
		tmpP[0]/=sumProb;tmpP[1]/=sumProb;

		float distance=(tmpP[0]-curP[0])*(tmpP[0]-curP[0])+(tmpP[1]-curP[1])*(tmpP[1]-curP[1]);
		//cout<<distance<<endl;
		if (distance<0.00001)
		{

			isstop=true;
		}
		else
		{
			curP[0]=tmpP[0];curP[1]=tmpP[1];
		}
		/*times++;
		if (times==2)
		{
			break;
		}*/
		
	}

	featureLocations[offset]=curP[0];
	featureLocations[offset+featureNum]=curP[1];

}

__global__ void cu_meanshift_fast(int featureNum,int height,float *featureLocations)
{

	int offset=blockIdx.x;
	int gridID=threadIdx.x;

	if (offset>=featureNum||gridID>=meanShiftGridArea)
	{
		return;
	}

	float ttx=featureLocations[offset];
	float tty=featureLocations[offset+featureNum];

	__shared__ float curP[2];
	__shared__ int center[2];
	if (gridID==0)
	{
		
		curP[0]=ttx;
		curP[1]=tty;
		center[0]=ttx;
		center[1]=tty;
	}

	
	
	const int localWindowSize=20;//20
	const int localWindowRealSize=localWindowSize+1;
	__shared__ float imgData[51][51];
	__shared__ float expResult[localWindowRealSize][localWindowRealSize];
	__shared__ int times;
	//__shared__ float expResult[1][1];
	//if (threadIdx.x==0)
	//{
	//	for (int i=-25;i<=25;i++)
	//	{
	//		for (int j=-25;j<=25;j++)
	//		{
	//			imgData[i+25][j+25]=tex2D(detectionResult,curP[0]+i,offset*height+curP[1]+j);
	//		}
	//	}
	//}
	int threadID=gridID;
	while (threadID<meanShiftGridArea)
	{
		int i=threadID/51-25;
		int j=threadID%51-25;
		imgData[i+25][j+25]=tex2D(detectionResult,ttx+i,offset*height+tty+j);
		threadID+=meanShiftThreadNum;
	}
	
	__shared__ bool isstop;
	if (gridID==0)
	{
		isstop=false;
		times=0;
	}

	__syncthreads();
	

	/*if (gridID==0)
	{
		if (isstop)
		{
			featureLocations[offset]=offset;
			featureLocations[offset+featureNum]=offset+featureNum;
		}
		else
		{
			featureLocations[offset]=0;
			featureLocations[offset+featureNum]=0;
		}
		
	}*/

	//featureLocations[offset]=offset;
	//featureLocations[offset+featureNum]=offset+featureNum;

	//return;

	
	
	float tmpP[2]; float sumProb;float curDis;float ttProb;

	
	float sigma=3;
	float curProb;
	
	while(!isstop)
	{
		int row=gridID/localWindowRealSize;
		int col=gridID%localWindowRealSize;
		if (gridID<localWindowRealSize*localWindowRealSize)
		{
			
			int m=curP[1]+row-localWindowSize/2;int n=curP[0]+col-localWindowSize/2;
			int ccX=n-center[0]+25;int ccY=m-center[1]+25;
			if (ccX<0||ccX>50||ccY<0||ccY>50)
			{
				isstop=true;
			}
			else
			{
				curProb=imgData[ccX][ccY];
				curDis=((float)m-curP[1])*((float)m-curP[1])+
					((float)n-curP[0])*((float)n-curP[0]);
				ttProb=powf(2.7173f,-curDis/(2*sigma*sigma))*curProb;

				expResult[row][col]=ttProb;
			}
			
			//tmpP[0]=ttProb;
		}
		__syncthreads();
		
		if (isstop==true)
		{
			break;
		}
		//break;
		if (gridID==0)
		{
			//for (int m=curP[1]-localWindowSize/2;m<=curP[1]+localWindowSize/2;m++)
			//{
			//	for (int n=curP[0]-localWindowSize/2;n<=curP[0]+localWindowSize/2;n++)
			//	{
			//		//curProb=tex2D(detectionResult,n,offset*height+m);
			//		curProb=imgData[n-center[0]+25][m-center[1]+25];
			//		if (curProb>0)
			//		{
			//			curDis=((float)m-curP[1])*((float)m-curP[1])+
			//				((float)n-curP[0])*((float)n-curP[0]);
			//			ttProb=powf(2.7173f,-curDis/(2*sigma*sigma))*curProb;
			//			tmpP[0]+=ttProb*(float)n;
			//			tmpP[1]+=ttProb*(float)m;
			//			sumProb+=ttProb;
			//		}

			//	}
			//}

			//isstop=true;
			tmpP[0]=tmpP[1]=0;
			sumProb=0;
			int cx,cy;
			cx=0;
			for (int m=curP[1]-localWindowSize/2;m<=curP[1]+localWindowSize/2;m++)
			{
				cy=0;
				for (int n=curP[0]-localWindowSize/2;n<=curP[0]+localWindowSize/2;n++)
				{
						ttProb=expResult[cx][cy];
						tmpP[0]+=ttProb*(float)n;
						tmpP[1]+=ttProb*(float)m;
						sumProb+=ttProb;
						cy++;
				}
				cx++;
			}
			//isstop=true;

			float distance;
			if (sumProb==0)
			{
				isstop=true;
			}
			else
			{
				tmpP[0]/=sumProb;tmpP[1]/=sumProb;

				distance=(tmpP[0]-curP[0])*(tmpP[0]-curP[0])+(tmpP[1]-curP[1])*(tmpP[1]-curP[1]);
				if (distance<0.00001)
				{

					isstop=true;
				}
				else
				{
					curP[0]=tmpP[0];curP[1]=tmpP[1];
				}
				//cout<<distance<<endl;
				
			}
			times++;

		
			//__threadfence_block();
			
		}
		__syncthreads();

		if (times>60)
		{
			break;
		}
			//tmpP[0]=tmpP[1]=0;
			//sumProb=0;
			//for (int m=curP[1]-localWindowSize/2;m<=curP[1]+localWindowSize/2;m++)
			//{
			//	for (int n=curP[0]-localWindowSize/2;n<=curP[0]+localWindowSize/2;n++)
			//	{
			//		//curProb=tex2D(detectionResult,n,offset*height+m);
			//		curProb=imgData[n-center[0]+25][m-center[1]+25];
			//		if (curProb>0)
			//		{
			//			curDis=((float)m-curP[1])*((float)m-curP[1])+
			//				((float)n-curP[0])*((float)n-curP[0]);
			//			ttProb=powf(2.7173f,-curDis/(2*sigma*sigma))*curProb;
			//			tmpP[0]+=ttProb*(float)n;
			//			tmpP[1]+=ttProb*(float)m;
			//			sumProb+=ttProb;
			//		}

			//	}
			//}
			//if (sumProb==0)
			//{
			//	return;
			//}
			//tmpP[0]/=sumProb;tmpP[1]/=sumProb;

			//float distance=(tmpP[0]-curP[0])*(tmpP[0]-curP[0])+(tmpP[1]-curP[1])*(tmpP[1]-curP[1]);
			////cout<<distance<<endl;
			//if (distance<0.00001)
			//{

			//	isstop=true;
			//}
			//else
			//{
			//	curP[0]=tmpP[0];curP[1]=tmpP[1];
			//}
	}

	
	if (gridID==0)
	{
		featureLocations[offset]=curP[0];
		featureLocations[offset+featureNum]=curP[1];
	}
	
	


}


extern "C" bool predict_GPU_separated_combination(int width,int height,float *host_result,float **finalPos,float *maxProb,int sx,int ex,int sy,int ey,
	bool isInitial,bool showProb,float lastTheta)
{
	//load the trained tree
	RandmizedTree_CUDA *data_color=trackEngine.randomizedTrees_Color;
	RandmizedTree_CUDA *data_depth=trackEngine.randomizedTrees_Depth;

	
	if (abs(lastTheta)<0.087) //5
	{
		lastTheta=0;
	}

	float cT=cos(lastTheta);
	float sT=sin(lastTheta);
	//GTB("AA");
	////for (int i=0;i<60;i++)
	//{

	int threadsPerBlock=64;


	predict_prob_withDepth_color_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_color->cu_LabelFullResult,data_color->labelNum,data_color->max_num_of_trees_in_the_forest,width,height,
				data_color->windowSize,data_color->cu_vectorTrees,(data_color->leafnode),data_color->MaxNumber,sx,ex,sy,ey,cT,sT);
	if (!isInitial||1)
	{
		predict_prob_withDepth_depth_combination_withCombination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_color->cu_LabelFullResult,data_depth->labelNum,data_depth->max_num_of_trees_in_the_forest,width,height,
		data_depth->windowSize,data_depth->cu_vectorTrees,(data_depth->leafnode),data_depth->MaxNumber,data_color->cu_LabelFullResult,sx,ex,sy,ey,cT,sT);
	}
	else
	{
		predict_prob_withDepth_depth_combination_withLastFrame<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_color->cu_LabelFullResult,data_depth->labelNum,data_depth->max_num_of_trees_in_the_forest,width,height,
			data_depth->windowSize,data_depth->cu_vectorTrees,(data_depth->leafnode),data_depth->MaxNumber,data_color->cu_LabelFullResult,
			trackEngine.AAM_ENGINE->cu_currentShape,trackEngine.cu_usedIndex,trackEngine.AAM_ENGINE->ptsNum,sx,ex,sy,ey);

		/*predict_prob_withDepth_depth_combination_withLastFrame<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_color->cu_LabelFullResult,data_depth->labelNum,data_depth->max_num_of_trees_in_the_forest,width,height,
			data_depth->windowSize,data_depth->cu_vectorTrees,(data_depth->leafnode),data_depth->MaxNumber,data_color->cu_LabelFullResult,
			trackEngine.cu_detectedFeatureLocations,trackEngine.cu_usedIndex,data_color->labelNum-1,sx,ex,sy,ey);*/
	}

	//predict_prob_withDepth_depth_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_depth->cu_LabelFullResult,data_depth->labelNum,data_depth->max_num_of_trees_in_the_forest,width,height,
	//	data_depth->windowSize,data_depth->cu_vectorTrees,(data_depth->leafnode),data_depth->MaxNumber,sx,ex,sy,ey);
	////combine the two output
	//colorDepthCombination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_depth->cu_LabelFullResult,data_color->cu_LabelFullResult,data_color->cu_LabelFullResult,data_depth->max_num_of_trees_in_the_forest,
	//	data_color->max_num_of_trees_in_the_forest,data_color->labelNum,width,height);

	//direct calculation in one kernel
	/*threadsPerBlock=128;
	predict_prob_withDepth_color_depth_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data_color->cu_LabelFullResult,data_color->labelNum,data_color->max_num_of_trees_in_the_forest,width,height,
		data_color->windowSize,data_color->cu_vectorTrees,data_color->MaxNumber,data_depth->max_num_of_trees_in_the_forest,data_depth->cu_vectorTrees,sx,ex,sy,ey);*/

	/*predict_prob_withDepth_color_combination_eachTree<<<perBlockForRT,data_color->max_num_of_trees_in_the_forest>>>(data_color->cu_LabelFullResult,data_color->labelNum,data_color->max_num_of_trees_in_the_forest,width,height,
		data_color->windowSize,data_color->cu_vectorTrees,(data_color->leafnode),data_color->MaxNumber,sx,ex,sy,ey);
	predict_prob_withDepth_depth_combination_withCombination_EachTree<<<perBlockForRT,data_color->max_num_of_trees_in_the_forest>>>(data_color->cu_LabelFullResult,data_depth->labelNum,data_depth->max_num_of_trees_in_the_forest,width,height,
		data_depth->windowSize,data_depth->cu_vectorTrees,(data_depth->leafnode),data_depth->MaxNumber,data_color->cu_LabelFullResult,sx,ex,sy,ey);*/
	//update whenever there is new data
	CUDA_CALL(cudaBindTexture2D( NULL, detectionResult,
		data_color->cu_LabelFullResult,
		desc_combination, 640 ,(1+MAX_LABEL_NUMBER)*480,
		sizeof(float) * 640));

	if(showProb)
		CUDA_CALL(cudaMemcpy(host_result,data_color->cu_LabelFullResult, (width*height)*(data_color->labelNum)*sizeof(float),cudaMemcpyDeviceToHost));
	/*float *tmpPos=new float[640*480];
	CUDA_CALL(cudaMemcpy(tmpPos,data_color->cu_LabelFullResult, 640*480*sizeof(float),cudaMemcpyDeviceToHost));
	for(int i=0;i<200;i++)
	{
		if (tmpPos[i]!=0)
		{
			cout<<tmpPos[i]<<" ";
		}
	}
	cout<<width<<" "<<height<<" "<<data_color->labelNum<<endl;*/
//}
//GTE("AA");
//gCodeTimer.printTimeTree();
//double time = total_fps;
//cout<<"used time per iteration: "<<time<<"  /60= "<<time/60<<endl;
	

	////////////////////////////////find the maximum value and position////////////////////////////////////////////////////
	int threadNum=threadsPerBlock_global;
	int blockNum=min(blocksPerGrid,(width*height+threadNum-1)/threadNum);

	////find maximum index on GPU
	////findMaxValue<<<blockNum,threadNum>>>(width,height,data->cu_LabelResult,data->cu_maxIndPerBlock);
	////cout<<blockNum<<" "<<threadNum<<endl;
	//
	//////////////////////new version: slower than the old one, seems//////////////////////////
	//for (int i=0;i<data_color->labelNum-1;i++)
	//{
	//	//findMaxValue<<<blockNum,threadNum>>>(width,height,i,data->cu_maxIndPerBlock+blocksPerGrid*2*i);
	//	findMaxValue_full<<<blockNum,threadNum>>>(width,height,i,data_color->cu_maxIndPerBlock+blocksPerGrid*2*i,trackEngine.cu_maximumPtsPos);

	//	/*float *result=new float[2];
	//	CUDA_CALL(cudaMemcpy(result,finalPosAndIndex, 2*sizeof(float),cudaMemcpyDeviceToHost));
	//	cout<<result[0]<<" "<<result[1]<<endl;*/
	//}

	//CUDA_CALL(cudaMemcpy(trackEngine.host_maximumPtsPos,trackEngine.cu_maximumPtsPos, maximumPtsNumPitch*2*sizeof(float),cudaMemcpyDeviceToHost));
	//float *tmpMaxPtsPos=trackEngine.host_maximumPtsPos;
	//for (int i=0;i<data_color->labelNum-1;i++)
	//{
	//	int posInd=tmpMaxPtsPos[i];
	//	finalPos[i][0]=posInd%width;
	//	finalPos[i][1]=posInd/width;
	//	maxProb[i]=tmpMaxPtsPos[i+maximumPtsNumPitch];
	//	cout<<posInd<<" "<<maxProb[i]<<endl;
	//}

	
	////////////////////////////////////////////////////////////
	
	
	//////////////////old slow version////////////////////////////
	//for (int i=0;i<data_color->labelNum-1;i++)
	//{
	//	findMaxValue<<<blockNum,threadNum>>>(width,height,i,data_color->cu_maxIndPerBlock+blocksPerGrid*2*i);
	//	//findMaxValue<<<blockNum,threadNum>>>(width,height,i,data_color->cu_maxIndPerBlock+blocksPerGrid*2*i,sx,ex,sy,ey);
	//}
	dim3 grid(blockNum,data_color->labelNum-1,1);
	findMaxValue_full<<<grid,threadNum>>>(width,height,data_color->cu_maxIndPerBlock);
	CUDA_CALL(cudaMemcpy(data_color->maxIndPerBlock,data_color->cu_maxIndPerBlock, blocksPerGrid*2*MAX_LABEL_NUMBER*sizeof(float),cudaMemcpyDeviceToHost));

	
	int finalMaxInd;
	float finalMaxV=0;
	for (int l=0;l<data_color->labelNum-1;l++)
	{
		finalMaxV=0;
		for (int i=0;i<blockNum;i++)
		{
			//cout<<i<<" "<<data->maxIndPerBlock[i]<<" "<<data->maxIndPerBlock[blocksPerGrid+i]<<endl;
			if (data_color->maxIndPerBlock[blocksPerGrid+i+blocksPerGrid*2*l]>finalMaxV)
			{
				finalMaxInd=data_color->maxIndPerBlock[i+blocksPerGrid*2*l];
				finalMaxV=data_color->maxIndPerBlock[blocksPerGrid*(2*l+1)+i];
			}
		}
		finalPos[l][0]=finalMaxInd%width;
		finalPos[l][1]=finalMaxInd/width;
		maxProb[l]=finalMaxV;

		//cout<<finalMaxInd<<" "<<finalMaxV<<endl;
	}
	//cout<<maxProb[6]<<" "<<maxProb[12]<<endl;
	//check the probability, if all are small, then return false
	int goodNum=0;
	for (int l=0;l<data_color->labelNum-1;l++)
	{
		if (maxProb[l]>0.1)
		{
			goodNum++;
		}
	}
	if (goodNum<6)
	{
		return false;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////




	//////////////////thrust version////////////////////////////////
	//thrust::device_ptr<float> probMap_vec(data_color->cu_LabelFullResult);
	//int totoalNum=width*height;
	//thrust::device_vector<float>::iterator iter;
	//for (int i=0;i<data_color->labelNum-1;i++)
	//{
	//	iter= thrust::max_element(probMap.begin()+totoalNum*i, probMap.begin()+totoalNum*(i+1));
	//	unsigned int finalMaxInd = (iter - probMap.begin()-totoalNum*i);
	//	//cout<<finalMaxInd<<endl;
	//	finalPos[i][0]=finalMaxInd%width;
	//	finalPos[i][1]=finalMaxInd/width;
	//	maxProb[i]=*iter;
	//}
	///////////////////////////////////////////////////////////////////

	/////////then find local maxima////////////////
	///////////////////////////////////////////////

	//return true;
	////////////////finally, mean shift///////////////////////////////

	
	int featureNum=data_color->labelNum-1;
	float *tmpPos=new float[featureNum*2];
	for(int i=0;i<featureNum;i++)
	{
		tmpPos[i]=finalPos[i][0];
		tmpPos[i+featureNum]=finalPos[i][1];
		
		//cout<<tmpPos[i]<<" "<<tmpPos[i+featureNum]<<endl;
	}
	
	CUDA_CALL(cudaMemcpy(trackEngine.cu_detectedFeatureLocations,tmpPos, featureNum*2*sizeof(float),cudaMemcpyHostToDevice));
	//cu_meanshift<<<64,512>>>(featureNum,height,trackEngine.cu_detectedFeatureLocations);
	
//	GTB("A");
	cu_meanshift_fast<<<24,meanShiftThreadNum>>>(featureNum,height,trackEngine.cu_detectedFeatureLocations);

	//GTE("A");
	//gCodeTimer.printTimeTree();
	//double time = total_fps;
	//cout<<"used time per iteration: "<<time<<endl;

	CUDA_CALL(cudaMemcpy(tmpPos,trackEngine.cu_detectedFeatureLocations, featureNum*2*sizeof(float),cudaMemcpyDeviceToHost));
	for(int i=0;i<featureNum;i++)
	{
		finalPos[i][0]=tmpPos[i];
		finalPos[i][1]=tmpPos[i+featureNum];
		/*if (maxProb[i]<0.1)
		{
			finalPos[i][0]=finalPos[i][1]=0;;
		}*/
	}
	delete []tmpPos;
	
	return true;
	//////////////////////////////////////////////////////////////////

	//CUDA_CALL(cudaEventRecord( stop, 0 ));
	//CUDA_CALL(cudaEventSynchronize( stop ));
	//float elapsedTime;
	//CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
	//	start, stop ) );
	//cout<<"time: "<<elapsedTime<<" ms"<<endl;

}


extern "C" void setCPUDetection2GPU(float *cpuResult)
{
	CUDA_CALL(cudaMemcpy(trackEngine.randomizedTrees_Color->cu_LabelFullResult,cpuResult, MPN*(1+MAX_LABEL_NUMBER)*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaBindTexture2D( NULL, detectionResult,
		trackEngine.randomizedTrees_Color->cu_LabelFullResult,
		desc_combination, 640 ,(1+MAX_LABEL_NUMBER)*480,
		sizeof(float) * 640));
	//currentImg.filterMode=cudaFilterModePoint;
	//currentColorImg_combination.filterMode=cudaFilterModePoint;
	//currentDepthImg_combination.filterMode=cudaFilterModePoint;
}

extern "C" void predict_GPU_withDepth_combination(int width,int height,float *host_result,int trainStyle,int **finalPos,float *maxProb)
{
	//load the trained tree
	RandmizedTree_CUDA *data=trackEngine.randomizedTrees_Color;

	dim3 grid(width,height,1);

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord( start, 0 ));
	int threadsPerBlock=64;
	if (trainStyle==0)
	{
		predict_prob_withDepth_depth_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelFullResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	else if (trainStyle==1)
	{
		/*predict_prob_withDepth_color_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);*/
		predict_prob_withDepth_color_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelFullResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	else if (trainStyle==2)
	{
		//predict_prob_withDepth_depth_color_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
		//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);

		predict_prob_withDepth_depth_color_ImageResult_combination<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelFullResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	/*predict_prob_withDepth<<<width*height/64+1,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
		data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber,trainStyle);*/
	//predict_prob_fullParral<<<data->max_num_of_trees_in_the_forest,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	//predict_prob_fullParral_pixel_tree<<<grid,16>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);

	//update whenever there is new data
	CUDA_CALL(cudaBindTexture2D( NULL, detectionResult,
		data->cu_LabelFullResult,
		desc_combination, 640 ,(1+MAX_LABEL_NUMBER)*480,
		sizeof(float) * 640));

	CUDA_CALL(cudaEventRecord( stop, 0 ));
	CUDA_CALL(cudaEventSynchronize( stop ));
	float elapsedTime;
	CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
		start, stop ) );
	cout<<"time: "<<elapsedTime<<" ms"<<endl;

	
	//we need to find out the maximum points and the locations. Others are not neccary to transfer.
	//CUDA_CALL(cudaMemcpy(data->cu_LabelResult,host_result,MPN*sizeof(LabelResult),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(host_result,data->cu_LabelFullResult, MPN*(1+MAX_LABEL_NUMBER)*sizeof(float),cudaMemcpyDeviceToHost));

	//return;
	int threadNum=threadsPerBlock_global;
	int blockNum=min(blocksPerGrid,(width*height+threadNum-1)/threadNum);
	//find maximum index on GPU
	//findMaxValue<<<blockNum,threadNum>>>(width,height,data->cu_LabelResult,data->cu_maxIndPerBlock);
	//cout<<blockNum<<" "<<threadNum<<endl;
	
	for (int i=0;i<data->labelNum-1;i++)
	{
		findMaxValue<<<blockNum,threadNum>>>(width,height,i,data->cu_maxIndPerBlock+blocksPerGrid*2*i);
		
	}
	

	CUDA_CALL(cudaMemcpy(data->maxIndPerBlock,data->cu_maxIndPerBlock, blocksPerGrid*2*MAX_LABEL_NUMBER*sizeof(float),cudaMemcpyDeviceToHost));

	int finalMaxInd;
	float finalMaxV=0;
	for (int l=0;l<data->labelNum-1;l++)
	{
		finalMaxV=0;
		for (int i=0;i<blockNum;i++)
		{
			//cout<<i<<" "<<data->maxIndPerBlock[i]<<" "<<data->maxIndPerBlock[blocksPerGrid+i]<<endl;
			if (data->maxIndPerBlock[blocksPerGrid+i+blocksPerGrid*2*l]>finalMaxV)
			{
				finalMaxInd=data->maxIndPerBlock[i+blocksPerGrid*2*l];
				finalMaxV=data->maxIndPerBlock[blocksPerGrid*(2*l+1)+i];
			}
		}
		finalPos[l][0]=finalMaxInd%width;
		finalPos[l][1]=finalMaxInd/width;
		maxProb[l]=finalMaxV;
	//	cout<<finalMaxInd%width<<" "<<finalMaxInd/width<<endl;
	}
}

///////////////////////////AAM///////////////////////////////////////////////////
int MCN_combination=MAX_COUNT_NUM;
int MAXIMUMPOINTDIM_combination=MAX_PIXEL_NUM;
int MPD_combination=MAX_POINT_DIM;



__global__ void getJacobians_Preprocess_combination(float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,int pixelNum,float *warp_tabel,
	float *t_vec,float *fowardIndex,float *Jacobian_transpose)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			float tmp;
			int allDim=s_dim+t_dim+4;
			int i=0;
			for (i=s_dim;i<s_dim+t_dim;i++)
			{
				tmp=t_vec[(i-s_dim)*pixelNum+pixelID];
				Jacobians[pixelID*allDim+i]=tmp;
				Jacobian_transpose[i*pixelNum+pixelID]=tmp;
			}
		}
		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}

		//}
	}
}

__global__ void calculateConv_preCalculation(int width,int height,int windowSize,int visibleNum,int totalNum,float *cu_locations,float *indList,float *output)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<totalNum)
	{
		output[offset]=0;
	}
	__syncthreads();

	if (offset<visibleNum)
	{
		float pos[2];
		pos[0]=cu_locations[offset];pos[1]=cu_locations[offset+visibleNum];
		float maxV;
		float sumV;
		float conv[2][2];
		float inv_conv[2][2];float determint;
		float cur_p;
		int cind;
		int labelID=indList[offset];
	//	for (int i=0;i<labelNum;i++)
		{
			//maxV=tex2D(detectionResult,centerX,centerY+labelID*totalNum);
			//get sum
			sumV=0;
			conv[0][0]=conv[0][1]=conv[1][0]=conv[1][1]=0;
			for (int k=pos[0]-windowSize/2;k<=pos[0]+windowSize/2;k++)
			{
				for (int l=pos[1]-windowSize/2;l<=pos[1]+windowSize/2;l++)
				{
					cur_p=tex2D(detectionResult,k,l+labelID*height);
					//cur_p=globalData[(l+labelID*height)*width+k];//(detectionResult,k,l+labelID*height);
					sumV+=cur_p;
					conv[0][0]+=cur_p*(k-pos[0])*(k-pos[0]);
					conv[0][1]+=cur_p*(k-pos[0])*(l-pos[1]);
					conv[1][0]+=cur_p*(k-pos[0])*(l-pos[1]);
					conv[1][1]+=cur_p*(l-pos[1])*(l-pos[1]);
				}
			}
			conv[0][0]/=sumV;conv[0][1]/=sumV;conv[1][0]/=sumV;conv[1][1]/=sumV;
			determint=conv[0][0]*conv[1][1]-conv[0][1]*conv[1][0];
		/*	inv_conv[0][0]=conv[1][1]/determint;
			inv_conv[0][1]=-conv[0][1]/determint;
			inv_conv[1][0]=-conv[1][0]/determint;
			inv_conv[1][1]=conv[0][0]/determint;*/

			/*cind=pos[1]*4*width+labelID*totalNum*4;*/
			int totalDim=offset*visibleNum*4+offset*2;
			output[totalDim]=conv[1][1]/determint;
			output[totalDim+1]=-conv[0][1]/determint;
			output[totalDim+visibleNum*2]=-conv[1][0]/determint;
			output[totalDim+visibleNum*2+1]=conv[0][0]/determint;

	/*		output[4*offset]=determint;
			output[4*offset+1]=determint;
			output[4*offset+2]=determint;
			output[4*offset+2+1]=determint;*/


			/*output[pos[0]+cind]=conv[0][0];
			output[pos[0]+width+cind]=conv[0][1];
			output[pos[0]+width*2+cind]=conv[1][0];
			output[pos[0]+width*3+cind]=conv[1][1];*/
		}
	}
}
//size of output : MPN * 4*max_labelNUm
__global__ void calculateConv(int windowSize,int centerX,int centerY,int labelID,int width,int height,float *output)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<width*height)
	{
		int totalNum=width*height;
		int curLabelInd;
		int pos[2];
		pos[0]=offset%width;
		pos[1]=offset/width;
		

		if(tex2D(currentDepthImg_combination,pos[0],pos[1])==0)
			return;

		if (pos[0]<=windowSize/2||pos[0]>=width-windowSize/2||
			pos[1]<=windowSize/2||pos[1]>=height-windowSize/2)
			return;
		

		float maxV;
		float sumV;
		float conv[2][2];
		float inv_conv[2][2];float determint;
		float cur_p;
		int cind;
	//	for (int i=0;i<labelNum;i++)
		{
			//maxV=tex2D(detectionResult,centerX,centerY+labelID*totalNum);
			//get sum
			sumV=0;
			conv[0][0]=conv[0][1]=conv[1][0]=conv[1][1]=0;
			for (int k=pos[0]-windowSize/2;k<=pos[0]+windowSize/2;k++)
			{
				for (int l=pos[1]-windowSize/2;l<=pos[1]+windowSize/2;l++)
				{
					cur_p=tex2D(detectionResult,k,l+labelID*height);
					//cur_p=globalData[(l+labelID*height)*width+k];//(detectionResult,k,l+labelID*height);
					sumV+=cur_p;
					conv[0][0]+=cur_p*(k-centerX)*(k-centerX);
					conv[0][1]+=cur_p*(k-centerX)*(l-centerY);
					conv[1][0]+=cur_p*(k-centerX)*(l-centerY);
					conv[1][1]+=cur_p*(l-centerY)*(l-centerY);
				}
			}
			conv[0][0]/=sumV;conv[0][1]/=sumV;conv[1][0]/=sumV;conv[1][1]/=sumV;
			determint=conv[0][0]*conv[1][1]-conv[0][1]*conv[1][0];
		/*	inv_conv[0][0]=conv[1][1]/determint;
			inv_conv[0][1]=-conv[0][1]/determint;
			inv_conv[1][0]=-conv[1][0]/determint;
			inv_conv[1][1]=conv[0][0]/determint;*/

			cind=pos[1]*4*width+labelID*totalNum*4;
			output[pos[0]+cind]=conv[1][1]/determint;
			output[pos[0]+width+cind]=-conv[0][1]/determint;
			output[pos[0]+width*2+cind]=-conv[1][0]/determint;
			output[pos[0]+width*3+cind]=conv[0][0]/determint;

		/*	output[pos[0]+cind]=sumV;
			output[pos[0]+width+cind]=centerX;
			output[pos[0]+width*2+cind]=centerY;
			output[pos[0]+width*3+cind]=conv[0][0]/determint;*/


			/*output[pos[0]+cind]=conv[0][0];
			output[pos[0]+width+cind]=conv[0][1];
			output[pos[0]+width*2+cind]=conv[1][0];
			output[pos[0]+width*3+cind]=conv[1][1];*/
		}
	}

}

extern "C" void setupConv(int width, int height,int windowSize,int **shapeLoc,vector<int> &indList,float *host_output)
{
	int threads_block=128;
	//float *output;

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord( start, 0 ));

	int pointNum=indList.size();
	int cx,cy;
	int cid;
	for (int i=0;i<pointNum;i++)
	{
	//	cout<<"-------------------------"<<shapeLoc[indList[i]][0]<<" "<<shapeLoc[indList[i]][1]<<endl;
		
		calculateConv<<<(width*height)/threads_block+1,threads_block>>>(windowSize,shapeLoc[indList[i]][0],shapeLoc[indList[i]][1],
		indList[i],width,height,trackEngine.cu_precalculatedConv);

		//cx=shapeLoc[indList[i]][0];cy=shapeLoc[indList[i]][1];cid=indList[i];
		//calculateConv<<<(width*height)/threads_block+1,threads_block>>>(windowSize,cx,cy,
		//	cid,width,height,trackEngine.randomizedTrees_EGGINE->cu_LabelFullResult,trackEngine.cu_precalculatedConv);

	}

	CUDA_CALL(cudaEventRecord( stop, 0 ));
	CUDA_CALL(cudaEventSynchronize( stop ));
	float elapsedTime;
	CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
		start, stop ) );
	cout<<"precalculate time: "<<elapsedTime<<" ms"<<endl;

	CUDA_CALL(cudaBindTexture2D( NULL, texture_preConv,
		trackEngine.cu_precalculatedConv,
		desc_combination, width*4 ,height*MAX_LABEL_NUMBER,
		sizeof(float) * width*4));
	texture_preConv.filterMode=cudaFilterModePoint;

	//then set up visible num, visible ind and cu_conv
	trackEngine.visibleNum=pointNum;
	for (int i=0;i<pointNum;i++)
	{
		trackEngine.visibleIndex[i]=indList[i];
	}
	CUDA_CALL(cudaMemcpy(trackEngine.cu_visibleIndex,trackEngine.visibleIndex,MAX_LABEL_NUMBER*sizeof(float),cudaMemcpyHostToDevice));
	

	CUDA_CALL(cudaMemcpy(host_output,trackEngine.cu_precalculatedConv, MAX_LABEL_NUMBER*MPN*4*sizeof(float),cudaMemcpyDeviceToHost));
}



extern "C" void setupConv_featurePts(int width, int height,int windowSize,float **shapeLoc,vector<int> &indList,float *absVisibleIndex)
{
	int threads_block=128;
	//float *output;

	//cudaEvent_t start, stop;
	//CUDA_CALL(cudaEventCreate(&start));
	//CUDA_CALL(cudaEventCreate(&stop));
	//CUDA_CALL(cudaEventRecord( start, 0 ));

	int pointNum=indList.size();
	//int cx,cy;
	//int cid;
	//for (int i=0;i<pointNum;i++)
	//{
	//	//	cout<<"-------------------------"<<shapeLoc[indList[i]][0]<<" "<<shapeLoc[indList[i]][1]<<endl;

	//	calculateConv<<<(width*height)/threads_block+1,threads_block>>>(windowSize,shapeLoc[indList[i]][0],shapeLoc[indList[i]][1],
	//		indList[i],width,height,trackEngine.cu_precalculatedConv);

	//	//cx=shapeLoc[indList[i]][0];cy=shapeLoc[indList[i]][1];cid=indList[i];
	//	//calculateConv<<<(width*height)/threads_block+1,threads_block>>>(windowSize,cx,cy,
	//	//	cid,width,height,trackEngine.randomizedTrees_EGGINE->cu_LabelFullResult,trackEngine.cu_precalculatedConv);

	//}

	//CUDA_CALL(cudaEventRecord( stop, 0 ));
	//CUDA_CALL(cudaEventSynchronize( stop ));
	//float elapsedTime;
	//CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
	//	start, stop ) );
	//cout<<"precalculate time: "<<elapsedTime<<" ms"<<endl;

	//CUDA_CALL(cudaBindTexture2D( NULL, texture_preConv,
	//	trackEngine.cu_precalculatedConv,
	//	desc_combination, width*4 ,height*MAX_LABEL_NUMBER,
	//	sizeof(float) * width*4));
	//texture_preConv.filterMode=cudaFilterModePoint;

	//then set up visible num, visible ind and cu_conv
	for (int i=0;i<pointNum;i++)
	{
	//	cout<<"-------------------------"<<shapeLoc[indList[i]][0]<<" "<<shapeLoc[indList[i]][1]<<endl;
		
		//calculateConv<<<(width*height)/threads_block+1,threads_block>>>(windowSize,shapeLoc[indList[i]][0],shapeLoc[indList[i]][1],
		//indList[i],width,height,trackEngine.cu_precalculatedConv);

		trackEngine.host_detectedFeatureLocations[i]=shapeLoc[indList[i]][0];
		trackEngine.host_detectedFeatureLocations[i+pointNum]=shapeLoc[indList[i]][1];
		//cx=shapeLoc[indList[i]][0];cy=shapeLoc[indList[i]][1];cid=indList[i];
		//calculateConv<<<(width*height)/threads_block+1,threads_block>>>(windowSize,cx,cy,
		//	cid,width,height,trackEngine.randomizedTrees_EGGINE->cu_LabelFullResult,trackEngine.cu_precalculatedConv);

	}

	//cout<<"ptsNum "<<pointNum<<endl;
	trackEngine.visibleNum=pointNum;
	for (int i=0;i<pointNum;i++)
	{
		trackEngine.visibleIndex[i]=indList[i];
	}
	//put it into paged memory
	CUDA_CALL(cudaMemcpy(trackEngine.cu_absVisibleIndex,absVisibleIndex,(MAX_LABEL_NUMBER*2+pointNum*2)*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(trackEngine.cu_visibleIndex,trackEngine.visibleIndex,(MAX_LABEL_NUMBER+pointNum*2)*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(trackEngine.cu_detectedFeatureLocations,trackEngine.host_detectedFeatureLocations,pointNum*2*sizeof(float),cudaMemcpyHostToDevice));
	int totalNum=MAX_LABEL_NUMBER*4*MAX_LABEL_NUMBER;
	calculateConv_preCalculation<<<totalNum/128+1,128>>>(width,height,windowSize,pointNum,totalNum,trackEngine.cu_detectedFeatureLocations,trackEngine.cu_visibleIndex,trackEngine.cu_conv);

	

	//if (trackEngine.AAM_ENGINE->showSingleStep)
	//{
	//	cout<<"conv:\n";
	//	float *cpu_conv=new float[MAX_LABEL_NUMBER*2*MAX_LABEL_NUMBER*2];
	//	CUDA_CALL(cudaMemcpy(cpu_conv, trackEngine.cu_conv, MAX_LABEL_NUMBER*2*MAX_LABEL_NUMBER*2*sizeof(float), cudaMemcpyDeviceToHost ));
	//	for (int i=0;i<pointNum;i++)
	//	{
	//		int cind=i*pointNum*2*2+i*2;
	//		cout<<cpu_conv[cind]<<" "<<cpu_conv[cind+1]<<" "<<cpu_conv[cind+2*pointNum]<<" "<<cpu_conv[cind+2*pointNum+1]<<endl;
	//	}
	//	delete []cpu_conv;
	//}
	
//	CUDA_CALL(cudaMemcpy(host_output,trackEngine.cu_precalculatedConv, MAX_LABEL_NUMBER*MPN*4*sizeof(float),cudaMemcpyDeviceToHost));

	//for (int i=0;i<pointNum;i++)
	//{
	//	cout<<"-------------------------"<<shapeLoc[indList[i]][0]<<" "<<shapeLoc[indList[i]][1]<<endl;
	//}
}

__global__ void gradientCalculation(float *gradientX,float *gradientY,int width,int height)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;

	if (offset<width*height)
	{
		int cx=offset%width;
		int cy=offset/width;
		if (cx>=1&&cx<width-1&&cy>=1&&cy<height-1)
		{
			gradientX[offset]=0.5*(tex2D(currentColorImg_combination,cx+1,cy)-tex2D(currentColorImg_combination,cx-1,cy));
			gradientY[offset]=0.5*(tex2D(currentColorImg_combination,cx,cy+1)-tex2D(currentColorImg_combination,cx,cy-1));
		}
	}
}

extern "C" void setData_onRun_AAM(float *parameters,int width,int height,vector<int>&finalInd)
{
	AAM_Search_RealGlobal_CUDA *data=trackEngine.AAM_ENGINE;
	
	if (parameters[0]!=-1000000000)
	{
		CUDA_CALL(cudaMemcpy(data->cu_parameters,parameters,(data->s_dim+data->t_dim+4)*sizeof(float),cudaMemcpyHostToDevice));
		//CUDA_CALL(cudaMemcpy(data->cu_parameters,parameters,(data->s_dim+4)*sizeof(float),cudaMemcpyHostToDevice));

	}
	else
	{
		//cout<<"not updating parameters\n";
		//CUDA_CALL(cudaMemcpy(data->cu_parameters+data->s_dim,parameters+data->s_dim,4*sizeof(float),cudaMemcpyHostToDevice));
	}
	gradientCalculation<<<width*height/256+1,256>>>(data->cu_gradientX,data->cu_gradientY,width,height);
	//CUDA_CALL(cudaMemcpy(data->cu_gradientX,InputGradientX,MAXIMUMPOINTDIM_combination*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(data->cu_gradientY,InputGradientY,MAXIMUMPOINTDIM_combination*sizeof(float),cudaMemcpyHostToDevice));

	//int pointNum=finalInd.size();
	//trackEngine.visibleNum=pointNum;
	//for (int i=0;i<pointNum;i++)
	//{
	//	trackEngine.visibleIndex[i]=finalInd[i];
	//	//trackEngine.visibleIndex[i]=indList[i];
	//}
	//CUDA_CALL(cudaMemcpy(trackEngine.cu_visibleIndex,trackEngine.visibleIndex,MAX_LABEL_NUMBER*3*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(trackEngine.cu_detectedFeatureLocations,trackEngine.host_detectedFeatureLocations,MAX_LABEL_NUMBER*2*sizeof(float),cudaMemcpyHostToDevice));

	//no need to input image, already done in the shared parameters
	//CUDA_CALL(cudaMemcpy(data->cu_inputImg,inputImage,MAXIMUMPOINTDIM*sizeof(float),cudaMemcpyHostToDevice));


	//CUDA_CALL(cudaMemcpy(data->cu_gradientX,InputGradientX,MAXIMUMPOINTDIM_combination*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(data->cu_gradientY,InputGradientY,MAXIMUMPOINTDIM_combination*sizeof(float),cudaMemcpyHostToDevice));
}

extern "C" void setShowSingle(bool showSingle)
{
	AAM_Search_RealGlobal_CUDA *data=trackEngine.AAM_ENGINE;
	data->showSingleStep=showSingle;
}

extern "C" void setData_AAM_combination(float *s_vec,float *t_vec,float *s_mean,float *t_mean,
	float *warpTabel,float *triangle_indexTabel,int s_dim,int t_dim,int ptsNum,int pix_num,
	int t_width,int t_height,float *shapeJacobians,float *maskTabel,float *fowardIndex,bool showSingleStep,bool isAptive,float *dataAddress)
{
	AAM_Search_RealGlobal_CUDA *data=trackEngine.AAM_ENGINE;

	/*CUDA_CALL( cudaMalloc(&data->cu_s_mean, MPD_combination * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_s_mean,s_mean,MPD_combination*sizeof(float),cudaMemcpyHostToDevice));

	CUDA_CALL( cudaMalloc(&data->cu_t_mean, MAXIMUMPOINTDIM_combination * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_t_mean,t_mean,MAXIMUMPOINTDIM_combination*sizeof(float),cudaMemcpyHostToDevice));*/
	float *host_SMean_Tmean=new float[ptsNum*2+pix_num];
	for (int i=0;i<ptsNum*2;i++)
	{
		host_SMean_Tmean[i]=s_mean[i];
	}
	for (int i=ptsNum*2;i<ptsNum*2+pix_num;i++)
	{
		host_SMean_Tmean[i]=t_mean[i-ptsNum*2];
	}

	//CUDA_CALL( cudaMalloc(&data->cu_t_vec, t_dim*MAXIMUMPOINTDIM_combination * sizeof(float)) );

	CUDA_CALL( cudaMalloc(&data->cu_sMean_T_mean, (ptsNum*2+pix_num+t_dim*pix_num) * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_sMean_T_mean,host_SMean_Tmean,(ptsNum*2+pix_num)*sizeof(float),cudaMemcpyHostToDevice));
	data->cu_s_mean=data->cu_sMean_T_mean;
	data->cu_t_mean=data->cu_sMean_T_mean+ptsNum*2;

	data->cu_t_vec=data->cu_t_mean+pix_num;
	CUDA_CALL(cudaMemcpy(data->cu_t_vec,t_vec,t_dim*pix_num*sizeof(float),cudaMemcpyHostToDevice));
	data->newMeanAndTVec=new float[t_dim*pix_num+pix_num];

	delete []host_SMean_Tmean;

	



	/*CUDA_CALL( cudaMalloc(&data->cu_currentLocalShape, MPD_combination * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_currentTemplate, MAXIMUMPOINTDIM_combination * sizeof(float)) );*/
	data->vec_curShape_curTemplate.resize(ptsNum*2+pix_num);
	data->cu_curLocalShape_curTemplate = thrust::raw_pointer_cast(&(data->vec_curShape_curTemplate[0]));
	//CUDA_CALL( cudaMalloc(&data->cu_curLocalShape_curTemplate, (ptsNum*2+pix_num) * sizeof(float)) );
	data->cu_currentLocalShape=data->cu_curLocalShape_curTemplate;
	data->cu_currentTemplate=data->cu_curLocalShape_curTemplate+ptsNum*2;

	CUDA_CALL( cudaMalloc(&data->cu_fullCurrentTexture, MAXIMUMPOINTDIM_combination * sizeof(float)) );


	//for shape
	//CUDA_CALL( cudaMalloc(&data->cu_currentShape, MPD_combination* sizeof(float)) );
	data->Vec_currentShape.resize(ptsNum*2);
	data->cu_currentShape = thrust::raw_pointer_cast(&(data->Vec_currentShape[0]));

	//CUDA_CALL( cudaMalloc(&data->cu_lastShape, MPD_combination* sizeof(float)) );
	data->Vec_lastShape.resize(ptsNum*2);
	data->cu_lastShape = thrust::raw_pointer_cast(&(data->Vec_lastShape[0]));


	/*CUDA_CALL( cudaMalloc(&data->cu_currentTexture, MAXIMUMPOINTDIM_combination * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_errorImage, (pix_num+MAX_LABEL_NUMBER*2) * sizeof(float)) );*/
	data->vec_curTexture.resize(pix_num+t_width*t_height*2);
	data->vec_errorImage.resize(pix_num+MAX_LABEL_NUMBER*2);
	data->cu_currentTexture = thrust::raw_pointer_cast(&(data->vec_curTexture[0]));
	data->cu_errorImage = thrust::raw_pointer_cast(&(data->vec_errorImage[0]));
	data->cu_warpedGx_warpedGy=data->cu_currentTexture+pix_num;
	data->cu_WarpedGradientX=data->cu_warpedGx_warpedGy;
	data->cu_WarpedGradientY=data->cu_warpedGx_warpedGy+t_width*t_height;

	float *allZerosImg=new float[t_width*t_height];
	for (int i=0;i<t_width*t_height;i++)
	{
		allZerosImg[i]=0;
	}
	CUDA_CALL(cudaMemcpy(data->cu_fullCurrentTexture,allZerosImg, t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(data->cu_WarpedGradientX,allZerosImg, t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(data->cu_WarpedGradientY,allZerosImg, t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));
	delete []allZerosImg;

	/*CUDA_CALL( cudaMalloc(&data->cu_WarpedGradientX, MAXIMUMPOINTDIM_combination * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_WarpedGradientY, MAXIMUMPOINTDIM_combination * sizeof(float)) );*/




	CUDA_CALL( cudaMalloc(&data->cu_s_vec, MCN_combination * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_s_vec,s_vec,MCN_combination*sizeof(float),cudaMemcpyHostToDevice));




	

	CUDA_CALL( cudaMalloc(&data->cu_warp_tabel, MAXIMUMPOINTDIM_combination*3 * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_warp_tabel,warpTabel,MAXIMUMPOINTDIM_combination*3*sizeof(float),cudaMemcpyHostToDevice));


	CUDA_CALL( cudaMalloc(&data->cu_triangle_indexTabel, MAXIMUMPOINTDIM_combination * 3 * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_triangle_indexTabel,triangle_indexTabel,MAXIMUMPOINTDIM_combination*3*sizeof(float),cudaMemcpyHostToDevice));

	CUDA_CALL( cudaMalloc(&data->cu_warp_tabel_transpose, t_width*t_height*3 * sizeof(float)) );
	float *cu_warpTable_t=new float[t_width*t_height*3];
	for (int i=0;i<t_height;i++)
	{
		for (int j=0;j<t_width;j++)
		{
			int cID=i*t_width+j;
			cu_warpTable_t[cID]=warpTabel[3*cID];
			cu_warpTable_t[cID+t_width*t_height]=warpTabel[3*cID+1];
			cu_warpTable_t[cID+2*t_width*t_height]=warpTabel[3*cID+2];
		}
	}
	CUDA_CALL(cudaMemcpy(data->cu_warp_tabel_transpose,cu_warpTable_t, t_width*t_height*3*sizeof(float),cudaMemcpyHostToDevice));
	delete []cu_warpTable_t;

	CUDA_CALL( cudaMalloc(&data->cu_triangle_indexTabel_transpose, t_width*t_height * 3 * sizeof(float)) );
	float *cu_triangle_indexTabel_t=new float[t_width*t_height*3];
	for (int i=0;i<t_height;i++)
	{
		for (int j=0;j<t_width;j++)
		{
			int cID=i*t_width+j;
			cu_triangle_indexTabel_t[cID]=triangle_indexTabel[3*cID];
			cu_triangle_indexTabel_t[cID+t_width*t_height]=triangle_indexTabel[3*cID+1];
			cu_triangle_indexTabel_t[cID+2*t_width*t_height]=triangle_indexTabel[3*cID+2];
		}
	}
	CUDA_CALL(cudaMemcpy(data->cu_triangle_indexTabel_transpose,cu_triangle_indexTabel_t, t_width*t_height*3*sizeof(float),cudaMemcpyHostToDevice));
	delete []cu_triangle_indexTabel_t;

	CUDA_CALL( cudaMalloc(&data->cu_shapeJacobians, MAXIMUMPOINTDIM_combination*s_dim*2 * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_shapeJacobians,shapeJacobians,MAXIMUMPOINTDIM_combination*s_dim*2*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL( cudaMalloc(&data->cu_shapeJacobians_transpose, t_width*t_height*s_dim*2 * sizeof(float)) );
	float *shapeJT=new float[t_width*t_height*s_dim*2];
	for (int i=0;i<t_height;i++)
	{
		for (int j=0;j<t_width;j++)
		{
			int cID=i*t_width+j;
	/*		if (fowardIndex[cID]==-1)
			{
				for (int k=0;k<s_dim;k++)
				{
					shapeJT[k*2*t_height*t_width+cID]=0;
					shapeJT[(k*2+1)*t_height*t_width+cID]=0;
				}
			}
			else*/
			{
				for (int k=0;k<s_dim;k++)
				{
					shapeJT[k*2*t_height*t_width+cID]=shapeJacobians[cID*s_dim*2+k];
					shapeJT[(k*2+1)*t_height*t_width+cID]=shapeJacobians[cID*s_dim*2+k+s_dim];
				}
			}
		
		}
	}
	//cout<<"calculation done\n";
	CUDA_CALL(cudaMemcpy(data->cu_shapeJacobians_transpose,shapeJT,t_width*t_height*s_dim*2*sizeof(float),cudaMemcpyHostToDevice));
	delete []shapeJT;

	//s_weight,t_weight
	CUDA_CALL( cudaMalloc(&data->cu_s_weight, MPD_combination * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_t_weight, MPD_combination * sizeof(float)) );
	
	
	
	//CUDA_CALL( cudaMalloc(&data->cu_errorImage, pix_num * sizeof(float)) );

	//set up the error image, which contains the difference of texture values and feature locations
	

	

	CUDA_CALL( cudaMalloc(&data->cu_parameters, MPD_combination * sizeof(float)) );
//	CUDA_CALL( cudaMalloc(&data->cu_deltaParameters, MPD_combination * sizeof(float)) );

	//this image is shared, no need to allocate it here
	//CUDA_CALL( cudaMalloc(&data->cu_inputImg, MAXIMUMPOINTDIM_combination * sizeof(float)) );

	
	

	CUDA_CALL( cudaMalloc(&data->cu_MaskTabel, t_width*t_height * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_MaskTabel,maskTabel,t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));

	CUDA_CALL( cudaMalloc(&data->cu_gradientX, MAXIMUMPOINTDIM_combination * sizeof(float)) );
	CUDA_CALL( cudaMalloc(&data->cu_gradientY, MAXIMUMPOINTDIM_combination * sizeof(float)) );

	
	CUDA_CALL( cudaMalloc(&data->cu_Hessian, (s_dim+t_dim+4)*(s_dim+t_dim+4+1) * sizeof(float)) );
	data->cu_deltaParameters=data->cu_Hessian+(s_dim+t_dim+4)*(s_dim+t_dim+4);
	CUDA_CALL( cudaMalloc(&data->cu_RTHessian, (s_dim+t_dim+4)*(s_dim+t_dim+4) * sizeof(float)) );

	

	CUDA_CALL( cudaMalloc(&data->cu_Jacobians, (t_width*t_height+MAX_LABEL_NUMBER*2)*(s_dim+t_dim+4) * sizeof(float)) );
	data->cu_RT_Jacobian_transpose=data->cu_Jacobians+pix_num*(s_dim+t_dim+4);

	/*cout<<(s_dim+t_dim+4)<<" "<<data->pix_num<<" "<<"asssss\n";
	cout<<data->cu_RT_Jacobian_transpose<<" "<<data->cu_Jacobians+(s_dim+t_dim+4)*data->pix_num<<endl;*/

	CUDA_CALL( cudaMalloc(&data->cu_Jacobian_transpose, (t_width*t_height+MAX_LABEL_NUMBER)*(s_dim+t_dim+4) * sizeof(float)) );
	float *tmp=new float[ (t_width*t_height+MAX_LABEL_NUMBER*2)*(s_dim+t_dim+4)];
	for (int i=0;i<(t_width*t_height+MAX_LABEL_NUMBER*2)*(s_dim+t_dim+4);i++)
	{
		tmp[i]=0;
	}
	CUDA_CALL(cudaMemcpy(data->cu_Jacobians,tmp, (t_width*t_height+MAX_LABEL_NUMBER*2)*(s_dim+t_dim+4)*sizeof(float),cudaMemcpyHostToDevice));
	delete []tmp;

	CUDA_CALL( cudaMalloc(&data->cu_fullErrorImage, t_width*t_height* sizeof(float)) );
	
	//cout<<t_width*t_height*(s_dim+t_dim+4)<<endl;


	//CUDA_CALL( cudaMalloc(&data->cu_fowardIndexTabel, t_width*t_height * sizeof(float)) );
	//CUDA_CALL(cudaMemcpy(data->cu_fowardIndexTabel,fowardIndex,t_width*t_height*sizeof(float),cudaMemcpyHostToDevice));

	data->cu_fowardIndexTabel=data->cu_MaskTabel;

	//float *cJacobians=new float[(s_dim+t_dim+4)*t_width*t_height];
	//CUDA_CALL(cudaMemcpy(cJacobians, data->cu_Jacobians, (s_dim+t_dim+4)*t_width*t_height*sizeof(float), cudaMemcpyDeviceToHost ));
	//for (int i=0;i<50;i++)
	//{
	//	cout<<cJacobians[i]<<" ";
	//}
	//cout<<endl;

	float *allones=new float[MAXIMUMPOINTDIM_combination];
	for (int i=0;i<pix_num;i++)
	{
		allones[i]=1;
	}
	CUDA_CALL(cudaMalloc(&data->cu_allOnesForImg,MAXIMUMPOINTDIM_combination*sizeof(float)));
	CUDA_CALL(cudaMemcpy(data->cu_allOnesForImg,allones,MAXIMUMPOINTDIM_combination*sizeof(float),cudaMemcpyHostToDevice));


	data->s_dim=s_dim;
	data->t_dim=t_dim;
	data->ptsNum=ptsNum;
	data->pix_num=pix_num;
	data->t_width=t_width;
	data->t_height=t_height;

	data->fullPix_num=data->t_width*data->t_height;

	delete []allones;

	getJacobians_Preprocess_combination<<<t_width*t_height/128+1,128>>>(data->cu_Jacobians,t_width,t_height,s_dim,t_dim,data->pix_num,data->cu_warp_tabel,data->cu_t_vec,data->cu_fowardIndexTabel,data->cu_Jacobian_transpose);

	/*data->host_Hessian=new float[(t_dim+s_dim+4)*(t_dim+s_dim+4)];
	data->host_inv_Hessian=new float[(t_dim+s_dim+4)*(t_dim+s_dim+4)];*/

	CUDA_CALL(cudaHostAlloc(&data->host_Hessian,(t_dim+s_dim+4)*(t_dim+s_dim+4+1)*sizeof(float),cudaHostAllocMapped));
	CUDA_CALL(cudaHostAlloc(&data->host_inv_Hessian,(t_dim+s_dim+4)*(t_dim+s_dim+4)*sizeof(float),cudaHostAllocMapped));
	cudaHostGetDevicePointer((void **)&data->cu_inv_Hessian, (void *)data->host_inv_Hessian, 0);

	data->host_b=data->host_Hessian+(t_dim+s_dim+4)*(t_dim+s_dim+4);
	//CUDA_CALL(cudaHostAlloc(&data->host_b,(t_dim+s_dim+4)*sizeof(float),cudaHostAllocMapped));
	cudaHostGetDevicePointer((void **)&data->cu_deltaB, (void *)data->host_b, 0);

	data->showSingleStep=showSingleStep;

	//CUDA_CALL(cudaBindTexture2D( NULL, currentColorImg_combination,
	//	data->cu_colorImage,
	//	desc_combination,  width,height,
	//	sizeof(float) * width));
	
	//CUDA_CALL(cudaBindTexture2D( NULL, currentColorImg_combination,
	//	trackEngine.cu_colorImage,
	//	desc_combination, ptsNum*2,480,
	//	sizeof(float) * 128));

	/*texture_s_vec.filterMode=cudaFilterModePoint;
	CUDA_CALL(cudaBindTexture2D( NULL, texture_s_vec,
		data->cu_s_vec,
		desc_combination, ptsNum*2 ,s_dim,
		sizeof(float) * 128));*/

	size_t pitch;
	CUDA_CALL(cudaMallocPitch((void**)&data->cu_s_vec_2D,&pitch, ptsNum*2 * sizeof(float), s_dim));

	CUDA_CALL(cudaMemcpy2D((void*)data->cu_s_vec_2D,pitch,(void*)s_vec,sizeof(float)*ptsNum*2,
		sizeof(float)*ptsNum*2,s_dim,cudaMemcpyHostToDevice));

	//cout<<pitch/sizeof(float);
	texture_s_vec.filterMode=cudaFilterModePoint;
	CUDA_CALL(cudaBindTexture2D( NULL, texture_s_vec,
		data->cu_s_vec_2D,
		desc_combination, ptsNum*2 ,s_dim,
		pitch));
	

	CUDA_CALL(cudaMalloc(&data->cu_JConv,MAX_LABEL_NUMBER*2*(s_dim+t_dim+4)*sizeof(float)));


	//prior part
	//prior and prior hessian
	CUDA_CALL(cudaMalloc(&trackEngine.cuPriorMean,(s_dim+(s_dim+t_dim+4)*(s_dim+t_dim+4))*sizeof(float)));
	trackEngine.cuHessianPrior=trackEngine.cuPriorMean+s_dim;
	//CUDA_CALL(cudaMalloc(&trackEngine.cuHessianPrior,(s_dim+t_dim+4)*(s_dim+t_dim+4)*sizeof(float)));
	CUDA_CALL(cudaMalloc(&trackEngine.cu_priorDifference,(s_dim+t_dim+4)*sizeof(float)));
	

	float *tmpMean=new float[s_dim+t_dim+4];
	for (int i=0;i<s_dim+t_dim+4;i++)
	{
		tmpMean[i]=0;
	}
	CUDA_CALL(cudaMemcpy(trackEngine.cuPriorMean,tmpMean, s_dim*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(trackEngine.cu_priorDifference,tmpMean, (s_dim+t_dim+4)*sizeof(float),cudaMemcpyHostToDevice));
	delete []tmpMean;

	 CUDA_CALL(cudaMalloc((void **)&data->ipiv, (s_dim+t_dim+4)*sizeof(float)));

	 //saving the maximum value and their positions
	 CUDA_CALL(cudaMalloc((void **)&trackEngine.cu_maximumPtsPos, (maximumPtsNumPitch*2)*sizeof(float)));
	 CUDA_CALL(cudaHostAlloc((void **)&trackEngine.host_maximumPtsPos,(maximumPtsNumPitch*2)*sizeof(float),cudaHostAllocMapped));


	 /////for trackEngine
	 CUDA_CALL(cudaHostAlloc((void **)&trackEngine.p_mean,(s_dim+(s_dim+t_dim+4)*(s_dim+t_dim+4))*sizeof(float),cudaHostAllocMapped));
	 //CUDA_CALL(cudaHostAlloc((void **)&trackEngine.p_sigma,(s_dim+t_dim+4)*(s_dim+t_dim+4)*sizeof(float),cudaHostAllocMapped));
	 trackEngine.p_sigma=trackEngine.p_mean+s_dim;
	 for (int i=0;i<(s_dim+t_dim+4)*(s_dim+t_dim+4);i++)
	 {
		 trackEngine.p_sigma[i]=0;
	 }
	 setPMeanHessian(trackEngine.p_mean,trackEngine.p_sigma);
	 CUDA_CALL(cudaHostAlloc((void **)&trackEngine.host_colorImage,MAXIMUMPOINTDIM_COM*sizeof(float),cudaHostAllocMapped));
	 CUDA_CALL(cudaHostAlloc((void **)&trackEngine.host_depthImage,MAXIMUMPOINTDIM_COM*sizeof(float),cudaHostAllocMapped));
	 setColorDepthData(trackEngine.host_colorImage,trackEngine.host_depthImage);

	 setAbsIndex(trackEngine.absInd);

	 CUDA_CALL( cudaMalloc(&data->cu_blockTextureData, pix_num*data->blockNum* sizeof(float)) );
	 data->isAdptive=isAptive;
	 data->host_blockTextureData=dataAddress;

	 //set up the streams
	cudaStreamCreate ( &stream1) ;
	cudaStreamCreate ( &stream2) ;
	
	 showCUDAMemoryUsage();
}


extern "C" void setLocalPrior(float *LocalPriorMean,float *localPriorConv,int s_dim,int t_dim)
{
	//CUDA_CALL(cudaMemcpy(trackEngine.cuPriorMean,LocalPriorMean, s_dim*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(trackEngine.cuHessianPrior,localPriorConv, (s_dim+t_dim+4)*(s_dim+t_dim+4)*sizeof(float),cudaMemcpyHostToDevice));

	//cout<<"cpu mean\n";
	//for (int i=0;i<s_dim;i++)
	//{
	//	cout<<LocalPriorMean[i]<<" ";
	//}

	//cout<<endl;
	CUDA_CALL(cudaMemcpy(trackEngine.cuPriorMean,LocalPriorMean, (s_dim+(s_dim+t_dim+4)*(s_dim+t_dim+4))*sizeof(float),cudaMemcpyHostToDevice));

	//CUDA_CALL(cudaMemcpy(trackEngine.cuPriorMean,LocalPriorMean, s_dim*sizeof(float),cudaMemcpyHostToDevice));
	//CUDA_CALL(cudaMemcpy(trackEngine.cuHessianPrior,localPriorConv, (s_dim+t_dim+4)*(s_dim+t_dim+4)*sizeof(float),cudaMemcpyHostToDevice));
	//then calculate the Hessian

	//float *cpuPara=new float[s_dim];
	//CUDA_CALL(cudaMemcpy(cpuPara, trackEngine.cuPriorMean, s_dim*sizeof(float), cudaMemcpyDeviceToHost ));
	//
	//cout<<"mean\n";
	//for (int i=0;i<s_dim;i++)
	//{
	//	cout<<cpuPara[i]<<" ";
	//}
	//cout<<endl;


	//cpuPara=new float[(s_dim+t_dim+4)*(s_dim+t_dim+4)];
	//CUDA_CALL(cudaMemcpy(cpuPara, trackEngine.cuHessianPrior, (s_dim+t_dim+4)*(s_dim+t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));
	//cout<<"hessian\n";
	//for (int i=0;i<(s_dim+t_dim+4)*(s_dim+t_dim+4);i++)
	//{
	//	if (cpuPara[i]!=0)
	//	{
	//		cout<<cpuPara[i]<<" ";
	//	}
	//	
	//	if ((i+1)%(s_dim+t_dim+4)==0)
	//	{
	//		cout<<endl;
	//	}
	//}
	//cout<<endl;


}

const int blockDIM_combination=16;
const int blockDIMX_combination=blockDIM_combination;

__global__ void getJacobians_combination(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			int totalDim=s_dim+t_dim;

			float theta=parameters[totalDim];
			float scale=parameters[totalDim+1];
			float translationX=parameters[totalDim+2];
			float translationY=parameters[totalDim+3];
		
			int tInd0=triangle_indexTabel[3*offset];
			int tInd1=triangle_indexTabel[3*offset+1];
			int tInd2=triangle_indexTabel[3*offset+2];


			float alpha=warp_tabel[offset*3+0];
			float beta=warp_tabel[offset*3+1];
			float gamma=warp_tabel[offset*3+2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape[tInd0]+beta*currentLocalShape[tInd1]+
				gamma*currentLocalShape[tInd2];
			float sumty=alpha*currentLocalShape[tInd0+ptsNum]+beta*currentLocalShape[tInd1+ptsNum]+
				gamma*currentLocalShape[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;
			for (i=0;i<s_dim;i++)
			{
				Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
				//Jacobians[cTotalID+i]=0;
			}

			Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);

			
			
			Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians[cTotalID+s_dim+t_dim+3]=-gy;

			//smooth weight
		}
		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}
		//	
		//}
	}
}

const int threadPerBlock_Jacobians=128;
__global__ void getJacobians_RT_combination_sharedMemory(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum,int pixel_num,double AAMWeight,double RTWeight,float *usedIndex,int usedLabelNum,float *Jacobians_transpos=NULL)
{
	__shared__ float globalTransformationParameters[4];
	__shared__ float warpTabel_shared[threadPerBlock_Jacobians*3];
	__shared__ float triangleIndex_shared[threadPerBlock_Jacobians*3];
	__shared__ float fowardIndex_shared[threadPerBlock_Jacobians];
	__shared__ float currentLocalShape_shared[200];

	__shared__ float gradientX_shared[threadPerBlock_Jacobians];
	__shared__ float gradientY_shared[threadPerBlock_Jacobians];

	//__shared__ float usedIndex_shared[30];

	//__shared__ float shapeJacobians_shared[2*threadPerBlock_Jacobians];
	int threadId=threadIdx.x;
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	//if (offset<t_width*t_height)
	//{
	//	gradientX_shared[threadId]=gradientX[offset];
	//	gradientY_shared[threadId]=gradientY[offset];

	//}
		if (threadId<4)
		{
			globalTransformationParameters[threadId]=parameters[s_dim+t_dim+threadId];
		}

		if (threadId<ptsNum)
		{
			currentLocalShape_shared[threadId]=currentLocalShape[threadId];
			currentLocalShape_shared[threadId+ptsNum]=currentLocalShape[threadId+ptsNum];
		}

		fowardIndex_shared[threadId]=fowardIndex[offset];

		int totalDim=t_width*t_height;
		if (fowardIndex_shared[threadId]!=-1)
		{
			triangleIndex_shared[threadId]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[3*offset+2];

		/*	triangleIndex_shared[threadId*3]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId*3+1]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId*3+2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId*3]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId*3+1]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId*3+2]=warp_tabel[3*offset+2];*/

			/*triangleIndex_shared[threadId]=triangle_indexTabel[offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[offset+totalDim];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[offset+2*totalDim];

			warpTabel_shared[threadId]=warp_tabel[offset];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[offset+totalDim];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[offset+2*totalDim];*/
		}

	/*	if (threadId<usedLabelNum)
		{
			usedIndex_shared[threadId]=usedIndex[threadId];
		}*/
	
	
	


	__syncthreads();

	
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			//int totalDim=s_dim+t_dim;

			//float theta=parameters[totalDim];
			//float scale=parameters[totalDim+1];
			//float translationX=parameters[totalDim+2];
			//float translationY=parameters[totalDim+3];


			float theta=globalTransformationParameters[0];
			float scale=globalTransformationParameters[1];
			float translationX=globalTransformationParameters[2];
			float translationY=globalTransformationParameters[3];
		
			int tInd0=triangleIndex_shared[threadId];
			int tInd1=triangleIndex_shared[threadId+threadPerBlock_Jacobians];
			int tInd2=triangleIndex_shared[threadId+threadPerBlock_Jacobians*2];


			float alpha=warpTabel_shared[threadId];
			float beta=warpTabel_shared[threadId+threadPerBlock_Jacobians];
			float gamma=warpTabel_shared[threadId+threadPerBlock_Jacobians*2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape_shared[tInd0]+beta*currentLocalShape_shared[tInd1]+
				gamma*currentLocalShape_shared[tInd2];
			float sumty=alpha*currentLocalShape_shared[tInd0+ptsNum]+beta*currentLocalShape_shared[tInd1+ptsNum]+
				gamma*currentLocalShape_shared[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			//float gx=gradientX_shared[threadId];float gy=gradientY_shared[threadId];
			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;
			for (i=0;i<s_dim;i++)
			{
				//Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					//cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);

				Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
				//Jacobians[cTotalID+i]=0;
			}

			

			//Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			Jacobians_transpos[(s_dim+t_dim+1)*pixel_num+pixelID]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			//Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			Jacobians_transpos[(s_dim+t_dim)*pixel_num+pixelID]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			
			
			//Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians_transpos[(s_dim+t_dim+2)*pixel_num+pixelID]=-gx;
			//Jacobians[cTotalID+s_dim+t_dim+3]=-gy;
			Jacobians_transpos[(s_dim+t_dim+3)*pixel_num+pixelID]=-gy;

		


		}
		//return;
		//smooth weight
		if (offset<usedLabelNum)
		{
			int allDim=s_dim+t_dim+4;
			

			//get current feature index
			//int currentId=usedIndex[offset];
			int currentFeatureIndex=usedIndex[offset];;
			//int globalCurrentID=

			int cTotalID=(pixel_num+offset*2)*allDim;

			int totalDim=s_dim+t_dim;

			float theta=globalTransformationParameters[0];
			float scale=globalTransformationParameters[1];
			float translationX=globalTransformationParameters[2];
			float translationY=globalTransformationParameters[3];

			float costheta=cos(theta);float sintheta=sin(theta);

			for (int i=0;i<s_dim;i++)
			{
				Jacobians[cTotalID+i]=scale*(costheta*tex2D(texture_s_vec,currentFeatureIndex,i)-sintheta*
					tex2D(texture_s_vec,currentFeatureIndex+ptsNum,i));

				//Jacobians[cTotalID+i]=tex2D(texture_s_vec,currentFeatureIndex,i);
				
				Jacobians[cTotalID+allDim+i]=scale*(sintheta*tex2D(texture_s_vec,currentFeatureIndex,i)+costheta*
					tex2D(texture_s_vec,currentFeatureIndex+ptsNum,i));
				//Jacobians[cTotalID+allDim+i]=tex2D(texture_s_vec,currentFeatureIndex,i);
			}
			//Jacobians[cTotalID+0]=scale;Jacobians[cTotalID+1]=costheta;
			//Jacobians[cTotalID+2]=sintheta;//Jacobians[cTotalID+i]=costheta;
			//Jacobians[cTotalID+0]=currentFeatureIndex;

			float cx=currentLocalShape_shared[currentFeatureIndex];
			float cy=currentLocalShape_shared[currentFeatureIndex+ptsNum];
			
			Jacobians[cTotalID+s_dim+t_dim]=scale*(-sintheta*cx-costheta*
				cy);
			Jacobians[cTotalID+s_dim+t_dim+allDim]=scale*(costheta*cx-sintheta*
				cy);

			Jacobians[cTotalID+s_dim+t_dim+1]=costheta*cx-sintheta*cy;
			Jacobians[cTotalID+s_dim+t_dim+allDim+1]=sintheta*cx+costheta*cy;

			//Jacobians[cTotalID+s_dim+t_dim]=currentLocalShape[currentFeatureIndex];
			//Jacobians[cTotalID+s_dim+t_dim+1]=currentLocalShape[currentFeatureIndex+ptsNum];

			//put theses two in pre-calculation
			Jacobians[cTotalID+s_dim+t_dim+2]=1;
			Jacobians[cTotalID+s_dim+t_dim+allDim+2]=0;	

			Jacobians[cTotalID+s_dim+t_dim+3]=0;
			Jacobians[cTotalID+s_dim+t_dim+allDim+3]=1;

		}



		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}
		//	
		//}
	}
}


__global__ void getJacobians_RT_combination_transpose(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians_T,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum,int pixel_num,double AAMWeight,double RTWeight,float *usedIndex,int usedLabelNum,float *Jacobians_transpos=NULL)
{
	__shared__ float globalTransformationParameters[4];
	__shared__ float warpTabel_shared[threadPerBlock_Jacobians*3];
	__shared__ float triangleIndex_shared[threadPerBlock_Jacobians*3];
	__shared__ float fowardIndex_shared[threadPerBlock_Jacobians];
	__shared__ float currentLocalShape_shared[200];

	__shared__ float gradientX_shared[threadPerBlock_Jacobians];
	__shared__ float gradientY_shared[threadPerBlock_Jacobians];

	//__shared__ float usedIndex_shared[30];

	//__shared__ float shapeJacobians_shared[2*threadPerBlock_Jacobians];
	int threadId=threadIdx.x;
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	//if (offset<t_width*t_height)
	//{
	//	gradientX_shared[threadId]=gradientX[offset];
	//	gradientY_shared[threadId]=gradientY[offset];

	//}
		if (threadId<4)
		{
			globalTransformationParameters[threadId]=parameters[s_dim+t_dim+threadId];
		}

		if (threadId<ptsNum)
		{
			currentLocalShape_shared[threadId]=currentLocalShape[threadId];
			currentLocalShape_shared[threadId+ptsNum]=currentLocalShape[threadId+ptsNum];
		}

		fowardIndex_shared[threadId]=fowardIndex[offset];

		int totalDim=t_width*t_height;
		if (fowardIndex_shared[threadId]!=-1)
		{
			triangleIndex_shared[threadId]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[3*offset+2];

		/*	triangleIndex_shared[threadId*3]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId*3+1]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId*3+2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId*3]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId*3+1]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId*3+2]=warp_tabel[3*offset+2];*/

			/*triangleIndex_shared[threadId]=triangle_indexTabel[offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[offset+totalDim];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[offset+2*totalDim];

			warpTabel_shared[threadId]=warp_tabel[offset];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[offset+totalDim];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[offset+2*totalDim];*/
		}

	/*	if (threadId<usedLabelNum)
		{
			usedIndex_shared[threadId]=usedIndex[threadId];
		}*/
	
	
	


	__syncthreads();

	
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			//int totalDim=s_dim+t_dim;

			//float theta=parameters[totalDim];
			//float scale=parameters[totalDim+1];
			//float translationX=parameters[totalDim+2];
			//float translationY=parameters[totalDim+3];


			float theta=globalTransformationParameters[0];
			float scale=globalTransformationParameters[1];
			float translationX=globalTransformationParameters[2];
			float translationY=globalTransformationParameters[3];
		
			int tInd0=triangleIndex_shared[threadId];
			int tInd1=triangleIndex_shared[threadId+threadPerBlock_Jacobians];
			int tInd2=triangleIndex_shared[threadId+threadPerBlock_Jacobians*2];


			float alpha=warpTabel_shared[threadId];
			float beta=warpTabel_shared[threadId+threadPerBlock_Jacobians];
			float gamma=warpTabel_shared[threadId+threadPerBlock_Jacobians*2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape_shared[tInd0]+beta*currentLocalShape_shared[tInd1]+
				gamma*currentLocalShape_shared[tInd2];
			float sumty=alpha*currentLocalShape_shared[tInd0+ptsNum]+beta*currentLocalShape_shared[tInd1+ptsNum]+
				gamma*currentLocalShape_shared[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			//float gx=gradientX_shared[threadId];float gy=gradientY_shared[threadId];
			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			//int allDim=s_dim+t_dim+4;
			//int cTotalID=pixelID*allDim;
			//int i=0;
			//for (i=0;i<s_dim;i++)
			//{
			//	//Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
			//		//cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);

			//	Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
			//		cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
			//	//Jacobians[cTotalID+i]=0;
			//}


				//put shapeJacobian into column major!
			int cWH=t_width*t_height;
			for (int i=0;i<s_dim;i++)
			{
				//Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					//cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);

				//Jacobians_transpos[i*pixel_num+pixelID]=tmp;
				float tJx=shapeJacobians_T[i*2*cWH+offset];
				float tJy=shapeJacobians_T[(i*2+1)*cWH+offset];
				Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*tJx+
					cgradient[1]*tJy);
				/*Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);*/
				//Jacobians[cTotalID+i]=0;
			}
			

			//Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			Jacobians_transpos[(s_dim+t_dim+1)*pixel_num+pixelID]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			//Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			Jacobians_transpos[(s_dim+t_dim)*pixel_num+pixelID]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			
			
			//Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians_transpos[(s_dim+t_dim+2)*pixel_num+pixelID]=-gx;
			//Jacobians[cTotalID+s_dim+t_dim+3]=-gy;
			Jacobians_transpos[(s_dim+t_dim+3)*pixel_num+pixelID]=-gy;

		


		}
		//return;
		//smooth weight
		//if (offset<usedLabelNum)
		//{
		//	int allDim=s_dim+t_dim+4;
		//	

		//	//get current feature index
		//	//int currentId=usedIndex[offset];
		//	int currentFeatureIndex=usedIndex[offset];;
		//	//int globalCurrentID=

		//	int cTotalID=(pixel_num+offset*2)*allDim;

		//	int totalDim=s_dim+t_dim;

		//	float theta=globalTransformationParameters[0];
		//	float scale=globalTransformationParameters[1];
		//	float translationX=globalTransformationParameters[2];
		//	float translationY=globalTransformationParameters[3];

		//	float costheta=cos(theta);float sintheta=sin(theta);

		//	for (int i=0;i<s_dim;i++)
		//	{
		//		float ttx=tex2D(texture_s_vec,currentFeatureIndex,i);
		//		float tty=tex2D(texture_s_vec,currentFeatureIndex+ptsNum,i);
		//		Jacobians[cTotalID+i]=scale*(costheta*ttx-sintheta*tty);

		//		//Jacobians[cTotalID+i]=tex2D(texture_s_vec,currentFeatureIndex,i);
		//		
		//		Jacobians[cTotalID+allDim+i]=scale*(sintheta*ttx+costheta*tty);
		//		//Jacobians[cTotalID+allDim+i]=tex2D(texture_s_vec,currentFeatureIndex,i);
		//	}
		//	//Jacobians[cTotalID+0]=scale;Jacobians[cTotalID+1]=costheta;
		//	//Jacobians[cTotalID+2]=sintheta;//Jacobians[cTotalID+i]=costheta;
		//	//Jacobians[cTotalID+0]=currentFeatureIndex;

		//	float cx=currentLocalShape_shared[currentFeatureIndex];
		//	float cy=currentLocalShape_shared[currentFeatureIndex+ptsNum];
		//	
		//	Jacobians[cTotalID+s_dim+t_dim]=scale*(-sintheta*cx-costheta*
		//		cy);
		//	Jacobians[cTotalID+s_dim+t_dim+allDim]=scale*(costheta*cx-sintheta*
		//		cy);

		//	Jacobians[cTotalID+s_dim+t_dim+1]=costheta*cx-sintheta*cy;
		//	Jacobians[cTotalID+s_dim+t_dim+allDim+1]=sintheta*cx+costheta*cy;

		//	//Jacobians[cTotalID+s_dim+t_dim]=currentLocalShape[currentFeatureIndex];
		//	//Jacobians[cTotalID+s_dim+t_dim+1]=currentLocalShape[currentFeatureIndex+ptsNum];

		//	//put theses two in pre-calculation
		//	Jacobians[cTotalID+s_dim+t_dim+2]=1;
		//	Jacobians[cTotalID+s_dim+t_dim+allDim+2]=0;	

		//	Jacobians[cTotalID+s_dim+t_dim+3]=0;
		//	Jacobians[cTotalID+s_dim+t_dim+allDim+3]=1;

		//}
		//return;

		if (offset<usedLabelNum)
		{
			int allDim=s_dim+t_dim+4;


			//get current feature index
			//int currentId=usedIndex[offset];
			int currentFeatureIndex=usedIndex[offset];
			//int globalCurrentID=
			int curDim=usedLabelNum*2;
			int cTotalID=pixel_num*allDim+offset*2;
			//int step=usedLabelNum*allDim;
			
			float theta=globalTransformationParameters[0];
			float scale=globalTransformationParameters[1];
			float translationX=globalTransformationParameters[2];
			float translationY=globalTransformationParameters[3];

			float costheta=cos(theta);float sintheta=sin(theta);

			for (int i=0;i<s_dim;i++)
			{
				float ttx=tex2D(texture_s_vec,currentFeatureIndex,i);
				float tty=tex2D(texture_s_vec,currentFeatureIndex+ptsNum,i);
				Jacobians[cTotalID+i*curDim]=scale*(costheta*ttx-sintheta*tty);

				//Jacobians[cTotalID+i]=tex2D(texture_s_vec,currentFeatureIndex,i);

				Jacobians[cTotalID+i*curDim+1]=scale*(sintheta*ttx+costheta*tty);
				//Jacobians[cTotalID+allDim+i]=tex2D(texture_s_vec,currentFeatureIndex,i);
			}

			for (int i=s_dim;i<s_dim+t_dim;i++)
			{
				Jacobians[cTotalID+i*curDim]=Jacobians[cTotalID+i*curDim+1]=0;
			}
			//Jacobians[cTotalID+0]=scale;Jacobians[cTotalID+1]=costheta;
			//Jacobians[cTotalID+2]=sintheta;//Jacobians[cTotalID+i]=costheta;
			//Jacobians[cTotalID+0]=currentFeatureIndex;

			float cx=currentLocalShape_shared[currentFeatureIndex];
			float cy=currentLocalShape_shared[currentFeatureIndex+ptsNum];

			Jacobians[(s_dim+t_dim)*curDim+cTotalID]=scale*(-sintheta*cx-costheta*
				cy);
			Jacobians[(s_dim+t_dim)*curDim+cTotalID+1]=scale*(costheta*cx-sintheta*
				cy);

			Jacobians[(s_dim+t_dim+1)*curDim+cTotalID]=costheta*cx-sintheta*cy;
			Jacobians[(s_dim+t_dim+1)*curDim+cTotalID+1]=sintheta*cx+costheta*cy;

			//Jacobians[cTotalID+s_dim+t_dim]=currentLocalShape[currentFeatureIndex];
			//Jacobians[cTotalID+s_dim+t_dim+1]=currentLocalShape[currentFeatureIndex+ptsNum];

			//put theses two in pre-calculation
			Jacobians[(s_dim+t_dim+2)*curDim+cTotalID]=1;
			Jacobians[(s_dim+t_dim+2)*curDim+cTotalID+1]=0;	

			Jacobians[(s_dim+t_dim+3)*curDim+cTotalID]=0;
			Jacobians[(s_dim+t_dim+3)*curDim+cTotalID+1]=1;

		}



		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}
		//	
		//}
	}
}

__global__ void getJacobians_combination_shapeTranspose(float *gradientX,float *gradientY,float *Jacobians_transpos,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians_T,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum,int pixel_num=0)
{

		__shared__ float globalTransformationParameters[4];
	__shared__ float warpTabel_shared[threadPerBlock_Jacobians*3];
	__shared__ float triangleIndex_shared[threadPerBlock_Jacobians*3];
	__shared__ float fowardIndex_shared[threadPerBlock_Jacobians];
	__shared__ float currentLocalShape_shared[200];

	__shared__ float gradientX_shared[threadPerBlock_Jacobians];
	__shared__ float gradientY_shared[threadPerBlock_Jacobians];

	//__shared__ float usedIndex_shared[30];

	//__shared__ float shapeJacobians_shared[2*threadPerBlock_Jacobians];
	int threadId=threadIdx.x;
	int offset=threadIdx.x+blockIdx.x*blockDim.x;

		if (threadId<4)
		{
			globalTransformationParameters[threadId]=parameters[s_dim+t_dim+threadId];
		}

		if (threadId<ptsNum)
		{
			currentLocalShape_shared[threadId]=currentLocalShape[threadId];
			currentLocalShape_shared[threadId+ptsNum]=currentLocalShape[threadId+ptsNum];
		}

		fowardIndex_shared[threadId]=fowardIndex[offset];

		int totalDim=t_width*t_height;
		if (fowardIndex_shared[threadId]!=-1)
		{
			triangleIndex_shared[threadId]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[3*offset+2];
		}
		__syncthreads();

	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID==-1)
		{
			return;
		}
		//if (pixelID!=-1)
		{
			float theta=globalTransformationParameters[0];
			float scale=globalTransformationParameters[1];
			float translationX=globalTransformationParameters[2];
			float translationY=globalTransformationParameters[3];
		
			int tInd0=triangleIndex_shared[threadId];
			int tInd1=triangleIndex_shared[threadId+threadPerBlock_Jacobians];
			int tInd2=triangleIndex_shared[threadId+threadPerBlock_Jacobians*2];


			float alpha=warpTabel_shared[threadId];
			float beta=warpTabel_shared[threadId+threadPerBlock_Jacobians];
			float gamma=warpTabel_shared[threadId+threadPerBlock_Jacobians*2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape_shared[tInd0]+beta*currentLocalShape_shared[tInd1]+
				gamma*currentLocalShape_shared[tInd2];
			float sumty=alpha*currentLocalShape_shared[tInd0+ptsNum]+beta*currentLocalShape_shared[tInd1+ptsNum]+
				gamma*currentLocalShape_shared[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			//float gx=gradientX_shared[threadId];float gy=gradientY_shared[threadId];
			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;

			//float tmp=0;

			//put shapeJacobian into column major!
			int cWH=t_width*t_height;
			for (i=0;i<s_dim;i++)
			{
				//Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					//cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);

				//Jacobians_transpos[i*pixel_num+pixelID]=tmp;
				float tJx=shapeJacobians_T[i*2*cWH+offset];
				float tJy=shapeJacobians_T[(i*2+1)*cWH+offset];
				Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*tJx+
					cgradient[1]*tJy);
				/*Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);*/
				//Jacobians[cTotalID+i]=0;
			}

			

			//Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			Jacobians_transpos[(s_dim+t_dim+1)*pixel_num+pixelID]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			//Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			Jacobians_transpos[(s_dim+t_dim)*pixel_num+pixelID]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			
			
			//Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians_transpos[(s_dim+t_dim+2)*pixel_num+pixelID]=-gx;
			//Jacobians[cTotalID+s_dim+t_dim+3]=-gy;
			Jacobians_transpos[(s_dim+t_dim+3)*pixel_num+pixelID]=-gy;
		}
	}
}

__global__ void getJacobians_combination_sharedMemory(float *gradientX,float *gradientY,float *Jacobians_transpos,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum,int pixel_num=0)
{

		__shared__ float globalTransformationParameters[4];
	__shared__ float warpTabel_shared[threadPerBlock_Jacobians*3];
	__shared__ float triangleIndex_shared[threadPerBlock_Jacobians*3];
	__shared__ float fowardIndex_shared[threadPerBlock_Jacobians];
	__shared__ float currentLocalShape_shared[200];

	__shared__ float gradientX_shared[threadPerBlock_Jacobians];
	__shared__ float gradientY_shared[threadPerBlock_Jacobians];

	//__shared__ float usedIndex_shared[30];

	//__shared__ float shapeJacobians_shared[2*threadPerBlock_Jacobians];
	int threadId=threadIdx.x;
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	//if (offset<t_width*t_height)
	//{
	//	gradientX_shared[threadId]=gradientX[offset];
	//	gradientY_shared[threadId]=gradientY[offset];

	//}
		if (threadId<4)
		{
			globalTransformationParameters[threadId]=parameters[s_dim+t_dim+threadId];
		}

		if (threadId<ptsNum)
		{
			currentLocalShape_shared[threadId]=currentLocalShape[threadId];
			currentLocalShape_shared[threadId+ptsNum]=currentLocalShape[threadId+ptsNum];
		}

		fowardIndex_shared[threadId]=fowardIndex[offset];

		int totalDim=t_width*t_height;
		if (fowardIndex_shared[threadId]!=-1)
		{
			triangleIndex_shared[threadId]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[3*offset+2];

		/*	triangleIndex_shared[threadId*3]=triangle_indexTabel[3*offset];
			triangleIndex_shared[threadId*3+1]=triangle_indexTabel[3*offset+1];
			triangleIndex_shared[threadId*3+2]=triangle_indexTabel[3*offset+2];

			warpTabel_shared[threadId*3]=warp_tabel[offset*3+0];
			warpTabel_shared[threadId*3+1]=warp_tabel[3*offset+1];
			warpTabel_shared[threadId*3+2]=warp_tabel[3*offset+2];*/

			/*triangleIndex_shared[threadId]=triangle_indexTabel[offset];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians]=triangle_indexTabel[offset+totalDim];
			triangleIndex_shared[threadId+threadPerBlock_Jacobians*2]=triangle_indexTabel[offset+2*totalDim];

			warpTabel_shared[threadId]=warp_tabel[offset];
			warpTabel_shared[threadId+threadPerBlock_Jacobians]=warp_tabel[offset+totalDim];
			warpTabel_shared[threadId+threadPerBlock_Jacobians*2]=warp_tabel[offset+2*totalDim];*/
		}

	/*	if (threadId<usedLabelNum)
		{
			usedIndex_shared[threadId]=usedIndex[threadId];
		}*/
	
	
	


	__syncthreads();

	
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			//int totalDim=s_dim+t_dim;

			//float theta=parameters[totalDim];
			//float scale=parameters[totalDim+1];
			//float translationX=parameters[totalDim+2];
			//float translationY=parameters[totalDim+3];


			float theta=globalTransformationParameters[0];
			float scale=globalTransformationParameters[1];
			float translationX=globalTransformationParameters[2];
			float translationY=globalTransformationParameters[3];
		
			int tInd0=triangleIndex_shared[threadId];
			int tInd1=triangleIndex_shared[threadId+threadPerBlock_Jacobians];
			int tInd2=triangleIndex_shared[threadId+threadPerBlock_Jacobians*2];


			float alpha=warpTabel_shared[threadId];
			float beta=warpTabel_shared[threadId+threadPerBlock_Jacobians];
			float gamma=warpTabel_shared[threadId+threadPerBlock_Jacobians*2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape_shared[tInd0]+beta*currentLocalShape_shared[tInd1]+
				gamma*currentLocalShape_shared[tInd2];
			float sumty=alpha*currentLocalShape_shared[tInd0+ptsNum]+beta*currentLocalShape_shared[tInd1+ptsNum]+
				gamma*currentLocalShape_shared[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			//float gx=gradientX_shared[threadId];float gy=gradientY_shared[threadId];
			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;
			for (i=0;i<s_dim;i++)
			{
				//Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					//cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);

				Jacobians_transpos[i*pixel_num+pixelID]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
				//Jacobians[cTotalID+i]=0;
			}

			

			//Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			Jacobians_transpos[(s_dim+t_dim+1)*pixel_num+pixelID]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			//Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			Jacobians_transpos[(s_dim+t_dim)*pixel_num+pixelID]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);
			
			
			//Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians_transpos[(s_dim+t_dim+2)*pixel_num+pixelID]=-gx;
			//Jacobians[cTotalID+s_dim+t_dim+3]=-gy;
			Jacobians_transpos[(s_dim+t_dim+3)*pixel_num+pixelID]=-gy;

		


		}
	}
}

//no need to add in the weight, it will be encoded in the error image
__global__ void getJacobians_RT_combination(float *gradientX,float *gradientY,float *Jacobians,int t_width,int t_height,int s_dim,int t_dim,float *warp_tabel,float *triangle_indexTabel,
							 float *shapeJacobians,float *t_vec,float *parameters,float *fowardIndex,float *currentLocalShape,int ptsNum,int pixel_num,double AAMWeight,double RTWeight,float *usedIndex,int usedLabelNum)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			//float *triangle_indexTabel=tdata->cu_triangle_indexTabel;
		//	float *warp_tabel=tdata->cu_warp_tabel;
			//float *currentLocalShape=tdata->cu_currentLocalShape;
			//float *shapeJacobians=tdata->cu_shapeJacobians;
			//float *t_vec=tdata->cu_t_vec;
			//float *parameters=tdata->cu_parameters;
			//float *fowardIndex=tdata->cu_fowardIndexTabel;


			//int s_dim=tdata->s_dim;
			//int t_dim=tdata->t_dim;
			int totalDim=s_dim+t_dim;

			float theta=parameters[totalDim];
			float scale=parameters[totalDim+1];
			float translationX=parameters[totalDim+2];
			float translationY=parameters[totalDim+3];
		
			int tInd0=triangle_indexTabel[3*offset];
			int tInd1=triangle_indexTabel[3*offset+1];
			int tInd2=triangle_indexTabel[3*offset+2];


			float alpha=warp_tabel[offset*3+0];
			float beta=warp_tabel[offset*3+1];
			float gamma=warp_tabel[offset*3+2];

	/*		int tInd0=0;
			int tInd1=0;
			int tInd2=0;


			float alpha=0;
			float beta=0;
			float gamma=0;*/

			float sumtx=alpha*currentLocalShape[tInd0]+beta*currentLocalShape[tInd1]+
				gamma*currentLocalShape[tInd2];
			float sumty=alpha*currentLocalShape[tInd0+ptsNum]+beta*currentLocalShape[tInd1+ptsNum]+
				gamma*currentLocalShape[tInd2+ptsNum];

			float cgradient[2];
			float costheta=cos(theta);float sintheta=sin(theta);

			float gx=gradientX[offset];float gy=gradientY[offset];
			cgradient[0]=-(gx*costheta+gy*sintheta);
			cgradient[1]=-(gx*(-sintheta)+gy*costheta);

			int allDim=s_dim+t_dim+4;
			int cTotalID=pixelID*allDim;
			int i=0;
			for (i=0;i<s_dim;i++)
			{
				Jacobians[cTotalID+i]=scale*(cgradient[0]*shapeJacobians[2*s_dim*offset+i]+
					cgradient[1]*shapeJacobians[2*s_dim*offset+i+s_dim]);
				//Jacobians[cTotalID+i]=0;
			}

			Jacobians[cTotalID+s_dim+t_dim+1]=cgradient[0]*sumtx+cgradient[1]*sumty;

			cgradient[0]=-(gx*-sintheta+gy*costheta);
			cgradient[1]=-(gx*-costheta+gy*-sintheta);
			Jacobians[cTotalID+s_dim+t_dim]=scale*(cgradient[0]*sumtx+cgradient[1]*sumty);

			
			
			Jacobians[cTotalID+s_dim+t_dim+2]=-gx;
			Jacobians[cTotalID+s_dim+t_dim+3]=-gy;

		


		}

		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}
		//	
		//}

		//smooth weight
		if (offset<usedLabelNum)
		{
			int allDim=s_dim+t_dim+4;
			

			//get current feature index
			int currentFeatureIndex=usedIndex[offset];

			int cTotalID=(pixel_num+offset*2)*allDim;

			int totalDim=s_dim+t_dim;

			float theta=parameters[totalDim];
			float scale=parameters[totalDim+1];
			float translationX=parameters[totalDim+2];
			float translationY=parameters[totalDim+3];

			float costheta=cos(theta);float sintheta=sin(theta);

			for (int i=0;i<s_dim;i++)
			{
				Jacobians[cTotalID+i]=scale*(costheta*tex2D(texture_s_vec,currentFeatureIndex,i)-sintheta*
					tex2D(texture_s_vec,currentFeatureIndex+ptsNum,i));
				Jacobians[cTotalID+allDim+i]=scale*(sintheta*tex2D(texture_s_vec,currentFeatureIndex,i)+costheta*
					tex2D(texture_s_vec,currentFeatureIndex+ptsNum,i));
			}
			
			Jacobians[cTotalID+s_dim+t_dim]=scale*(-sintheta*currentLocalShape[currentFeatureIndex]-costheta*
				currentLocalShape[currentFeatureIndex+ptsNum]);
			Jacobians[cTotalID+s_dim+t_dim+allDim]=scale*(costheta*currentLocalShape[currentFeatureIndex]-sintheta*
				currentLocalShape[currentFeatureIndex+ptsNum]);

			Jacobians[cTotalID+s_dim+t_dim+1]=costheta*currentLocalShape[currentFeatureIndex]-sintheta*currentLocalShape[currentFeatureIndex+ptsNum];
			Jacobians[cTotalID+s_dim+t_dim+allDim+1]=sintheta*currentLocalShape[currentFeatureIndex]+costheta*currentLocalShape[currentFeatureIndex+ptsNum];


			//put theses two in pre-calculation
			Jacobians[cTotalID+s_dim+t_dim+2]=1;
			Jacobians[cTotalID+s_dim+t_dim+allDim+2]=0;	

			Jacobians[cTotalID+s_dim+t_dim+3]=0;
			Jacobians[cTotalID+s_dim+t_dim+allDim+3]=1;

		}



	
	}
}

__global__ void getFullShape_combination(float *currentLocalShape, int ptsNum,int totalDim,float *parameters,float *finalShape,float *lastShape)
{

	__shared__ float theta;
	__shared__ float scale;
	__shared__ float translationX,translationY;
	if (threadIdx.x==0)
	{
		theta=parameters[totalDim];
		scale=parameters[totalDim+1];
		translationX=parameters[totalDim+2];
		translationY=parameters[totalDim+3];
	}

	__syncthreads();


	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<ptsNum)
	{
		lastShape[offset]=finalShape[offset];
		lastShape[offset+ptsNum]=finalShape[offset+ptsNum];
		/*float theta=parameters[totalDim];
		float scale=parameters[totalDim+1];
		float translationX=parameters[totalDim+2];
		float translationY=parameters[totalDim+3];*/
		finalShape[offset]=scale*(cos(theta)*currentLocalShape[offset]-sin(theta)*currentLocalShape[offset+ptsNum])+translationX;
		finalShape[offset+ptsNum]=scale*(sin(theta)*currentLocalShape[offset]+cos(theta)*currentLocalShape[offset+ptsNum])+translationY;
	}
}

void cu_getCurrentShape_combination(float *s_mean,float *s_weight,float *s_vec,int s_dim,int t_dim,int ptsNum,float *parameters,float *finalShape,float *lastShape)
{
	float alpha = 1.0f;
	float beta=1.0f;
	int m,n,k,lda,ldb,ldc;
	m=1;
	n=ptsNum*2;
	k=s_dim;

	lda=1;
	ldb=ptsNum*2;
	ldc=1;

	CUBLAS_CALL(cublasSgemm_v2(trackEngine.AAM_ENGINE->blas_handle_,CUBLAS_OP_N,CUBLAS_OP_T,m,n,k,&alpha,s_weight,lda,s_vec,ldb,&beta,s_mean,ldc));

	//now smean is the local shape
	int totaDim=s_dim+t_dim;
	getFullShape_combination<<<(ptsNum+128)/128,128>>>(s_mean,ptsNum,totaDim,parameters,finalShape,lastShape);

}

__global__ void updateTextureJacobian(int t_width,int t_height,int s_dim,int t_dim,int pixelNum,
	float *t_vec,float *fowardIndex,float *Jacobian_transpose)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<t_width*t_height)
	{
		int pixelID=fowardIndex[offset];
		if (pixelID!=-1)
		{
			int allDim=s_dim+t_dim+4;
			int i=0;
			for (i=s_dim;i<s_dim+t_dim;i++)
			{
				Jacobian_transpose[i*pixelNum+pixelID]=t_vec[(i-s_dim)*pixelNum+pixelID];
			}
		}
		//else
		//{
		//	int i=0;
		//	int allDim=s_dim+t_dim+4;
		//	for (i=0;i<allDim;i++)
		//	{
		//		Jacobians[offset*allDim+i]=0;
		//		//Jacobians[offset]=0;
		//	}

		//}
	}
}

void updateTextureModel_cuda(float *newMeanVec)
{
	//float *newMeanAndVec=AAM_DataEngine.newMeanAndTVec;
	int pixNum=trackEngine.AAM_ENGINE->pix_num;
	/*for (int i=0;i<pixNum;i++)
	{
		newMeanAndVec[i]=newMean[i];
	}
	for (int i=pixNum;i<pixNum*AAM_DataEngine.t_dim;i++)
	{
		newMeanAndVec[i]=newModel[i-pixNum];
	}*/

	CUDA_CALL(cudaMemcpy(trackEngine.AAM_ENGINE->cu_t_mean,newMeanVec,pixNum*(trackEngine.AAM_ENGINE->t_dim+1)*sizeof(float),cudaMemcpyHostToDevice));
	updateTextureJacobian<<<trackEngine.AAM_ENGINE->t_width*trackEngine.AAM_ENGINE->t_height/128+1,128>>>(trackEngine.AAM_ENGINE->t_width,trackEngine.AAM_ENGINE->t_height,
		trackEngine.AAM_ENGINE->s_dim,trackEngine.AAM_ENGINE->t_dim,pixNum,trackEngine.AAM_ENGINE->cu_t_vec,
		trackEngine.AAM_ENGINE->cu_fowardIndexTabel,trackEngine.AAM_ENGINE->cu_Jacobian_transpose);


}

void cu_getCurrentTexture_combination(float *t_mean,float *t_weight,float *t_vec,int t_dim,int pix_num)
{
	float alpha = 1.0f;
	float beta=1.0f;
	int m,n,k,lda,ldb,ldc;
	m=1;
	n=pix_num;
	k=t_dim;

	lda=1;
	ldb=pix_num;
	ldc=1;

	//CUBLAS_CALL(cublasSgemm_v2(trackEngine.AAM_ENGINE->blas_handle_,CUBLAS_OP_N,CUBLAS_OP_T,m,n,k,&alpha,t_weight,lda,t_vec,ldb,&beta,t_mean,ldc));	
	CUBLAS_CALL(cublasSgemv_v2(trackEngine.AAM_ENGINE->blas_handle_,CUBLAS_OP_N,pix_num,t_dim,&alpha,
		t_vec,pix_num,t_weight,1,&beta,t_mean,1));
}

__global__ void vectorSubVal_combination(float *vec,float val,int N)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<N)
	{
		vec[offset]-=val;
		//vec[offset]=floor(vec[offset]);
	}
}


#define warpBlockSize 512

__global__ void cu_PAWarping_float_shared_transpose(float *warp_tabel_t,float *triangle_indexTabel_t,float *pts_pos,int ptsNum,float *inputImg1,float *inputImg2,float *inputImg3,int width,int height,float *outputImg1,float *outputImg2,float *outputImg3,int outputWidth,int outputHeight,float *maskTable,float *compactTexture)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>=outputHeight*outputWidth)
	{
		return;
	}

	/*int localID=threadIdx.x;
	int beginId=blockIdx.x*blockDim.x;
	__shared__ float localIndexTabel[warpBlockSize][3];
	localIndexTabel[localID][0]=triangle_indexTabel[3*(beginId+localID)];
	localIndexTabel[localID][1]=triangle_indexTabel[3*(beginId+localID)+1];
	localIndexTabel[localID][2]=triangle_indexTabel[3*(beginId+localID)+2];

	__syncthreads();*/
	//if (offset<outputWidth*outputHeight)
	//{
	int totalNum=outputHeight*outputWidth;
		int ind1=triangle_indexTabel_t[offset];
		if (ind1==-1)
		{
			return;
		}
		//if (ind1!=-1)
		//{
		float output1,output2,output3;
			int ind2=triangle_indexTabel_t[totalNum+offset];
			int ind3=triangle_indexTabel_t[2*totalNum+offset];
			float x=0;
			float y=0;

			float w1,w2,w3;
			w1=warp_tabel_t[offset];
			w2=warp_tabel_t[totalNum+offset];
			w3=warp_tabel_t[totalNum*2+offset];
			x=w1*pts_pos[ind1]+
				w2*pts_pos[ind2]+
				w3*pts_pos[ind3];
			y=w1*pts_pos[ind1+ptsNum]+
				w2*pts_pos[ind2+ptsNum]+
				w3*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImg1[(intY*width+intX)]+ratioX*
				inputImg1[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg1[((intY+1)*width+intX)]+ratioX*
				inputImg1[((intY+1)*width+(intX+1))];

			outputImg1[offset]=((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on

			tpx1=(1-ratioX)*inputImg2[(intY*width+intX)]+ratioX*
				inputImg2[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg2[((intY+1)*width+intX)]+ratioX*
				inputImg2[((intY+1)*width+(intX+1))];
			outputImg2[offset]=((1-ratioY)*tpx1+ratioY*tpx2);

			int ccind=maskTable[offset];
			if (ccind!=-1)
			{
				tpx1=(1-ratioX)*inputImg3[(intY*width+intX)]+ratioX*
					inputImg3[(intY*width+(intX+1))];
				tpx2=(1-ratioX)*inputImg3[((intY+1)*width+intX)]+ratioX*
					inputImg3[((intY+1)*width+(intX+1))];

				compactTexture[ccind]=((1-ratioY)*tpx1+ratioY*tpx2);
			}
}

__global__ void cu_PAWarping_float_shared(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg1,float *inputImg2,float *inputImg3,int width,int height,float *outputImg1,float *outputImg2,float *outputImg3,int outputWidth,int outputHeight)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset>=outputHeight*outputWidth)
	{
		return;
	}

	/*int localID=threadIdx.x;
	int beginId=blockIdx.x*blockDim.x;
	__shared__ float localIndexTabel[warpBlockSize][3];
	localIndexTabel[localID][0]=triangle_indexTabel[3*(beginId+localID)];
	localIndexTabel[localID][1]=triangle_indexTabel[3*(beginId+localID)+1];
	localIndexTabel[localID][2]=triangle_indexTabel[3*(beginId+localID)+2];

	__syncthreads();*/
	//if (offset<outputWidth*outputHeight)
	//{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1==-1)
		{
			return;
		}
		//if (ind1!=-1)
		//{

		float output1,output2,output3;
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImg1[(intY*width+intX)]+ratioX*
				inputImg1[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg1[((intY+1)*width+intX)]+ratioX*
				inputImg1[((intY+1)*width+(intX+1))];

			outputImg1[offset]=((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on

			tpx1=(1-ratioX)*inputImg2[(intY*width+intX)]+ratioX*
				inputImg2[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg2[((intY+1)*width+intX)]+ratioX*
				inputImg2[((intY+1)*width+(intX+1))];
			outputImg2[offset]=((1-ratioY)*tpx1+ratioY*tpx2);

			tpx1=(1-ratioX)*inputImg3[(intY*width+intX)]+ratioX*
				inputImg3[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg3[((intY+1)*width+intX)]+ratioX*
				inputImg3[((intY+1)*width+(intX+1))];
			outputImg3[offset]=((1-ratioY)*tpx1+ratioY*tpx2);
			//outputImg[offset]=inputImg[offset];
		//}
	/*	else
		{
			outputImg[offset]=0;
		}*/

	//}
}

__global__ void cu_PAWarping_combination(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<outputWidth*outputHeight)  //use tempary varible
	{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1!=-1)
		{
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImg[(intY*width+intX)]+ratioX*
				inputImg[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg[((intY+1)*width+intX)]+ratioX*
				inputImg[((intY+1)*width+(intX+1))];

			outputImg[offset]=(int)((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on
			//outputImg[offset]=inputImg[offset];
		}
		else
		{
			outputImg[offset]=0;
		}

	}
}

__global__ void setCompactTexture_combination(float *fullTexture,float *compactTexture,float *MaskTabel,int width,int height) //try use shared memory, and do multiple positions
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<width*height)
	{
		int ind=MaskTabel[offset];
		if (ind!=-1)
		{
			compactTexture[ind]=fullTexture[offset];
		}
	}
}

__global__ void cu_PAWarping_float_combination(float *warp_tabel,float *triangle_indexTabel,float *pts_pos,int ptsNum,float *inputImg,int width,int height,float *outputImg,int outputWidth,int outputHeight)
{
	/*int x1=blockIdx.x;
	int y1=blockIdx.y;*/
	int offset=threadIdx.x+blockIdx.x*blockDim.x;


	if (offset<outputWidth*outputHeight)
	{
		int ind1=triangle_indexTabel[3*offset];
		if (ind1!=-1)
		{
			int ind2=triangle_indexTabel[3*offset+1];
			int ind3=triangle_indexTabel[3*offset+2];
			float x=0;
			float y=0;
			x=warp_tabel[offset*3+0]*pts_pos[ind1]+
				warp_tabel[offset*3+1]*pts_pos[ind2]+
				warp_tabel[offset*3+2]*pts_pos[ind3];
			y=warp_tabel[offset*3+0]*pts_pos[ind1+ptsNum]+
				warp_tabel[offset*3+1]*pts_pos[ind2+ptsNum]+
				warp_tabel[offset*3+2]*pts_pos[ind3+ptsNum]; 

			int intX,intY;
			float ratioX,ratioY;
			float tpx1,tpx2;

			intX=(int)x;
			intY=(int)y;
			ratioX=(x-intX);
			ratioY=y-intY;
			/*	tpx1=(1-ratioX)*m_img.at<uchar>(intY,intX)+ratioX*
			m_img.at<uchar>(intY,intX+1);
			tpx2=(1-ratioX)*m_img.at<uchar>(intY+1,intX)+ratioX*
			m_img.at<uchar>(intY+1,intX+1);*/
			tpx1=(1-ratioX)*inputImg[(intY*width+intX)]+ratioX*
				inputImg[(intY*width+(intX+1))];
			tpx2=(1-ratioX)*inputImg[((intY+1)*width+intX)]+ratioX*
				inputImg[((intY+1)*width+(intX+1))];

			outputImg[offset]=((1-ratioY)*tpx1+ratioY*tpx2);///////int can be deleted later on
			//outputImg[offset]=inputImg[offset];
		}
		else
		{
			outputImg[offset]=0;
		}

	}
}

__global__ void calculateDifference_withPrior(int width,int height,float *currentTexture,float *currentTemplate,int pix_num,float *currentShape,float *currentIndex,float *targetShape,int featureNum,int ptsNum,float w1,float w2,float *output,float *conv,int shape_dim,float*currentWeight,float *priorMean,float *priorDif)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<pix_num)
	{
		//return;
		output[offset]=w1*(currentTemplate[offset]-currentTexture[offset]); //thrust
		return;
	}
	//return;
	if (offset>=pix_num&&offset<pix_num+featureNum)  //split
	{
		int currentId=currentIndex[offset-pix_num];
		float cdifx=(currentShape[currentId]-targetShape[offset-pix_num]);
		float cdify=(currentShape[currentId+ptsNum]-targetShape[offset-pix_num+featureNum]);

		//sigma*E
		//int cx=currentShape[currentId];int cy=currentShape[currentId+ptsNum];
		int totalDim=(offset-pix_num)*featureNum*4+(offset-pix_num)*2;
		//int pos[2];
		//pos[0]=

		int cind=2*(offset-pix_num)+pix_num;
		float ot1=w2*(conv[totalDim]*cdifx+conv[totalDim+1]*cdify);
		float ot2=w2*(conv[totalDim+featureNum*2]*cdifx+conv[totalDim+1+featureNum*2]*cdify);

		output[cind]=ot1;
		output[cind+1]=ot2;
		return;
	//	output[cind]=targetShape[offset-pix_num];
	//	output[cind+1]=targetShape[offset-pix_num+featureNum];

		//output[offset]=currentShape[currentId];
		//output[offset+featureNum]=currentShape[currentId+ptsNum];
		/*output[offset]=w2*(tex2D(texture_preConv,cx,totalDim)*cdifx+tex2D(texture_preConv,cx+width,totalDim)*cdify);
		output[offset+featureNum]=w2*(tex2D(texture_preConv,cx+width*2,totalDim)*cdifx+tex2D(texture_preConv,cx+width*3,totalDim)*cdify);*/
	}
	if (offset>=pix_num+featureNum&&offset<pix_num+featureNum+shape_dim)
	{
		int cind=offset-pix_num-featureNum;
		priorDif[cind]=currentWeight[cind]-priorMean[cind];
		return;
	}

}

//currentShape can be replaced by textureShape
__global__ void calculateDifference(int width,int height,float *currentTexture,float *currentTemplate,int pix_num,float *currentShape,float *currentIndex,float *targetShape,int featureNum,int ptsNum,float w1,float w2,float *output,float *conv)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<pix_num)
	{
		output[offset]=w1*(currentTemplate[offset]-currentTexture[offset]);
	}
	if (offset>=pix_num&&offset<pix_num+featureNum)
	{
		int currentId=currentIndex[offset-pix_num];
		float cdifx=(currentShape[currentId]-targetShape[offset-pix_num]);
		float cdify=(currentShape[currentId+ptsNum]-targetShape[offset-pix_num+featureNum]);

		//sigma*E
		int cx=currentShape[currentId];int cy=currentShape[currentId+ptsNum];
		int totalDim=(offset-pix_num)*featureNum*4+(offset-pix_num)*2;
		int pos[2];
		//pos[0]=

		int cind=2*(offset-pix_num)+pix_num;
		output[cind]=w2*(conv[totalDim]*cdifx+conv[totalDim+1]*cdify);
		output[cind+1]=w2*(conv[totalDim+featureNum*2]*cdifx+conv[totalDim+1+featureNum*2]*cdify);

		//output[offset]=currentShape[currentId];
		//output[offset+featureNum]=currentShape[currentId+ptsNum];
		/*output[offset]=w2*(tex2D(texture_preConv,cx,totalDim)*cdifx+tex2D(texture_preConv,cx+width,totalDim)*cdify);
		output[offset+featureNum]=w2*(tex2D(texture_preConv,cx+width*2,totalDim)*cdifx+tex2D(texture_preConv,cx+width*3,totalDim)*cdify);*/
	}


	//no need to setup cu_conv, it is pre-calculated 
	/*if (offset>=pix_num+featureNum&&offset<pix_num+featureNum*2)
	{
		int absID=offset-pix_num-featureNum;
		int currentId=currentIndex[absID];
		float cdifx=(currentShape[currentId]-targetShape[offset-pix_num]);
		float cdify=(currentShape[currentId+ptsNum]-targetShape[offset-pix_num+featureNum]);
		int cx=currentShape[currentId];int cy=currentShape[currentId+ptsNum];
		int totalDim=width*height*4*currentId+cy*width*4;
		int totalDim_conv=absID*featureNum*4+absID*2;

		conv[totalDim_conv]=tex2D(texture_preConv,cx,totalDim);
		conv[totalDim_conv+1]=tex2D(texture_preConv,cx+width,totalDim);
		conv[totalDim_conv+featureNum*2]=tex2D(texture_preConv,cx+width*2,totalDim);
		conv[totalDim_conv+1*featureNum*2]=tex2D(texture_preConv,cx+width*3,totalDim);
	}*/
}

__global__ void copyJacobianWithWeight(float *jacobians,int allDim,int pix_num,int visibleNum,float w1,float w2,float *output)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<(pix_num+visibleNum)*allDim)
	{
		if (offset<pix_num)
		{
			output[offset]=sqrt(w1)*jacobians[offset];
		}
		if (offset>=pix_num)
		{
			output[offset]=sqrt(w2)*jacobians[offset];
		}
	}
}

//B=w1*A+w2*B;
__global__ void vectorAdd(float *A,float *B,float weight1,float weight2,int totalDim)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<totalDim)
	{
		B[offset]=weight1*A[offset]+weight2*B[offset];
	}
}

extern "C" int iterate_combination(int width,int height,int currentFrame,int startFrame,float &resultTheta,float *finalShape, int& finalShapePtsNum, bool isAAMOnly,bool showNN,bool updateOnly)
{
	// Peihong	This is where the code crashes
	//cout<<"in the cuda AAM\n";
	//return;
	int status=0;
	AAM_Search_RealGlobal_CUDA *data=trackEngine.AAM_ENGINE;

	if (data->isAdptive&&data->currentFrameID==data->blockNum)
	{
		//cout<<"update model\n";
		//GTB("updateModel");
		//update texture model
		CUDA_CALL(cudaMemcpy(data->host_blockTextureData, data->cu_blockTextureData, data->pix_num*data->blockNum*sizeof(float), cudaMemcpyDeviceToHost ));


		updateModelCPU(data->host_blockTextureData,data->blockNum,data->newMeanAndTVec);

		//data->exp->updateModel(data->host_blockTextureData,data->blockNum);
		updateTextureModel_cuda(data->newMeanAndTVec);
		data->currentFrameID=0;

		//return status;//no need to return now. still process the frame
		

	}

	//float *cu_inputImg=data->cu_inputImg;
	int allDim=data->s_dim+data->t_dim+4;
	int MaxIterNum=30;

	if (isAAMOnly)
	{
		MaxIterNum=30;
	}
	/*if (smoothWeight>0)
	{
		MaxIterNum=35;
	}*/
	
	int incx,incy;
	incx=incy=1;
	//float *cu_currentTexture;
//	float *cu_errorImage;
	float result;
	float textureScale;
	float tmp_scale,tex_scale;

	//device_vector<int> d_ipiv(allDim);
	//int *ipiv=raw_pointer_cast(&d_ipiv[0]);

	//float *lastParameters;
	//malloc Jacobians and Hessian
	//suppose parameters are already initialized

	//calculate the gradient of input image
	//By openCV first
	
	int full_pix_num=data->t_width*data->t_height;


	int times=0;

	float alpha,beta;
	float difference;

	float errorSum,lastError;

	
	dim3 grid((data->t_width*data->t_height)/blockDIM_combination+1,(data->s_dim+data->t_dim+4)/blockDIM_combination+1,1);
	dim3 threads(blockDIMX_combination,blockDIM_combination,1);
	//cout<<"lalala\n";
	if (isAAMOnly)
	{
		trackEngine.RTWeight=trackEngine.RTWeight_backup*0;
		trackEngine.priorWeight=trackEngine.priorWeight_backup*0;
		trackEngine.RTWeight=trackEngine.priorWeight=0;
		//trackEngine.RTWeight=0.0005;
		//trackEngine.priorWeight=0.001;
	}
	else
	{
		trackEngine.RTWeight=trackEngine.RTWeight_backup;
		trackEngine.priorWeight=trackEngine.priorWeight_backup;
	}

	float minV=0.0005;

	if (trackEngine.AAM_ENGINE->showSingleStep)
	{
		//trackEngine.AAMWeight=0.6;
		//trackEngine.RTWeight=0.005;
		//trackEngine.priorWeight=0.005;

		//minV=0.001;
		////trackEngine.priorWeight=0.005;
		//MaxIterNum=20;

		//trackEngine.AAMWeight=1;
		//trackEngine.RTWeight=0.05;
		//trackEngine.priorWeight=0.005;

		//minV=0.01;
		////trackEngine.priorWeight=0.005;
		//MaxIterNum=30;
	}



	//cudaEvent_t start, stop;
	//CUDA_CALL(cudaEventCreate(&start));
	//CUDA_CALL(cudaEventCreate(&stop));
	//CUDA_CALL(cudaEventRecord( start, 0 ));
	abs_diff<float>        binary_op2;

	while(1)
	{

		//show current image
		if (data->showSingleStep)
		//if(1)
		{
			cout<<"time "<<times<<endl;
			float *parameters=new float[MPD_combination];
			CUDA_CALL(cudaMemcpy(parameters, data->cu_parameters, MPD_combination*sizeof(float), cudaMemcpyDeviceToHost ));
			checkIterationResult(parameters,data->ptsNum,data->s_dim,data->t_dim,false);
			delete []parameters;

		}

		/*cudaEvent_t start, stop;
		CUDA_CALL(cudaEventCreate(&start));
		CUDA_CALL(cudaEventCreate(&stop));
		CUDA_CALL(cudaEventRecord( start, 0 ));

		for (int tesdt=0;tesdt<30;tesdt++)
		{*/
		
		//copy parameters to constant memory
		//CUDA_CALL(cudaMemcpyToSymbol(cu_currentParameters,data->cu_parameters,MPD_combination*sizeof(float),0,cudaMemcpyDeviceToDevice));

		//!!need to assign mean value to current shape
		/*CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->s_dim,data->cu_parameters,incx,data->cu_s_weight,incy));
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->t_dim,data->cu_parameters+data->s_dim,incx,data->cu_t_weight,incy));
	
		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->ptsNum*2,data->cu_s_mean,incx,data->cu_currentLocalShape,incy));
		cu_getCurrentShape_combination(data->cu_currentLocalShape,data->cu_s_weight,data->cu_s_vec,data->s_dim,data->t_dim,data->ptsNum,data->cu_parameters,data->cu_currentShape);


		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->pix_num,data->cu_t_mean,incx,data->cu_currentTemplate,incy));	
		cu_getCurrentTexture_combination(data->cu_currentTemplate,data->cu_t_weight,data->cu_t_vec,data->t_dim,data->pix_num);
	*/		
		
	/*	GTB("START");
		for(int i=0;i<60;i++)
		{*/

		CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->pix_num+data->ptsNum*2,data->cu_sMean_T_mean,incx,data->cu_curLocalShape_curTemplate,incy));
		//continue;
	
		cu_getCurrentShape_combination(data->cu_currentLocalShape,data->cu_parameters,data->cu_s_vec,data->s_dim,data->t_dim,data->ptsNum,data->cu_parameters,data->cu_currentShape,data->cu_lastShape);


		/*if (trackEngine.AAM_ENGINE->showSingleStep)
		{
			float *cpuPara=new float[data->ptsNum*2];
			CUDA_CALL(cudaMemcpy(cpuPara, data->cu_currentShape, data->ptsNum*2*sizeof(float), cudaMemcpyDeviceToHost ));
			
			cout<<"current shape:\n";
			for (int i=0;i<10*2;i++)
			{
				cout<<cpuPara[i]<<" ";
			}
			cout<<endl;

			
		}*/
		

		
		if (showNN)
		{
			CUDA_CALL(cudaMemcpy(finalShape, data->cu_currentShape, data->ptsNum*2*sizeof(float), cudaMemcpyDeviceToHost ));
			break;
		}
		//check shape difference and see whether to return
		
		

		/*vectorAdd<<<(data->ptsNum*2)/16,16>>>(data->cu_lastShape,data->cu_currentShape,1,
			-1,data->ptsNum*2);*/
	
		

		cu_getCurrentTexture_combination(data->cu_currentTemplate,data->cu_parameters+data->s_dim,data->cu_t_vec,data->t_dim,data->pix_num);
			//continue;
		
		//normalize and devide
		//normalize
		float sum;

		//CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_currentTemplate,incx,data->cu_t_mean,incy,&sum));
		//sum=1.0f/sum;
		//cout<<"sum: "<<sum<<endl;
		//CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,data->pix_num,&sum,data->cu_currentTemplate,incx));
		
		//////////////////template check point//////////////////////
		
		//cu_PAWarping_float_shared<<<data->t_width*data->t_height/warpBlockSize+1,warpBlockSize>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,
		//	data->cu_gradientX,data->cu_gradientY,data->cu_inputImg,width,height,
		//	data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_fullCurrentTexture,data->t_width,data->t_height);
		////cu_PAWarping_TEX<<<MAXIMUMPOINTDIM/256,256>>>(data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_currentShape,data->ptsNum,data->cu_inputImg,width,height,data->cu_fullCurrentTexture,data->t_width,data->t_height);
		//
		//setCompactTexture_combination<<<(data->t_width*data->t_height+128)/128,128>>>(data->cu_fullCurrentTexture,data->cu_currentTexture,
		//	data->cu_MaskTabel,data->t_width,data->t_height);

		cu_PAWarping_float_shared_transpose<<<data->t_width*data->t_height/512+1,512>>>(data->cu_warp_tabel_transpose,data->cu_triangle_indexTabel_transpose,data->cu_currentShape,data->ptsNum,
			data->cu_gradientX,data->cu_gradientY,data->cu_inputImg,width,height,
			data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_fullCurrentTexture,data->t_width,data->t_height,data->cu_MaskTabel,data->cu_currentTexture);



		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_currentTexture,incx,data->cu_allOnesForImg,incy,&sum));
		sum/=data->pix_num;
		vectorSubVal_combination<<<(data->pix_num+128)/128,128>>>(data->cu_currentTexture,sum,data->pix_num);
		
		//texture normalize
		CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_t_mean,incx,data->cu_currentTexture,incy,&result));
		
		tmp_scale=1.0f/result;
		tex_scale=tmp_scale;
		//CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,data->pix_num,&tmp_scale,data->cu_currentTexture,incx));
		CUBLAS_CALL(cublasSscal_v2(data->blas_handle_,data->pix_num+full_pix_num*2,&tex_scale,data->cu_currentTexture,incx));
		//////////////////texture check point//////////////////////	
		//GTE("START");

	//	GTB("START");
		if (trackEngine.priorWeight==0&&trackEngine.RTWeight==0)
		{
			//cout<<"fast difference\n";
			thrust::minus<float> op;
			thrust::transform(data->vec_curShape_curTemplate.begin()+data->ptsNum*2,
				data->vec_curShape_curTemplate.end(),data->vec_curTexture.begin(),data->vec_errorImage.begin(),
				op);
		}
		else
		{
			calculateDifference_withPrior<<<(data->pix_num+MAX_LABEL_NUMBER+data->s_dim)/128+1,128>>>(width,height,data->cu_currentTexture,data->cu_currentTemplate,data->pix_num,
				data->cu_currentShape,trackEngine.cu_absVisibleIndex,trackEngine.cu_detectedFeatureLocations,trackEngine.visibleNum,
				data->ptsNum,trackEngine.AAMWeight,trackEngine.RTWeight,data->cu_errorImage,trackEngine.cu_conv,data->s_dim,
				data->cu_parameters,trackEngine.cuPriorMean,trackEngine.cu_priorDifference);
		}




		//if (times>15)
		{
			//float sumSquare=thrust::inner_product(data->Vec_lastShape.begin(),data->Vec_lastShape.end(),data->Vec_currentShape.begin(),0,thrust::plus<float>(),binary_op2);
			CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_errorImage,incx,data->cu_errorImage,incy,&errorSum)); //thrust
			//cout<<"times: "<<times<<" error: "<<errorSum<<endl;
			if((errorSum<0.1&&abs(lastError-errorSum)<0.00001)||times>MaxIterNum)
			{

				// copy the results back to host
				CUDA_CALL(cudaMemcpy(finalShape, data->cu_currentShape, data->ptsNum*2*sizeof(float), cudaMemcpyDeviceToHost ));

				saveIterationResult(finalShape,data->ptsNum);
				finalShapePtsNum = data->ptsNum;

				if (data->isAdptive)
				{

					//cout<<errorSum<<endl;
					if (errorSum<0.6)
					{
						CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->pix_num,data->cu_currentTexture,1,data->cu_blockTextureData+data->currentFrameID*data->pix_num,1));
						data->currentFrameID++;
					}
					else
					{
						cout<<"bad data\n";
						//CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->pix_num,data->cu_t_mean,1,data->cu_blockTextureData+data->currentFrameID*data->pix_num,1));
					}
					
					if (data->currentFrameID==data->blockNum)
					{
						//data->currentFrameID=0;
						status=1;
					}
				}
				float *parameters=new float[4];
				CUDA_CALL(cudaMemcpy(parameters, data->cu_parameters+data->s_dim+data->t_dim, 4*sizeof(float), cudaMemcpyDeviceToHost ));
				resultTheta=parameters[0];
				
				cout<<"times: "<<times<<endl;
				break;
			}

		
		}


////GTE("START");
//		if (trackEngine.RTWeight>0||trackEngine.priorWeight>0)
		/*GTB("START");
		for (int ii=0;ii<60;ii++)
		{*/
		
			
			
	/*	}
		GTE("START");
*/

			//if (times>6)
			//{
			//	CUBLAS_CALL(cublasSdot_v2(data->blas_handle_,data->pix_num,data->cu_errorImage,incx,data->cu_errorImage,incy,&errorSum)); //thrust
			//	errorSum/=trackEngine.AAMWeight;
			//}
			//else
			//{
			//	errorSum=1;
			//}

		
		/*}
		GTE("START");*/

		////////////////////////////NEW ADDED, NNED TO CHECK////////////////////////////////////////////
		//worked version without constant memory fastest
		/*GTB("START");
		for (int ii=0;ii<60;ii++)
		{*/
	
	

		if (trackEngine.RTWeight>0)
		{
			/*getJacobians_RT_combination<<<(data->t_width*data->t_height)/32+1,32>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
				data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
				data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum,data->pix_num,trackEngine.AAMWeight,trackEngine.RTWeight,trackEngine.cu_absVisibleIndex,trackEngine.visibleNum);*/

			//getJacobians_RT_combination_sharedMemory<<<(data->t_width*data->t_height)/threadPerBlock_Jacobians+1,threadPerBlock_Jacobians>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
			//	data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
			//	data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum,data->pix_num,trackEngine.AAMWeight,trackEngine.RTWeight,trackEngine.cu_absVisibleIndex,trackEngine.visibleNum,data->cu_Jacobian_transpose);

		/*for (int ii=0;ii<200;ii++)
		{

			GTB("START");*/
			getJacobians_RT_combination_transpose<<<(data->t_width*data->t_height)/threadPerBlock_Jacobians+1,threadPerBlock_Jacobians>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
				data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians_transpose,data->cu_t_vec,data->cu_parameters,
				data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum,data->pix_num,trackEngine.AAMWeight,trackEngine.RTWeight,trackEngine.cu_absVisibleIndex,trackEngine.visibleNum,data->cu_Jacobian_transpose);
			/*GTE("START");
		}
			
			gCodeTimer.printTimeTree();
			double time1 = total_fps;
			cout<<"used time per iteration: "<<time1<<endl;
			break;*/
			/*	if (trackEngine.AAM_ENGINE->showSingleStep)
			{
			float *cpuPara=new float[trackEngine.visibleNum*2*allDim];
			CUDA_CALL(cudaMemcpy(cpuPara, data->cu_RT_Jacobian_transpose, (trackEngine.visibleNum*2*allDim)*sizeof(float), cudaMemcpyDeviceToHost ));

			cout<<"detection jacobian:\n";

			ofstream out("detJ_725.txt",ios::out);
			for (int i=0;i<trackEngine.visibleNum*2*allDim;i++)
			{

			out<<cpuPara[i]<<" ";


			}
			out.close();


			}*/
				//	float *cpuPara=new float[trackEngine.visibleNum*2*allDim];
				//CUDA_CALL(cudaMemcpy(cpuPara, data->cu_RT_Jacobian_transpose, (trackEngine.visibleNum*2*allDim)*sizeof(float), cudaMemcpyDeviceToHost ));
				//ofstream out("D:\\Fuhao\\cpu gpu validation\\J_RT_new.txt",ios::out);
				///*for (int i=0;i<trackEngine.visibleNum*2;i++)
				//{
				//	for (int j=0;j<allDim;j++)
				//	{
				//		out<<cpuPara[j*trackEngine.visibleNum*2+i]<<" ";
				//	}
				//	out<<endl;
				//}
				//out.close();*/

				//for (int i=0;i<allDim;i++)
				//{
				//	for (int j=0;j<trackEngine.visibleNum*2;j++)
				//	{
				//		out<<cpuPara[i*trackEngine.visibleNum*2+j]<<" ";
				//	}
				//	out<<endl;
				//}
				//out.close();
				//cout<<"done!\n";
		}
		else
		{
			/*getJacobians_combination<<<(data->t_width*data->t_height)/32+1,32>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
				data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
				data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum);*/

			/*getJacobians_combination_sharedMemory<<<(data->t_width*data->t_height)/threadPerBlock_Jacobians+1,threadPerBlock_Jacobians>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobian_transpose,data->t_width,
				data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
				data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum,data->pix_num);*/

			getJacobians_combination_shapeTranspose<<<(data->t_width*data->t_height)/threadPerBlock_Jacobians+1,threadPerBlock_Jacobians>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobian_transpose,data->t_width,
				data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians_transpose,data->cu_t_vec,data->cu_parameters,
				data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum,data->pix_num);
		}

		

	/*	}
		GTE("START");*/
		/////39-40 21
		/*if (times>20)
		{
			break;
		}
		times++;
		continue;*/

		//float *cpuJacobian=new float[(data->t_width*data->t_height+MAX_LABEL_NUMBER*2)*(data->s_dim+data->t_dim+4)];
		//CUDA_CALL(cudaMemcpy(cpuJacobian, data->cu_Jacobians, (data->t_width*data->t_height+MAX_LABEL_NUMBER*2)*(data->s_dim+data->t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));

		//ofstream out_J("D:\\Fuhao\\cpu gpu validation\\dj_GPU.txt",ios::out);
		//for (int i=data->pix_num;i<data->pix_num+trackEngine.visibleNum*2;i++)
		//{
		//	for (int j=0;j<data->s_dim+data->t_dim+4;j++)
		//	{
		//		out_J<<cpuJacobian[i*(data->s_dim+data->t_dim+4)+j]<<" ";
		//	}
		//	out_J<<endl;
		//}
		//out_J.close();

		////////////////////checked-0321/////////////////////////////////
		

		//Jacobians with texture memory, not correct and not as fast as the original one
	/*	getJacobians_TEX<<<(data->t_width*data->t_height)/32+1,32>>>(data->cu_WarpedGradientX,data->cu_WarpedGradientY,data->cu_Jacobians,data->t_width,
			data->t_height,data->s_dim,data->t_dim,data->cu_warp_tabel,data->cu_triangle_indexTabel,data->cu_shapeJacobians,data->cu_t_vec,data->cu_parameters,
			data->cu_fowardIndexTabel,data->cu_currentLocalShape,data->ptsNum);*/
		
		
			
		//////////////////////Jacobians checked///////////////////////////

	

		//MatrixMVector(data->cu_Jacobians,data->pix_num,allDim,data->cu_errorImage,data->cu_deltaParameters,data);
	//	compactError2Full<<<(data->t_width*data->t_height+128)/128,128>>>(data->cu_errorImage,data->cu_fullErrorImage,data->cu_MaskTabel,data->t_width,data->t_height);

		/*GTB("START");
		for (int ii=0;ii<60;ii++)
		{*/


	/*	GTB("START");
		for (int ii=0;ii<60;ii++)
		{*/
		alpha=1;beta=0;
		if (trackEngine.RTWeight>0||trackEngine.priorWeight>0)
		{
			//if (trackEngine.AAMWeight>0)
			//{
			//at least initialize it to 0
			CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_T,data->pix_num,allDim,&alpha,data->cu_Jacobian_transpose,data->pix_num,data->cu_errorImage,incx,&beta,
				data->cu_deltaParameters,incy));


		/*	if (trackEngine.AAM_ENGINE->showSingleStep)
			{
				float *cpuPara=new float[data->s_dim+data->t_dim+4];
				CUDA_CALL(cudaMemcpy(cpuPara, data->cu_deltaParameters, (data->s_dim+data->t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));

				cout<<"initial JE:\n";
				for (int i=0;i<data->s_dim+data->t_dim+4;i++)
				{

					cout<<cpuPara[i]<<" ";


				}
				cout<<endl;


			}*/
				
			//}

			if (trackEngine.RTWeight>0)
			{
				////////////////////////////NEW ADDED, NNED TO CHECK////////////////////////////////////////////
				//cu_errorImage and featureNum. Also, weight need to be added in the error image.
				/*CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_N,allDim,data->pix_num+trackEngine.visibleNum*2,&alpha,data->cu_Jacobians,allDim,data->cu_errorImage,incx,&beta,
					data->cu_deltaParameters,incy));*/

				beta=1;

			

				/*CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_N,allDim,trackEngine.visibleNum*2,&alpha,data->cu_RT_Jacobian_transpose,
					allDim,data->cu_errorImage+data->pix_num,incx,&beta,data->cu_deltaParameters,incy));*/
			
				/*for (int ii=0;ii<200;ii++)
				{

					GTB("START")*/;

				CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_T,trackEngine.visibleNum*2,allDim,&alpha,data->cu_RT_Jacobian_transpose,
					trackEngine.visibleNum*2,data->cu_errorImage+data->pix_num,incx,&beta,data->cu_deltaParameters,incy));

				//GTE("START");
				//}

				//gCodeTimer.printTimeTree();
				//double time1 = total_fps;
				//cout<<"used time per iteration: "<<time1<<endl;
				//break;


				beta=0;



				/*if (trackEngine.AAM_ENGINE->showSingleStep)
				{
					cout<<"visible Num: "<<trackEngine.visibleNum<<"\n";

					float *ptsDiff=new float[trackEngine.visibleNum*2];
					CUDA_CALL(cudaMemcpy(ptsDiff,data->cu_errorImage+data->pix_num, trackEngine.visibleNum*2*sizeof(float), cudaMemcpyDeviceToHost ));
					cout<<"detection diff:\n";
					for (int i=0;i<trackEngine.visibleNum*2;i++)
					{
						cout<<ptsDiff[i]<<" ";
					}
					cout<<endl;

					float *cpuPara=new float[data->s_dim+data->t_dim+4];
					CUDA_CALL(cudaMemcpy(cpuPara, data->cu_deltaParameters, (data->s_dim+data->t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));

					cout<<"detection JE:\n";
					for (int i=0;i<data->s_dim+data->t_dim+4;i++)
					{

						cout<<cpuPara[i]<<" ";


					}
					cout<<endl;


				}*/

			/*		float *cpuPara=new float[data->s_dim+data->t_dim+4];
		CUDA_CALL(cudaMemcpy(cpuPara, data->cu_deltaParameters, (data->s_dim+data->t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_J1("D:\\Fuhao\\cpu gpu validation\\je_new.txt",ios::out);
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
		out_J1<<cpuPara[i]<<" ";
		}
		out_J1<<endl;
		out_J1.close();*/

			}

			if (trackEngine.priorWeight>0)
			{

			
				
				/*CUBLAS_CALL( cublasSgemv_v2(data->blas_handle_, CUBLAS_OP_T, 
					data->s_dim+data->t_dim+4,data->s_dim,
					&alpha, trackEngine.cuHessianPrior, data->s_dim+data->t_dim+4, 
					trackEngine.cu_priorDifference,1,
					&beta,trackEngine.cu_priorDifference, 1) ); 

				CUBLAS_CALL(cublasSaxpy_v2(data->blas_handle_,allDim,&(trackEngine.priorWeight),trackEngine.cu_priorDifference,1,data->cu_deltaParameters,1));*/
				beta=1;
				CUBLAS_CALL( cublasSgemv_v2(data->blas_handle_, CUBLAS_OP_T, 
					data->s_dim+data->t_dim+4,data->s_dim,
					&(trackEngine.priorWeight), trackEngine.cuHessianPrior, data->s_dim+data->t_dim+4, 
					trackEngine.cu_priorDifference,1,
					&beta,data->cu_deltaParameters, 1) ); 
				/*CUBLAS_CALL( cublasSgemv_v2(data->blas_handle_, CUBLAS_OP_T, 
					data->s_dim,data->s_dim,
					&(trackEngine.priorWeight), trackEngine.cuHessianPrior, data->s_dim+data->t_dim+4, 
					trackEngine.cu_priorDifference,1,
					&beta,data->cu_deltaParameters, 1) ); */
				beta=0;
				

				/*if (trackEngine.AAM_ENGINE->showSingleStep)
				{
					float *cpuPara=new float[data->s_dim+data->t_dim+4];
					CUDA_CALL(cudaMemcpy(cpuPara, data->cu_deltaParameters, (data->s_dim+data->t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));

					cout<<"detection+prior JE:\n";
					for (int i=0;i<data->s_dim+data->t_dim+4;i++)
					{

						cout<<cpuPara[i]<<" ";


					}
					cout<<endl;


				}*/
				
			}
		}		
		else
		{
			/*CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_N,allDim,data->pix_num,&alpha,data->cu_Jacobians,allDim,data->cu_errorImage,incx,&beta,
			data->cu_deltaParameters,incy));*/

			CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_T,data->pix_num,allDim,&alpha,data->cu_Jacobian_transpose,data->pix_num,data->cu_errorImage,incx,&beta,
				data->cu_deltaParameters,incy));
		}


		
	/*		}
		GTE("START");
		gCodeTimer.printTimeTree();
		double time1 = total_fps;
		cout<<"used time per iteration: "<<time1<<endl;
		continue;*/
	
		/*	float *cpuPara=new float[data->s_dim+data->t_dim+4];
		CUDA_CALL(cudaMemcpy(cpuPara, data->cu_deltaParameters, (data->s_dim+data->t_dim+4)*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_J1("D:\\Fuhao\\cpu gpu validation\\dp_GPU.txt",ios::out);
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
		out_J1<<cpuPara[i]<<" ";
		}
		out_J1<<endl;
		out_J1.close();*/
		
		//break;
		///////////////////////J'E checked- 0321////////////////////////////////
		//45/21ms
	/*	if (times>20)
		{
			break;
		}
		times++;
		continue;*/

		

	/*	GTB("START");
		for (int ii=0;ii<60;ii++)
		{*/
		//getHessian.
		if (trackEngine.RTWeight>0||trackEngine.priorWeight>0)
		{
			CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
			allDim, allDim, data->pix_num, 
			&(trackEngine.AAMWeight), data->cu_Jacobian_transpose, data->pix_num, 
			data->cu_Jacobian_transpose, data->pix_num,
			&beta,
			data->cu_Hessian, allDim) );


			//using syemetric one, should be faster
			//at least initialize it to 0
			/*CUBLAS_CALL( cublasSsyrk_v2(data->blas_handle_,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T,
				allDim,data->pix_num,&(trackEngine.AAMWeight),data->cu_Jacobian_transpose,data->pix_num,&beta,
				data->cu_Hessian,allDim));*/
			if (trackEngine.RTWeight>0)
			{
				//hessian AAM first
				/*CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
				allDim, allDim, data->pix_num, 
				&alpha, data->cu_Jacobians, allDim, 
				data->cu_Jacobians, allDim,
				&beta,
				data->cu_Hessian, allDim) );*/

				//using transposed one




				/*CUBLAS_CALL( cublasSsyrk_v2(data->blas_handle_,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,
				allDim,data->pix_num,&alpha,data->cu_Jacobian_transpose,data->pix_num,&beta,
				data->cu_Hessian,allDim));*/


				//	float *cpuPara=new float[allDim*allDim];
				//CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
				//ofstream out_J("D:\\Fuhao\\cpu gpu validation\\ah_GPU.txt",ios::out);
				//for (int i=0;i<data->s_dim+data->t_dim+4;i++)
				//{
				//for (int j=0;j<data->s_dim+data->t_dim+4;j++)
				//out_J<<cpuPara[i*allDim+j]<<" ";
				//out_J<<endl;
				//}

				//out_J.close();

				////then Hessian RT
				//CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
				//	allDim, allDim, trackEngine.visibleNum*2, 
				//	&alpha, data->cu_Jacobians+(data->pix_num*allDim), allDim, 
				//	trackEngine.cu_conv, allDim,
				//	&beta,
				//	data->cu_JConv, allDim) );

				//then Hessian RT
				/*CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
					allDim,trackEngine.visibleNum*2,trackEngine.visibleNum*2,
					&alpha, data->cu_Jacobians+(data->pix_num*allDim), allDim, 
					trackEngine.cu_conv, trackEngine.visibleNum*2,
					&beta,
					data->cu_JConv, allDim) ); 
					CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
					allDim, allDim, trackEngine.visibleNum*2, 
					&alpha, data->cu_JConv, allDim, 
					data->cu_Jacobians+(data->pix_num*allDim), allDim,
					&beta,
					data->cu_RTHessian, allDim) );*/

				CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
					allDim,trackEngine.visibleNum*2,trackEngine.visibleNum*2,
					&alpha, data->cu_RT_Jacobian_transpose, trackEngine.visibleNum*2, 
					trackEngine.cu_conv, trackEngine.visibleNum*2,
					&beta,
					data->cu_JConv, allDim) ); 
			/*	CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
					allDim, allDim, trackEngine.visibleNum*2, 
					&alpha, data->cu_JConv, allDim, 
					data->cu_RT_Jacobian_transpose, trackEngine.visibleNum*2,
					&beta,
					data->cu_RTHessian, allDim) );
				vectorAdd<<<allDim*allDim/128+1,128>>>(data->cu_RTHessian,data->cu_Hessian,trackEngine.RTWeight,
					alpha,allDim*allDim);*/

				////J*Sigma*J'
				CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
					allDim, allDim, trackEngine.visibleNum*2, 
					&trackEngine.RTWeight, data->cu_JConv, allDim, 
					data->cu_RT_Jacobian_transpose, trackEngine.visibleNum*2,
					&alpha,
					data->cu_Hessian, allDim) );

				//J'*(J*Sigma)'
				/*CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_T, CUBLAS_OP_T,
					allDim, allDim, trackEngine.visibleNum*2, 
					&trackEngine.RTWeight, data->cu_RT_Jacobian_transpose, trackEngine.visibleNum*2, 
					data->cu_JConv, allDim,
					&alpha,
					data->cu_Hessian, allDim) );*/
		
				
				/*if (trackEngine.AAM_ENGINE->showSingleStep)
				{
					float *cpuPara=new float[allDim*allDim];
					CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));

					cout<<"detection Hessian:\n";
					for (int i=0;i<20;i++)
					{

						cout<<cpuPara[i]<<" ";


					}
					cout<<endl;


				}*/

				/*	float *cpuPara=new float[allDim*allDim];
				CUDA_CALL(cudaMemcpy(cpuPara, data->cu_RTHessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
				ofstream out_J("D:\\Fuhao\\cpu gpu validation\\rtH_new.txt",ios::out);
				for (int i=0;i<data->s_dim+data->t_dim+4;i++)
				{
				for (int j=0;j<data->s_dim+data->t_dim+4;j++)
				out_J<<cpuPara[i*allDim+j]<<" ";
				out_J<<endl;
				}
				out_J.close();
				cout<<"done!\n";*/

				//cout<<trackEngine.RTWeight<<" "<<trackEngine.AAMWeight<<endl;
				//add the two hessian together
			

				//copyJacobianWithWeight<<<(data->pix_num+trackEngine.visibleNum)*allDim+1,128>>>(data->cu_Jacobians,
				//	allDim,data->pix_num,trackEngine.visibleNum,trackEngine.AAMWeight,trackEngine.RTWeight,data->cu_FullJacobians);
				////hessian AAM
				//CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
				//	allDim, allDim, data->pix_num+trackEngine.visibleNum*2, 
				//	&alpha, data->cu_FullJacobians, allDim, 
				//	data->cu_Jacobians, allDim,
				//	&beta,
				//	data->cu_Hessian, allDim) );

				//hessian RT

				//weighted sum
			}





			if (trackEngine.priorWeight>0)
			{
				vectorAdd<<<allDim*data->s_dim/128+1,128>>>(trackEngine.cuHessianPrior,data->cu_Hessian,trackEngine.priorWeight,
					1,allDim*data->s_dim);


			/*	if (trackEngine.AAM_ENGINE->showSingleStep)
				{
					float *cpuPara=new float[allDim*allDim];
					CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));

					cout<<"full Hessian:\n";
					for (int i=0;i<20;i++)
					{

						cout<<cpuPara[i]<<" ";


					}
					cout<<endl;


				}*/

				/*vectorAdd<<<allDim*allDim/128+1,128>>>(trackEngine.cuHessianPrior,data->cu_Hessian,trackEngine.priorWeight,
					1,allDim*allDim);*/
			}
		}
			/*	float *cpuPara=new float[allDim*allDim];
				CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
				ofstream out_J("D:\\Fuhao\\cpu gpu validation\\h_GPU.txt",ios::out);
				for (int i=0;i<data->s_dim+data->t_dim+4;i++)
				{
				for (int j=0;j<data->s_dim+data->t_dim+4;j++)
				out_J<<cpuPara[i*allDim+j]<<" ";
				out_J<<endl;
				}

				out_J.close();*/
		else
		{
			/*CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
				allDim, allDim, data->pix_num, 
				&alpha, data->cu_Jacobians, allDim, 
				data->cu_Jacobians, allDim,
				&beta,
				data->cu_Hessian, allDim) );*/
			CUBLAS_CALL( cublasSgemm_v2(data->blas_handle_, CUBLAS_OP_T, CUBLAS_OP_N,
				allDim, allDim, data->pix_num, 
				&alpha, data->cu_Jacobian_transpose, data->pix_num, 
				data->cu_Jacobian_transpose, data->pix_num,
				&beta,
				data->cu_Hessian, allDim) );
		}

		
		/*}
		GTE("START");*/
		

		////////84/21//////////
		/*if (times>20)
		{
			break;
		}
		times++;
		continue;*/


	/*	float *cpuPara=new float[allDim*allDim];
		CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_JH("D:\\Fuhao\\cpu gpu validation\\Hessian_transpose.txt",ios::out);
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
			for (int j=0;j<data->s_dim+data->t_dim+4;j++)
				out_JH<<cpuPara[i*allDim+j]<<" ";
			out_JH<<endl;
		}

		out_JH.close();
		cout<<"done!\n";*/

		///////////////////////Hessian Checked 0321////////////////////////////

		//CULA_CALL( culaDeviceSgetrf(allDim, allDim, data->cu_Hessian, allDim, ipiv) );
		//CULA_CALL( culaDeviceSgetri(allDim, data->cu_Hessian, allDim, ipiv) );

		

		

	
		/*	CULA_CALL(culaDeviceSgesv(allDim,1,data->cu_Hessian,allDim,data->ipiv,
				data->cu_deltaParameters,allDim));*/
	
			/*	if (trackEngine.AAM_ENGINE->showSingleStep)
				{
					float *cpuPara=new float[allDim];
					CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian+allDim*allDim, allDim*sizeof(float), cudaMemcpyDeviceToHost ));

					cout<<"b:\n";
					for (int i=0;i<allDim;i++)
					{

						cout<<cpuPara[i]<<" ";


					}
					cout<<endl;


				}*/

				/////////////////////////mkl version////////////////////////////////
				//GTB("START");
				////for (int ii=0;ii<60;ii++)
				//{
				CUDA_CALL(cudaMemcpy(data->host_Hessian, data->cu_Hessian,allDim*(allDim+1)*sizeof(float), cudaMemcpyDeviceToHost ));
				//CUDA_CALL(cudaMemcpy(data->host_b, data->cu_deltaParameters,allDim*sizeof(float), cudaMemcpyDeviceToHost ));
				solveAxB(data->host_Hessian,data->host_Hessian+allDim*allDim,allDim);

				/*if (isAAMOnly||trackEngine.AAM_ENGINE->showSingleStep)
				{
					for (int k=0;k<10;k++)
					{
						cout<<data->host_b[k]<<" ";
					}
					cout<<endl;
				}*/

				alpha=-1.0;
				CUBLAS_CALL(cublasSaxpy_v2(data->blas_handle_,allDim,&alpha,data->cu_deltaB,1,data->cu_parameters,1));
				/*}
				GTE("START");*/
				//continue;
				////////////////////////////////////////////////////////////
	
		/////////////////////opencv version///////////////////////////////
	//	CUDA_CALL(cudaMemcpy(data->host_Hessian, data->cu_Hessian,allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
	//	invHessian(data->host_Hessian,data->host_inv_Hessian,allDim);
	////	//CUDA_CALL(cudaMemcpy(data->cu_Hessian,data->host_inv_Hessian,allDim*allDim*sizeof(float),cudaMemcpyHostToDevice));

	////	//get the delta
	/////*	CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_T,allDim,allDim,&alpha,data->cu_Hessian,allDim,data->cu_deltaParameters,1,&beta,
	////		data->cu_deltaParameters,1));//
	//	CUBLAS_CALL(cublasSgemv_v2(data->blas_handle_,CUBLAS_OP_T,allDim,allDim,&alpha,data->cu_inv_Hessian,allDim,data->cu_deltaParameters,1,&beta,
	//		data->cu_deltaParameters,1));
	//	alpha=-1.0;

	//	//update parameters
	//	CUBLAS_CALL(cublasSaxpy_v2(data->blas_handle_,allDim,&alpha,data->cu_deltaParameters,1,data->cu_parameters,1));
		//////////////////////////////////////////////////////////////


		//////////////////////////////////////////////////////////////////////

		/////////////////////direct solve 1//////////////////////////////////////////
		 
		
		/*CULA_CALL(culaDeviceSgesv(allDim,1,data->cu_Hessian,allDim,data->ipiv,
			data->cu_deltaParameters,allDim));*/
		///////////////////////////////////////////////////////////////

		//////////////////////////direct solve 2///////////////////////////////
		/*CUDA_CALL(cudaMemcpy(data->host_Hessian, data->cu_Hessian,allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
		CUDA_CALL(cudaMemcpy(data->host_b, data->cu_deltaParameters,allDim*sizeof(float), cudaMemcpyDeviceToHost ));
		solveAb(data->host_Hessian,data->host_b,data->host_dx,allDim);
		CUDA_CALL(cudaMemcpy(data->cu_deltaParameters,data->host_dx,allDim*sizeof(float),cudaMemcpyHostToDevice));*/
		/////////////////////////////////////////////////////////////////////

	/*	if (times>20)
		{
			break;
		}
		times++;
		continue;*/

	/*	float *cpuPara=new float[allDim*allDim];
		CUDA_CALL(cudaMemcpy(cpuPara, data->cu_Hessian, allDim*allDim*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_J("D:\\Fuhao\\cpu gpu validation\\HessianInv_GPU.txt",ios::out);
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
			for (int j=0;j<data->s_dim+data->t_dim+4;j++)
				out_J<<cpuPara[i*allDim+j]<<" ";
			out_J<<endl;
		}
		
		out_J.close();*/
		////////////////////////////inv checked//////////////////////////////////////////
		
		


		

	/*	float *cpuPara=new float[MPD_combination];
		CUDA_CALL(cudaMemcpy(cpuPara, data->cu_deltaParameters, MPD_combination*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_J1("D:\\Fuhao\\cpu gpu validation\\dp_GPU.txt",ios::out);
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
			out_J1<<-cpuPara[i]<<" ";
		}
		out_J1<<endl;
		out_J1.close();*/

	/*	ofstream out_Jj("D:\\Fuhao\\cpu gpu validation\\initialP_GPU.txt",ios::out);
		float *cpuPara=new float[MPD_combination];
		CUDA_CALL(cudaMemcpy(cpuPara, data->cu_parameters, MPD_combination*sizeof(float), cudaMemcpyDeviceToHost ));
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
			out_Jj<<cpuPara[i]<<" ";
		}
		out_Jj<<endl;
		out_Jj.close();*/

	
		


		//////////////////check parameters//////////////////////////
		//float *cpuPara=new float[MPD_combination];
	/*	CUDA_CALL(cudaMemcpy(cpuPara, data->cu_parameters, MPD_combination*sizeof(float), cudaMemcpyDeviceToHost ));
		ofstream out_J("D:\\Fuhao\\cpu gpu validation\\parameter_GPU.txt",ios::out);
		for (int i=0;i<data->s_dim+data->t_dim+4;i++)
		{
			out_J<<cpuPara[i]<<" ";
		}
		out_J<<endl;
		out_J.close();*/
		//check if we need to stop

		//if (times>MaxIterNum||errorSum<0.2&&abs(lastError-errorSum)<0.0001)
		//{
		//	/*cout<<"time "<<times<<endl;*/
		///*	float *parameters=new float[MPD_combination];
		//	CUDA_CALL(cudaMemcpy(parameters, data->cu_parameters, MPD_combination*sizeof(float), cudaMemcpyDeviceToHost ));
		//	checkIterationResult(parameters,data->ptsNum,data->s_dim,data->t_dim,true);
		//	delete []parameters;*/

		//	CUDA_CALL(cudaMemcpy(finalShape, data->cu_currentShape, data->ptsNum*2*sizeof(float), cudaMemcpyDeviceToHost ));
		//	break;
		//}
		//GTE("START");

		if (trackEngine.RTWeight_backup>minV)
		{
			trackEngine.RTWeight-=0.1*trackEngine.RTWeight_backup;
			if (trackEngine.RTWeight<=minV)
			{
				trackEngine.RTWeight=minV;
			}
		}
		
	/*	}
		GTE("START");
		gCodeTimer.printTimeTree();
		double time = total_fps;
		cout<<"used time per iteration: "<<time<<endl;*/
		/*if (times>40)
		{
			trackEngine.priorWeight=0.001;
		}*/
		//CUBLAS_CALL(cublasScopy_v2(data->blas_handle_,data->s_dim,data->cu_currentShape,1,data->cu_lastShape,1));
		lastError=errorSum;
		//cout<<"errorSum: "<<errorSum<<endl;
		times++;

	

	/*	if (times>20)
		{
				break;
		}*/

	

		
	}

	return status;
//	cout<<times<<endl;
	//CUDA_CALL(cudaEventRecord( stop, 0 ));
	//CUDA_CALL(cudaEventSynchronize( stop ));
	//float elapsedTime;
	//CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
	//	start, stop ) );
	//cout<<"time: "<<elapsedTime<<"/"<<times<<" ms"<<endl;


	/*cout<<"ok..\n";*/

}