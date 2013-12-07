#include <string>
#include <fstream>
#include "CUDA_basic.h"
#include <math.h>
//#include <cutil.h>
#include <helper_cuda.h>
#include <helper_math.h>
//#include "sharedDefination.h"
using namespace std;

#include "RandomizedTreeGPU.h"
RandmizedTree_CUDA RandomizedTreeEngine;
texture<float,2> currentImg;
texture<float,2> currentColorImg,currentDepthImg;
texture<float,2,cudaReadModeElementType> trees_device;
texture<float,2> detectionResult2D;

int maximumWidth;
//texture<float> trees_device_1D;
struct Lock {
	int *mutex;
	Lock( void ) {
		int state = 0;
		CUDA_CALL( cudaMalloc( (void**)& mutex,
			sizeof(int) ) );
		CUDA_CALL( cudaMemcpy( mutex, &state, sizeof(int),
			cudaMemcpyHostToDevice ) );
	}
	~Lock( void ) {
		cudaFree( mutex );
	}
	__device__ void lock( void ) {
		while( atomicCAS( mutex, 0, 1 ) != 0 );
	}
	__device__ void unlock( void ) {
		atomicExch( mutex, 1 );
	}
};


double *cu_label_prob_all;
cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
int currentID;

void convertTrees2Array(Node *root_CPU, Node_GPU **root_GPU,int labelNum)
{
	//if (root_CPU->label!=-1)
	//{
	//	cout<<currentID<<" "<<root_CPU->label<<endl;
	//}
	//cout<<currentID<<" ";
	int rootID=currentID;
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
		currentID++;
		root_GPU[rootID]->parameters[6]=currentID;
		convertTrees2Array(root_CPU->l_child, root_GPU,labelNum);
	}
	if (root_CPU->r_child!=NULL)
	{
		currentID++;
		root_GPU[rootID]->parameters[7]=currentID;
		convertTrees2Array(root_CPU->r_child, root_GPU,labelNum);
	}
}

void outputTrees(Node_GPU **tree,int ind,ofstream &out)
{
	out<<tree[ind]->parameters[5]<<" ";
	out<<tree[ind]->parameters[0]<<" "<<tree[ind]->parameters[1]<<" "<<
		tree[ind]->parameters[2]<<" "<<tree[ind]->parameters[3]<<" "<<tree[ind]->parameters[9]<<endl;
	if (tree[ind]->parameters[6]==-1&&tree[ind]->parameters[7]==-1)
	{
		bool ok=false;
		for (int i=0;i<RandomizedTreeEngine.labelNum;i++)
		{
			out<<tree[ind]->parameters[10+i]<<" ";
			if (tree[ind]->parameters[10+i]>0)
			{
				ok=true;
			}
		}
		out<<endl;

		if (!ok)
		{
			cout<<"bad leaf at "<<ind<<endl;
			cout<<tree[ind]->parameters[4]<<" "<<tree[ind]->parameters[5]<<endl;
		}
	}
	else
	{
		if (tree[ind]->parameters[6]!=-1)
		{
			outputTrees(tree,tree[ind]->parameters[6],out);
		}
		if (tree[ind]->parameters[7]!=-1)
		{
			outputTrees(tree,tree[ind]->parameters[7],out);
		}
	}
}

__global__ void outputTree(float *tree,int MaxNumber,int treeInd,int labelNum)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if (offset<MaxNumber)
	{
		//printf("%d %d\n",offset,(10+MAX_LABEL_NUMBER)*MaxNumber);
		int initialInd=(treeInd*MaxNumber+offset)*(10+MAX_LABEL_NUMBER);
		//printf("after getting values %d %d\n",tree[initialInd+6],tree[initialInd+7]);
		if (tree[initialInd+6]==-1&&tree[initialInd+7]==-1)
		{
			int i;
			for (i=0;i<labelNum;i++)
			{
				if(tree[initialInd+10+i]>0)
					break;
			}
			if (i==labelNum)
			{
				for (int i=0;i<labelNum;i++)
				{
					printf("%f ",tree[initialInd+10+i]);
				}
			}
			printf("\n");
		}
	}
}

extern "C" void setData_preprocess(int _max_depth,int _min_sample_count,double _regression_accuracy, int _max_num_of_trees_in_the_forest,int _windowSize, int labelNum, Node **trees_cpu,int treeNum,bool withDepth)
{
	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
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
		currentID=0;
		convertTrees2Array(trees_cpu[i],data->host_trees[i],labelNum);
		cout<<i<<" "<<currentID<<endl;
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
	
	if(!withDepth)
	{
		CUDA_CALL( cudaMalloc(&data->cu_currentImage, MPN * sizeof(float)) );
	}
	else
	{
		CUDA_CALL( cudaMalloc(&data->cu_colorImage, MPN * sizeof(float)) );
		CUDA_CALL( cudaMalloc(&data->cu_depthImage, MPN * sizeof(float)) );
	}

	

	CUDA_CALL( cudaMalloc(&data->cu_LabelResult, MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );


	//CUDA_CALL( cudaMalloc(&data->cu_LabelResultEachTree, treeNum*MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );
	/*CUDA_CALL(cudaBindTexture( NULL, detectionResult1D,
	treeNum*MPN*(1+MAX_LABEL_NUMBER) * sizeof(float));*/
	
	
	cout<<"labelResult GPU set"<<MPN*(1+MAX_LABEL_NUMBER)<<endl;
//	maximumWidth=640;
	CUDA_CALL( cudaMalloc(&data->cu_LabelFullResult, MPN*(1+MAX_LABEL_NUMBER) * sizeof(float)) );

	CUDA_CALL(cudaBindTexture2D( NULL, detectionResult2D,
		data->cu_LabelFullResult,
		desc, 480 ,(1+MAX_LABEL_NUMBER)*640,
		sizeof(float) * 480));



	currentImg.filterMode=cudaFilterModePoint;
	currentColorImg.filterMode=cudaFilterModePoint;
	currentDepthImg.filterMode=cudaFilterModePoint;
	//data->host_LabelResult=new LabelResult[MPN];
}








__device__ void getProb(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
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


		if (tex2D(currentImg,pos1[0]+pos[0],pos1[1]+pos[1])>
			tex2D(currentImg,pos2[0]+pos[0],pos2[1]+pos[1])+threshold)
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

__device__ void getProb_depth_depth(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
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

		float curDepth=1.0f/tex2D(currentDepthImg,pos[0],pos[1]);

		//according to the train style
		//if (trainStyle==0)
		{
			if (tex2D(currentDepthImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentDepthImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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

__device__ void getProb_depth_color(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
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

		float curDepth=1.0f/tex2D(currentDepthImg,pos[0],pos[1]);

		//according to the train style
	
		{
			if (tex2D(currentColorImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentColorImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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

__device__ void getProb_depth_depth_color(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
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

		float curDepth=1.0f/tex2D(currentDepthImg,pos[0],pos[1]);
		{
			if (threshold>=0)
			{
				if (tex2D(currentDepthImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentDepthImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
				if (tex2D(currentColorImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentColorImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1]))
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

__device__ void getProb_depth(float *trees,int treeInd,int *pos,int *ind,int currentInd,int MaxNumber,int trainStyle)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
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

		float curDepth=1.0f/tex2D(currentDepthImg,pos[0],pos[1]);

		//according to the train style
		if (trainStyle==0)
		{
			if (tex2D(currentDepthImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentDepthImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
			if (tex2D(currentColorImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentColorImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
				if (tex2D(currentDepthImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentDepthImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
				if (tex2D(currentColorImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentColorImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1]))
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

__device__ void getProb_depth_textureTrees(int treeInd,int *pos,int *ind,int currentInd,int MaxNumber,int trainStyle)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//Node_GPU *current=&(root[currentInd]);

//	printf("getProb in\n");

	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
	int initialInd=treeInd*MaxNumber;
	int startInd;
	//int currentInd=treeInd;
	while(1)
	//for(int i=0;i<2;i++)
	{
		//printf("%d\n",startInd);

		//2d texture
		startInd=initialInd+currentInd;
	
		pos1[0]=tex2D(trees_device,0,startInd);//trees[startInd+0];
		pos1[1]=tex2D(trees_device,1,startInd);//trees[startInd+1];
		pos2[0]=tex2D(trees_device,2,startInd);//trees[startInd+2];
		pos2[1]=tex2D(trees_device,3,startInd);//trees[startInd+3];
		label=tex2D(trees_device,4,startInd);//trees[startInd+4];
		l_child_ind=tex2D(trees_device,6,startInd);
		r_child_ind=tex2D(trees_device,7,startInd);
		threshold=tex2D(trees_device,9,startInd);//trees[startInd+9];
		
		/* l_child_ind=trees[startInd+6];
		 r_child_ind=trees[startInd+7];
		 pos1[0]=trees[startInd+0];
		 pos1[1]=trees[startInd+1];
		  pos2[0]=trees[startInd+2];
		   pos2[1]=trees[startInd+3];
		   label=trees[startInd+4];
		   threshold=trees[startInd+9];*/

		
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

		float curDepth=1.0f/tex2D(currentDepthImg,pos[0],pos[1]);

		//according to the train style
		if (trainStyle==0)
		{
			if (tex2D(currentDepthImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentDepthImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
			if (tex2D(currentColorImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
				tex2D(currentColorImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
				if (tex2D(currentDepthImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentDepthImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1])+threshold)
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
				if (tex2D(currentColorImg,(float)pos1[0]*curDepth+pos[0],(float)pos1[1]*curDepth+pos[1])>
					tex2D(currentColorImg,(float)pos2[0]*curDepth+pos[0],(float)pos2[1]*curDepth+pos[1]))
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

//suppose the maximum class number is 10
__global__ void predict_prob(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber)
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
	/*	for (int i=0;i<labelNum;i++)
		{
			label_prob_all[i]=0;
		}*/
		int ind[2]={0,0};
		int startInd;
		//for (i=0;i<treeNum;i++)
		for (i=0;i<treeNum;i++)
		{
			label_prob_all[i]=0;
			getProb(trees,i,pos,ind,0,MaxNumber);
			startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
			//printf("LI: %d, RI: %d",trees[startInd+6],trees[startInd+7]);
			if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			{
				
				for (j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=trees[startInd+10+j];
				}
				//label_prob_all[j]/=(float)treeNum;
			}
		}
		
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

__global__ void predict_prob_withDepth_textureTrees(float *result,int labelNum,int treeNum,int width,int height,int windowSize,LeafNode_GPU *leaf,int MaxNumber,int trainStyle)
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

		if(tex2D(currentDepthImg,pos[0],pos[1])==0)
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
			getProb_depth_textureTrees(i,pos,ind,0,MaxNumber,trainStyle);
			startInd=i*MaxNumber+ind[0];
			//printf("LI: %d, RI: %d",trees[startInd+6],trees[startInd+7]);
			if (tex2D(trees_device,6,startInd)==-1&&tex2D(trees_device,7,startInd)==-1) //reach a leaf
			{
				
				for (j=0;j<labelNum;j++)
				{
					label_prob_all[j]+=tex2D(trees_device,10+j,startInd);//trees[startInd+10+j];
				}
				//label_prob_all[j]/=(float)treeNum;
			}
		}
		
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

__global__ void predict_prob_withDepth_depth_color(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber)
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

		if(tex2D(currentDepthImg,pos[0],pos[1])==0)
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

			getProb_depth_depth_color(trees,i,pos,ind,0,MaxNumber);
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

__global__ void getProbMap(float *result,int labelNum,int treeNum,int width,int height,int windowSize)
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

		result[offset]=tex2D(detectionResult2D,pos[1],pos[0]);
	}
}

__global__ void predict_prob_withDepth_color(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber)
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

		if(tex2D(currentDepthImg,pos[0],pos[1])==0)
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

			getProb_depth_color(trees,i,pos,ind,0,MaxNumber);
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
		int maxInd=0;
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

__global__ void predict_prob_withDepth_depth(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber)
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

		if(tex2D(currentDepthImg,pos[0],pos[1])==0)
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

			getProb_depth_depth(trees,i,pos,ind,0,MaxNumber);
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

//suppose the maximum class number is 10
__global__ void predict_prob_withDepth(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,int trainStyle)
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

		if(tex2D(currentDepthImg,pos[0],pos[1])==0)
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

			getProb_depth(trees,i,pos,ind,0,MaxNumber,trainStyle);
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

__global__ void predict_prob_withDepth_eachTree(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,int MaxNumber,int trainStyle)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int treeId=blockIdx.y;
	if (offset<width*height&&treeId<treeNum)
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

		if(tex2D(currentDepthImg,pos[0],pos[1])==0)
		{

			if (treeId==0)
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
			}
			
			return;
		}

		int i,j;

		if (pos[0]<=windowSize||pos[0]>=width-windowSize||
			pos[1]<=windowSize||pos[1]>=height-windowSize)
		{
			if (treeId==0)
			{
				int currentInd=offset*(1+MAX_LABEL_NUMBER);
				//if (label_prob_all[maxInd]>threshold)
				{
					result[currentInd+0]=labelNum-1;;
					//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
					for (i=0;i<labelNum;i++)
					{
						result[currentInd+1+i]=0;
					}
					//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
				}
			}

			return;
		}

		//double label_prob_all[MAX_LABEL_NUMBER];
		//for (int i=0;i<labelNum;i++)
		//{
		//	label_prob_all[i]=0;
		//}
		int ind[2]={0,0};
		int startInd;

		////about 210ms
		////for (i=0;i<treeNum;i++)
		//{

		getProb_depth(trees,treeId,pos,ind,0,MaxNumber,trainStyle);
			startInd=treeId*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
		//	//printf("LI: %d, RI: %d",trees[startInd+6],trees[startInd+7]);
			//if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
			//{
			//	int currentInd=(treeId*MPN+offset)*(1+MAX_LABEL_NUMBER);

			//	for (i=0;i<labelNum;i++)
			//	{
			//		//lockEXP.lock();
			//		result[currentInd+1+i]=trees[startInd+10+i];
			//		//lockEXP.unlock();
			//	}
			//}
		
	}
}

////maximum Ind the probability
////1+MAX_LABEL_NUMBER
//__global__ findMaximumProb(float *detectionResult,int width,int height)
//{
//	__shared__ float cache[512];
//	int tid=blockIdx.x*blockDim.x+threadIdx.x;
//	int cacheIndex=threadIdx.x;
//	
//	int tmpLabel;
//
//	int maximumID;
//	if (tid<width*height)
//	{
//		tmpLabel=detectionResult[];
//		cache[cacheIndex]
//	}
//}

__device__ void getProb_FP(float *trees,int treeInd,int startPixelInd,int pixelStep,int width, int height, int MaxNumber,float *result,int labelNum,int treeNum)
{
	int i=0;
	int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
	int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;//;=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	int startInd;
	int currentInd;
	int pos[2];
	int times=0;
	for (i=startPixelInd;i<width*height;i+=pixelStep)
	{
		//int currentInd=treeInd;
		currentInd=0;
		pos[0]=i%width;
		pos[1]=i/width;
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
				/*ind[0]=currentInd;
				ind[1]=label;*/
				for (int k=1;k<=labelNum;k++)
				{
					atomicAdd(&result[i*(1+MAX_LABEL_NUMBER)+k],trees[startInd+9+k]/treeNum);
				}

				break;
			}


			if (tex2D(currentImg,pos1[0]+pos[0],pos1[1]+pos[1])>
				tex2D(currentImg,pos2[0]+pos[0],pos2[1]+pos[1])+threshold)
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

__global__ void predict_prob_fullParral_pixel_tree(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,const int MaxNumber)
{
	//int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int treeInd=threadIdx.x;
	int PixelInd=blockIdx.y*gridDim.x+blockIdx.x;
	//int pixelStep=blockDim.x;
	//const int TreeSize=MaxNumber*(10+10);
	//const int TreeSize=MaxNumber*(10+labelNum);
	//__shared__ float LocalParameters[32767*20];

	//int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;//;=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	//for (int ind=0;ind<(10+MAX_LABEL_NUMBER)*MaxNumber;ind++)
	//{
	//	LocalParameters[ind]=trees[initialInd+ind];
	//}
	//__syncthreads();
	//return;

	if (treeInd<treeNum&&PixelInd<width*height)
	{
		int i=PixelInd;
		int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
		
		int startInd;
		int currentInd;
		int pos[2];
		int k;
		int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;//;=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;;
		//int times=0;
	

		//int currentInd=treeInd;
		currentInd=0;
		pos[0]=i%width;
		pos[1]=i/width;

		if (pos[0]<=windowSize||pos[0]>=width-windowSize||
			pos[1]<=windowSize||pos[1]>=height-windowSize)
		{
			int currentInd=i*(1+MAX_LABEL_NUMBER);
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
				/*ind[0]=currentInd;
				ind[1]=label;*/
				for (k=1;k<=labelNum;k++)
				{
					atomicAdd(&result[i*(1+MAX_LABEL_NUMBER)+k],trees[startInd+9+k]/treeNum);
				}

				break;
			}


			if (tex2D(currentImg,pos1[0]+pos[0],pos1[1]+pos[1])>
				tex2D(currentImg,pos2[0]+pos[0],pos2[1]+pos[1])+threshold)
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
//suppose the maximum class number is 10
__global__ void predict_prob_fullParral(float *result,int labelNum,int treeNum,int width,int height,int windowSize,float *trees,LeafNode_GPU *leaf,const int MaxNumber)
{
	//int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int treeInd=blockIdx.x;
	int startPixelInd=threadIdx.x;
	int pixelStep=blockDim.x;
	//const int TreeSize=MaxNumber*(10+10);
	//const int TreeSize=MaxNumber*(10+labelNum);
	//__shared__ float LocalParameters[32767*20];

	//int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;//;=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;
	//for (int ind=0;ind<(10+MAX_LABEL_NUMBER)*MaxNumber;ind++)
	//{
	//	LocalParameters[ind]=trees[initialInd+ind];
	//}
	//__syncthreads();
	//return;

	if (treeInd<treeNum)
	{
		int i=0;
		int l_child_ind,r_child_ind,pos1[2],pos2[2],label,threshold;
		
		int startInd;
		int currentInd;
		int pos[2];
		int k;
		int initialInd=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;//;=treeInd*(10+MAX_LABEL_NUMBER)*MaxNumber;;
		//int times=0;
		for (i=startPixelInd;i<width*height;i+=pixelStep)
		{
			//int currentInd=treeInd;
			currentInd=0;
			pos[0]=i%width;
			pos[1]=i/width;
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
					/*ind[0]=currentInd;
					ind[1]=label;*/
					for (k=1;k<=labelNum;k++)
					{
						atomicAdd(&result[i*(1+MAX_LABEL_NUMBER)+k],trees[startInd+9+k]/treeNum);
					}

					break;
				}


				if (tex2D(currentImg,pos1[0]+pos[0],pos1[1]+pos[1])>
					tex2D(currentImg,pos2[0]+pos[0],pos2[1]+pos[1])+threshold)
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
		//getProb_FP(trees,treeInd,startPixelInd,pixelStep,width,height,MaxNumber);
		

	//	int pos[2];
	//	pos[0]=offset%width;
	//	pos[1]=offset/width;
	//	int i,j;

	//	if (pos[0]<=windowSize||pos[0]>=width-windowSize||
	//		pos[1]<=windowSize||pos[1]>=height-windowSize)
	//	{
	//		int currentInd=offset*(1+MAX_LABEL_NUMBER);
	//		//if (label_prob_all[maxInd]>threshold)
	//		{
	//			result[currentInd+0]=-1;
	//			//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
	//			for (i=0;i<labelNum;i++)
	//			{
	//				result[currentInd+1+i]=0;
	//			}
	//			//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	//		}

	//		return;
	//	}

	//	double label_prob_all[MAX_LABEL_NUMBER];
	///*	for (int i=0;i<labelNum;i++)
	//	{
	//		label_prob_all[i]=0;
	//	}*/
	//	int ind[2]={0,0};
	//	int startInd;
	//	//for (i=0;i<treeNum;i++)
	//	for (i=0;i<treeNum;i++)
	//	{
	//		label_prob_all[i]=0;
	//		getProb(trees,i,pos,ind,0,MaxNumber);
	//		startInd=i*(10+MAX_LABEL_NUMBER)*MaxNumber+ind[0]*(10+MAX_LABEL_NUMBER);
	//		//printf("LI: %d, RI: %d",trees[startInd+6],trees[startInd+7]);
	//		if (trees[startInd+6]==-1&&trees[startInd+7]==-1) //reach a leaf
	//		{
	//			
	//			for (j=0;j<labelNum;j++)
	//			{
	//				label_prob_all[j]+=trees[startInd+10+j];
	//			}
	//			//label_prob_all[j]/=(float)treeNum;
	//		}
	//	}
	//	
	//	////find the most frequent label
	//	int maxInd=0;
	//	double maxNum=-1;
	//	for (i=0;i<labelNum;i++)
	//	{
	//		if (label_prob_all[i]>maxNum)
	//		{
	//			maxNum=label_prob_all[i];
	//			maxInd=i;
	//		}
	//	}

	//	int currentInd=offset*(1+MAX_LABEL_NUMBER);
	//	//if (label_prob_all[maxInd]>threshold)
	//	{
	//		result[currentInd+0]=maxInd;
	//		//result[offset].prob=label_prob_all[maxInd]/(float)treeNum;
	//		for (i=0;i<labelNum;i++)
	//		{
	//			result[currentInd+1+i]=label_prob_all[i]/(float)treeNum;
	//		}
	//		//cout<<result.label<<" "<<label_prob_all[maxInd]<<" "<<treeNum<<" "<<result.prob<<endl;
	//	}
		
	//}
	
	
}



void setData_onrun(float *hostImg,int width,int height)
{
	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	CUDA_CALL(cudaMemcpy(data->cu_currentImage,hostImg,MPN*sizeof(float),cudaMemcpyHostToDevice));
	if (!data->hasImage)
	{
		CUDA_CALL(cudaBindTexture2D( NULL, currentImg,
					data->cu_currentImage,
					desc,  width,height,
					sizeof(float) * width));
		data->hasImage=true;
	}
}

void setData_onrun(float *colorImg,float *depthImg,int width,int height)
{
	//cout<<"setting data on the fly"<<endl;
	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	CUDA_CALL(cudaMemcpy(data->cu_colorImage,colorImg,MPN*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(data->cu_depthImage,depthImg,MPN*sizeof(float),cudaMemcpyHostToDevice));
	if (!data->hasImage)
	{
		CUDA_CALL(cudaBindTexture2D( NULL, currentColorImg,
			data->cu_colorImage,
			desc,  width,height,
			sizeof(float) * width));
		CUDA_CALL(cudaBindTexture2D( NULL, currentDepthImg,
			data->cu_depthImage,
			desc,  width,height,
			sizeof(float) * width));
		data->hasImage=true;
	}
}

extern "C" void setData_RT_onrun(float *colorImg,float *depthImg,int width,int height)
{
	//cout<<"setting data on the fly"<<endl;
	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	CUDA_CALL(cudaMemcpy(data->cu_colorImage,colorImg,MPN*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(data->cu_depthImage,depthImg,MPN*sizeof(float),cudaMemcpyHostToDevice));
	//if (!data->hasImage)
	{
		CUDA_CALL(cudaBindTexture2D( NULL, currentColorImg,
			data->cu_colorImage,
			desc,  width,height,
			sizeof(float) * width));
		CUDA_CALL(cudaBindTexture2D( NULL, currentDepthImg,
			data->cu_depthImage,
			desc,  width,height,
			sizeof(float) * width));
		//data->hasImage=true;
	}
}


extern "C" void predict_GPU(float *host_img,int width,int height,float *host_result)
{
	//load the trained tree
	setData_onrun(host_img,width,height);

	RandmizedTree_CUDA *data=&RandomizedTreeEngine;

	dim3 grid(width,height,1);

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord( start, 0 ));
	predict_prob<<<width*height/64+1,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
		data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	//predict_prob_fullParral<<<data->max_num_of_trees_in_the_forest,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	//predict_prob_fullParral_pixel_tree<<<grid,16>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	CUDA_CALL(cudaEventRecord( stop, 0 ));
	CUDA_CALL(cudaEventSynchronize( stop ));
	float elapsedTime;
	CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
		start, stop ) );
	cout<<"time: "<<elapsedTime<<" ms"<<endl;

	//CUDA_CALL(cudaMemcpy(data->cu_LabelResult,host_result,MPN*sizeof(LabelResult),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(host_result,data->cu_LabelResult, MPN*(1+MAX_LABEL_NUMBER)*sizeof(float),cudaMemcpyDeviceToHost));
}

extern "C" void predict_GPU_withDepth(float *color_img,float *depth_img,int width,int height,float *host_result,int trainStyle)
{
	//load the trained tree
	setData_onrun(color_img,depth_img,width,height);

	int threadsPerBlock;

	RandmizedTree_CUDA *data=&RandomizedTreeEngine;

	

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord( start, 0 ));
	threadsPerBlock=256;
	if (trainStyle==0)
	{
		predict_prob_withDepth_depth<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	else if (trainStyle==1)
	{
		predict_prob_withDepth_color<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	else if (trainStyle==2)
	{
		predict_prob_withDepth_depth_color<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}

//	predict_prob_withDepth<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber,trainStyle);


	//each tree goes independently first
	//threadsPerBlock=256;
	////Lock lock;
	//dim3 grid(width*height/threadsPerBlock+1,data->max_num_of_trees_in_the_forest,1);
	//predict_prob_withDepth_eachTree<<<grid,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber,trainStyle);

	//predict_prob_fullParral<<<data->max_num_of_trees_in_the_forest,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	//predict_prob_fullParral_pixel_tree<<<grid,16>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	//predict_prob_withDepth_textureTrees<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
		//data->windowSize,(data->leafnode),data->MaxNumber,trainStyle);
	CUDA_CALL(cudaEventRecord( stop, 0 ));
	CUDA_CALL(cudaEventSynchronize( stop ));
	float elapsedTime;
	CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
		start, stop ) );
	cout<<"time: "<<elapsedTime<<" ms"<<endl;

	////////////////////////////////////////////////////////////
	//find the maximum probability for each label

	////////////////////////////////////////////////////////////

	//CUDA_CALL(cudaMemcpy(data->cu_LabelResult,host_result,MPN*sizeof(LabelResult),cudaMemcpyHostToDevice));
	
	CUDA_CALL(cudaMemcpy(host_result,data->cu_LabelResult, MPN*(1+MAX_LABEL_NUMBER)*sizeof(float),cudaMemcpyDeviceToHost));

}

extern "C" void predict_GPU_withDepth_clean(int width,int height,float *host_result,int trainStyle)
{
	//load the trained tree
	RandmizedTree_CUDA *data=&RandomizedTreeEngine;

	dim3 grid(width,height,1);

	cudaEvent_t start, stop;
	CUDA_CALL(cudaEventCreate(&start));
	CUDA_CALL(cudaEventCreate(&stop));
	CUDA_CALL(cudaEventRecord( start, 0 ));
	int threadsPerBlock=256;
	if (trainStyle==0)
	{
		predict_prob_withDepth_depth<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	else if (trainStyle==1)
	{
		predict_prob_withDepth_color<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	else if (trainStyle==2)
	{
		predict_prob_withDepth_depth_color<<<width*height/threadsPerBlock+1,threadsPerBlock>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
			data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	}
	/*predict_prob_withDepth<<<width*height/64+1,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
		data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber,trainStyle);*/
	//predict_prob_fullParral<<<data->max_num_of_trees_in_the_forest,64>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	//predict_prob_fullParral_pixel_tree<<<grid,16>>>(data->cu_LabelResult,data->labelNum,data->max_num_of_trees_in_the_forest,width,height,
	//	data->windowSize,data->cu_vectorTrees,(data->leafnode),data->MaxNumber);
	CUDA_CALL(cudaEventRecord( stop, 0 ));
	CUDA_CALL(cudaEventSynchronize( stop ));
	float elapsedTime;
	CUDA_CALL( cudaEventElapsedTime( &elapsedTime,
		start, stop ) );
	cout<<"time: "<<elapsedTime<<" ms"<<endl;

	//we need to find out the maximum points and the locations. Others are not neccary to transfer.
	//CUDA_CALL(cudaMemcpy(data->cu_LabelResult,host_result,MPN*sizeof(LabelResult),cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(host_result,data->cu_LabelResult, MPN*(1+MAX_LABEL_NUMBER)*sizeof(float),cudaMemcpyDeviceToHost));

}


//training examples copy
//training data: images 
//cu_currentInterestIndex: current index list: [x1+ y1*width] and [x2 +y2*width]
extern "C" void setData_Training_Preprocess(float *trainingData,int num,int maximumDepth)
{
	RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	CUDA_CALL( cudaMalloc(&data->cu_trainingData, num * sizeof(float)) );
	CUDA_CALL(cudaMemcpy(data->cu_trainingData,trainingData,num*sizeof(float),cudaMemcpyHostToDevice));

	//100d
	CUDA_CALL( cudaMalloc(&data->cu_currentInterestIndex, maximumDepth*100*2 * sizeof(int)) );
}

//testData: current training data
//indexList: candidate lists
//sampleNum: the number of candidates
//startIndex: starting index of each image
__global__ void getGain(float *testData,int *indexList,int sampleNum,int *startingIndex)
{
	int offset=threadIdx.x+blockIdx.x*blockDim.x;
	int threashold=blockIdx.y;
	
	int i;
	int l_size,r_size;
	l_size=r_size=0;

	float labelNum[10];

	int c_startIndex;
	for(i=0;i<sampleNum;i++)
	{
		c_startIndex=startingIndex[i];
		if (testData[c_startIndex+indexList[i]]>testData[c_startIndex+indexList[i+sampleNum]]+threashold)
		{
			
		}
	}
}

//currentInd: [x1 y1] [x2  y2]
//length: the number of candidates
extern "C" void Split_MaximumGain_GPU(int *currentInd,int length)
{
	//RandmizedTree_CUDA *data=&RandomizedTreeEngine;
	//CUDA_CALL(cudaMemcpy(data->cu_currentInterestIndex,currentInd,num*sizeof(float),cudaMemcpyHostToDevice));
	//
	////calculate gain for each possible criteria
	////try every possible value first, then range
	//dim3 dim(length/32+1,255);
	//getGain<<<(dim,32>>>(data->cu_trainingData,data->cu_currentInterestIndex,length);
}