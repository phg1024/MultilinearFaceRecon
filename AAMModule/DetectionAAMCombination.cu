#include <string>
#include <fstream>
#include "CUDA_basic.h"
#include <math.h>

#include <cuda.h>
#include <helper_cuda.h>
#include <helper_math.h>

using namespace std;

#include "RandomizedTreeGPU.h"
#include "AAM_RealGlobal_CUDA.h"

AAM_Search_RealGlobal_CUDA AAM_DataEngine;

RandmizedTree_CUDA RandomizedTreeEngine;
texture<float,2> currentColorImg,currentDepthImg;

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

extern "C" void setData_RT_Preprocess(int _max_depth,int _min_sample_count,double _regression_accuracy, int _max_num_of_trees_in_the_forest,int _windowSize, int labelNum, Node **trees_cpu,int treeNum,bool withDepth)
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

	CUDA_CALL(cudaMalloc((void **)&data->cu_vectorTrees,(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float)));
	float *host_vectorTrees=new float[(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum];

	//root_GPU[currentID]->parameters[0]=root_CPU->pos1[0];
	//root_GPU[currentID]->parameters[1]=root_CPU->pos1[1];
	//root_GPU[currentID]->parameters[2]=root_CPU->pos2[0];
	//root_GPU[currentID]->parameters[3]=root_CPU->pos2[1];
	//root_GPU[currentID]->parameters[4]=root_CPU->label;
	//root_GPU[currentID]->parameters[5]=root_CPU->nLevel;
	//root_GPU[currentID]->parameters[8]=root_CPU->num_all;
	//root_GPU[currentID]->parameters[9]=root_CPU->threshold;

	//cout<<"tree num: "<<treeNum<<endl;
	cout<<MaxNumber<<endl;
	cout<<"assigning values\n";
	for (int i=0;i<treeNum;i++)
	{
		cout<<i<<endl;
		for (int j=0;j<MaxNumber;j++)
		{
			//cout<<i<<" "<<j<<endl;
		/*	for (int k=0;k<)
			{
			}*/
			for (int k=0;k<10+MAX_LABEL_NUMBER;k++)
			{
				host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+k]=data->host_trees[i][j]->parameters[k];
			}

	/*		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+0]=data->host_trees[i][j].pos1[0];
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+1]=data->host_trees[i][j].pos1[1];
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+2]=data->host_trees[i][j].pos2[0];
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+3]=data->host_trees[i][j].pos2[1];
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+4]=data->host_trees[i][j].label;
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+5]=data->host_trees[i][j].nLevel;
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+8]=data->host_trees[i][j].num_all;
			host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+9]=data->host_trees[i][j].threshold;*/
			
			

			//if (trees_cpu[i][j].l_child==NULL)
			//{
			//	host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+6]=-1;
			//}
			//if (trees_cpu[i][j].r_child==NULL)
			//{
			//	host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+7]=-1;
			//}

			//for (int i=0;i<MAX_LABEL_NUMBER;i++)
			//{
			//	host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+10+i]=0;
			//}

			//if ((trees_cpu[i][j].l_child==NULL)&&trees_cpu[i][j].r_child==NULL)//root
			//{
			//	for (int i=0;i<labelNum;i++)
			//	{
			//		host_vectorTrees[(i*MaxNumber+j)*(10+MAX_LABEL_NUMBER)+10+i]=trees_cpu[i][j].num_of_each_class[i];
			//	}
			//}
		}
	}

	cout<<"copying values\n";
	CUDA_CALL(cudaMemcpy(data->cu_vectorTrees,host_vectorTrees,(10+MAX_LABEL_NUMBER)*MaxNumber*treeNum*sizeof(float),cudaMemcpyHostToDevice));

	delete []host_vectorTrees;
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
	cout<<"labelResult GPU set"<<MPN*(1+MAX_LABEL_NUMBER)<<endl;
	//CUDA_CALL( cudaMalloc(&data->cu_LabelFullResult, MPN*treeNum*(1+MAX_LABEL_NUMBER) * sizeof(float)) );

	currentImg.filterMode=cudaFilterModePoint;
	currentColorImg.filterMode=cudaFilterModePoint;
	currentDepthImg.filterMode=cudaFilterModePoint;
	//data->host_LabelResult=new LabelResult[MPN];
}