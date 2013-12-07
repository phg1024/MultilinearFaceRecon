#include <fstream>
#include "CUDA_basic.h"
#include "sharedDefination.h"
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>
using namespace std;

#define LEFT_NULL 99999
#define RIGHT_NULL 11111
#define LEFT_NODE 999990
#define RIGHT_NODE 111110
#define TREE_START 888888
#define LEAF_NODE_LEFT 123456
#define LEAF_NODE_RIGHT 654321
#define ROOT_NODE 666666
#define END_TREE 798798



struct Node_CPU
{
	Node_CPU()
	{
		l_child=r_child=NULL;
		sampleInd=new vector<int>;
		label=-1;
		num_of_each_class=NULL;
		num_all=0;
		pos1[0]=pos1[1]=pos2[0]=pos2[1]=0;
		threshold=0;
	}
	int pos1[2];
	int pos2[2];
	//float *posteriorProb;//only used for leaves
	int label;
	int nLevel;
	Node_CPU *l_child,*r_child;
	vector <int> *sampleInd;

	//only for leaves
	int num_all;
	float *num_of_each_class;
	int threshold;
};



//struct LabelResult
//{
//	LabelResult()
//	{
//		label=-1;
//		prob=0;
//		//prob_all=new float[10];
//	}
//
//	int label;
//	float prob;
//	float prob_all[MAX_LABEL_NUMBER];
//};


struct Node_GPU
{
	Node_GPU()
	{
		//l_child_ind=r_child_ind=-2;
		////sampleInd=new vector<int>;
		//label=-1;
		////num_of_each_class=NULL;
		//num_all=0;
		//pos1[0]=pos1[1]=pos2[0]=pos2[1]=0;
		//threshold=0;
		num_of_each_class=NULL;
	/*	parameters[6]=parameters[7]=-2;
		parameters[4]=-1;
		parameters[8]=0;
		parameters[0]=parameters[1]=parameters[2]=parameters[3]=0;
		parameters[9]=0;*/
	}
	//int pos1[2];
	//int pos2[2];
	////float *posteriorProb;//only used for leaves
	//int label;
	//int nLevel;
	//int l_child_ind,r_child_ind;
	////vector <int> *sampleInd;

	////only for leaves
	//int num_all;
	//int threshold;
	//pos1[0] pos1[1] pos2[0] pos2[1] label nlevel lind rind num_all, threshold 
	float *parameters;
	float *cu_parameters;
	float *num_of_each_class;
	
};

struct LeafNode_GPU
{
	LeafNode_GPU()
	{
		leaf_node=NULL;
		label=-1;
	}
	Node_GPU *leaf_node;
	int label;
};

//struct LeafNode
//{
//	LeafNode(){leaf_node=NULL;label=-1;};
//	Node_GPU *leaf_node;
//	int label;
//};

struct RandmizedTree_CUDA
{
	RandmizedTree_CUDA()
	{
		hasImage=false;
		cu_treesTexture=new float *[30];
	}
	bool hasImage;
	int max_depth;
	int min_sample_count;
	double regression_accuracy;
	int max_num_of_trees_in_the_forest;
	int windowSize;
	int labelNum;
	int **indexList;

	LeafNode_GPU *leafnode;
	float *cu_currentImage;

	float *cu_colorImage;
	float *cu_depthImage;

	//label + prob_ALL[MAX_LABEL_NUMBER]
	float *cu_LabelResult;
	float *cu_LabelResultEachTree;
	float *cu_LabelFullResult;

	
	//LabelResult *host_LabelResult;
	//int width,height;

	Node_GPU **trees;
	Node_GPU ***host_trees;

	float *cu_vectorTrees;//trees in vectors
	float **cu_treesTexture;
	int MaxNumber;
	int *cu_currentInterestIndex;
	float *cu_trainingData;

	float *cu_maxIndPerBlock;
	float *maxIndPerBlock;

	//float *cu_maxValuePerBlock;
};