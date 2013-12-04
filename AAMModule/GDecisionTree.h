//
// coded by Xiaolin (Kevin) Wei
//

#ifndef GDECISIONTREE_H
#define GDECISIONTREE_H

#include "G/GBinaryTree.h"
#include "G/GSingleList.h"
#include "G/GSkeleton.h"
#include "G/GViewpoint.h"

#include <vector>
#include <iostream>
using std::vector;

class GMatrix;

class GDecisionTree
{
public:
	GDecisionTree() {}

	int GetNumClasses() const { return C; }

	void SetOptions(int frame_width, int frame_height,
		int window_length, int max_tree_depth, int min_node_size, 
		int num_testing_candidates, int num_threshold_candidates);
	void Learn_DFS(vector< GMatrix > *depthMap, vector< GMatrix > *labelMap, 
		const vector<int> &dataSamples, int C);

	void Induce(const GMatrix &depthMap, int frame_idx, double *probMap);
	
	int getFrameWidth() const { return _frame_width; }  
	int getFrameHeight() const{ return _frame_height; } 
	int getWindowLength() const { return _window_length; } 

	// use for copying data to GPU
	void TreeIterToRoot() const;
	void TreeIterToLeft() const;
	void TreeIterToRight() const;
	void TreeIterToParent() const;
	bool TreeIterHasLeft() const;
	bool TreeIterHasRight() const;
	int getCurNodeVar1() const;
	int getCurNodeVar2() const;
	double getCurNodeBestcut() const;
	int getCurNodeProbCount() const;
	const double* getCurNodeProb() const;

private:
	struct SplitRule{
		int var1, var2;	// variable 1 & 2
		double bestcut;
		int max_prob_class;		// only for leaf nodes
		vector<double> prob;	// only for leaf nodes
		GSingleList<int> work; // (tree-depth) + (purity?) + (indices to the data to split)
	};

public:
	typedef GBinaryTree<SplitRule>::Iter TreeIter;

	struct NodeListData
	{
		TreeIter treePointer;
		int parentIndex;
		bool isLeftNode;
		NodeListData(const TreeIter& iter, int parentIdx, bool isLeft)
		{
			treePointer = iter;
			parentIndex = parentIdx;
			isLeftNode = isLeft;
		}
		NodeListData() {}
	};
	int getTreeNodeList(vector<GDecisionTree::NodeListData> *outNodeList) const;

private:
	//struct SplitRule{
	//	int var1, var2;	// variable 1 & 2
	//	double bestcut;
	//	int max_prob_class;		// only for leaf nodes
	//	vector<double> prob;	// only for leaf nodes
	//	GSingleList<int> work; // (tree-depth) + (purity?) + (indices to the data to split)
	//};
	typedef GSingleList<int>::Iter ListIter;

	// tree
	GBinaryTree<SplitRule> tree;

	// used by TreeIterTo*() 
	mutable TreeIter tree_iter_;

	// training data
	vector< GMatrix > *_depthMap;
	vector< GMatrix > *_labelMap;
	const int *_dataSamples;	// 3 x N (frame_num, u, v)
	int N;	// sample #
	int C;	// class #

	// options
	int _frame_width;
	int _frame_height;
	int _window_length;	// odd number
	int _max_tree_depth;
	int _min_node_size;
	int _num_testing_candidates;
	int _num_threshold_candidates;

	int frame_size;  // _frame_width x _frame_height
	int window_size; // _window_length^2

	bool DetermineBestCut_DFS(const TreeIter &iter);
	void SplitNode(const TreeIter &iter);

	void SampleAndSortThresholds(const TreeIter &iter, int var1, int var2, vector<double> &thresholds);
	int FetchClass(int sample_idx);
	double FeatureFunction(int sample_idx, int var1, int var2);
	double FeatureFunction(const GMatrix &depthMap, int x, int y, int var1, int var2);
	double InformationGain(const int *l_histogram, int l_n, const int *r_histogram, int r_n);

	friend std::ostream& operator<<(std::ostream& os, const GDecisionTree& dt);
	friend std::istream& operator>>(std::istream& is, GDecisionTree& dt);
	friend std::ostream& operator<<(std::ostream& os, const SplitRule& v);
	friend std::istream& operator>>(std::istream& is, SplitRule& v);
};

class GRandomForest {

public:
	GRandomForest() : trees(NULL), _num_trees(0) {}
	~GRandomForest() { if (_num_trees) delete[] trees; }
	int GetNumClasses() const { return C; }
	//void SetTEMP_C(int C_in) { C = C_in; }		// this is just temparory
	void GRandomForest::SetOptions(int ntrees, double percent_select,
		int frame_w, int frame_h,
		int window_len, int max_tree_depth, int min_node_size,
		int n_testing_candidates, int n_threshold_candidates);
	void Learn_DFS(vector< GMatrix > *depthMap, vector< GMatrix > *labelMap, int C_in);
	void Learn_DFS(char* asf_name, char* amc_name, GViewPers view_in, int C_in);
	void Learn_DFS(char* dir_name, GViewPers view_in, int C_in);
	vector<GSkeleton> Learn_DFS_SkeletonCalibration(char* asf_dir, const GMotion &rand_poses, GViewPers view_in, int C_in);
	void Induce(const GMatrix &depthMap, GMatrix &labelMap, double *probMap) const;
	void Induce(int t, double *probMap);

	void DrawDepthPoints_RF(int t) const;

	int  LoadDepthMapMat(char *fname);
	void SaveInductionResult(char *fname);

	int getTreeNum() const { return _num_trees; }
	int getFrameWidth() const { return _frame_width; }
	int getFrameHeight() const { return _frame_height; }
	const GDecisionTree *getDecisionTree(int i) const
	{
		return &(trees[i]);
	}
	void setLabelMap(const vector<GMatrix> &mat)
	{
		_labelMap = mat;
	}


private:
	GDecisionTree *trees;

	// options
	int _num_trees;
	double _percentage_of_selection;	// 0~1, by how much percentage a pixel will be included in the learning
	int _frame_width;
	int _frame_height;

	int C;

	// internal for Learn_DFS()
	vector< GMatrix > _depthMap;
	vector< GMatrix > _labelMap;
	GViewPers view;

	// internal vectors for Project()
	vector<float> data_depth;
	vector<float> data_color;		// vector<char> will humiliate you!

	void Project(int t, const GSkeleton& sk);

	friend std::ostream& operator<<(std::ostream& os, const GRandomForest& rf);
	friend std::istream& operator>>(std::istream& is, GRandomForest& rf);
};

#endif



