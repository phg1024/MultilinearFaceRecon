//
// coded by Xiaolin (Kevin) Wei
//

#include "GDecisionTree.h"
#include "G/GRandom.h"
//#include "G/GViewpoint.h"


#include <windows.h>

#include <vector>
#include <stack>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <string>
using namespace std;

#include "matrix.h"
#include "mat.h"
#include <ctime>

#define myround(x) int(floor(float(x)+0.5))

#define BACKGROUND_DEPTH 10
#define BACKGROUND_CLASS 0

void GDecisionTree::SetOptions(int frame_width, int frame_height, 
							   int window_length, int max_tree_depth, int min_node_size, 
							   int num_testing_candidates, int num_threshold_candidates ) {
	_frame_width = frame_width;
	_frame_height = frame_height;
	_window_length = window_length;
	_max_tree_depth = max_tree_depth;
	_min_node_size = min_node_size;
	_num_testing_candidates = num_testing_candidates;
	_num_threshold_candidates = num_threshold_candidates;

	frame_size = _frame_width * _frame_height;
	window_size = _window_length * _window_length;
}

void GDecisionTree::Learn_DFS(vector< GMatrix > *depthMap, vector< GMatrix > *labelMap, 
							  const vector<int> &dataSamples, int C_in) {
	// set training data
	_depthMap = depthMap;
	_labelMap = labelMap;
	_dataSamples = &dataSamples[0];
	N = int(dataSamples.size() / 3);
	C = C_in;

	// initialize 'work'
	tree.SetRoot(SplitRule());
	TreeIter cur = tree.RootIter();
	cur->work.PushBack(0);	// depth
	cur->work.PushBack(0);	// purity
	for (int i = 0; i < N; i++) {
		cur->work.PushBack(i);
	}

	// DFS
	stack<TreeIter> s;
	s.push(cur);
	while(!s.empty()) {
		cur = s.top();  s.pop();
		int itmp = cur->work.Size() - 2;
		bool is_split = DetermineBestCut_DFS(cur);
		if (is_split) {
			SplitNode(cur);
			if (itmp > 10000) {
				printf("%d => %d | %d\n", itmp, cur.R()->work.Size() - 2, cur.L()->work.Size() - 2);
			}
			s.push(cur.R());
			s.push(cur.L());
		} else {
			//printf("%d \t\t\t=== %d \t", 
			//	itmp, cur->max_prob_class);
			//for (int i = 0; i < round(cur->prob[cur->max_prob_class]*10); i++) {
			//	printf("|");
			//}
			//printf("\n");
		}
	}
}

bool GDecisionTree::DetermineBestCut_DFS(const TreeIter &iter) {
	// depth?, size?, purity?
	int tree_depth = iter->work.PopFront();
	int is_pure = iter->work.PopFront();
	int data_size = iter->work.Size();
	if (tree_depth >= _max_tree_depth
		|| data_size <= _min_node_size
		|| is_pure) {
		// form 'prob' and delete 'work'
		iter->prob = vector<double>(C, 0);
		int xxx = 0;
		for (ListIter li = iter->work.Begin(); li != iter->work.End();) {
			int sample_idx = iter->work.PopFront(li);
			//printf("%d %d\n", xxx++, sample_idx);
			iter->prob[FetchClass(sample_idx)]++;
		}
		double max_prob = 0;
		for (int c = 0; c < C; c++) {
			iter->prob[c] /= data_size;
			if (iter->prob[c] > max_prob) {
				max_prob = iter->prob[c];
				iter->max_prob_class = c;
			}
		}
		return false;
	}

	// sample testing certerions
	vector<int> tests(_num_testing_candidates * 2);
	RandIntVector_cABc(0, window_size, tests);

	// look for the best cut
	vector<int> l_histogram(_num_threshold_candidates * C);
	vector<int> r_histogram(_num_threshold_candidates * C);
	vector<int> l_n(_num_threshold_candidates);
	double min_tests_info = DBL_MAX;
	vector<double> thresholds;
	for (int i = 0; i < (int)tests.size(); i+=2) {
		// sample thresholds
		SampleAndSortThresholds(iter, tests[i], tests[i + 1], thresholds);

		memset(&l_n[0], 0, _num_threshold_candidates * sizeof(int));
		memset(&l_histogram[0], 0, _num_threshold_candidates * C * sizeof(int));
		memset(&r_histogram[0], 0, _num_threshold_candidates * C * sizeof(int));

		// build histograms from training data
		for (ListIter li = iter->work.Begin(); li != iter->work.End(); ++li) {
			int c = FetchClass(*li);
			double feature_value = FeatureFunction(*li, tests[i], tests[i + 1]);
			bool always_left = false;
			for (int k = 0; k < (int)thresholds.size(); k++) {
				if (always_left || feature_value < thresholds[k]) {
					l_histogram[k * C + c]++;
					l_n[k]++;
					always_left = true;
				} else {
					r_histogram[k * C + c]++;
				}
			}
		}

		// determine the best cut for this testing
		double info;
		double min_info = DBL_MAX;
		double min_threshold = -1010;
		for (int k = 0; k < (int)thresholds.size(); k++) {
			if (l_n[k] == 0 || data_size - l_n[k] == 0) {
				continue;
			}
			info = InformationGain(&l_histogram[k * C], l_n[k], 
				&r_histogram[k * C], data_size - l_n[k]);
			if (info < min_info) {
				min_info = info;
				min_threshold = thresholds[k];
			}
		}

		if (min_info < min_tests_info) {
			min_tests_info = min_info;
			iter->var1 = tests[i];
			iter->var2 = tests[i + 1];
			iter->bestcut = min_threshold;
		}
	}

	// recover
	iter->work.PushFront(is_pure);
	iter->work.PushFront(tree_depth);

	return true;
}

void GDecisionTree::SplitNode(const TreeIter &iter) {
	TreeIter l = iter.InsertL(SplitRule());
	TreeIter r = iter.InsertR(SplitRule());

	// tree depth
	int tree_depth = iter->work.PopFront() + 1;
	iter->work.PopFront();
	
	// split work list
	double threshold = iter->bestcut;
	int var1 = iter->var1;
	int var2 = iter->var2;
	bool l_purity(true), r_purity(true);
	int l_class(-1), r_class(-1);
	for (ListIter i = iter->work.Begin(); i != iter->work.End();) {
		int sample_idx = iter->work.PopFront(i);
		if (FeatureFunction(sample_idx, var1, var2) < threshold) {
			l->work.PushBack(sample_idx);
			if (l_purity) {
				if (l_class == -1) {
					l_class = FetchClass(sample_idx);
				} else if (l_class != FetchClass(sample_idx)) {
					l_purity = false;
				}
			}
		} else {
			r->work.PushBack(sample_idx);
			if (r_purity) {
				if (r_class == -1) {
					r_class = FetchClass(sample_idx);
				} else if (r_class != FetchClass(sample_idx)) {
					r_purity = false;
				}
			}
		}
	}

	assert(l->work.Size());
	assert(r->work.Size());

	l->work.PushFront(l_purity?1:0);
	l->work.PushFront(tree_depth);
	r->work.PushFront(r_purity?1:0);
	r->work.PushFront(tree_depth);
}

void GDecisionTree::SampleAndSortThresholds(const TreeIter &iter, 
											int var1, int var2,
											vector<double> &thresholds) {
	int data_size = iter->work.Size();
	if (data_size <= _num_threshold_candidates) {
		thresholds.resize(data_size);
		int kk = 0;
		for (ListIter li = iter->work.Begin(); li != iter->work.End(); ++li) {
			thresholds[kk++] = FeatureFunction(*li, var1, var2);
		}
	} else {
		thresholds.resize(_num_threshold_candidates);
		vector<int> tmp_vec;
		RandSample(data_size, _num_threshold_candidates, tmp_vec);
		sort(tmp_vec.begin(), tmp_vec.end());
		int kk = 0, h = 1;
		for (ListIter li = iter->work.Begin(); li != iter->work.End(); ++li, h++) {
			if (tmp_vec[kk] == h) {
				thresholds[kk] = FeatureFunction(*li, var1, var2);
				kk++;
			}
			if (kk >= (int)tmp_vec.size()) {
				break;
			}
		}
	}
	sort(thresholds.begin(), thresholds.end());
	unique(thresholds.begin(), thresholds.end());
}

double GDecisionTree::FeatureFunction(int sample_idx, int var1, int var2) {
	assert(sample_idx < N);

	int frame_idx = _dataSamples[sample_idx * 3];
	int x = _dataSamples[sample_idx * 3 + 1];
	int y = _dataSamples[sample_idx * 3 + 2];

	return FeatureFunction((*_depthMap)[frame_idx], x, y, var1, var2);
}

int g_printf_feature_function;

double GDecisionTree::FeatureFunction(const GMatrix &depthMap, int x, int y, int var1, int var2) {
	double dx = depthMap(x,y);
	double half_dx = dx / 2.0f;

	int half_window_length = (_window_length - 1) / 2;
	int x1 = x + myround(((var1 / _window_length) - half_window_length) / (half_dx));
	int y1 = y + myround(((var1 % _window_length) - half_window_length) / (half_dx));
	int x2 = x + myround(((var2 / _window_length) - half_window_length) / (half_dx));
	int y2 = y + myround(((var2 % _window_length) - half_window_length) / (half_dx));

	double d1, d2;
	if (x1 >= 0 && x1 < _frame_width && y1 >= 0 && y1 < _frame_height) {
		d1 = depthMap(x1,y1);
	} else {
		d1 = BACKGROUND_DEPTH;
	}
	if (x2 >= 0 && x2 < _frame_width && y2 >= 0 && y2 < _frame_height) {
		d2 = depthMap(x2,y2);
	} else {
		d2 = BACKGROUND_DEPTH;
	}

	if(g_printf_feature_function == 7)
	{
		printf("%d %d %d %d %.03f %0.3f\n", x1, y1, x2, y2, d1, d2);
		printf("%d %d %d %d %d %lf %lf %d\n", y, var1, var2, _window_length, half_window_length, dx, half_dx, myround(((var2 % _window_length) - half_window_length) / (half_dx)));
	}

	return d1 - d2;
}

double GDecisionTree::InformationGain(const int *l_histogram, int l_n, const int *r_histogram, int r_n) {
	assert(l_n);
	assert(r_n);
	double l_p, r_p;
	double l_entropy(0), r_entropy(0);
	for (int c = 0; c < C; c++) {
		if (l_histogram[c]) {
			l_p = l_histogram[c] / (double)l_n;
			l_entropy -= l_p * log(l_p) / log(2.0);
		}
		if (r_histogram[c]) {
			r_p = r_histogram[c] / (double)r_n;
			r_entropy -= r_p * log(r_p) / log(2.0);
		}
	}

	int total_n = l_n + r_n;
	return (l_n / (double)total_n) * l_entropy + (r_n / (double)total_n) * r_entropy;
}

int GDecisionTree::FetchClass(int sample_idx) {
	assert(sample_idx < N);
	int frame_idx = _dataSamples[sample_idx * 3];
	int x = _dataSamples[sample_idx * 3 + 1];
	int y = _dataSamples[sample_idx * 3 + 2];
	int label = (int)(*_labelMap)[frame_idx](x,y);
	assert(label <= C && label > 0);
	return label - 1;
}

int g_cur_tree_idx;

void GDecisionTree::Induce(const GMatrix &depthMap, int frame_idx, double *accumulated_probMap) {

	TreeIter iter = tree.RootIter();
	int level = 0;
	while (iter.HasL() && iter.HasR()) {
		if (FeatureFunction(depthMap, frame_idx/_frame_height, frame_idx%_frame_height, 
			iter->var1, iter->var2) < iter->bestcut) {
			//if(frame_idx == 15145/* && g_cur_tree_idx == 8 && level == 7*/)
			//{
			//	g_printf_feature_function = 0;//level;
			//	double ffv = FeatureFunction(depthMap, frame_idx/_frame_height, frame_idx%_frame_height, 
			//		iter->var1, iter->var2);
			//	printf("%0.2lf ", ffv);
			//	g_printf_feature_function = 0;
			//}
			iter.GoToL();
		} else {
			//if(frame_idx == 15145 /*&& g_cur_tree_idx == 8 && level == 7*/)
			//{
			//	g_printf_feature_function = 0;//level;
			//	double ffv = FeatureFunction(depthMap, frame_idx/_frame_height, frame_idx%_frame_height, 
			//		iter->var1, iter->var2);
			//	printf("%0.2lf ", ffv);
			//	g_printf_feature_function = 0;
			//}
			iter.GoToR();
		}
		level++;
	}
	//if(frame_idx == 15145 /*&& g_cur_tree_idx == 8*/)
	//{
	//	printf("\n");
	//}
//	printf("\n");
	//if(frame_idx == 309792/21)
	//{
	//	for(int ii = 0; ii < 21; ii++)
	//	{
	//		printf("%0.2f ", iter->prob[ii]);
	//	}
	//	printf("\n");
	//}
	for (int c = 0; c < C; c++) {
		accumulated_probMap[c] += (float)(iter->prob[c]);
//		printf("%0.3lf ", iter->prob[c]);
//		printf("%0.3lf ", accumulated_probMap[c]);
	}
//	printf("\n");
}

void GDecisionTree::TreeIterToRoot() const
{
	tree_iter_ = tree.RootIter();
}

void GDecisionTree::TreeIterToLeft() const
{
	tree_iter_ = tree_iter_.GoToL();
}

void GDecisionTree::TreeIterToRight() const
{
	tree_iter_ = tree_iter_.GoToR();
}

void GDecisionTree::TreeIterToParent() const
{
	tree_iter_ = tree_iter_.GoToParent();
}

int GDecisionTree::getCurNodeVar1() const
{
	return tree_iter_->var1;
}

int GDecisionTree::getCurNodeVar2() const
{
	return tree_iter_->var2;
}

double GDecisionTree::getCurNodeBestcut() const
{
	return tree_iter_->bestcut;
}

int GDecisionTree::getCurNodeProbCount() const
{
	return tree_iter_->prob.size();
}

const double* GDecisionTree::getCurNodeProb() const
{
	if(tree_iter_->prob.size() == 0)
	{
		return NULL;
	}
	return &(tree_iter_->prob[0]);
}

bool GDecisionTree::TreeIterHasLeft() const
{
	return tree_iter_.HasL();
}

bool GDecisionTree::TreeIterHasRight() const
{
	return tree_iter_.HasR();
}

int GDecisionTree::getTreeNodeList( vector<GDecisionTree::NodeListData> *outNodeList ) const
{
	outNodeList->clear();
	outNodeList->reserve(100000);
	TreeIter iter = tree.RootIter();
	outNodeList->push_back(NodeListData(iter, -1, true));
	vector<NodeListData>::iterator li = outNodeList->begin();

	NodeListData curr_node;
	int cur_idx = 0;
	while(li != outNodeList->end())
	{
		if(li->treePointer.HasL())
		{
			iter = li->treePointer.L();
			outNodeList->push_back(NodeListData(iter, cur_idx, true));
		}
		if(li->treePointer.HasR())
		{
			iter = li->treePointer.R();
			outNodeList->push_back(NodeListData(iter, cur_idx, false));
		}
		li++;
		cur_idx++;
	}
	return cur_idx;
}

std::ostream& operator<<(std::ostream& os, const GDecisionTree& dt) {
	os<<"N\t"<<dt.N<<endl;
	os<<"C\t"<<dt.C<<endl;
	os<<"_frame_width\t"<<dt._frame_width<<endl;
	os<<"_frame_height\t"<<dt._frame_height<<endl;
	os<<"_window_length\t"<<dt._window_length<<endl;
	os<<"_max_tree_depth\t"<<dt._max_tree_depth<<endl;
	os<<"_min_node_size\t"<<dt._min_node_size<<endl;
	os<<"_num_testing_candidates\t"<<dt._num_testing_candidates<<endl;
	os<<"_num_threshold_candidates\t"<<dt._num_threshold_candidates<<endl;
	os<<dt.tree;
	return os;
}

std::istream& operator>>(std::istream& is, GDecisionTree& dt) {
	string stmp;
	is>>stmp>>dt.N
		>>stmp>>dt.C
		>>stmp>>dt._frame_width
		>>stmp>>dt._frame_height
		>>stmp>>dt._window_length
		>>stmp>>dt._max_tree_depth
		>>stmp>>dt._min_node_size
		>>stmp>>dt._num_testing_candidates
		>>stmp>>dt._num_threshold_candidates;
	is>>dt.tree;
	return is;
}

std::ostream& operator<<(std::ostream& os, const GDecisionTree::SplitRule& v) {
	if (v.prob.empty()) {
		os<<"nonleaf "<<v.var1<<' '<<v.var2<<' '<<v.bestcut;
	} else {
		os<<"leaf "<<v.max_prob_class<<"    "<<v.prob.size()<<"  ";
		for (int c = 0; c < (int)v.prob.size(); c++) {
			os<<' '<<v.prob[c];
		}
	}
	return os;
}

std::istream& operator>>(std::istream& is, GDecisionTree::SplitRule& v) {
	string stmp;
	is>>stmp;
	if (stmp == "leaf") {
		int itmp;
		is>>v.max_prob_class>>itmp;
		v.prob.resize(itmp);
		for (int c = 0; c < (int)v.prob.size(); c++) {
			is>>v.prob[c];
		}
	} else {
		is>>v.var1>>v.var2>>v.bestcut;
	}
	return is;
}

void GRandomForest::SetOptions(int num_trees, double percentage_of_selection,
							   int frame_width, int frame_height,
							   int window_length, int max_tree_depth, int min_node_size,
							   int num_testing_candidates, int num_threshold_candidates)
{
	_num_trees = num_trees;
	trees = new GDecisionTree[_num_trees];

	_percentage_of_selection = percentage_of_selection;

	_frame_width = frame_width;
	_frame_height = frame_height;
	
	for (int k = 0; k < _num_trees; k++) {
		trees[k].SetOptions(frame_width, frame_height,
			window_length, max_tree_depth, min_node_size, 
			num_testing_candidates, num_threshold_candidates);
	}
}

void GRandomForest::Learn_DFS(vector< GMatrix > *depthMap, vector< GMatrix > *labelMap, int C_in)
{
	C = C_in;

	vector< vector<int> > dataSamples(_num_trees);
	for (int k = 0; k < _num_trees; k++) {
		dataSamples[k].reserve(3000);
	}

	// sample training data into different trees, dataSamples
	for (int t = 0; t < (int)depthMap->size(); t++) {
		for (int i = 0; i < _frame_width; i++) {
			for (int j = 0; j < _frame_height; j++) {
				if ((*depthMap)[t](i,j) == BACKGROUND_DEPTH) {
					continue;
				}
				int itmp = RandInt_cABc(0, int(_num_trees / _percentage_of_selection));
				if (itmp >= _num_trees) {	// not used for learning
					continue;
				}
				dataSamples[itmp].push_back(t);
				dataSamples[itmp].push_back(i);
				dataSamples[itmp].push_back(j);
			}
		}
	}

	for (int k = 0; k < _num_trees; k++) {
		printf("%d: %d\n", k, dataSamples[k].size() / 3);
	}
	printf("\n");

	// learn
	for (int k = 0; k < _num_trees; k++) {
		printf("%d\n", k);
		trees[k].Learn_DFS(depthMap, labelMap, dataSamples[k], C);
	}
}

void GRandomForest::Learn_DFS(char* asf_name, char* amc_name, GViewPers view_in, int C_in)
{
	view = view_in;

	// load motion
	GSkeleton sk;
	sk.LoadASF(asf_name);
	GMotion mot = sk.LoadAMC(amc_name);

	// initialize for synthsis
	int T = mot.size();
	_depthMap.resize(T);
	_labelMap.resize(T);
	data_depth = vector<float>(view.npixel_v * view.npixel_u, 0);
	data_color = vector<float>(view.npixel_v * view.npixel_u, 0);

	// synthsize
	for (int t=0; t<T; t++) {
		sk.UpdatePose(mot[t]);
		Project(t, sk);
	}

	// learn
	Learn_DFS(&_depthMap, &_labelMap, C_in);
}

void GRandomForest::Learn_DFS( char* dir_name, GViewPers view_in, int C_in ) 
{
	view = view_in;

	string dir(dir_name), stmp;
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;
	
	// sk
	GSkeleton sk;
	stmp = dir + "/*.asf";
	hFind = FindFirstFile(stmp.c_str(), &ffd);
	if (INVALID_HANDLE_VALUE == hFind) {
		printf("FindFirstFile(): no such asf file!\n");
		return;
	}
	printf("%s\n", ffd.cFileName);
	string asf_name = dir + "/" + string(ffd.cFileName);
	sk.LoadASF(asf_name.c_str());

	// initialize for synthsis
	data_depth = vector<float>(view.npixel_v * view.npixel_u, 0);
	data_color = vector<float>(view.npixel_v * view.npixel_u, 0);
	_depthMap.resize(0);
	_labelMap.resize(0);

	// amc
	GMotion mot;
	stmp = dir + "/*.amc";
	hFind = FindFirstFile(stmp.c_str(), &ffd);
	do {
		printf("%s\n", ffd.cFileName);
		string amc_name = dir + "/" + string(ffd.cFileName);
		mot = sk.LoadAMC(amc_name.c_str());

		int old_size = _depthMap.size();
		int T = old_size + mot.size();
		_depthMap.resize(T);
		_labelMap.resize(T);

		// synthsize
		for (int t = old_size; t < T; t++) {
			sk.UpdatePose(mot[t - old_size]);
			Project(t, sk);
		}

	} while (FindNextFile(hFind, &ffd) != 0);
	FindClose(hFind);

	// learn
	Learn_DFS(&_depthMap, &_labelMap, C_in);
}

vector<GSkeleton> GRandomForest::Learn_DFS_SkeletonCalibration(char* asf_dir, const GMotion &rand_poses, GViewPers view_in, int C_in)
{
	view = view_in;

	string dir(asf_dir), stmp;
	WIN32_FIND_DATA ffd;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	// initialize for synthsis
	int T = rand_poses.size();
	_depthMap.resize(T);
	_labelMap.resize(T);
	data_depth = vector<float>(view.npixel_v * view.npixel_u, 0);
	data_color = vector<float>(view.npixel_v * view.npixel_u, 0);

	// asf
	vector<GSkeleton> sk(T);
	stmp = dir + "/*.asf";
	hFind = FindFirstFile(stmp.c_str(), &ffd);
	int t = 0;
	do {
		printf("%s\n", ffd.cFileName);
		string asf_name = dir + "/" + string(ffd.cFileName);
		sk[t].LoadASF(asf_name.c_str());

		// synthsize
		sk[t].UpdatePose(rand_poses[t]);
		Project(t, sk[t]);
		t++;
	} while (FindNextFile(hFind, &ffd) != 0);
	FindClose(hFind);

	// learn
	Learn_DFS(&_depthMap, &_labelMap, C_in);

	return sk;
}

static double my_round(double number)
{
	return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

//double my_round(double r, int precision) {
//	double pw = pow(10, precision);
//	double tr = r * pw;
//	double ret = (tr > 0.0) ? floor(tr + 0.5) : ceil(tr - 0.5);
//	ret / = pw;
//	return ret;
//}

//static clock_t InducingTime = 0;

void GRandomForest::Induce(const GMatrix &depthMap, GMatrix &labelMap, double *probMap) const
{
//	clock_t start_time = clock();
	labelMap.Resize(_frame_width, _frame_height);

	memset(probMap, 0, _frame_width * _frame_height * C * sizeof(double));
	for (int i = 0; i < _frame_width * _frame_height; i++) 
	{
//		printf("cpu frame_idx = %d, val = %0.3lf\n", i, depthMap(i));
		if (depthMap(i) == BACKGROUND_DEPTH) {
			labelMap(i) = BACKGROUND_CLASS;
			continue;
		}

		// induce
		int itmp = i * C;
//		int k = 8;
//		int idx[] = {3, 4, 5, 6, 7, 8, 9};
		
		
		//double *prob_test_1 = new double[C*10];
		//double *prob_test_2 = new double[C];
		//double *prob_test_2_sep = new double[C*10];
		//memset(prob_test_1, 0, sizeof(double)*C*10);
		//memset(prob_test_2, 0, sizeof(double)*C);
		//memset(prob_test_2_sep, 0, sizeof(double)*C*10);

		int num_tree = _num_trees;
		for (int k = 0; k < num_tree/*_num_trees*/; k++)
		{
//			if(k == 2 || k == 8) continue;
			g_cur_tree_idx = k;
			trees[k/*idx[k]*/].Induce(depthMap, i, &probMap[itmp]);
//			trees[k/*idx[k]*/].Induce(depthMap, i, &prob_test_1[k*C]);
		}

		//if(i == 15145)
		//{
		//	for(int kk = 0; kk < 21; kk++)
		//	{
		//		printf("%lf ", probMap[itmp+kk]);
		//	}
		//	printf("\n");
		//}

		//for (int k = num_tree-1; k >= 0/*_num_trees*/; k--)
		//{
		//	//			if(k == 2 || k == 8) continue;
		//	trees[k/*idx[k]*/].Induce(depthMap, i, prob_test_2);
		//	trees[k/*idx[k]*/].Induce(depthMap, i, &prob_test_2_sep[k*C]);
		//}
		//for(int iii = 0; iii < C; iii++)
		//{
		//	if(fabs(probMap[itmp+iii]-prob_test_2[iii]) > 1E-15 )
		//	{
		//		printf("POS: ");
		//		for(int jjj = 0; jjj < C; jjj++)
		//		{
		//			printf("%0.5lf ", probMap[itmp+jjj]);
		//		}
		//		printf("\n");
		//		printf("NEG: ");
		//		for(int jjj = 0; jjj < C; jjj++)
		//		{
		//			printf("%0.5lf ", prob_test_2[jjj]);
		//		}
		//		printf("\n");
		//	}
		//	break;
		//}

		float precision = 1000.0f;
		float max_prob = -1;
		for (int c = 0; c < C; c++) {
			if (my_round(probMap[itmp + c]*precision) > max_prob) {
				max_prob = my_round(probMap[itmp + c]*precision);
				labelMap(i) = c + 1;
				assert(labelMap(i) <=C && labelMap(i) > 0);
			}
			probMap[itmp + c] /= num_tree/*_num_trees*/;
		}
		
		//double labe_test;
		//double test_max = -5;
		//for(int iii = 0; iii < C; iii++)
		//{
		//	if(my_round(prob_test_2[iii]*precision) > test_max)
		//	{
		//		test_max = my_round(prob_test_2[iii]*precision);
		//		labe_test = iii+1;
		//	}
		//}
		//if(labelMap(i) != labe_test)
		//{ 
		//	printf("no identical!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		//	printf("ORG: ");
		//	for(int iii = 0; iii < C; iii++)
		//	{
		//		printf("%0.3lf ", probMap[itmp+iii]);	
		//	}
		//	printf("\n");
		//	printf("NEG: ");
		//	for(int iii = 0; iii < C; iii++)
		//	{
		//		printf("%0.3lf ", prob_test_2[iii]);	
		//	}
		//	printf("\n");
		//	printf("org lab %lf, new lab %lf\n", labelMap(i), labe_test);
		//	printf("org val %0.20lf, new val %0.20lf\n", probMap[(int)(itmp+labelMap(i)-1)], prob_test_2[(int)(labe_test-1)]);
		//	printf("sep data labemMap(i)\n");
		//	double s1, s2;
		//	s1 = 0;
		//	s2 = 0;
		//	for(int iii = 0; iii < num_tree; iii++)
		//	{
		//		int idx_old = labelMap(i)-1;
		//		int idx_new = labe_test - 1;
		//		printf("%0.20lf %0.20lf\n", prob_test_1[iii*C+idx_old], prob_test_1[iii*C+idx_new]);
		//		s1 += prob_test_1[iii*C+idx_old];
		//		s2 += prob_test_1[iii*C+idx_new];
		//	}
		//	printf("sum %0.20lf %0.20lf\n", s1, s2);
		//	s1 = 0;
		//	s2 = 0;
		//	printf("sep data labe_test\n");
		//	for(int iii = 0; iii < num_tree; iii++)
		//	{
		//		int idx_old = labelMap(i)-1;
		//		int idx_new = labe_test - 1;
		//		printf("%0.20lf %0.20lf\n", prob_test_2_sep[iii*C+idx_old], prob_test_2_sep[iii*C+idx_new]);
		//		s1 += prob_test_2_sep[iii*C+idx_old];
		//		s2 += prob_test_2_sep[iii*C+idx_new];
		//	}
		//	printf("sum %0.20lf %0.20lf\n", s1, s2);
		//	s1 = 0;
		//	s2 = 0;
		//	printf("sep data labe_test inverse\n");
		//	for(int iii = num_tree-1; iii >= 0; iii--)
		//	{
		//		int idx_old = labelMap(i)-1;
		//		int idx_new = labe_test - 1;
		//		printf("%0.20lf %0.20lf\n", prob_test_2_sep[(num_tree-iii-1)*C+idx_old], prob_test_2_sep[(num_tree-iii-1)*C+idx_new]);
		//		s1 += prob_test_2_sep[iii*C+idx_old];
		//		s2 += prob_test_2_sep[iii*C+idx_new];
		//	}
		//	printf("sum %0.20lf %0.20lf\n", s1, s2);
		//	return;
		//}

		//delete[] prob_test_1;
		//delete[] prob_test_2;
		//delete[] prob_test_2_sep;
	}
//	clock_t end_time = clock();
//	InducingTime += (end_time - start_time);
}

//void printCPUInducingTime()
//{
//	printf("Inducing time: %0.3lf\n", ((double)InducingTime)/CLOCKS_PER_SEC);
//}

void GRandomForest::Induce( int t, double *probMap )
{
	Induce(_depthMap[t], _labelMap[t], probMap);
}

void GRandomForest::Project(int t, const GSkeleton& sk)
{
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	GLfloat clearcolor[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, clearcolor);

	// set view
	view.SetUpViewport();
	glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity();
	view.SetUpProjection();
	glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
	view.SetUpModelView();

	// draw
	glDisable(GL_LIGHTING);
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	sk.Draw_CylindersBalls_Fast_Color_21();

	// read depth pixels
	glReadBuffer(GL_BACK);
	glReadPixels(0, 0, view.npixel_u, view.npixel_v, GL_DEPTH_COMPONENT, GL_FLOAT, &data_depth[0]);
	glReadPixels(0, 0, view.npixel_u, view.npixel_v, GL_RED, GL_FLOAT, &data_color[0]);

	// recover view
	glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
	glMatrixMode(GL_PROJECTION); glPopMatrix();
	glMatrixMode(GL_MODELVIEW); glPopMatrix();
	glClearColor(clearcolor[0], clearcolor[1], clearcolor[2], clearcolor[3]);

	_depthMap[t].Resize(view.npixel_u, view.npixel_v);
	_labelMap[t] = GMatrix(view.npixel_u, view.npixel_v, double(0));
	for (int j=0; j<view.npixel_v; j++)
	{
		for (int i=0; i<view.npixel_u; i++)
		{
			float ftmp = data_depth[j*view.npixel_u + i];
			if (ftmp != 1) {
				_depthMap[t](i,view.npixel_v-1-j) = view.GLz2Eyez(ftmp);
			} else {
				_depthMap[t](i,view.npixel_v-1-j) = BACKGROUND_DEPTH;
			}
			_labelMap[t](i,view.npixel_v-1-j) = int(data_color[j*view.npixel_u + i]*256);
		}
	}
}

void GRandomForest::DrawDepthPoints_RF(int t) const
{
	// for drawing round points
	glEnable(GL_ALPHA_TEST);
	glAlphaFunc(GL_NOTEQUAL, 0);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

	glPointSize(6);
	glBegin(GL_POINTS);{
		// STZPZ
		//for (int i=0; i<view.npixel_u; i++)	{
		//	for (int j=0; j<view.npixel_v; j++) {
		for (int i=0; i<_frame_width; i++)	{
			for (int j=0; j<_frame_height; j++) {
				switch (int(_labelMap[t](i,j))) {
					//case -1:
					//	continue; break;
					case 0:
						//glColor3d(0.95, 0.95, 0.95); break;
						continue; break;
					case 1:
						glColor3d(1, 0, 0); break;
					case 2:
						glColor3d(0, 1, 0); break;
					case 3:
						glColor3d(0, 0, 1); break;
					case 4:
						glColor3d(1, 1, 0); break;
					case 5:
						glColor3d(0.5, 0, 0); break;
					case 6:
						glColor3d(0, 0.5, 0); break;
					case 7:
						glColor3d(0, 0, 0.5); break;
					case 8:
						glColor3d(0.5, 0.5, 0); break;
					case 9:
						glColor3d(0, 0, 0); break;
					case 10:
						glColor3d(0, 1, 1); break;
					case 11:
						glColor3d(1, 0, 1); break;
					case 12:
						glColor3d(0.6, 1, 0.6); break;
					case 13:
						glColor3d(1, 0.6, 0.6); break;
					case 14:
						glColor3d(0.6, 0.6, 1); break;
					case 15:
						glColor3d(1, 1, 0.5); break;
					case 16:
						glColor3d(1, 0.7, 1); break;
					case 17:
						glColor3d(0.7, 0.3, 1); break;
					case 18:
						glColor3d(0.2, 0, 0.8); break;
					case 19:
						glColor3d(0.5, 0.7, 0); break;
					case 20:
						glColor3d(0.7, 0, 0.3); break;
					case 21:
						glColor3d(0.7, 0.3, 0); break;
					case -2:
						glColor3d(0.945, 0.945, 0.945); break;
					default:
						printf("what %d\n", int(_labelMap[t](i,j)));
				} // switch
				if (_depthMap[t](i,j) != BACKGROUND_DEPTH) {
					GVector3 v = view.UnProject_Eyez(GVector2(i,j), _depthMap[t](i,j));
					glVertex3dv(&v.x);
				}
			}
		}
	}glEnd();

	glDisable(GL_POINT_SMOOTH);
	glBlendFunc(GL_NONE, GL_NONE);
	glDisable(GL_BLEND);

}

int GRandomForest::LoadDepthMapMat(char *fname) {
	// read MAT file
	MATFile *pmat = matOpen(fname, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", fname);
	}

	const char *name;
	const mxArray *depthMap_mx;
	mxArray *tmp;
	for (int i=0; i<100; i++) {
		tmp = matGetNextVariable(pmat, &name);
		if (tmp == NULL) break;
		if (strcmp(name,"depthMap")==0)
			depthMap_mx = tmp;
	}
	if (matClose(pmat) != 0)
		printf("Error closing file %s\n",fname);

	// depthMap
	int T = (int)mxGetNumberOfElements(depthMap_mx);
	printf("%d\n", T);

	_depthMap.resize(T);
	for (int t=0; t<T; t++)
	{
		mxArray* tmpArray = mxGetCell(depthMap_mx,t);
		_depthMap[t].Resize(mxGetN(tmpArray),mxGetM(tmpArray));
		memcpy(_depthMap[t].GetArray(), mxGetPr(tmpArray), _depthMap[t].NumElements() * sizeof(double) );
	}

	_labelMap.resize(T);
	return T;
}

void GRandomForest::SaveInductionResult(char *fname) {
	// open MAT file
	MATFile *pmat =matOpen(fname, "w");
	if (pmat == NULL)
		printf("Error opening file %s\n", fname);

	// inducedMap
	int T = _labelMap.size();
	mxArray *hit = mxCreateCellMatrix(1,T);
	vector<mxArray*> tmpMX_inducedMap(T);
	for (int t=0; t<T; t++)
	{
		//tmpMX_inducedMap[t] = mxCreateDoubleMatrix(0,0,mxREAL);
		//mxSetM (tmpMX_inducedMap[t], _frame_height);
		//mxSetN (tmpMX_inducedMap[t], _frame_width);
		//mxSetPr(tmpMX_inducedMap[t], _labelMap[t].GetArray());
		tmpMX_inducedMap[t] = mxCreateDoubleMatrix(_frame_height, _frame_width, mxREAL);
		double *tmp = mxGetPr(tmpMX_inducedMap[t]);
		_labelMap[t].CopyToArray(tmp);
		mxSetCell(hit, t, tmpMX_inducedMap[t]);
	}
	matPutVariable(pmat, "inducedMap", hit);

	// close
	if (matClose(pmat) != 0)
		printf("Error closing file %s\n",fname);

	// destroy
	for (int t=0; t<T; t++)
	{
		mxSetM(tmpMX_inducedMap[t], 0);
		mxSetN(tmpMX_inducedMap[t], 0);
		mxSetPr(tmpMX_inducedMap[t], NULL);
	}
	mxDestroyArray(hit);
}

std::ostream& operator<<(std::ostream& os, const GRandomForest& rf) {
	os<<"_num_trees\t"<<rf._num_trees<<endl;
	os<<"_percentage_of_selection\t"<<rf._percentage_of_selection<<endl;
	os<<"_frame_width\t"<<rf._frame_width<<endl;
	os<<"_frame_height\t"<<rf._frame_height<<endl;
	os<<endl;

	for (int k = 0; k < rf._num_trees; k++) {
		os<<"Tree "<<k<<endl;
		os<<rf.trees[k];
		os<<endl;
	}
	return os;
}

std::istream& operator>>(std::istream& is, GRandomForest& rf) {
	string stmp;
	is>>stmp>>rf._num_trees
		>>stmp>>rf._percentage_of_selection
		>>stmp>>rf._frame_width
		>>stmp>>rf._frame_height;

	rf.trees = new GDecisionTree[rf._num_trees];
	for (int k = 0; k < rf._num_trees; k++) {
		is>>stmp>>stmp>>rf.trees[k];
	}
	rf.C = rf.trees[0].GetNumClasses();
	return is;
}





