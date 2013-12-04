
#ifndef GRANDOM_H
#define GRANDOM_H

#include <stdlib.h>
#include <cmath>
#include <vector>

// double [0,1]
inline double RandDouble_c01c() {
	return ((double)rand()) / RAND_MAX;
}

// double [0,1)
inline double RandDouble_c01o() {
	return ((double)rand()) / (RAND_MAX + 1);
}

// double [A,B]
inline double RandDouble_cABc(double A, double B) {
	return ((double)rand()) / RAND_MAX * (B - A) + A;
}

// double [A,B)
inline double RandDouble_cABo(double A, double B) {
	return ((double)rand()) / (RAND_MAX + 1) * (B - A) + A;
}

// int [A,B]
inline int RandInt_cABc(int A, int B) {
	return (int)floor( ((double)rand()) / (RAND_MAX + 1) * (B + 1 - A) ) + A;
}

// vector of int [A,B]
inline void RandIntVector_cABc(int A, int B, std::vector<int> &ret) {
	double tmp = (B + 1 - A) / double(RAND_MAX + 1);
	for (int i = 0; i < (int)ret.size(); i++) {
		ret[i] = (int)floor( ((double)rand()) * tmp ) + A;
	}
}

// return K numbers from 1~N without replacement
// ***** NOTE: this can be faster with a hash-table or BST. *******
inline void RandSample(int N, int K, std::vector<int> &ret) {
	//double count=0;
	//count+=rand()%500000;
	//srand(GetTickCount()+count);
//	srand(time(NULL));
	ret.resize(K);
	int k = 0;
	while (k < K) {
		int tmp = RandInt_cABc(1, N);
		bool found = false;
		for (int i = 0; i < k; i++) {
			if (ret[i] == tmp) {
				found = true;
				break;
			}
		}
		if (!found) {
			ret[k++] = tmp;
		}
	}
}

// return K numbers from 0~N-1 without replacement
// ***** NOTE: this can be faster with a hash-table or BST. *******
inline void RandSample_V1(int N, int K, std::vector<int> &ret) {
	//double count=0;
	//count+=rand()%500000;
	//srand(GetTickCount()+count);
	//	srand(time(NULL));
	ret.resize(K);
	int k = 0;
	while (k < K) {
		int tmp = RandInt_cABc(0, N-1);
		bool found = false;
		for (int i = 0; i < k; i++) {
			if (ret[i] == tmp) {
				found = true;
				break;
			}
		}
		if (!found) {
			ret[k++] = tmp;
		}
	}
}




#endif