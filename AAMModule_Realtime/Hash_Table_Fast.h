#ifndef HASHTABLEFAST_H
#define HASHTABLEFAST_H
#include <sparsehash/dense_hash_map>
#include <iostream>
#include <map>


using namespace std;
using google::dense_hash_map;
using tr1::hash;


//using std::dense_hash_map;      // namespace where class lives by default

struct eqstr
{
	bool operator()(const pair<float,float> s1, const pair<float,float> s2) const
	{
		return s1.first==s2.first&&s1.second==s2.second;
	}
};

struct hash_pair {
	template <typename T, typename U>
	std::size_t operator ()(std::pair<T, U> const& p) const {
		//   using std::hash;
		return hash<T>()(p.first) ^ hash<T>()(p.second);
	}
};

typedef dense_hash_map<pair<float,float>, int, hash_pair, eqstr> GeoHashFast;

class HashTableFast
{
public:
	HashTableFast();
	//int curID;
	GeoHashFast mappings;

	//int curSize;
	bool insert(float * pos);
	int isInside(float *pos);
//	int *isInsideFast(float *pos);
	//GeoHash *mappings;
protected:
private:
};


#endif