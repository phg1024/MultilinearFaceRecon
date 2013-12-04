#ifndef HASHTABLE_H
#define HASHTABLE_H
#include <map>
#include <vector>
using namespace std;
typedef map<map<float,float>,int> GeoHash;




class HashTable
{
public:
	HashTable();
	bool insert(float * pos);
	GeoHash::iterator isInside(float *pos);
	GeoHash::iterator *isInsideFast(float *pos);
	GeoHash *mappings;
protected:
private:
};


#endif