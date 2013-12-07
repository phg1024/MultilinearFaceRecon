#include "HashTable.h"

HashTable::HashTable()
{
	mappings=new GeoHash;
}

bool HashTable::insert(float *pos)
{
	map<float,float> basis;
	basis.insert(make_pair(pos[0],pos[1]));

	GeoHash::iterator iter;
	iter = mappings->find(basis);
	if (iter==mappings->end())
	{
		int cid=mappings->size();
		mappings->insert(make_pair(basis,cid));
		return true;
	}


	return false;


	
}

GeoHash::iterator HashTable::isInside(float *pos)
{
	map<float,float> basis;
	basis.insert(make_pair(pos[0],pos[1]));

	GeoHash::iterator iter;
	iter = mappings->find(basis);
	return iter;
}

GeoHash::iterator *HashTable::isInsideFast(float *pos)
{
	map<float,float> basis;
	basis.insert(make_pair(pos[0],pos[1]));

	GeoHash::iterator iter;
	iter = mappings->find(basis);
	return &iter;
}