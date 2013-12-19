#include "Hash_Table_Fast.h"

HashTableFast::HashTableFast()
{
	//mappings=new GeoHash;
	//curID=1;
	pair<float,float> emptyPair=make_pair(-1,-1);
	mappings.set_empty_key(emptyPair);


}

bool HashTableFast::insert(float *pos)
{
	//cout<<mappings.size()<<endl;
	pair<float,float> tmpPair=make_pair(pos[0],pos[1]);
	bool nonExist=(mappings.find(tmpPair)==mappings.end());
	

	if (nonExist)
	{
		int curID=mappings.size();
		mappings[tmpPair]=curID;
		
		return true;
	}

	return false;
	//map<float,float> basis;
	//basis.insert(make_pair(pos[0],pos[1]));

	//GeoHash::iterator iter;
	//iter = mappings->find(basis);
	//if (iter==mappings->end())
	//{
	//	int cid=mappings->size();
	//	mappings->insert(make_pair(basis,cid));
	//	return true;
	//}


	//return false;



}

int HashTableFast::isInside(float *pos)
{
	pair<float,float> tmpPair=make_pair(pos[0],pos[1]);
	

	GeoHashFast::iterator iter=mappings.find(tmpPair);
	if (iter!=mappings.end())
	{
		return iter->second;
	}
	return -1;
	/*map<float,float> basis;
	basis.insert(make_pair(pos[0],pos[1]));

	GeoHash::iterator iter;
	iter = mappings->find(basis);
	return iter;*/
}

//HashTableFast::iterator *HashTableFast::isInsideFast(float *pos)
//{
//	map<float,float> basis;
//	basis.insert(make_pair(pos[0],pos[1]));
//
//	GeoHash::iterator iter;
//	iter = mappings->find(basis);
//	return &iter;
//}