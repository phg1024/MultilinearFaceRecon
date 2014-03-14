#pragma once

// settings and parameters for multilinear reconstruction
#include "phgutils.h"
#include "Utils/utility.hpp"

class MRSetting
{
public:
	typedef PhGUtils::param_t param_t;

	MRSetting(void);
	~MRSetting(void);

	bool load(const string& filename);
	bool save(const string& filename);

	void print();

	template <typename T>
	tuple<bool, T> getParam(const string& name) {
		map<string, param_t>::iterator it = params.find(name);
		if( it != params.end() ) {
			return make_tuple(true, (*it).second.getValue<T>());
		}
		else {
			return make_tuple(false, T(0));
		}
	}
	
private:
	map<string, param_t> params;
};

