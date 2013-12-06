#pragma once

#include "phgutils.h"
#include "../AAMModule/AAM_Detection_Combination.h"

// an interface layer for AAM

class AAMWrapper
{
public:
	AAMWrapper(void);
	~AAMWrapper(void);

	// tracking interface
	void track();

protected:
	void setup();

private:
	AAM_Detection_Combination *engine;
};

