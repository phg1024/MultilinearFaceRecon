#include "AAM_common.h"

AAM_Common::AAM_Common()
{
	ref=new Delaunay_Tri();
	//warp=new PieceAffineWarpping();
}

AAM_Common::~AAM_Common()
{
	delete ref;
}



