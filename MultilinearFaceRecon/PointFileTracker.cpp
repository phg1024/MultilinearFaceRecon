#include "PointFileTracker.h"


PointFileTracker::PointFileTracker()
{
}


PointFileTracker::~PointFileTracker()
{
}

const vector<float>& PointFileTracker::track(const unsigned char *cimg, const unsigned char *dimg, int w, int h, const string &filename)
{
  ifstream fin(filename);
  int npts;
  fin >> npts;
  fpts.resize(npts * 2);
  for (int i = 0; i < npts; ++i) {
    fin >> fpts[i] >> fpts[i + npts];
    cout << fpts[i] << ", " << fpts[i + npts] << endl;
  }
  return fpts;
}
