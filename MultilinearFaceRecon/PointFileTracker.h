#pragma once

#include "phgutils.h"

class PointFileTracker
{
public:
  PointFileTracker();
  ~PointFileTracker();

  void reset() {}
  void resize(int w, int h) {}
  float getTrackingError() const { return 0; }

  const vector<float>& track(const unsigned char *cimg, const unsigned char *dimg, int w, int h, const string &filename);
  void printTimeStats(){}  

private:
  vector<float> fpts;
};

