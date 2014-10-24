#pragma once

#include "phgutils.h"

template <class Engine>
class FaceAligner
{
public:
  FaceAligner();
  ~FaceAligner();

  void setImageWidth(int width){ w = width; }
  void setImageHeight(int height){ h = height; }
  const vector<float>& track(const unsigned char* cimg,
                             const unsigned char* dimg);

  void reset();
protected:
  void initializeEngine();

private:
  int w, h;
  unique_ptr<Engine> engine;
};

template <typename Engine>
FaceAligner<Engine>::FaceAligner()
{
  initializeEngine();
}

template <typename Engine>
FaceAligner<Engine>::~FaceAligner()
{
}

template <typename Engine>
void FaceAligner<Engine>::initializeEngine() {
  engine.reset(new Engine());
}

template <typename Engine>
void FaceAligner<Engine>::reset() {
  engine->reset();
}

template <typename Engine>
const vector<float>& FaceAligner<Engine>::track(const unsigned char* cimg,
  const unsigned char* dimg) {
  return engine->track(cimg, dimg, w, h);
}

