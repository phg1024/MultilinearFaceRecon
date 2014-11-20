#pragma once

#include "phgutils.h"

template <class Engine>
class FaceAligner
{
public:
  FaceAligner();
  ~FaceAligner();

  void setImageSize(int width, int height) { w = width; h = height; engine->resize(w, h); }
  void setImageWidth(int width){ w = width; }
  void setImageHeight(int height){ h = height; }
  void setGroudTruthFile(const string& filename) { truthfile = filename; }
  int getImageWidth() const { return w; }
  int getImageHeight() const { return h; }
  const vector<float>& track(const unsigned char* cimg,
                             const unsigned char* dimg);
  void printTimeStats();
  void reset();
protected:
  void initializeEngine();

private:
  int w, h;
  string truthfile;
  unique_ptr<Engine> engine;
};

template <class Engine>
void FaceAligner<Engine>::printTimeStats()
{
  engine->printTimeStats();
}

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
  return engine->track(cimg, dimg, w, h, truthfile);
}

