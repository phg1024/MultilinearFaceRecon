#pragma once

#include "Math/Tensor.hpp"
#include "Geometry/geometryutils.hpp"

// the class related to the loading and processing the multilinear model
template <typename T>
struct MultilinearModel {
  MultilinearModel(){}
  MultilinearModel(const string &filename) {
    // load the model from tensor files
    core.read(filename);
  }

  MultilinearModel project(const vector<int> &indices) const {
    //cout << "creating projected tensors..." << endl;
    // create a projected version of the model
    MultilinearModel newmodel;
    newmodel.core.resize(core.dim(0), core.dim(1), indices.size() * 3);

    for (int i = 0; i < core.dim(0); i++) {
      for (int j = 0; j < core.dim(1); j++) {
        for (int k = 0, idx = 0; k < indices.size(); k++, idx += 3) {
          int vidx = indices[k] * 3;
          newmodel.core(i, j, idx) = core(i, j, vidx);
          newmodel.core(i, j, idx + 1) = core(i, j, vidx + 1);
          newmodel.core(i, j, idx + 2) = core(i, j, vidx + 2);
        }
      }
    }

    return newmodel;
  }

  void unfold() {
    tu0 = core.unfold(0);
    tu1 = core.unfold(1);
  }

  void transformTM0(const PhGUtils::Matrix3x3<T> &R) {
    int npts = tm0.dim(1) / 3;
    // don't use the assignment operator, it actually moves the data
    //tm0cRT = tm0c;
    for (int i = 0; i < tm0.dim(0); i++) {
      for (int j = 0, vidx = 0; j < npts; j++, vidx += 3) {
        tm0RT(i, vidx) = tm0(i, vidx);
        tm0RT(i, vidx + 1) = tm0(i, vidx + 1);
        tm0RT(i, vidx + 2) = tm0(i, vidx + 2);
        // rotation only!!!
        PhGUtils::rotatePoint(tm0RT(i, vidx), tm0RT(i, vidx + 1), tm0RT(i, vidx + 2), R);
      }
    }
  }

  void transformTM1(const PhGUtils::Matrix3x3<T> &R) {
    int npts = tm1.dim(1) / 3;
    // don't use the assignment operator, it actually moves the data
    //tm1cRT = tm1c;
    for (int i = 0; i < tm1.dim(0); i++) {
      for (int j = 0, vidx = 0; j < npts; j++, vidx += 3) {

        tm1RT(i, vidx) = tm1(i, vidx);
        tm1RT(i, vidx + 1) = tm1(i, vidx + 1);
        tm1RT(i, vidx + 2) = tm1(i, vidx + 2);

        // rotation only!!!
        PhGUtils::rotatePoint(tm1RT(i, vidx), tm1RT(i, vidx + 1), tm1RT(i, vidx + 2), R);
      }
    }
  }

  void updateTM0(const Tensor1<T> &w) {
    tm0 = core.modeProduct(w, 0);
  }

  void updateTM1(const Tensor1<T> &w) {
    tm1 = core.modeProduct(w, 1);
  }

  void updateTMWithMode0(const Tensor1<T> &w) {    
    tm = tm0.modeProduct(w, 0);
  }

  void updateTMWithMode1(const Tensor1<T> &w) {
    tm = tm1.modeProduct(w, 0);
  }

  void applyWeights(const Tensor1<T> &w0, const Tensor1<T> &w1) {
    //cout << "applying weights ..." << endl;
    updateTM0(w0);
    //cout << tm0.dim(0) << "x" << tm0.dim(1) << endl;
    updateTM1(w1);
    //cout << tm1.dim(0) << "x" << tm1.dim(1) << endl;
    updateTMWithMode1(w0);
    //cout << tm.length() << endl;
    //cout << "done." << endl;
  }

  // original tensors
  Tensor3<T> core;      // core tensor
  Tensor2<T> tu0, tu1;  // unfolded tensor of core
  Tensor2<T> tm0, tm1;  // tensor after mode product
  Tensor2<T> tm0RT, tm1RT;  // tensor after mode product
  Tensor1<T> tm;        // tensor after 2 mode product
};
