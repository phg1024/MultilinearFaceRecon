#pragma once

#include "Geometry/geometryutils.hpp"

struct PointConstraint {
  PointConstraint() :weight(1.0){}

  int vidx; // vertex index on the mesh
  float weight;

  bool hasSameId(const PointConstraint &other) {
    return vidx == other.vidx;
  }
};

struct Constraint_2D : PointConstraint {
  Constraint_2D():PointConstraint(){}
  ~Constraint_2D(){}

  PhGUtils::Point2f q;  // 2D point

  float residue(const PhGUtils::Point3f &p, const PhGUtils::Matrix3x3f &Mproj) const {
    float u, v;
    PhGUtils::projectPoint(u, v, p.x, p.y, p.z, Mproj);
    float dx = q.x - u, dy = q.y - v;
    return sqrt(dx*dx + dy*dy) * this->weight;
  }
};

struct Constraint_2DAndDepth : public PointConstraint{
  Constraint_2DAndDepth():PointConstraint(){}
  ~Constraint_2DAndDepth(){}

  PhGUtils::Point3f q;  // 2D point plus depth
  float w_depth;

  float residue(const PhGUtils::Point3f &p, const PhGUtils::Matrix3x3f &Mproj) {
    float u, v, d;
    PhGUtils::projectPoint(u, v, d, p.x, p.y, p.z, Mproj);
    float dx = q.x - u, dy = q.y - v, dz = q.z - d;
    return (sqrt(dx*dx + dy*dy) + fabs(dz) * w_depth) * this->weight;
  }
};

struct Constraint_3D : public PointConstraint{
  Constraint_3D():PointConstraint(){}
  ~Constraint_3D(){}

  PhGUtils::Point3f q;  // 3D point

  float residue(const PhGUtils::Point3f &p) {
    return p.distanceTo(q) * this->weight;
  }
};