#pragma once

#include "MultilinearModel.h"
#include "Geometry/geometryutils.hpp"
#include "Parameters.h"

template <typename T>
struct EnergyFunction2D {
  EnergyFunction2D():model(MultilinearModel<T>()), params(SingleImageParameters()), targets_2d(vector<Constraint_2D>()){}
  EnergyFunction2D(SingleImageParameters &params) :
    model(params.reconParams.model_projected), params(params), targets_2d(params.reconParams.cons){
    f_pose = &EnergyFunction2D::cost_pose;
    jac_pose = &EnergyFunction2D::jacobian_pose;

    f_id = &EnergyFunction2D::cost_identity;
    jac_id = &EnergyFunction2D::jacobian_identity;

    f_exp = &EnergyFunction2D::cost_expression;
    jac_exp = &EnergyFunction2D::jacobian_expression;
  }

  typedef typename T value_t;
  typedef void (EnergyFunction2D::*fptr)(T *p, T *hx, int m, int n, void* adata);

  T error() {
    int npts = targets_2d.size();
    T E = 0;
    T scale = 100.0/targets_2d[29].q.distanceTo(targets_2d[33].q);

    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();    
    auto &tmc = model.tm;
    for (int i = 0; i < npts; i++) {
      int vidx = i * 3;
      // should change this to a fast version
      PhGUtils::Point3<T> p(tmc(vidx), tmc(vidx + 1), tmc(vidx + 2));
      //cout << p << endl;
      /*p = Rmat * p + Tvec;*/
      PhGUtils::transformPoint(p.x, p.y, p.z, params.modelParams.Rmat, params.modelParams.Tvec);

      // project the point
      T u, v;
      PhGUtils::projectPoint(u, v, p.x, p.y, p.z, Mproj);

      T dx, dy;
      dx = u - targets_2d[i].q.x;
      dy = v - targets_2d[i].q.y;
      E += sqrt(dx * dx + dy * dy) * scale;
    }

    E /= npts;
    return E;
  }

  void cost_pose(T *p, T *hx, int m, int n, void* adata) {
    T s, rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto &tmc = model.tm;

    // set up rotation matrix and translation vector
    PhGUtils::Point3<T> Tvec(tx, ty, tz);
    PhGUtils::Matrix3x3<T> R = PhGUtils::rotationMatrix(rx, ry, rz);
    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();
    
    //cout << Mproj << endl;

    // compute the scale of the input
    T scale = 100.0/targets_2d[29].q.distanceTo(targets_2d[33].q);

    // apply the new global transformation
    for (int i = 0, vidx = 0; i < npts; i++) {
      //PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
      T wpt = 1.0;

      // exclude mouth region and chin region
      /*
      if (!params.fitExpression) {
        if (i >= 46 || (i >= 4 && i <= 11))
          wpt = 0.9;
      }
      else {
        if (i < 15) wpt = 1.0;
        else if (i >= 68) wpt = 0.5;
      }
      */      

      T px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      const PhGUtils::Point2f& q = targets_2d.at(i).q;

      // PhGUtils::Point3f pp = R * p + T;
      PhGUtils::transformPoint(px, py, pz, R, Tvec);
      //cout << px << " " << py << " " << pz << endl;

      // Projection to image plane
      T u, v;
      PhGUtils::projectPoint(u, v, px, py, pz, Mproj);

      //cout << i << "\t" << u << ", " << v << endl;

      T dx = q.x - u, dy = q.y - v;
      //cout << i << "\t" << dx << ", " << dy << ", " << dz << endl;
      //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
      hx[i] = sqrt(dx * dx + dy * dy) * wpt * scale;
      //cout << hx[i] << ' ';
    }
    //cout << endl;
    //system("pause");
  }

  void jacobian_pose(T *p, T *J, int m, int n, void* adata) {
    // J is a n-by-m matrix
    T rx, ry, rz, tx, ty, tz;
    rx = p[0], ry = p[1], rz = p[2], tx = p[3], ty = p[4], tz = p[5];

    int npts = targets_2d.size();
    auto &tmc = model.tm;

    // set up rotation matrix and translation vector
    PhGUtils::Matrix3x3<T> R = PhGUtils::rotationMatrix(rx, ry, rz);
    PhGUtils::Matrix3x3<T> Jx, Jy, Jz;
    PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);
    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();

    //cout << Jx << endl;
    //cout << Jy << endl;
    //cout << Jz << endl;

    // compute the scale of the input
    T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

    // for Jacobian of projection/viewport transformation
    const T f_x = params.modelParams.camparams.fx, f_y = params.modelParams.camparams.fy;
    T Jf[6] = { 0 };

    // apply the new global transformation
    for (int i = 0, vidx = 0, jidx = 0; i<npts; i++) {
      T wpt = 1.0;

      // exclude mouth region and chin region
      
      /*
      if (!params.fitExpression) {
        if (i >= 46 || (i >= 4 && i <= 11))
          wpt = 0.9;
      }
      else{
        if (i < 15) wpt = 1.0;
        else if (i >= 68) wpt = 0.5;
      }
      */
      

      // point p
      T px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
      //cout << px << ", " << py << ", " << pz << endl;

      // point q
      const PhGUtils::Point2f& q = targets_2d[i].q;

      // R * p
      T Rpx = px, Rpy = py, Rpz = pz;
      PhGUtils::rotatePoint(Rpx, Rpy, Rpz, R);

      // P = R * p + t
      T Pkx = Rpx + tx, Pky = Rpy + ty, Pkz = Rpz + tz;

      // Jf
      T inv_z = 1.0 / Pkz;
      T inv_z2 = inv_z * inv_z;

      Jf[0] = f_x * inv_z; Jf[2] = -f_x * Pkx * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * Pky * inv_z2;

      /*
      Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
      */

      // project p to color image plane
      T pu, pv;
      PhGUtils::projectPoint(pu, pv, Pkx, Pky, Pkz, Mproj);

      // residue
      T rkx = pu - q.x, rky = pv - q.y;
      T rk = sqrt(rkx*rkx + rky*rky);
      T inv_rk = 1.0 / rk;
      // normalize it
      rkx *= inv_rk; rky *= inv_rk;

      // J_? * p_k
      T jpx, jpy, jpz;
      // J_f * J_? * p_k
      T jfjpx, jfjpy;

      jpx = px, jpy = py, jpz = pz;
      PhGUtils::rotatePoint(jpx, jpy, jpz, Jx);
      jfjpx = Jf[0] * jpx + Jf[2] * jpz;
      jfjpy = Jf[4] * jpy + Jf[5] * jpz;
      /*
      cout << "jf\t";
      PhGUtils::printArray(Jf, 9);
      cout << "j\t" << jpx << ", " << jpy << ", "  << jpz << endl;
      cout << "jj\t" << jfjpx << ", " << jfjpy << ", "  << jfjpz << endl;
      */

      // \frac{\partial r_i}{\partial \theta_x}
      J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt * scale;

      jpx = px, jpy = py, jpz = pz;
      PhGUtils::rotatePoint(jpx, jpy, jpz, Jy);
      jfjpx = Jf[0] * jpx + Jf[2] * jpz;
      jfjpy = Jf[4] * jpy + Jf[5] * jpz;
      // \frac{\partial r_i}{\partial \theta_y}
      J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt * scale;

      jpx = px, jpy = py, jpz = pz;
      PhGUtils::rotatePoint(jpx, jpy, jpz, Jz);
      jfjpx = Jf[0] * jpx + Jf[2] * jpz;
      jfjpy = Jf[4] * jpy + Jf[5] * jpz;
      // \frac{\partial r_i}{\partial \theta_z}
      J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt * scale;

      // \frac{\partial r_i}{\partial \t_x}
      J[jidx++] = (Jf[0] * rkx) * wpt * scale;

      // \frac{\partial r_i}{\partial \t_y}
      J[jidx++] = (Jf[4] * rky) * wpt * scale;

      // \frac{\partial r_i}{\partial \t_z}
      J[jidx++] = (Jf[2] * rkx + Jf[5] * rky) * wpt * scale;
    }

    /*
    ofstream fout("jacobian.txt");

    for(int i=0, jidx=0;i<npts;i++) {
    for(int j=0;j<7;j++) {
    fout << J[jidx++] << '\t';
    }
    fout << endl;
    }
    fout.close();


    ::system("pause");
    */
  }

  void cost_identity(T *p, T *hx, int m, int n, void* adata) {

    int npts = targets_2d.size();
    auto tm1cRT = model.tm1RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3<T>& Tvec = params.modelParams.Tvec;
    const PhGUtils::Matrix3x3<T>& R = params.modelParams.Rmat;
    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();

    T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);
    //cout << "npts = " << npts << endl;
    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      T wpt = 1.0;

      T x = 0, y = 0, z = 0;
      for (int j = 0; j < m; j++) {
        x += tm1cRT(j, vidx) * p[j];
        y += tm1cRT(j, vidx + 1) * p[j];
        z += tm1cRT(j, vidx + 2) * p[j];
      }
      const PhGUtils::Point2f& q = targets_2d[i].q;

      T u, v;
      PhGUtils::projectPoint(u, v, x + Tvec.x, y + Tvec.y, z + Tvec.z, Mproj);

      T dx = q.x - u, dy = q.y - v;
      //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
      hx[i] = sqrt(dx*dx + dy*dy) * scale;
      //cout << hx[i] << ", ";
    }

    // regularization term
    arma::vec wmmu(m);
    for (int i = 0; i < m; ++i) {
      wmmu(i) = p[i] - params.priorParams.mu_wid_orig(i);
    }
    hx[n - 1] = sqrt(arma::dot(wmmu, params.priorParams.sigma_wid * wmmu)) * params.priorParams.w_prior_id;
    //cout << hx[n] << endl;
  }

  void jacobian_identity(T *p, T *J, int m, int n, void* adata) {

    int npts = targets_2d.size();
    auto tm1cRT = model.tm1RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3<T>& Tvec = params.modelParams.Tvec;
    const PhGUtils::Matrix3x3<T>& R = params.modelParams.Rmat;
    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();

    T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

    //cout << "hx: ";
    T Jf[6] = { 0 };
    T f_x = params.modelParams.camparams.fx, f_y = params.modelParams.camparams.fy;

    int jidx = 0;
    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      T wpt = 1.0;

      // Pk
      T Px = 0, Py = 0, Pz = 0;
      for (int j = 0; j < m; j++) {
        Px += tm1cRT(j, vidx) * p[j];
        Py += tm1cRT(j, vidx + 1) * p[j];
        Pz += tm1cRT(j, vidx + 2) * p[j];
      }
      Px += Tvec.x; Py += Tvec.y; Pz += Tvec.z;

      const PhGUtils::Point2f& q = targets_2d[i].q;
      T u, v;
      PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

      T rkx = u - q.x, rky = v - q.y;

      T rk = sqrt(rkx*rkx + rky*rky);
      T inv_rk = 1.0 / rk;
      rkx *= inv_rk; rky *= inv_rk;

      // Jf
      T inv_z = 1.0 / Pz;
      T inv_z2 = inv_z * inv_z;

      Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

      for (int j = 0; j < m; ++j) {
        // R * Jm * e_j
        T jmex = tm1cRT(j, vidx), jmey = tm1cRT(j, vidx + 1), jmez = tm1cRT(j, vidx + 2);

        // Jf * R * Jm * e_j
        float jfjmx = Jf[0] * jmex + Jf[2] * jmez;
        float jfjmy = Jf[4] * jmey + Jf[5] * jmez;
        // data term and prior term
        J[jidx++] = (jfjmx * rkx + jfjmy * rky) * scale;
      }
    }

    // regularization term
    arma::vec wmmu(m);
    for (int j = 0; j < m; ++j) {
      wmmu(j) = p[j] - params.priorParams.mu_wid_orig(j);
    }
    arma::vec Jprior = params.priorParams.sigma_wid * wmmu;
    T Eprior = sqrt(arma::dot(wmmu, Jprior));
    T inv_Eprior = fabs(Eprior) < 1e-12 ? 1e-12 : 1.0 / Eprior;

    for (int j = 0; j < m; ++j) {
      J[jidx++] = inv_Eprior * Jprior(j) * params.priorParams.w_prior_id;
    }
  }

  void cost_expression(T *p, T *hx, int m, int n, void* adata) {

    int npts = targets_2d.size();
    auto tm0cRT = model.tm0RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3<T>& Tvec = params.modelParams.Tvec;
    const PhGUtils::Matrix3x3<T>& R = params.modelParams.Rmat;
    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();

    T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      T wpt = 1.0;

      T x = 0, y = 0, z = 0;
      for (int j = 0; j < m; j++) {
        x += tm0cRT(j, vidx) * p[j];
        y += tm0cRT(j, vidx + 1) * p[j];
        z += tm0cRT(j, vidx + 2) * p[j];
      }
      const PhGUtils::Point2f& q = targets_2d[i].q;

      T u, v;
      PhGUtils::projectPoint(u, v, x + Tvec.x, y + Tvec.y, z + Tvec.z, Mproj);

      T dx = q.x - u, dy = q.y - v;
      //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
      hx[i] = sqrt(dx*dx + dy*dy) * scale;
    }

    // regularization term
    int nparams = params.priorParams.mu_wexp.n_elem;
    arma::vec wmmu(nparams);
    for (int i = 0; i < nparams; ++i) {
      wmmu(i) = p[i] - params.priorParams.mu_wexp_orig(i);
    }
    hx[n - 1] = sqrt(arma::dot(wmmu, params.priorParams.sigma_wexp * wmmu)) * params.priorParams.w_prior_exp;
  }

  void jacobian_expression(T *p, T *J, int m, int n, void* adata) {

    int npts = targets_2d.size();
    auto tm0cRT = model.tm0RT;

    // set up rotation matrix and translation vector
    const PhGUtils::Point3<T>& Tvec = params.modelParams.Tvec;
    const PhGUtils::Matrix3x3<T>& R = params.modelParams.Rmat;
    PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix();
    T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

    T Jf[6] = { 0 };
    T f_x = params.modelParams.camparams.fx, f_y = params.modelParams.camparams.fy;

    int jidx = 0;
    for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
      T wpt = 1.0;

      T x = 0, y = 0, z = 0;
      for (int j = 0; j < m; j++) {
        x += tm0cRT(j, vidx) * p[j];
        y += tm0cRT(j, vidx + 1) * p[j];
        z += tm0cRT(j, vidx + 2) * p[j];
      }
      const PhGUtils::Point2f& q = targets_2d[i].q;

      T Px = x + Tvec.x, Py = y + Tvec.y, Pz = z + Tvec.z;
      T u, v;
      PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

      T dx = u - q.x, dy = v - q.y;

      T rk = sqrt(dx*dx + dy*dy);
      T inv_rk = 1.0 / rk;

      // Jf
      T inv_z = 1.0 / Pz;
      T inv_z2 = inv_z * inv_z;

      Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
      Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

      for (int j = 0; j < m; ++j) {
        // Jm * e_j
        T jmex = tm0cRT(j, vidx), jmey = tm0cRT(j, vidx + 1), jmez = tm0cRT(j, vidx + 2);
        // Jf * Jm * e_j
        T jfjmx = Jf[0] * jmex + Jf[2] * jmez;
        T jfjmy = Jf[4] * jmey + Jf[5] * jmez;
        // data term and prior term
        J[jidx++] = (jfjmx * dx + jfjmy * dy) * inv_rk * scale;
      }
    }

    // regularization term
    int nparams = params.priorParams.mu_wexp_orig.n_elem;
    arma::vec wmmu(nparams);
    for (int i = 0; i < nparams; ++i) {
      wmmu(i) = p[i] - params.priorParams.mu_wexp_orig(i);
    }
    arma::vec Jprior = params.priorParams.sigma_wexp * wmmu;
    T Eprior = sqrt(arma::dot(wmmu, Jprior));
    T inv_Eprior = fabs(Eprior) < 1e-12 ? 1e-12 : 1.0 / Eprior;

    for (int j = 0; j < m; ++j) {
      J[jidx++] = inv_Eprior * Jprior(j) * params.priorParams.w_prior_exp;
    }
  }

  fptr f_pose, jac_pose, f_id, jac_id, f_exp, jac_exp;

  MultilinearModel<T> &model;
  SingleImageParameters &params;
  const vector<Constraint_2D> &targets_2d;
};

template <typename T>
struct MultiImageEngergyFunction2D{
  MultiImageEngergyFunction2D() :model(MultilinearModel<T>()), params(MultiImageParameters()), targets_2d(vector<Constraint_2D>()){}
  MultiImageEngergyFunction2D(MultiImageParameters &params) :params(params){}

  void cost_pose(T *p, T *hx, int m, int n, void *adata) {
    //cout << m << ", " << n << endl;
    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {
      int offset = imgidx * MultilinearModelParameters::nRTparams;
      T s, rx, ry, rz, tx, ty, tz;
      rx = p[offset + 0], ry = p[offset + 1], rz = p[offset + 2];
      tx = p[offset + 3], ty = p[offset + 4], tz = p[offset + 5];

      auto &targets_2d = params.reconParams[imgidx].cons;
      int npts = targets_2d.size();
      int hxoffset = npts * imgidx;
      auto &tmc = params.reconParams[imgidx].model_projected.tm;

      //cout << "tmc length = " << tmc.length() << endl;

      // set up rotation matrix and translation vector
      PhGUtils::Point3<T> Tvec(tx, ty, tz);
      PhGUtils::Matrix3x3<T> R = PhGUtils::rotationMatrix(rx, ry, rz);
      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);

      //cout << Mproj << endl;

      // compute the scale of the input
      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

      // apply the new global transformation
      for (int i = 0, vidx=0; i < npts; i++) {
        //PhGUtils::Point3f p(tmc(vidx), tmc(vidx+1), tmc(vidx+2));
        T wpt = 1.0;

        // exclude mouth region and chin region
        /*
        if (!params.fitExpression) {
        if (i >= 46 || (i >= 4 && i <= 11))
        wpt = 0.9;
        }
        else {
        if (i < 15) wpt = 1.0;
        else if (i >= 68) wpt = 0.5;
        }
        */

        T px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
        const PhGUtils::Point2f& q = targets_2d.at(i).q;

        // PhGUtils::Point3f pp = R * p + T;
        PhGUtils::transformPoint(px, py, pz, R, Tvec);
        //cout << px << " " << py << " " << pz << endl;

        // Projection to image plane
        T u, v;
        PhGUtils::projectPoint(u, v, px, py, pz, Mproj);

        //cout << i << "\t" << u << ", " << v << endl;

        T dx = q.x - u, dy = q.y - v;
        //cout << i << "\t" << dx << ", " << dy << endl;
        //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
        hx[hxoffset+i] = sqrt(dx * dx + dy * dy) * wpt * scale;
        //cout << hx[i] << ' ';
      }
      //cout << endl;
    }
    //system("pause");
  }

  void jacobian_pose(T *p, T *J, int m, int n, void *adata) {
    // J is a npts*nImage-by-nRTparams*nImage matrix
    // [J1,  0, ...,  0]
    // [ 0, J2, ...,  0]
    // [ 0,  0, ...,  0]
    // [ 0,  0, ..., Jn]
    // Ji is the jacobian of for the ith image in row major order
    
    // set all 0 first, check if this is valid with the compiler implementation
    memset(J, 0, sizeof(T)*n*m);

    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {
      int offset = imgidx * MultilinearModelParameters::nRTparams;
      T s, rx, ry, rz, tx, ty, tz;
      rx = p[offset + 0], ry = p[offset + 1], rz = p[offset + 2];
      tx = p[offset + 3], ty = p[offset + 4], tz = p[offset + 5];

      auto &targets_2d = params.reconParams[imgidx].cons;
      int npts = targets_2d.size();
      auto &tmc = params.reconParams[imgidx].model_projected.tm;

      // set up rotation matrix and translation vector
      PhGUtils::Matrix3x3<T> R = PhGUtils::rotationMatrix(rx, ry, rz);
      PhGUtils::Matrix3x3<T> Jx, Jy, Jz;
      PhGUtils::jacobian_rotationMatrix(rx, ry, rz, Jx, Jy, Jz);
      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);

      //cout << Jx << endl;
      //cout << Jy << endl;
      //cout << Jz << endl;

      // compute the scale of the input
      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

      // for Jacobian of projection/viewport transformation
      const T f_x = params.modelParams[imgidx].camparams.fx, f_y = params.modelParams[imgidx].camparams.fy;
      T Jf[6] = { 0 };

      int roffset = npts * imgidx;
      int coffset = offset; // same as offset for p
      int rowsize = params.nImages * MultilinearModelParameters::nRTparams;

      // apply the new global transformation
      for (int i = 0, vidx = 0; i<npts; i++) {
        T wpt = 1.0;

        // exclude mouth region and chin region

        /*
        if (!params.fitExpression) {
        if (i >= 46 || (i >= 4 && i <= 11))
        wpt = 0.9;
        }
        else{
        if (i < 15) wpt = 1.0;
        else if (i >= 68) wpt = 0.5;
        }
        */


        // point p
        T px = tmc(vidx++), py = tmc(vidx++), pz = tmc(vidx++);
        //cout << px << ", " << py << ", " << pz << endl;

        // point q
        const PhGUtils::Point2f& q = targets_2d[i].q;

        // R * p
        T Rpx = px, Rpy = py, Rpz = pz;
        PhGUtils::rotatePoint(Rpx, Rpy, Rpz, R);

        // P = R * p + t
        T Pkx = Rpx + tx, Pky = Rpy + ty, Pkz = Rpz + tz;

        // Jf
        T inv_z = 1.0 / Pkz;
        T inv_z2 = inv_z * inv_z;

        Jf[0] = f_x * inv_z; Jf[2] = -f_x * Pkx * inv_z2;
        Jf[4] = f_y * inv_z; Jf[5] = -f_y * Pky * inv_z2;

        /*
        Jf[0] = -f_x * inv_z; Jf[2] = f_x * pkx * inv_z2;
        Jf[4] = f_y * inv_z; Jf[5] = -f_y * pky * inv_z2;
        */

        // project p to color image plane
        T pu, pv;
        PhGUtils::projectPoint(pu, pv, Pkx, Pky, Pkz, Mproj);

        // residue
        T rkx = pu - q.x, rky = pv - q.y;
        T rk = sqrt(rkx*rkx + rky*rky);
        T inv_rk = 1.0 / rk;
        // normalize it
        rkx *= inv_rk; rky *= inv_rk;

        // J_? * p_k
        T jpx, jpy, jpz;
        // J_f * J_? * p_k
        T jfjpx, jfjpy;

        jpx = px, jpy = py, jpz = pz;
        PhGUtils::rotatePoint(jpx, jpy, jpz, Jx);
        jfjpx = Jf[0] * jpx + Jf[2] * jpz;
        jfjpy = Jf[4] * jpy + Jf[5] * jpz;
        /*
        cout << "jf\t";
        PhGUtils::printArray(Jf, 9);
        cout << "j\t" << jpx << ", " << jpy << ", "  << jpz << endl;
        cout << "jj\t" << jfjpx << ", " << jfjpy << ", "  << jfjpz << endl;
        */

        int jidx = (roffset + i) * rowsize + coffset;

        // \frac{\partial r_i}{\partial \theta_x}
        J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt * scale;

        jpx = px, jpy = py, jpz = pz;
        PhGUtils::rotatePoint(jpx, jpy, jpz, Jy);
        jfjpx = Jf[0] * jpx + Jf[2] * jpz;
        jfjpy = Jf[4] * jpy + Jf[5] * jpz;
        // \frac{\partial r_i}{\partial \theta_y}
        J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt * scale;

        jpx = px, jpy = py, jpz = pz;
        PhGUtils::rotatePoint(jpx, jpy, jpz, Jz);
        jfjpx = Jf[0] * jpx + Jf[2] * jpz;
        jfjpy = Jf[4] * jpy + Jf[5] * jpz;
        // \frac{\partial r_i}{\partial \theta_z}
        J[jidx++] = (jfjpx * rkx + jfjpy * rky) * wpt * scale;

        // \frac{\partial r_i}{\partial \t_x}
        J[jidx++] = (Jf[0] * rkx) * wpt * scale;

        // \frac{\partial r_i}{\partial \t_y}
        J[jidx++] = (Jf[4] * rky) * wpt * scale;

        // \frac{\partial r_i}{\partial \t_z}
        J[jidx++] = (Jf[2] * rkx + Jf[5] * rky) * wpt * scale;
      }
    }
  }

  void cost_identity(T *p, T *hx, int m, int n, void *adata) {
    double wimg = sqrt(1.0 / params.nImages);
    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {
      auto &targets_2d = params.reconParams[imgidx].cons;
      int npts = targets_2d.size();
      int hxoffset = imgidx * (npts + 1);
      int nparams = params.priorParams.mu_wid.n_elem;
      auto &tm1cRT = params.reconParams[imgidx].model_projected.tm1RT;

      // set up rotation matrix and translation vector
      const PhGUtils::Point3<T>& Tvec = params.modelParams[imgidx].Tvec;
      const PhGUtils::Matrix3x3<T>& R = params.modelParams[imgidx].Rmat;
      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);

      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);
      //cout << "npts = " << npts << endl;
      for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
        T wpt = 1.0;

        T x = 0, y = 0, z = 0;
        for (int j = 0; j < nparams; j++) {
          x += tm1cRT(j, vidx) * p[j];
          y += tm1cRT(j, vidx + 1) * p[j];
          z += tm1cRT(j, vidx + 2) * p[j];
        }
        const PhGUtils::Point2f& q = targets_2d[i].q;

        T u, v;
        PhGUtils::projectPoint(u, v, x + Tvec.x, y + Tvec.y, z + Tvec.z, Mproj);

        T dx = q.x - u, dy = q.y - v;
        //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
        hx[hxoffset + i] = sqrt(dx*dx + dy*dy) * scale;
        //cout << hx[i] << ", ";
      }

      // regularization term
      arma::vec wmmu(nparams);
      for (int i = 0; i < nparams; ++i) {
        wmmu(i) = p[i] - params.priorParams.mu_wid_orig(i);
      }
      hx[hxoffset + npts] = sqrt(arma::dot(wmmu, params.priorParams.sigma_wid * wmmu)) * params.priorParams.w_prior_id * wimg;
      //cout << hx[n] << endl;
    }
  }

  void jacobian_identity(T *p, T *J, int m, int n, void *adata) {
    // J is a (npts+1)*nImage-by-nparams_id matrix
    // [J1]
    // [J2]
    // [...]
    // [Jn]
    // Ji is the jacobian of for the ith image in row major order

    // set all 0 first, check if this is valid with the compiler implementation
    memset(J, 0, sizeof(T)*n*m);

    double wimg = sqrt(1.0 / params.nImages);
    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {

      auto &targets_2d = params.reconParams[imgidx].cons;
      int npts = targets_2d.size();
      auto tm1cRT = params.reconParams[imgidx].model_projected.tm1RT;

      // set up rotation matrix and translation vector
      const PhGUtils::Point3<T>& Tvec = params.modelParams[imgidx].Tvec;
      const PhGUtils::Matrix3x3<T>& R = params.modelParams[imgidx].Rmat;
      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);

      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

      //cout << "hx: ";
      T Jf[6] = { 0 };
      T f_x = params.modelParams[imgidx].camparams.fx, f_y = params.modelParams[imgidx].camparams.fy;

      int roffset = (npts + 1)*imgidx;
      int nparams = params.priorParams.mu_wid0.n_elem;      
      for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
        T wpt = 1.0;

        // Pk
        T Px = 0, Py = 0, Pz = 0;
        for (int j = 0; j < nparams; j++) {
          Px += tm1cRT(j, vidx) * p[j];
          Py += tm1cRT(j, vidx + 1) * p[j];
          Pz += tm1cRT(j, vidx + 2) * p[j];
        }
        Px += Tvec.x; Py += Tvec.y; Pz += Tvec.z;

        const PhGUtils::Point2f& q = targets_2d[i].q;
        T u, v;
        PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

        T rkx = u - q.x, rky = v - q.y;

        T rk = sqrt(rkx*rkx + rky*rky);
        T inv_rk = 1.0 / rk;
        rkx *= inv_rk; rky *= inv_rk;

        // Jf
        T inv_z = 1.0 / Pz;
        T inv_z2 = inv_z * inv_z;

        Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
        Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

        int jidx = (roffset+i)*nparams;
        for (int j = 0; j < nparams; ++j) {
          // R * Jm * e_j
          T jmex = tm1cRT(j, vidx), jmey = tm1cRT(j, vidx + 1), jmez = tm1cRT(j, vidx + 2);

          // Jf * R * Jm * e_j
          float jfjmx = Jf[0] * jmex + Jf[2] * jmez;
          float jfjmy = Jf[4] * jmey + Jf[5] * jmez;
          // data term and prior term
          J[jidx++] = (jfjmx * rkx + jfjmy * rky) * scale;
        }
      }

      // regularization term
      arma::vec wmmu(nparams);
      for (int j = 0; j < nparams; ++j) {
        wmmu(j) = p[j] - params.priorParams.mu_wid_orig(j);
      }
      arma::vec Jprior = params.priorParams.sigma_wid * wmmu;
      T Eprior = sqrt(arma::dot(wmmu, Jprior));
      T inv_Eprior = fabs(Eprior) < 1e-12 ? 1e-12 : 1.0 / Eprior;

      for (int j = 0, jidx=(roffset+npts)*nparams; j < nparams; ++j) {
        J[jidx++] = inv_Eprior * Jprior(j) * params.priorParams.w_prior_id * wimg;
      }
    }

  }

  void cost_expression(T *p, T *hx, int m, int n, void *adata) {
    double wimg = sqrt(1.0 / params.nImages);

    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {
      auto &targets_2d = params.reconParams[imgidx].cons;

      int npts = targets_2d.size();
      int nparams = params.priorParams.mu_wexp.n_elem;
      int hxoffset = imgidx * (npts + 1);
      int poffset = imgidx * nparams;
      auto &tm0cRT = params.reconParams[imgidx].model_projected.tm0RT;

      // set up rotation matrix and translation vector
      const PhGUtils::Point3<T>& Tvec = params.modelParams[imgidx].Tvec;
      const PhGUtils::Matrix3x3<T>& R = params.modelParams[imgidx].Rmat;
      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);

      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

      for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
        T wpt = 1.0;

        T x = 0, y = 0, z = 0;
        for (int j = 0; j < nparams; j++) {
          x += tm0cRT(j, vidx) * p[poffset + j];
          y += tm0cRT(j, vidx + 1) * p[poffset + j];
          z += tm0cRT(j, vidx + 2) * p[poffset + j];
        }
        const PhGUtils::Point2f& q = targets_2d[i].q;

        T u, v;
        PhGUtils::projectPoint(u, v, x + Tvec.x, y + Tvec.y, z + Tvec.z, Mproj);

        T dx = q.x - u, dy = q.y - v;
        //hx[i] = (dx * dx + dy * dy) + dz * dz * wpt;
        hx[hxoffset+i] = sqrt(dx*dx + dy*dy) * scale;
      }

      // regularization term      
      arma::vec wmmu(nparams);
      for (int i = 0; i < nparams; ++i) {
        wmmu(i) = p[poffset+i] - params.priorParams.mu_wexp_orig(i);
      }
      hx[hxoffset + npts] = sqrt(arma::dot(wmmu, params.priorParams.sigma_wexp * wmmu)) * params.priorParams.w_prior_exp * wimg;
    }
  }

  void jacobian_expression(T *p, T *J, int m, int n, void *adata) {
    // J is a (npts+1)*nImage-by-nparams_id matrix
    // [J1,  0, ...,  0]
    // [ 0, J2, ...,  0]
    // [ 0,  0, ...,  0]
    // [ 0,  0, ..., Jn]
    // Ji is the jacobian of for the ith image in row major order

    // set all 0 first, check if this is valid with the compiler implementation
    memset(J, 0, sizeof(T)*n*m);
    double wimg = sqrt(1.0 / params.nImages);

    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {
      auto &targets_2d = params.reconParams[imgidx].cons;
      int npts = targets_2d.size();
      auto tm0cRT = params.reconParams[imgidx].model_projected.tm0RT;

      // set up rotation matrix and translation vector
      const PhGUtils::Point3<T>& Tvec = params.modelParams[imgidx].Tvec;
      const PhGUtils::Matrix3x3<T>& R = params.modelParams[imgidx].Rmat;
      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);
      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

      T Jf[6] = { 0 };
      T f_x = params.modelParams[imgidx].camparams.fx, f_y = params.modelParams[imgidx].camparams.fy;

      int nparams = params.priorParams.mu_wexp0.n_elem;
      int rowsize = params.nImages * nparams;
      int roffset = (npts + 1)*imgidx;      
      int poffset = nparams * imgidx;
      int coffset = poffset;

      for (int i = 0, vidx = 0; i < npts; i++, vidx += 3) {
        T wpt = 1.0;

        T x = 0, y = 0, z = 0;
        for (int j = 0; j < nparams; j++) {
          x += tm0cRT(j, vidx) * p[poffset+j];
          y += tm0cRT(j, vidx + 1) * p[poffset + j];
          z += tm0cRT(j, vidx + 2) * p[poffset + j];
        }
        const PhGUtils::Point2f& q = targets_2d[i].q;

        T Px = x + Tvec.x, Py = y + Tvec.y, Pz = z + Tvec.z;
        T u, v;
        PhGUtils::projectPoint(u, v, Px, Py, Pz, Mproj);

        T dx = u - q.x, dy = v - q.y;

        T rk = sqrt(dx*dx + dy*dy);
        T inv_rk = 1.0 / rk;

        // Jf
        T inv_z = 1.0 / Pz;
        T inv_z2 = inv_z * inv_z;

        Jf[0] = f_x * inv_z; Jf[2] = -f_x * Px * inv_z2;
        Jf[4] = f_y * inv_z; Jf[5] = -f_y * Py * inv_z2;

        int jidx = (roffset + i)*rowsize + coffset;
        for (int j = 0; j < nparams; ++j) {
          // Jm * e_j
          T jmex = tm0cRT(j, vidx), jmey = tm0cRT(j, vidx + 1), jmez = tm0cRT(j, vidx + 2);
          // Jf * Jm * e_j
          T jfjmx = Jf[0] * jmex + Jf[2] * jmez;
          T jfjmy = Jf[4] * jmey + Jf[5] * jmez;
          // data term and prior term
          J[jidx++] = (jfjmx * dx + jfjmy * dy) * inv_rk * scale;
        }
      }

      // regularization term
      arma::vec wmmu(nparams);
      for (int i = 0; i < nparams; ++i) {
        wmmu(i) = p[poffset + i] - params.priorParams.mu_wexp_orig(i);
      }
      arma::vec Jprior = params.priorParams.sigma_wexp * wmmu;
      T Eprior = sqrt(arma::dot(wmmu, Jprior));
      T inv_Eprior = fabs(Eprior) < 1e-12 ? 1e-12 : 1.0 / Eprior;

      for (int j = 0, jidx=(roffset+npts)*rowsize+coffset; j < nparams; ++j) {
        J[jidx++] = inv_Eprior * Jprior(j) * params.priorParams.w_prior_exp * wimg;
      }
    }
  }

  T error() {
    T E = 0;
    int npts = params.reconParams.front().cons.size();
    for (int imgidx = 0; imgidx < params.nImages; ++imgidx) {
      auto &targets_2d = params.reconParams[imgidx].cons;

      int npts = targets_2d.size();
      T scale = 100.0 / targets_2d[29].q.distanceTo(targets_2d[33].q);

      PhGUtils::Matrix3x3<T> Mproj = params.getProjectionMatrix(imgidx);
      auto &tmc = params.reconParams[imgidx].model_projected.tm;

      for (int i = 0; i < npts; i++) {
        int vidx = i * 3;
        // should change this to a fast version
        PhGUtils::Point3<T> p(tmc(vidx), tmc(vidx + 1), tmc(vidx + 2));
        //cout << p << endl;
        /*p = Rmat * p + Tvec;*/
        PhGUtils::transformPoint(p.x, p.y, p.z, params.modelParams[imgidx].Rmat, params.modelParams[imgidx].Tvec);

        // project the point
        T u, v;
        PhGUtils::projectPoint(u, v, p.x, p.y, p.z, Mproj);

        T dx, dy;
        dx = u - targets_2d[i].q.x;
        dy = v - targets_2d[i].q.y;
        E += sqrt(dx * dx + dy * dy) * scale;
      }
    }

    E /= (npts * params.nImages);
    return E;
  }

  MultiImageParameters &params;
};