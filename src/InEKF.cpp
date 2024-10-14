/* ----------------------------------------------------------------------------
 * Copyright 2018, Ross Hartley <m.ross.hartley@gmail.com>
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file   InEKF.cpp
 *  @author Ross Hartley
 *  @brief  Source file for Invariant EKF
 *  @date   September 25, 2018
 **/

#include "src/InEKF.h"
#include <chrono>
#include <Eigen/Sparse>

namespace inekf {

using namespace std;

void removeRowAndColumn(Eigen::MatrixXd& M, int index);
void remove3RowsAndColumns(Eigen::MatrixXd& M, int start_index);

// ------------ Observation -------------
// Default constructor
Observation::Observation(Eigen::VectorXd& Y, Eigen::VectorXd& b,
                         Eigen::MatrixXd& H, Eigen::MatrixXd& N,
                         Eigen::MatrixXd& PI)
    : Y(Y), b(b), H(H), N(N), PI(PI) {}

// Check if empty
bool Observation::empty() { return Y.rows() == 0; }

ostream& operator<<(ostream& os, const Observation& o) {
  os << "---------- Observation ------------" << endl;
  os << "Y:\n" << o.Y << endl << endl;
  os << "b:\n" << o.b << endl << endl;
  os << "H:\n" << o.H << endl << endl;
  os << "N:\n" << o.N << endl << endl;
  os << "PI:\n" << o.PI << endl;
  os << "-----------------------------------";
  return os;
}

// ------------ InEKF -------------
// Default constructor
InEKF::InEKF() {}

// Constructor with noise params
InEKF::InEKF(NoiseParams params)
    : noise_params_(params) {}

// Constructor with initial state
InEKF::InEKF(RobotState state)
    : state_(state) {}

// Constructor with initial state and noise params
InEKF::InEKF(RobotState state, NoiseParams params)
    : state_(state),
      noise_params_(params) {}

// Assignment operator
InEKF& InEKF::operator = (const InEKF& old_inekf) {
  state_ = old_inekf.getState();
  noise_params_ = old_inekf.getNoiseParams();
  prior_landmarks_ = old_inekf.getPriorLandmarks();
  estimated_landmarks_ = old_inekf.getEstimatedLandmarks();
  contacts_ = old_inekf.getContacts();
  estimated_contact_positions_ = old_inekf.getEstimatedContactPositions();
  return *this;
}

// Return robot's current state
const RobotState& InEKF::getState() const { return state_; }

// Sets the robot's current state
void InEKF::setState(RobotState state) { state_ = state; }

// Return noise params
const NoiseParams& InEKF::getNoiseParams() const { return noise_params_; }

// Sets the filter's noise parameters
void InEKF::setNoiseParams(NoiseParams params) { noise_params_ = params; }

// Return filter's prior (static) landmarks
mapIntVector3d InEKF::getPriorLandmarks() const { return prior_landmarks_; }

// Set the filter's prior (static) landmarks
void InEKF::setPriorLandmarks(const mapIntVector3d& prior_landmarks) {
  prior_landmarks_ = prior_landmarks;
}

// Return filter's estimated landmarks
map<int, int> InEKF::getEstimatedLandmarks() const {
  return estimated_landmarks_;
}

// Return filter's estimated landmarks
const map<int, int>& InEKF::getEstimatedContactPositions() const {
  return estimated_contact_positions_;
}

// Set the filter's contact state
void InEKF::setContacts(vector<pair<int, bool> > contacts) {
  // Insert new measured contact states
  for (vector<pair<int, bool> >::iterator it = contacts.begin();
       it != contacts.end(); ++it) {
    pair<map<int, bool>::iterator, bool> ret = contacts_.insert(*it);
    // If contact is already in the map, replace with new value
    if (ret.second == false) {
      ret.first->second = it->second;
    }
  }
  return;
}

// Return the filter's contact state
std::map<int, bool> InEKF::getContacts() const {
  return contacts_;
}

// InEKF Propagation - Inertial Data
void InEKF::Propagate(const Eigen::Matrix<double, 6, 1>& m, double dt) {
  Eigen::Vector3d w =
      m.head(3) - state_.getGyroscopeBias();  // Angular Velocity
  Eigen::Vector3d a =
      m.tail(3) - state_.getAccelerometerBias();  // Linear Acceleration

  const Eigen::MatrixXd& X = state_.getX();
  const Eigen::MatrixXd& P = state_.getP();

  // Extract State
  const Eigen::Matrix3d& R = state_.getRotation();
  const Eigen::Vector3d& v = state_.getVelocity();
  const Eigen::Vector3d& p = state_.getPosition();

  // Strapdown IMU motion model
  Eigen::Vector3d phi = w * dt;
  Eigen::Matrix3d R_pred = R * Exp_SO3(phi);
  Eigen::Vector3d v_pred = v + (R * a + g_) * dt;
  Eigen::Vector3d p_pred = p + v * dt + 0.5 * (R * a + g_) * dt * dt;

  // Set new state (bias has constant dynamics)
  state_.setRotation(R_pred);
  state_.setVelocity(v_pred);
  state_.setPosition(p_pred);

  // ---- Linearized invariant error dynamics -----
  int dimX = state_.dimX();
  int dimP = state_.dimP();
  int dimTheta = state_.dimTheta();

  // Actually compute the elements of A * dt directly to
  // avoid unnecessary dt * 0 multiplications later
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(dimP, dimP);
  // Inertial terms
  A.block<3, 3>(3, 0) = dt * gskew_;
  A.block<3, 3>(6, 3) = dt * Eigen::Matrix3d::Identity();
  // Bias terms
  A.block<3, 3>(0, dimP - dimTheta) = -dt * R;
  A.block<3, 3>(3, dimP - dimTheta + 3) = -dt * R;
  for (int i = 3; i < dimX; ++i) {
    A.block<3, 3>(3 * i - 6, dimP - dimTheta) = -dt * skew(X.block<3, 1>(0, i)) * R;
  }

  // Telling eigen Qk is diagonal speeds up PhiAdj * Qk * PhiAdj.T by 2x
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Qk = MakeQk(dimP, dimTheta, dt);

  Eigen::MatrixXd Adj = Eigen::MatrixXd::Identity(dimP, dimP);
  Adj.block(0, 0, dimP - dimTheta, dimP - dimTheta) = Adjoint_SEK3(X);

  // (A dt + I) * adj = A dt * adj + adj has more advantageous sparsity
  // That we can exploit to save a few tens of us
  Eigen::SparseMatrix<double> A_sparse = A.sparseView();
  Eigen::SparseMatrix<double> Adj_sparse = Adj.sparseView();
  Eigen::MatrixXd PhiAdj = A_sparse * Adj_sparse + Adj;

  // Approximated discretized noise matrix
  Eigen::MatrixXd Qk_hat = (PhiAdj * Qk) * PhiAdj.transpose(); //

  // Propagate Covariance - start by correcting Phi to A* dt + I
  Eigen::MatrixXd& Phi = A;
  for (int i = 0; i < Phi.cols(); ++i) {
    Phi(i, i) += 1;
  }
  Eigen::MatrixXd P_pred = Phi * P * Phi.transpose() + Qk_hat;

  // Set new covariance
  state_.setP(P_pred);

  return;
}

inline Eigen::DiagonalMatrix<double, Eigen::Dynamic> InEKF::MakeQk(
    int dimP, int dimTheta, double dt) const {
  // Noise terms
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> Qk(dimP);
  Qk.setZero();

  Qk.diagonal().segment<3>(0) = dt * noise_params_.getGyroscopeCov().diagonal();
  Qk.diagonal().segment<3>(3) = dt * noise_params_.getAccelerometerCov().diagonal();

  for (auto& it :estimated_contact_positions_) {
    Qk.diagonal().segment<3>(3 + 3 * (it.second - 3)) =
      dt * noise_params_.getContactCov().diagonal();  // Contact noise terms
  }
  Qk.diagonal().segment<3>(dimP - dimTheta) =
      dt * noise_params_.getGyroscopeBiasCov().diagonal();
  Qk.diagonal().segment<3>(dimP - dimTheta + 3) =
      dt * noise_params_.getAccelerometerBiasCov().diagonal();

  return Qk;
}

// Correct State: Right-Invariant Observation
void InEKF::Correct(const Observation& obs) {
  // Compute Kalman Gain
  Eigen::MatrixXd P = state_.getP();
  Eigen::MatrixXd PHT = P * obs.H.transpose();
  Eigen::MatrixXd S = obs.H * PHT + obs.N;
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(S.rows(), S.cols());
  Eigen::MatrixXd K = PHT * S.ldlt().solve(I);

  // Compute correction terms
  int dimX = state_.dimX();
  const Eigen::MatrixXd& X = state_.getX();

  // Construct  PI * Z = PI * (X * Y - b) taking advantage of block structure of
  // X and noting that for both kinematic and landmark updates:
  //   1. PI just selects the top 3 rows of each segment of Z
  //   2. obs.b[0:3] = 0
  Eigen::VectorXd PI_Z = Eigen::VectorXd::Zero(3 * obs.Y.rows() / dimX);
  for (int i = 0; i < obs.Y.rows() / state_.dimX(); ++i) {
    int startIndex = i * dimX;
    PI_Z.segment<3>(3*i) =
        X.topRows<3>() * obs.Y.segment(startIndex, dimX) -
            obs.b.segment<3>(startIndex);
  }
  //  Eigen::MatrixXd Z = BigX * obs.Y - obs.b;
  Eigen::VectorXd delta = K * PI_Z;
  Eigen::MatrixXd dX =
      Exp_SEK3(delta.segment(0, delta.rows() - state_.dimTheta()));

  Eigen::VectorXd dTheta =
      delta.segment(delta.rows() - state_.dimTheta(), state_.dimTheta());

  // Update state
  Eigen::MatrixXd X_new = dX * state_.getX();  // Right-Invariant Update
  Eigen::VectorXd Theta_new = state_.getTheta() + dTheta;
  state_.setX(X_new);
  state_.setTheta(Theta_new);

  // Update Covariance
  Eigen::MatrixXd IKH = - K * obs.H;
  for (int i = 0; i < state_.dimP(); ++i) {
    IKH(i,i) += 1;
  }
  Eigen::MatrixXd P_new = IKH * P * IKH.transpose() +
                          K * obs.N * K.transpose();  // Joseph update form

  state_.setP(P_new);
}

// Create Observation from vector of landmark measurements
void InEKF::CorrectLandmarks(const vectorLandmarks& measured_landmarks) {
  Eigen::VectorXd Y;
  Eigen::VectorXd b;
  Eigen::MatrixXd H;
  Eigen::MatrixXd N;
  Eigen::MatrixXd PI;

  Eigen::Matrix3d R = state_.getRotation();
  vectorLandmarks new_landmarks;
  vector<int> used_landmark_ids;

  for (vectorLandmarksIterator it = measured_landmarks.begin();
       it != measured_landmarks.end(); ++it) {
    // Detect and skip if an ID is not unique (this would cause singularity
    // issues in InEKF::Correct)
    if (find(used_landmark_ids.begin(), used_landmark_ids.end(), it->id) !=
        used_landmark_ids.end()) {
      cout << "Duplicate landmark ID detected! Skipping measurement.\n";
      continue;
    } else {
      used_landmark_ids.push_back(it->id);
    }

    // See if we can find id in prior_landmarks or estimated_landmarks
    mapIntVector3dIterator it_prior = prior_landmarks_.find(it->id);
    map<int, int>::iterator it_estimated = estimated_landmarks_.find(it->id);
    if (it_prior != prior_landmarks_.end()) {
      // Found in prior landmark set
      int dimX = state_.dimX();
      int dimP = state_.dimP();
      int startIndex;

      // Fill out Y
      startIndex = Y.rows();
      Y.conservativeResize(startIndex + dimX, Eigen::NoChange);
      Y.segment(startIndex, dimX) = Eigen::VectorXd::Zero(dimX);
      Y.segment(startIndex, 3) = it->position;  // p_bl
      Y(startIndex + 4) = 1;

      // Fill out b
      startIndex = b.rows();
      b.conservativeResize(startIndex + dimX, Eigen::NoChange);
      b.segment(startIndex, dimX) = Eigen::VectorXd::Zero(dimX);
      b.segment(startIndex, 3) = it_prior->second;  // p_wl
      b(startIndex + 4) = 1;

      // Fill out H
      startIndex = H.rows();
      H.conservativeResize(startIndex + 3, dimP);
      H.block(startIndex, 0, 3, dimP) = Eigen::MatrixXd::Zero(3, dimP);
      H.block(startIndex, 0, 3, 3) = skew(it_prior->second);  // skew(p_wl)
      H.block(startIndex, 6, 3, 3) = -Eigen::Matrix3d::Identity();  // -I

      // Fill out N
      startIndex = N.rows();
      N.conservativeResize(startIndex + 3, startIndex + 3);
      N.block(startIndex, 0, 3, startIndex) =
          Eigen::MatrixXd::Zero(3, startIndex);
      N.block(0, startIndex, startIndex, 3) =
          Eigen::MatrixXd::Zero(startIndex, 3);
      N.block(startIndex, startIndex, 3, 3) =
          R * noise_params_.getLandmarkCov() * R.transpose();

      // Fill out PI
      startIndex = PI.rows();
      int startIndex2 = PI.cols();
      PI.conservativeResize(startIndex + 3, startIndex2 + dimX);
      PI.block(startIndex, 0, 3, startIndex2) =
          Eigen::MatrixXd::Zero(3, startIndex2);
      PI.block(0, startIndex2, startIndex, dimX) =
          Eigen::MatrixXd::Zero(startIndex, dimX);
      PI.block(startIndex, startIndex2, 3, dimX) =
          Eigen::MatrixXd::Zero(3, dimX);
      PI.block(startIndex, startIndex2, 3, 3) = Eigen::Matrix3d::Identity();

    } else if (it_estimated != estimated_landmarks_.end()) {
      // Found in estimated landmark set
      int dimX = state_.dimX();
      int dimP = state_.dimP();
      int startIndex;

      // Fill out Y
      startIndex = Y.rows();
      Y.conservativeResize(startIndex + dimX, Eigen::NoChange);
      Y.segment(startIndex, dimX) = Eigen::VectorXd::Zero(dimX);
      Y.segment(startIndex, 3) = it->position;  // p_bl
      Y(startIndex + 4) = 1;
      Y(startIndex + it_estimated->second) = -1;

      // Fill out b
      startIndex = b.rows();
      b.conservativeResize(startIndex + dimX, Eigen::NoChange);
      b.segment(startIndex, dimX) = Eigen::VectorXd::Zero(dimX);
      b(startIndex + 4) = 1;
      b(startIndex + it_estimated->second) = -1;

      // Fill out H
      startIndex = H.rows();
      H.conservativeResize(startIndex + 3, dimP);
      H.block(startIndex, 0, 3, dimP) = Eigen::MatrixXd::Zero(3, dimP);
      H.block(startIndex, 6, 3, 3) = -Eigen::Matrix3d::Identity();  // -I
      H.block(startIndex, 3 * it_estimated->second - 6, 3, 3) =
          Eigen::Matrix3d::Identity();  // I

      // Fill out N
      startIndex = N.rows();
      N.conservativeResize(startIndex + 3, startIndex + 3);
      N.block(startIndex, 0, 3, startIndex) =
          Eigen::MatrixXd::Zero(3, startIndex);
      N.block(0, startIndex, startIndex, 3) =
          Eigen::MatrixXd::Zero(startIndex, 3);
      N.block(startIndex, startIndex, 3, 3) =
          R * noise_params_.getLandmarkCov() * R.transpose();

      // Fill out PI
      startIndex = PI.rows();
      int startIndex2 = PI.cols();
      PI.conservativeResize(startIndex + 3, startIndex2 + dimX);
      PI.block(startIndex, 0, 3, startIndex2) =
          Eigen::MatrixXd::Zero(3, startIndex2);
      PI.block(0, startIndex2, startIndex, dimX) =
          Eigen::MatrixXd::Zero(startIndex, dimX);
      PI.block(startIndex, startIndex2, 3, dimX) =
          Eigen::MatrixXd::Zero(3, dimX);
      PI.block(startIndex, startIndex2, 3, 3) = Eigen::Matrix3d::Identity();

    } else {
      // First time landmark as been detected (add to list for later state
      // augmentation)
      new_landmarks.push_back(*it);
    }
  }

  // Correct state using stacked observation
  Observation obs(Y, b, H, N, PI);
  if (!obs.empty()) {
    this->Correct(obs);
  }

  // Augment state with newly detected landmarks
  if (new_landmarks.size() > 0) {
    Eigen::MatrixXd X_aug = state_.getX();
    Eigen::MatrixXd P_aug = state_.getP();
    Eigen::Vector3d p = state_.getPosition();
    for (vectorLandmarksIterator it = new_landmarks.begin();
         it != new_landmarks.end(); ++it) {
      // Initialize new landmark mean
      int startIndex = X_aug.rows();
      X_aug.conservativeResize(startIndex + 1, startIndex + 1);
      X_aug.block(startIndex, 0, 1, startIndex) =
          Eigen::MatrixXd::Zero(1, startIndex);
      X_aug.block(0, startIndex, startIndex, 1) =
          Eigen::MatrixXd::Zero(startIndex, 1);
      X_aug(startIndex, startIndex) = 1;
      X_aug.block(0, startIndex, 3, 1) = p + R * it->position;

      // Initialize new landmark covariance - TODO:speed up
      Eigen::MatrixXd F =
          Eigen::MatrixXd::Zero(state_.dimP() + 3, state_.dimP());
      F.block(0, 0, state_.dimP() - state_.dimTheta(),
              state_.dimP() - state_.dimTheta()) =
          Eigen::MatrixXd::Identity(
              state_.dimP() - state_.dimTheta(),
              state_.dimP() - state_.dimTheta());  // for old X
      F.block(state_.dimP() - state_.dimTheta(), 6, 3, 3) =
          Eigen::Matrix3d::Identity();  // for new landmark
      F.block(state_.dimP() - state_.dimTheta() + 3,
              state_.dimP() - state_.dimTheta(), state_.dimTheta(),
              state_.dimTheta()) =
          Eigen::MatrixXd::Identity(state_.dimTheta(),
                                    state_.dimTheta());  // for theta
      Eigen::MatrixXd G = Eigen::MatrixXd::Zero(F.rows(), 3);
      G.block(G.rows() - state_.dimTheta() - 3, 0, 3, 3) = R;
      P_aug = (F * P_aug * F.transpose() +
               G * noise_params_.getLandmarkCov() * G.transpose())
                  .eval();

      // Update state and covariance
      state_.setX(X_aug);
      state_.setP(P_aug);

      // Add to list of estimated landmarks
      estimated_landmarks_.insert(pair<int, int>(it->id, startIndex));
    }
  }
  return;
}


// Correct state using kinematics measured between imu and contact point
void InEKF::CorrectKinematics(const vectorKinematics& measured_kinematics) {
  Eigen::VectorXd Y;
  Eigen::VectorXd b;
  Eigen::MatrixXd H;
  Eigen::MatrixXd N;
  Eigen::MatrixXd PI;

  Eigen::Matrix3d R = state_.getRotation();
  vector<pair<int, int> > remove_contacts;
  vectorKinematics new_contacts;
  vector<int> used_contact_ids;

  for (vectorKinematicsIterator it = measured_kinematics.begin();
       it != measured_kinematics.end(); ++it) {
    // Detect and skip if an ID is not unique (this would cause singularity
    // issues in InEKF::Correct)
    if (find(used_contact_ids.begin(), used_contact_ids.end(), it->id) !=
        used_contact_ids.end()) {
      cout << "Duplicate contact ID detected! Skipping measurement.\n";
      continue;
    } else {
      used_contact_ids.push_back(it->id);
    }

    // Find contact indicator for the kinematics measurement
    map<int, bool>::iterator it_contact = contacts_.find(it->id);
    if (it_contact == contacts_.end()) {
      continue;
    }  // Skip if contact state is unknown
    bool contact_indicated = it_contact->second;

    // See if we can find id estimated_contact_positions
    map<int, int>::iterator it_estimated =
        estimated_contact_positions_.find(it->id);
    bool found = it_estimated != estimated_contact_positions_.end();

    // If contact is not indicated and id is found in estimated_contacts_, then
    // remove state
    if (!contact_indicated && found) {
      remove_contacts.push_back(*it_estimated);  // Add id to remove list
      //  If contact is indicated and id is not found i n estimated_contacts_,
      //  then augment state
    } else if (contact_indicated && !found) {
      new_contacts.push_back(*it);  // Add to augment list

      // If contact is indicated and id is found in estimated_contacts_, then
      // correct using kinematics
    } else if (contact_indicated && found) {
      int dimX = state_.dimX();
      int dimP = state_.dimP();
      int startIndex;

      // Fill out Y
      startIndex = Y.rows();
      Y.conservativeResize(startIndex + dimX, Eigen::NoChange);
      Y.segment(startIndex, dimX) = Eigen::VectorXd::Zero(dimX);
      Y.segment(startIndex, 3) = it->pose.block<3, 1>(0, 3);  // p_bc
      Y(startIndex + 4) = 1;
      Y(startIndex + it_estimated->second) = -1;

      // Fill out b
      startIndex = b.rows();
      b.conservativeResize(startIndex + dimX, Eigen::NoChange);
      b.segment(startIndex, dimX) = Eigen::VectorXd::Zero(dimX);
      b(startIndex + 4) = 1;
      b(startIndex + it_estimated->second) = -1;

      // Fill out H
      startIndex = H.rows();
      H.conservativeResize(startIndex + 3, dimP);
      H.block(startIndex, 0, 3, dimP) = Eigen::MatrixXd::Zero(3, dimP);
      H.block(startIndex, 6, 3, 3) = -Eigen::Matrix3d::Identity();  // -I
      H.block(startIndex, 3 * it_estimated->second - 6, 3, 3) =
          Eigen::Matrix3d::Identity();  // I

      // Fill out N
      startIndex = N.rows();
      N.conservativeResize(startIndex + 3, startIndex + 3);
      N.block(startIndex, 0, 3, startIndex) =
          Eigen::MatrixXd::Zero(3, startIndex);
      N.block(0, startIndex, startIndex, 3) =
          Eigen::MatrixXd::Zero(startIndex, 3);
      N.block(startIndex, startIndex, 3, 3) =
          R * it->covariance.block<3, 3>(3, 3) * R.transpose();

      // Fill out PI
      startIndex = PI.rows();
      int startIndex2 = PI.cols();
      PI.conservativeResize(startIndex + 3, startIndex2 + dimX);
      PI.block(startIndex, 0, 3, startIndex2) =
          Eigen::MatrixXd::Zero(3, startIndex2);
      PI.block(0, startIndex2, startIndex, dimX) =
          Eigen::MatrixXd::Zero(startIndex, dimX);
      PI.block(startIndex, startIndex2, 3, dimX) =
          Eigen::MatrixXd::Zero(3, dimX);
      PI.block(startIndex, startIndex2, 3, 3) = Eigen::Matrix3d::Identity();

      //  If contact is not indicated and id is found in estimated_contacts_,
      //  then skip
    } else {
      continue;
    }
  }

  // Correct state using stacked observation
  Observation obs(Y, b, H, N, PI);
  if (!obs.empty()) {
    this->Correct(obs);
  }

  // Remove contacts from state
  if (remove_contacts.size() > 0) {
    Eigen::MatrixXd X_rem = state_.getX();
    Eigen::MatrixXd P_rem = state_.getP();
    for (vector<pair<int, int> >::iterator it = remove_contacts.begin();
         it != remove_contacts.end(); ++it) {
      // Remove from list of estimated contact positions
      estimated_contact_positions_.erase(it->first);

      // Remove row and column from X
      removeRowAndColumn(X_rem, it->second);

      // Remove 3 rows and columns from P
      int startIndex = 3 + 3 * (it->second - 3);
      remove3RowsAndColumns(P_rem, startIndex);

      // Update all indices for estimated_landmarks and
      // estimated_contact_positions
      for (map<int, int>::iterator it2 = estimated_landmarks_.begin();
           it2 != estimated_landmarks_.end(); ++it2) {
        if (it2->second > it->second) it2->second -= 1;
      }
      for (map<int, int>::iterator it2 = estimated_contact_positions_.begin();
           it2 != estimated_contact_positions_.end(); ++it2) {
        if (it2->second > it->second) it2->second -= 1;
      }
      // We also need to update the indices of remove_contacts in the case where
      // multiple contacts are being removed at once
      for (vector<pair<int, int> >::iterator it2 = it;
           it2 != remove_contacts.end(); ++it2) {
        if (it2->second > it->second) it2->second -= 1;
      }

      // Update state and covariance
      state_.setX(X_rem);
      state_.setP(P_rem);
    }
  }

  // Augment state with newly detected contacts
  if (new_contacts.size() > 0) {
    Eigen::MatrixXd X_aug = state_.getX();
    Eigen::MatrixXd P_aug = state_.getP();
    Eigen::Vector3d p = state_.getPosition();
    for (vectorKinematicsIterator it = new_contacts.begin();
         it != new_contacts.end(); ++it) {
      // Initialize new landmark mean
      int startIndex = X_aug.rows();
      X_aug.conservativeResize(startIndex + 1, startIndex + 1);
      X_aug.block(startIndex, 0, 1, startIndex) =
          Eigen::MatrixXd::Zero(1, startIndex);
      X_aug.block(0, startIndex, startIndex, 1) =
          Eigen::MatrixXd::Zero(startIndex, 1);
      X_aug(startIndex, startIndex) = 1;
      X_aug.block(0, startIndex, 3, 1) = p + R * it->pose.block<3, 1>(0, 3);
      // cout << "new contact position = " <<
      //     (p + R * it->pose.block<3, 1>(0, 3)).transpose() << endl;

      // Initialize new landmark covariance - TODO:speed up
      Eigen::MatrixXd F =
          Eigen::MatrixXd::Zero(state_.dimP() + 3, state_.dimP());
      F.block(0, 0, state_.dimP() - state_.dimTheta(),
              state_.dimP() - state_.dimTheta()) =
          Eigen::MatrixXd::Identity(
              state_.dimP() - state_.dimTheta(),
              state_.dimP() - state_.dimTheta());  // for old X
      F.block(state_.dimP() - state_.dimTheta(), 6, 3, 3) =
          Eigen::Matrix3d::Identity();  // for new landmark
      F.block(state_.dimP() - state_.dimTheta() + 3,
              state_.dimP() - state_.dimTheta(), state_.dimTheta(),
              state_.dimTheta()) =
          Eigen::MatrixXd::Identity(state_.dimTheta(),
                                    state_.dimTheta());  // for theta
      Eigen::MatrixXd G = Eigen::MatrixXd::Zero(F.rows(), 3);
      G.block(G.rows() - state_.dimTheta() - 3, 0, 3, 3) = R;
      P_aug = (F * P_aug * F.transpose() +
               G * it->covariance.block<3, 3>(3, 3) * G.transpose())
                  .eval();

      // Update state and covariance
      state_.setX(X_aug);
      state_.setP(P_aug);

      // Add to list of estimated contact positions
      estimated_contact_positions_.insert(pair<int, int>(it->id, startIndex));
    }
  }

  return;
}

void InEKF::RemoveLandmarks(const std::vector<int64_t> &to_remove) {

  vector<int> columns_to_remove{};
  columns_to_remove.reserve(to_remove.size());

  // remove landmark ids from the landmark id map
  for (const auto& id: to_remove) {
    if (estimated_landmarks_.count(id) > 0) {
      columns_to_remove.push_back(estimated_landmarks_.at(id));
      estimated_landmarks_.erase(id);
    }
  }

  // Need to account for the fact that column indices will decrease as we
  // remove columns
  std::sort(columns_to_remove.begin(), columns_to_remove.end());
  for (int i = 0; i < columns_to_remove.size(); ++i) {
    columns_to_remove[i] -= i;
  }

  Eigen::MatrixXd X_rem = state_.getX();
  Eigen::MatrixXd P_rem = state_.getP();
  for (const auto& col: columns_to_remove) {
    // Remove row and column from X
    removeRowAndColumn(X_rem, col);

    // Remove 3 rows and columns from P
    int startIndex = 3 + 3 * (col - 3);
    remove3RowsAndColumns(P_rem, startIndex);

    // Update indices for estimated_landmarks and contacts that
    // remain in the filter
    for (map<int, int>::iterator it2 = estimated_landmarks_.begin();
         it2 != estimated_landmarks_.end(); ++it2) {
      it2->second -= static_cast<int>(it2->second > col);
    }
    for (map<int, int>::iterator it2 = estimated_contact_positions_.begin();
         it2 != estimated_contact_positions_.end(); ++it2) {
      it2->second -= static_cast<int>(it2->second > col);
    }
  }

  // Update state and covariance
  state_.setX(X_rem);
  state_.setP(P_rem);

}

void removeRowAndColumn(Eigen::MatrixXd& M, int index) {
  unsigned int dimX = M.cols();
  // cout << "Removing index: " << index<< endl;
  M.block(index, 0, dimX - index - 1, dimX) =
      M.bottomRows(dimX - index - 1).eval();
  M.block(0, index, dimX, dimX - index - 1) =
      M.rightCols(dimX - index - 1).eval();
  M.conservativeResize(dimX - 1, dimX - 1);
}

void remove3RowsAndColumns(Eigen::MatrixXd& M, int index) {
  unsigned int dim = M.cols();
  M.block(index, 0, dim - index - 3, dim) =
      M.bottomRows(dim - index - 3).eval();
  M.block(0, index, dim, dim - index - 3) =
      M.rightCols(dim - index - 3).eval();
  M.conservativeResize(dim - 3, dim - 3);
}

}  // namespace inekf
