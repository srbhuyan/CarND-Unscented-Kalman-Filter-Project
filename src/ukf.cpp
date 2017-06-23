#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define EPSILON 0.001

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  ///* time when the state is true, in us
  time_us_;

  ///* State dimension
  n_x_ = x_.size();

  ///* Augmented state dimension
  n_aug_ = n_x_ + 2;

  ///* Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  n_sig_ = 2 * n_aug_ + 1;

  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  ///* Weights of sigma points
  weights_ = VectorXd(n_sig_);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  if(!is_initialized_){
    // covariance matrix initialization
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
	  0, 0, 1, 0, 0,
	  0, 0, 0, 1, 0,
	  0, 0, 0, 0, 1;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
      
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];

      // convert to cartesian
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      float vx = rho_dot * cos(phi);
      float vy = rho_dot * sin(phi);
      float v = sqrt(vx*vx + vy*vy);

      x_ << px, py, v, 0, 0;

    }else if(meas_package.sensor_type_ == MeasurementPackage::LASER){

      float px = meas_package.raw_measurements_[0];
      float py = meas_package.raw_measurements_[1];

      if(fabs(px) < EPSILON && fabs(py) < EPSILON){
        px = EPSILON;
	py = EPSILON;
      }

      x_ << px, py, 0, 0, 0;
    }

    // weights initialization
    weights_(0) = lambda_ / (lambda_ + n_aug_);

    for(int i=1; i<n_sig_; i++){
      weights_(i) = (double)0.5 / (lambda_ + n_aug_);
    }

    // timestamp
    time_us_ = meas_package.timestamp_;    

    // init done
    is_initialized_ = true;

    return;
  }


  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  // predict
  Prediction(dt);

  // update
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
  
  if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_){
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  double delta_t_sq = delta_t * delta_t;


  // --------------- generate augmented sigma points ---------------
  VectorXd x_aug = VectorXd(n_aug_);            // augmented mean vector
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);    // augmented state covariance
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_); // sigma points matrix

  x_aug.fill(0.0);
  x_aug.head(n_x_) = x_;

  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_*std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_*std_yawdd_;

  // square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // generation
  Xsig_aug.col(0) = x_aug;
  for(int i=0; i<n_aug_; i++){
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

  // --------------- predict sigma points ---------------
  
  for(int i=0; i<n_sig_; i++){
    double px       = Xsig_aug(0, i);
    double py       = Xsig_aug(1, i);
    double v        = Xsig_aug(2, i);
    double yaw      = Xsig_aug(3, i);
    double yawd     = Xsig_aug(4, i);
    double nu_a     = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    // division by zero
    if(fabs(yawd) < EPSILON){
      px_p = px + v * delta_t * cos(yaw);
      py_p = py + v * delta_t * sin(yaw);
    }else{
      px_p = px + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = py + v/yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }

    double v_p   = v;
    double yaw_p  = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p   = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p   = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p    = v_p + nu_a * delta_t;
    yaw_p  = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;
 
    // write predicted sigma points
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p; 
  }

  // --------------- predicted state mean and covariance ---------------
  //mean
  x_.fill(0.0);
  for(int i=0; i<n_sig_; i++){
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  //covariance
  P_.fill(0.0);
  for(int i=0; i<n_sig_; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle cormalization
    while(x_diff(3) > M_PI)  x_diff(3) -= 2. * M_PI;
    while(x_diff(3) < -M_PI) x_diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // measurement_dimension
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);

  Update(meas_package, Zsig, n_z);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  
  // measurement dimension
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // transform sigma points into measurement space
  for(int i=0; i<n_sig_; i++){
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v  = Xsig_pred_(2, i);
    double yaw = Xsig_pred_(3, i);

    double v1 = v * cos(yaw);
    double v2 = v * sin(yaw);

    // measurement model
    Zsig(0, i) = sqrt(px*px + py*py);
    Zsig(1, i) = atan2(py, px);
    Zsig(2, i) = (px*v1 + py*v2) / Zsig(0, i);
  }

  Update(meas_package, Zsig, n_z);
}

void UKF::Update(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for(int i=0; i<n_sig_; i++){
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0);
  for(int i=0; i<n_sig_; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while(z_diff(1) > M_PI)  z_diff(1) -= 2.*M_PI;
    while(z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z, n_z);
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    R << std_radr_*std_radr_, 0, 0,
	 0, std_radphi_*std_radphi_, 0,
	 0, 0, std_radrd_*std_radrd_;

  }else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    R << std_laspx_*std_laspx_, 0,
	 0, std_laspy_*std_laspy_;
  }

  S = S + R;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i=0; i<n_sig_; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = Tc * S.inverse();

  // raw measurement
  VectorXd z = meas_package.raw_measurements_;

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1) >  M_PI) z_diff(1) -= 2.*M_PI;
  while (z_diff(1) < -M_PI) z_diff(1) += 2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  // calculate NIS
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
    NIS_radar_ = z.transpose() * S.inverse() * z;
  }else if(meas_package.sensor_type_ == MeasurementPackage::LASER){
    NIS_laser_ = z.transpose() * S.inverse() * z;
  }

}

