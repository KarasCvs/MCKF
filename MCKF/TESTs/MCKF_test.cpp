
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <iostream>
int main()
{
  unsigned int STATE_SIZE = 3;
  unsigned int updateSize = 2;
    // MC part
  Eigen::VectorXd state_(STATE_SIZE);
  Eigen::MatrixXd estimateErrorCovarianceSqrt(STATE_SIZE, STATE_SIZE);  // S_p, cholesky of P(estimateErrorCovariance_)
  Eigen::MatrixXd measurementCovarianceSqrt;                            // S_r, cholesky of R
  Eigen::MatrixXd S(STATE_SIZE+updateSize, STATE_SIZE+updateSize);      // A diagonal matrix builded by S_p and S_r
  Eigen::MatrixXd H;                                                    // P_xx/P_xz
  Eigen::MatrixXd W(STATE_SIZE+updateSize, updateSize);                 // S^-1*[I; H]
  Eigen::VectorXd D(STATE_SIZE+STATE_SIZE);                             //
  Eigen::MatrixXd Cx;                                                   // A diagonal builded by G(e_1~n)
  Eigen::MatrixXd Cy;
  Eigen::MatrixXd estimateErrorCovariance_(STATE_SIZE, STATE_SIZE);                         // P_hat                                        // A diagonal builded by G(e_n~n+m)
  Eigen::MatrixXd estimateErrorCovariance_temp(STATE_SIZE, STATE_SIZE);                         // P_hat
  Eigen::MatrixXd measurementErrorCovariance_temp(updateSize, updateSize);                      // R_hat
  Eigen::MatrixXd predictedStateCovar(STATE_SIZE, STATE_SIZE);          // P_xx
  Eigen::VectorXd entropy_x(STATE_SIZE);
  Eigen::VectorXd entropy_y(updateSize);
  Eigen::VectorXd state_temp(STATE_SIZE);

  Eigen::VectorXd stateSubset(updateSize);                              // x (in most literature)
  Eigen::VectorXd measurementSubset(updateSize);                        // z
  Eigen::MatrixXd measurementCovarianceSubset(updateSize, updateSize);  // R
  Eigen::MatrixXd stateToMeasurementSubset(updateSize, STATE_SIZE);     // H
  Eigen::MatrixXd kalmanGainSubset(STATE_SIZE, updateSize);             // K
  Eigen::VectorXd innovationSubset(updateSize);                         // z - Hx
  Eigen::VectorXd predictedMeasurement(updateSize);
  Eigen::VectorXd sigmaDiff(updateSize);
  Eigen::MatrixXd predictedMeasCovar(updateSize, updateSize);
  Eigen::MatrixXd crossCovar(STATE_SIZE, updateSize);
  // consts
  // const double PI = 3.141592653589793;
  // const double TAU = 6.283185307179587;
  const double epsilon_ = 1e-3;
  const int sigma_ = 6;
  // parameters

  estimateErrorCovariance_ << 2, 0, 0,
                              0, 2, 0,
                              0, 0, 2;
  measurementCovarianceSubset << 1, 0,
                                 0, 1;
  crossCovar << 1, 1, 1,
                1, 1, 1;
  state_ << 1,
            1,
            1;
  predictedMeasurement << 2,
                          3;
  measurementSubset << 4,
                       1;

  // MC part.1 calculate S= dig(S_p, S_r)
  estimateErrorCovarianceSqrt = estimateErrorCovariance_.llt().matrixL();
  measurementCovarianceSqrt = measurementCovarianceSubset.llt().matrixL();
  S << estimateErrorCovarianceSqrt, Eigen::MatrixXd::Zero(STATE_SIZE, updateSize),
       Eigen::MatrixXd::Zero(updateSize, STATE_SIZE), measurementCovarianceSqrt;

  // MC part.2 build H, W, D
  // estimateErrorCovariance_: P_xx
  H.noalias() = (estimateErrorCovariance_.inverse() * crossCovar).transpose();
  W << Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE),
       H;
  W = S.inverse() * W;
  D << state_,
       measurementSubset - predictedMeasurement + H*state_;
  D = S.inverse() * D;
  // MC part.3 fixed-point repeat
  state_temp = (W.transpose()*W).inverse()*W.transpose()*D;
  double fixed_point_evaluation = 1.0;
  int repeat_count;
  repeat_count = 0;
  while(fixed_point_evaluation > epsilon_)
  {
  // MC part.4 entropy
  Eigen::VectorXd E = D - W * state_temp;
  for (size_t Ind = 0; Ind < STATE_SIZE; ++Ind)
    {
      entropy_x[Ind] = exp(-(pow(E[Ind], 2)/(2*pow(sigma_, 2))));    //kernel function.
      if (entropy_x[Ind] < 1e-9)
      {
          entropy_x[Ind] += 1e-9;
      }
    }
  for (size_t Ind = 0; Ind < updateSize; ++Ind)
    {
      entropy_y[Ind] = exp(-(pow(E[Ind+STATE_SIZE], 2)/(2*pow(sigma_, 2))));
      if (entropy_y[Ind] < 1e-9)
      {
          entropy_y[Ind] += 1e-9;
      }
    }
  Cx = entropy_x.asDiagonal();
  Cy = entropy_y.asDiagonal();
  estimateErrorCovariance_temp = estimateErrorCovarianceSqrt * Cx.inverse() * estimateErrorCovarianceSqrt.transpose();
  measurementErrorCovariance_temp = measurementCovarianceSqrt * Cy.inverse() * measurementCovarianceSqrt.transpose();
  // (3) Compute the Kalman gain, making sure to use the actual measurement covariance:
  Eigen::MatrixXd invInnovCov = (H*estimateErrorCovariance_temp*H.transpose() + measurementErrorCovariance_temp).inverse(); //我可以直接用这个吗?
  kalmanGainSubset = estimateErrorCovariance_temp * H.transpose() * invInnovCov;

  // (4) Apply the gain to the difference between the actual and predicted measurements: x = x + K(z - z_hat)
  innovationSubset = (measurementSubset - predictedMeasurement);
  // Wrap angles in the innovation
  // for (size_t i = 0; i < updateSize; ++i)
  // {
  // if (updateIndices[i] == StateMemberRoll  ||
  //     updateIndices[i] == StateMemberPitch ||
  //     updateIndices[i] == StateMemberYaw)
  // {
  //   while (innovationSubset(i) < -PI)
  //   {
  //     innovationSubset(i) += TAU;
  //   }

  //   while (innovationSubset(i) > PI)
  //   {
  //     innovationSubset(i) -= TAU;
  //   }
  // }
  // }
  // (5) Check Mahalanobis distance of innovation
  // if (checkMahalanobisThreshold(innovationSubset, invInnovCov, measurement.mahalanobisThresh_))
  // {
  state_.noalias() = state_temp + kalmanGainSubset * innovationSubset;
  // }
  fixed_point_evaluation = (state_ - state_temp).norm() / state_temp.norm();
  state_temp = state_;
  repeat_count += 1;
  }
// (6) Compute the new estimate error covariance P = (eye(L)-K*H)*P1*(eye(L)-K*H)' + K*R*K'
  estimateErrorCovariance_.noalias() = (Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE)-kalmanGainSubset*H)*\
  estimateErrorCovariance_*(Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE)-kalmanGainSubset*H).transpose() + \
  (kalmanGainSubset * measurementCovarianceSubset * kalmanGainSubset.transpose());
  std::cout << repeat_count << std::endl;
  std::cout << state_ << std::endl;
  std::cout << estimateErrorCovariance_ << std::endl;
}