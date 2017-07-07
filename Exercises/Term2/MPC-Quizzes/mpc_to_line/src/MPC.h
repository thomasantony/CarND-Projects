#ifndef MPC_H
#define MPC_H

#include <vector>
#include <tuple>
#include <cppad/cppad.hpp>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

// Evaluate a polynomial.
template<typename scalar_t>
scalar_t polyeval_cppad(Eigen::VectorXd coeffs, scalar_t x) {
  scalar_t result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {
    result += coeffs[i] * CppAD::pow(x, i);
  }
  return result;
}

// Evaluate a polynomial slope.
template<typename scalar_t>
scalar_t polyeval_slope_cppad(Eigen::VectorXd coeffs, scalar_t x) {
  scalar_t result = 0.0;
  for (int i = 1; i < coeffs.size(); i++) {
    result += coeffs[i] * CppAD::pow(x, i-1);
  }
  return result;
}

/**
 * Converts a point from global coordinates to local coordinates
 * @param input
 * @return
 */
template<typename scalar_t, typename vector_t>
tuple<scalar_t, scalar_t> local_transform(tuple<scalar_t, scalar_t> input, vector_t vehicle)
{
  // First find relative coordinates and then rotate them
  scalar_t x, y;
  scalar_t x0 = vehicle[0], y0 = vehicle[1], theta0 = vehicle[2];
  std::tie(x, y) = input;

  x = x - x0;
  y = y - y0;

  return std::make_tuple(x * CppAD::cos(theta0) + y * CppAD::sin(theta0), -x * CppAD::sin(theta0) + y * CppAD::cos(theta0));
}

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state.
  // Return the next state and actuations as a
  // vector.
  vector<double> Solve(Eigen::VectorXd x0, Eigen::VectorXd coeffs);
};

#endif /* MPC_H */
