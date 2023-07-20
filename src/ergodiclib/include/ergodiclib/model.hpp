#ifndef MODEL_INCLUDE_GUARD_HPP
#define MODEL_INCLUDE_GUARD_HPP
/// \file
/// \brief Model Template Concept defintion

#if defined(__APPLE__)
#include </opt/homebrew/include/armadillo>
#else
#include <armadillo>
#endif

namespace ergodiclib
{
/// The template interface for Dynamic Model Systems
template<typename ModelTemplate>
concept ModelConcept = requires(ModelTemplate modeltemplate)
{
  // Ensure that the Template is a class
  requires std::is_class_v<ModelTemplate>;
  // Returns A Matrix (Jacobian with respect to state)
  {modeltemplate.getA(
      std::declval<arma::vec>(),
      std::declval<arma::vec>())}->std::same_as<arma::mat>;
  // Returns B Matrix (Jacobian with respect to control)
  {modeltemplate.getB(
      std::declval<arma::vec>(),
      std::declval<arma::vec>())}->std::same_as<arma::mat>;
  // Returns derivative of state with respect to time, xdot
  {modeltemplate.dynamics(
    std::declval<arma::vec>(),
    std::declval<arma::vec>())}->std::same_as<arma::vec>;
  // Resolves state x
  // An example would be keeping a rotation variable between -PI and PI
  {modeltemplate.resolveState(
    std::declval<arma::mat>())}->std::same_as<arma::mat>;
  // Intiial state of system
  std::same_as<decltype(modeltemplate.x0), arma::vec>;
  // Intiial control of system
  std::same_as<decltype(modeltemplate.u0), arma::vec>;
  // Number of iterations used for file creation
  std::same_as<decltype(modeltemplate.n_iter), int>;
  // Timestep
  std::same_as<decltype(modeltemplate.dt), double>;
  // Initial t0 state
  std::same_as<decltype(modeltemplate.t0), double>;
  // Final tf state
  std::same_as<decltype(modeltemplate.tf), double>;
};
}


#endif
