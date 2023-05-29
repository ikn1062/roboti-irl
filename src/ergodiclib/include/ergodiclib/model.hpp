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
  requires std::is_class_v<ModelTemplate>;
  {modeltemplate.getA(
      std::declval<arma::vec>(),
      std::declval<arma::vec>())}->std::same_as<arma::mat>;
  {modeltemplate.getB(
      std::declval<arma::vec>(),
      std::declval<arma::vec>())}->std::same_as<arma::mat>;
  {modeltemplate.createTrajectory()}->std::same_as<std::pair<arma::mat, arma::mat>>;
  {modeltemplate.createTrajectory(
      std::declval<arma::vec>(),
      std::declval<arma::mat>())}->std::same_as<arma::mat>;
  {modeltemplate.createTrajectory(
      std::declval<arma::vec>(),
      std::declval<arma::mat>(),
      std::declval<unsigned int>())}->std::same_as<arma::mat>;
  std::same_as<decltype(modeltemplate.x0), arma::vec>;
  std::same_as<decltype(modeltemplate.u0), arma::vec>;
  std::same_as<decltype(modeltemplate.dt), double>;
  std::same_as<decltype(modeltemplate.t0), double>;
  std::same_as<decltype(modeltemplate.tf), double>;
};
}


#endif
