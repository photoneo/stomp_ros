#ifndef INDUSTRIAL_MOVEIT_STOMP_MOVEIT_INCLUDE_STOMP_MOVEIT_NOISE_GENERATORS_CARTESIAN_CONSTRAINTS_SAMPLING_H_
#define INDUSTRIAL_MOVEIT_STOMP_MOVEIT_INCLUDE_STOMP_MOVEIT_NOISE_GENERATORS_CARTESIAN_CONSTRAINTS_SAMPLING_H_

#include <stomp_moveit/noise_generators/stomp_noise_generator.h>
#include <stomp_moveit/utils/multivariate_gaussian.h>
#include "stomp_moveit/utils/kinematics.h"

namespace stomp_moveit
{
namespace noise_generators
{

typedef boost::mt19937 RGNType;
typedef boost::variate_generator< RGNType, boost::normal_distribution<> > RandomGenerator;

/**
 * @class stomp_moveit::noise_generators::CartesianNormalDistributionSampling
 * @brief This class generates noisy trajectories to an under-constrained cartesian waypoints.
 *
 */
class CartesianNormalDistributionSampling: public StompNoiseGenerator
{
public:
  CartesianNormalDistributionSampling();
  virtual ~CartesianNormalDistributionSampling();

  /**
   * @brief Initializes and configures.
   * @param robot_model_ptr A pointer to the robot model.
   * @param group_name      The designated planning group.
   * @param config          The configuration data.  Usually loaded from the ros parameter server
   * @return true if succeeded, false otherwise.
   */
  virtual bool initialize(moveit::core::RobotModelConstPtr robot_model_ptr,
                          const std::string& group_name,const XmlRpc::XmlRpcValue& config) override;

  /**
   * @brief Sets internal members of the plugin from the configuration data.
   * @param config  The configuration data.  Usually loaded from the ros parameter server
   * @return  true if succeeded, false otherwise.
   */
  virtual bool configure(const XmlRpc::XmlRpcValue& config) override;

  /**
   * @brief Stores the planning details.
   * @param planning_scene      A smart pointer to the planning scene
   * @param req                 The motion planning request
   * @param config              The  Stomp configuration.
   * @param error_code          Moveit error code.
   * @return  true if succeeded, false otherwise.
   */
  virtual bool setMotionPlanRequest(const planning_scene::PlanningSceneConstPtr& planning_scene,
                   const moveit_msgs::MotionPlanRequest &req,
                   const stomp_core::StompConfiguration &config,
                   moveit_msgs::MoveItErrorCodes& error_code) override;

  /**
   * @brief Generates a noisy trajectory from the parameters.
   * @param parameters        The current value of the optimized parameters [num_dimensions x num_parameters]
   * @param start_timestep    Start index into the 'parameters' array, usually 0.
   * @param num_timesteps     The number of elements to use from 'parameters' starting from 'start_timestep'
   * @param iteration_number  The current iteration count in the optimization loop
   * @param rollout_number    The index of the noisy trajectory.
   * @param parameters_noise  The parameters + noise
   * @param noise             The noise applied to the parameters
   * @return true if cost were properly computed, false otherwise.
   */
  virtual bool generateNoise(const Eigen::MatrixXd& parameters,
                                       std::size_t start_timestep,
                                       std::size_t num_timesteps,
                                       int iteration_number,
                                       int rollout_number,
                                       Eigen::MatrixXd& parameters_noise,
                                       Eigen::MatrixXd& noise) override;

  /**
   * @brief Called by the Stomp at the end of the optimization process
   *
   * @param success           Whether the optimization succeeded
   * @param total_iterations  Number of iterations used
   * @param final_cost        The cost value after optimizing.
   * @param parameters        The parameters generated at the end of current iteration[num_dimensions x num_timesteps]
   */
  virtual void done(bool success,int total_iterations,double final_cost,const Eigen::MatrixXd& parameters){}


  virtual std::string getName() const
  {
    return name_ + "/" + group_;
  }


  virtual std::string getGroupName() const
  {
    return group_;
  }

protected:

  /**
   * @brief Genereates a random tool pose by apply noise on the redundant axis to a reference tool pose;
   * @param reference_joint_pose  Joint position used in computing the reference tool pose with FK
   * @param result       The joint position corresponding to the randomized tool pose
   * @return noisy tool position
   */
  virtual Eigen::Affine3d applyCartesianNoise(const Eigen::VectorXd& reference_joint_pose);

protected:

  // names
  std::string name_;
  std::string group_;

  // tool link and ik tolerance
  std::string tool_link_;
  const std::array<double, 6> ik_tolerance_ = {0.001, 0.001, 0.001, 0.01, 0.01, 0.01};

  // ros parameters
  std::array<double, 6> stddev_;                                      /**< @brief The standard deviations applied to each cartesian DOF **/

  // random goal generation
  boost::shared_ptr<RandomGenerator> goal_rand_generator_;            /**< @brief Random generator for the tool pose **/

  // robot
  moveit::core::RobotModelConstPtr robot_model_;
  moveit::core::RobotStatePtr state_;
  stomp_moveit::utils::kinematics::IKSolverPtr ik_solver_;
};

} /* namespace noise_generators */
} /* namespace stomp_moveit */
#endif /* INDUSTRIAL_MOVEIT_STOMP_MOVEIT_INCLUDE_STOMP_MOVEIT_NOISE_GENERATORS_CARTESIAN_CONSTRAINTS_SAMPLING_H_ */
