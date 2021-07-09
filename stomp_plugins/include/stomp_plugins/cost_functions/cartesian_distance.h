/**
 * @file cartesian_distance.h
 * @brief This defines a cost function for minimize cartesian translation and angular distance
 *
 * @author Michal Dobis
 * @date July 7, 2021
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2021, Photoneo s.r.o
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef INDUSTRIAL_MOVEIT_STOMP_MOVEIT_INCLUDE_STOMP_MOVEIT_COST_FUNCTIONS_CARTESIAN_DISTANCE_H_
#define INDUSTRIAL_MOVEIT_STOMP_MOVEIT_INCLUDE_STOMP_MOVEIT_COST_FUNCTIONS_CARTESIAN_DISTANCE_H_

#include <Eigen/Sparse>
#include <moveit/robot_model/robot_model.h>
#include <stomp_moveit/utils/kinematics.h>
#include "stomp_moveit/cost_functions/stomp_cost_function.h"

namespace stomp_moveit
{
namespace cost_functions
{

/**
 * @class stomp_moveit::cost_functions::CartesianDistance
 * @brief Evaluates the cost of the tool pose by determining how far the tool pose
 *        is from the shortest cartesian trajectory
 */
class CartesianDistance: public StompCostFunction
{
public:
    CartesianDistance();
  virtual ~CartesianDistance();

  /**
   * @brief Initializes and configures the Cost Function.
   * @param robot_model_ptr A pointer to the robot model.
   * @param group_name      The designated planning group.
   * @param config          The configuration data.  Usually loaded from the ros parameter server
   * @return true if succeeded, false otherwise.
   */
  virtual bool initialize(moveit::core::RobotModelConstPtr robot_model_ptr,
                          const std::string& group_name,XmlRpc::XmlRpcValue& config) override;

  /**
   * @brief Sets internal members of the plugin from the configuration data.
   * @param config  The configuration data .  Usually loaded from the ros parameter server
   * @return  true if succeeded, false otherwise.
   */
  virtual bool configure(const XmlRpc::XmlRpcValue& config) override;

  /**
   * @brief Stores the planning details which will be used during the costs calculations.
   * @param planning_scene  A smart pointer to the planning scene
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
   * @brief computes the goal state costs as a function of the distance from the desired task manifold.
   * @param parameters        The parameter values to evaluate for state costs [num_dimensions x num_parameters]
   * @param start_timestep    start index into the 'parameters' array, usually 0.
   * @param num_timesteps     number of elements to use from 'parameters' starting from 'start_timestep'   *
   * @param iteration_number  The current iteration count in the optimization loop
   * @param rollout_number    index of the noisy trajectory whose cost is being evaluated.   *
   * @param costs             vector containing the state costs per timestep.  Only the array's last entry is set. [num_parameters x 1]
   * @param validity          whether or not the trajectory is valid
   * @return true if cost were properly computed
   */
  virtual bool computeCosts(const Eigen::MatrixXd& parameters,
                            std::size_t start_timestep,
                            std::size_t num_timesteps,
                            int iteration_number,
                            int rollout_number,
                            Eigen::VectorXd& costs,
                            bool& validity) override;

  virtual std::string getGroupName() const override
  {
    return group_name_;
  }


  virtual std::string getName() const override
  {
    return name_ + "/" + group_name_;
  }

  /**
   * @brief Called by the Stomp Task at the end of the optimization process
   *
   * @param success           Whether the optimization succeeded
   * @param total_iterations  Number of iterations used
   * @param final_cost        The cost value after optimizing.
   * @param parameters        The parameters generated at the end of current iteration[num_dimensions x num_timesteps]
   */
  virtual void done(bool success,int total_iterations,double final_cost,const Eigen::MatrixXd& parameters) override;

protected:

  std::string name_;

  // robot details
  std::string group_name_;
  std::string tool_link_;
  moveit::core::RobotModelConstPtr robot_model_;
  moveit::core::RobotStatePtr state_;

  // ros parameters
  double position_cost_weight_;                       /**< @brief factor multiplied to the scaled position error **/
  double orientation_cost_weight_;                    /**< @brief factor multiplied to the scaled orientation error **/
  double translation_tolerance_;                      /**< @brief maximal valid distance **/
  double rotation_tolerance_;                         /**< @brief maximal valid rotation **/

  // initial states used for evaluation
  Eigen::Affine3d start_;                            /**< @brief Start pose from MotionPlanRequest **/
  Eigen::Affine3d goal_;                             /**< @brief Goal pose from MotionPlanRequest **/

};

} /* namespace cost_functions */
} /* namespace stomp_moveit */

#endif /* INDUSTRIAL_MOVEIT_STOMP_MOVEIT_INCLUDE_STOMP_MOVEIT_COST_FUNCTIONS_CARTESIAN_DISTANCE_H_ */
