/**
 * @file cartesian_constraints_sampling.cpp
 * @brief This a normal distribution noisy trajectory update generator in cartesian space
 *
 * @author Michal Dobis
 * @date June 6, 2021
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2021, Photoneo
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
#include <stomp_plugins/noise_generators/cartesian_normal_distribution_sampling.h>
#include <stomp_moveit/utils/multivariate_gaussian.h>
#include <XmlRpcException.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>

PLUGINLIB_EXPORT_CLASS(stomp_moveit::noise_generators::CartesianNormalDistributionSampling,stomp_moveit::noise_generators::StompNoiseGenerator);

static const int CARTESIAN_DOF_SIZE = 6;

namespace stomp_moveit
{
namespace noise_generators
{

CartesianNormalDistributionSampling::CartesianNormalDistributionSampling():
  name_("CartesianNormalDistributionSampling"),
  goal_rand_generator_(new RandomGenerator(RGNType(),boost::uniform_real<>(-1,1)))
{

}

CartesianNormalDistributionSampling::~CartesianNormalDistributionSampling()
{

}

bool CartesianNormalDistributionSampling::initialize(moveit::core::RobotModelConstPtr robot_model_ptr,
                        const std::string& group_name,const XmlRpc::XmlRpcValue& config)
{
  using namespace moveit::core;

  // robot model details
  group_ = group_name;
  robot_model_ = robot_model_ptr;

  // robot state
  state_ = std::make_shared<RobotState>(robot_model_);

  const JointModelGroup* joint_group = robot_model_ptr->getJointModelGroup(group_name);
  if(!joint_group)
  {
    ROS_ERROR("Invalid joint group %s",group_name.c_str());
    return false;
  }

  // kinematics
  ik_solver_.reset(new stomp_moveit::utils::kinematics::IKSolver(robot_model_ptr,group_name));

  return configure(config);
}

bool CartesianNormalDistributionSampling::configure(const XmlRpc::XmlRpcValue& config)
{
  using namespace XmlRpc;

  try
  {
    XmlRpcValue params = config;

    // noise generation parameters
    XmlRpcValue stddev_param = params["stddev"];
    tool_link_ = static_cast<std::string>(params["link_id"]);

    // Check link_id
    if (tool_link_.empty())
    {
      ROS_ERROR("%s the 'link_id' parameter is empty",getName().c_str());
      return false;
    }

    // check  stddev
    if(stddev_param.size() != stddev_.size())
    {
      ROS_ERROR("%s the 'stddev' parameter has incorrect number of cartesian DOF (6)",getName().c_str());
      return false;
    }

    // parsing parameters
    for(auto i = 0u; i < stddev_param.size(); i++)
    {
      stddev_[i] = static_cast<double>(stddev_param[i]);
    }
  }
  catch(XmlRpc::XmlRpcException& e)
  {
    ROS_ERROR("%s failed to load parameters",getName().c_str());
    return false;
  }

  return true;
}

bool CartesianNormalDistributionSampling::setMotionPlanRequest(const planning_scene::PlanningSceneConstPtr& planning_scene,
                 const moveit_msgs::MotionPlanRequest &req,
                 const stomp_core::StompConfiguration &config,
                 moveit_msgs::MoveItErrorCodes& error_code)
{
  using namespace moveit::core;
  using namespace utils::kinematics;

  // Update robot state
  robotStateMsgToRobotState(req.start_state,*state_);

  // update kinematic model
  ik_solver_->setKinematicState(*state_);

  return true;
}

bool CartesianNormalDistributionSampling::generateNoise(const Eigen::MatrixXd& parameters,
                                     std::size_t start_timestep,
                                     std::size_t num_timesteps,
                                     int iteration_number,
                                     int rollout_number,
                                     Eigen::MatrixXd& parameters_noise,
                                     Eigen::MatrixXd& noise)
{
  using namespace Eigen;
  using namespace stomp_moveit::utils;
  using namespace moveit::core;

  if(parameters.rows() != stddev_.size())
  {
    ROS_ERROR("%s Number of rows in parameters %i differs from expected number of joints",getName().c_str(),int(parameters.rows()));
    return false;
  }

  for(auto t = 0u; t < parameters.cols();t++)
  {
    const auto noisy_tool_pose = applyCartesianNoise(parameters.col(t));
    Eigen::VectorXd result = Eigen::VectorXd::Zero(parameters.rows());

    const Eigen::VectorXd ik_tolerance_eigen = Map<const VectorXd>(ik_tolerance_.data(), CARTESIAN_DOF_SIZE);
    if(!ik_solver_->solve(parameters.col(t),noisy_tool_pose,result,ik_tolerance_eigen))
    {
      ROS_DEBUG("%s could not compute ik, returning noiseless goal pose",getName().c_str());
      parameters_noise.col(t) = parameters.col(t);
    }
    else
    {
      parameters_noise.col(t) = result;
    }
  }

  // generating noise
  noise = parameters_noise - parameters;
  return true;
}

Eigen::Affine3d CartesianNormalDistributionSampling::applyCartesianNoise(const Eigen::VectorXd& reference_joint_pose)
{
  using namespace Eigen;
  using namespace moveit::core;
  using namespace stomp_moveit::utils;

  const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_);

  Eigen::VectorXd raw_noise = Eigen::VectorXd::Zero(CARTESIAN_DOF_SIZE);
  for(auto d = 0u; d < raw_noise.size(); d++)
  {
      raw_noise(d) = stddev_[d]*(*goal_rand_generator_)();
  }

  state_->setJointGroupPositions(joint_group,reference_joint_pose);
  Eigen::Affine3d tool_pose = state_->getFrameTransform(tool_link_);

  const auto& n = raw_noise;
  return tool_pose * Translation3d(Vector3d(n(0),n(1),n(2)))*AngleAxisd(n(3),Vector3d::UnitX())*AngleAxisd(n(4),Vector3d::UnitY())*AngleAxisd(n(5),Vector3d::UnitZ());
}

} /* namespace noise_generators */
} /* namespace stomp_moveit */
