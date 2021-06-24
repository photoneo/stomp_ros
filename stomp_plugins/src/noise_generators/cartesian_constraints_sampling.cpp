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
#include <stomp_plugins/noise_generators/cartesian_constraints_sampling.h>
#include <stomp_moveit/utils/multivariate_gaussian.h>
#include <XmlRpcException.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>

PLUGINLIB_EXPORT_CLASS(stomp_moveit::noise_generators::CartesianConstraintsSampling,stomp_moveit::noise_generators::StompNoiseGenerator);

static const std::vector<double> ACC_MATRIX_DIAGONAL_VALUES = {-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0};
static const std::vector<int> ACC_MATRIX_DIAGONAL_INDICES = {-2, -1, 0 ,1, 2};
static const int CARTESIAN_DOF_SIZE = 6;
static const double DEFAULT_IK_POS_TOLERANCE = 0.001;
static const double DEFAULT_IK_ROT_TOLERANCE = 0.01;


namespace stomp_moveit
{
namespace noise_generators
{

CartesianConstraintsSampling::CartesianConstraintsSampling():
  name_("CartesianConstraintsSampling"),
  goal_rand_generator_(new RandomGenerator(RGNType(),boost::uniform_real<>(-1,1)))
{

}

CartesianConstraintsSampling::~CartesianConstraintsSampling()
{

}

bool CartesianConstraintsSampling::initialize(moveit::core::RobotModelConstPtr robot_model_ptr,
                        const std::string& group_name,const XmlRpc::XmlRpcValue& config)
{
  using namespace moveit::core;

  // robot model details
  group_ = group_name;
  robot_model_ = robot_model_ptr;
  const JointModelGroup* joint_group = robot_model_ptr->getJointModelGroup(group_name);
  if(!joint_group)
  {
    ROS_ERROR("Invalid joint group %s",group_name.c_str());
    return false;
  }

  // kinematics
  ik_solver_.reset(new stomp_kinematics::kinematics::IKSolver(robot_model_ptr,group_name));

  // trajectory noise generation
  stddev_.resize(CARTESIAN_DOF_SIZE);

  // creating default cartesian tolerance for IK solver
  tool_goal_tolerance_.resize(CARTESIAN_DOF_SIZE);
  double ptol = DEFAULT_IK_POS_TOLERANCE;
  double rtol = DEFAULT_IK_ROT_TOLERANCE;
  tool_goal_tolerance_ << ptol, ptol, ptol, rtol, rtol, rtol;

  is_first_trajectory_ = true;

  return configure(config);
}

bool CartesianConstraintsSampling::configure(const XmlRpc::XmlRpcValue& config)
{
  using namespace XmlRpc;

  try
  {
    XmlRpcValue params = config;

    // noise generation parameters
    XmlRpcValue stddev_param = params["stddev"];
    translation_tolerance_ =  static_cast<double>(params["translation_tolerance"]);
    rotation_tolerance_ =  static_cast<double>(params["rotation_tolerance"]);

    // check  stddev
    if(stddev_param.size() != stddev_.size())
    {
      ROS_ERROR("%s the 'stddev' parameter has incorrect number of cartesian DOF (6)",getName().c_str());
      return false;
    }
    if (translation_tolerance_ <= 0)
    {
      ROS_ERROR("%s the 'translation_tolerance' parameter must not have non positive value",getName().c_str());
      return false;
    }

    if (rotation_tolerance_ <= 0)
    {
      ROS_ERROR("%s the 'rotation_tolerance_' parameter must not have non positive value",getName().c_str());
      return false;
    }

    // parsing parameters
    for(auto i = 0u; i < stddev_param.size(); i++)
    {
      stddev_[i] = static_cast<double>(stddev_param[i]);
    }

    // Update tolerance according to IK tolerance
    Eigen::VectorXd ik_trans_tolerance = Eigen::VectorXd::Zero(3);
    Eigen::VectorXd ik_rot_tolerance = Eigen::VectorXd::Zero(3);
    ik_trans_tolerance << tool_goal_tolerance_[0], tool_goal_tolerance_[1], tool_goal_tolerance_[2];
    ik_rot_tolerance << tool_goal_tolerance_[3], tool_goal_tolerance_[4], tool_goal_tolerance_[5];
    translation_tolerance_ -= ik_trans_tolerance.norm();
    rotation_tolerance_ -= ik_rot_tolerance.norm();
  }
  catch(XmlRpc::XmlRpcException& e)
  {
    ROS_ERROR("%s failed to load parameters",getName().c_str());
    return false;
  }

  return true;
}

bool CartesianConstraintsSampling::setMotionPlanRequest(const planning_scene::PlanningSceneConstPtr& planning_scene,
                 const moveit_msgs::MotionPlanRequest &req,
                 const stomp_core::StompConfiguration &config,
                 moveit_msgs::MoveItErrorCodes& error_code)
{
  is_first_trajectory_ = true;
  bool succeed = setupNoiseGeneration(planning_scene,req,config,error_code) &&
          setupRobotState(planning_scene,req,config,error_code);
  return succeed;
}

bool CartesianConstraintsSampling::setupNoiseGeneration(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                               const moveit_msgs::MotionPlanRequest &req,
                                               const stomp_core::StompConfiguration &config,
                                               moveit_msgs::MoveItErrorCodes& error_code)
{
  using namespace Eigen;

  // convenience lambda function to fill matrix
  auto fill_diagonal = [](Eigen::MatrixXd& m,double coeff,int diag_index)
  {
    std::size_t size = m.rows() - std::abs(diag_index);
    m.diagonal(diag_index) = VectorXd::Constant(size,coeff);
  };

  // creating finite difference acceleration matrix
  std::size_t num_timesteps = config.num_timesteps;
  Eigen::MatrixXd A = MatrixXd::Zero(num_timesteps,num_timesteps);
  int num_elements = (int((ACC_MATRIX_DIAGONAL_INDICES.size() -1)/2.0) + 1)* num_timesteps ;
  for(auto i = 0u; i < ACC_MATRIX_DIAGONAL_INDICES.size() ; i++)
  {
    fill_diagonal(A,ACC_MATRIX_DIAGONAL_VALUES[i],ACC_MATRIX_DIAGONAL_INDICES[i]);
  }

  // create and scale covariance matrix
  Eigen::MatrixXd covariance = MatrixXd::Identity(num_timesteps,num_timesteps);
  covariance = A.transpose() * A;
  covariance = covariance.fullPivLu().inverse();
  double max_val = covariance.array().abs().matrix().maxCoeff();
  covariance /= max_val;

  // preallocating noise data
  raw_noise_ = Eigen::VectorXd::Zero(CARTESIAN_DOF_SIZE);

  error_code.val = error_code.SUCCESS;

  return true;
}

bool CartesianConstraintsSampling::setupRobotState(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                               const moveit_msgs::MotionPlanRequest &req,
                                               const stomp_core::StompConfiguration &config,
                                               moveit_msgs::MoveItErrorCodes& error_code)
{
  using namespace moveit::core;
  using namespace stomp_kinematics::kinematics;

  // robot state
  const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_);
  tool_link_ = joint_group->getLinkModelNames().back();
  state_.reset(new RobotState(robot_model_));
  robotStateMsgToRobotState(req.start_state,*state_);

  // update kinematic model
  ik_solver_->setKinematicState(*state_);

  return true;
}

bool CartesianConstraintsSampling::generateNoise(const Eigen::MatrixXd& parameters,
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

  if (is_first_trajectory_)
  {
    const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_);
    initial_trajectory_.resize(parameters.cols());
    for (int i = 0; i < parameters.cols();i++)
    {
      state_->setJointGroupPositions(joint_group,parameters.col(i));
      initial_trajectory_[i] = state_->getFrameTransform(tool_link_);
    }
    is_first_trajectory_ = false;
  }

  for(auto t = 0u; t < parameters.cols();t++)
  {
    const auto noisy_tool_pose = applyCartesianNoise(parameters.col(t));

    if (!inTolerance(noisy_tool_pose, initial_trajectory_.at(t)))
    {
      ROS_DEBUG("%s noisy tool pose is out of tolerance, returning noiseless goal pose",getName().c_str());
      parameters_noise.col(t) = parameters.col(t);
      continue;
    }

    Eigen::VectorXd result = Eigen::VectorXd::Zero(parameters.rows());
    if(!ik_solver_->solve(parameters.col(t),noisy_tool_pose,result,tool_goal_tolerance_))
    {
      ROS_ERROR("%s could not compute ik, returning noiseless goal pose",getName().c_str());
      parameters_noise.col(t) = parameters.col(t);
    }
    else
    {
      parameters_noise.col(t) = result;
    }
  }

  // generating noise
  noise = parameters_noise - parameters;

  // TODO test
    // Evaluation experiment
    std::ofstream outfile ("/home/michaldobis/michal_ws/new2.txt", std::fstream::app);
    double translation_tolerance = 1.0;
    double rotation_tolerance = 0.2;
    bool out_of_bounds = false;
//    ROS_INFO_STREAM("Init: " << )
    const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_);

    for (int i=0; i < parameters_noise.cols(); i++) {
//        Eigen::Isometry3d init_pose, result_pose;
//        state_->setJointGroupPositions(joint_group,initial_trajectory_.col(i));
//        Eigen::Affine3d init_pose = state_->getFrameTransform(tool_link_);

        const auto& init_pose = initial_trajectory_[i];
        state_->setJointGroupPositions(joint_group,parameters_noise.col(i));
        Eigen::Affine3d result_pose = state_->getFrameTransform(tool_link_);

//        fk_solver_->solve( initial_trajectory_.col(i),init_pose);
//        fk_solver_->solve( parameters_noise.col(i),result_pose);

        const auto diff = init_pose.inverse() * result_pose;
        const Eigen::AngleAxisd rv(diff.rotation());
        if ( !inTolerance(result_pose, init_pose)) {
            outfile << std::to_string(i) << ". timestep is out of tolerance . Distance: "<< diff.translation().norm() << " Angle: " << rv.angle() << std::endl;
            ROS_WARN_STREAM(std::to_string(i) << ". timestep is out of tolerance . Distance: "<< diff.translation().norm() << " Angle: " << rv.angle());
            out_of_bounds = true;
        }
    }

    if (out_of_bounds) {
        outfile << std::endl;
    }
  //-------------------------
  return true;
}

Eigen::Affine3d CartesianConstraintsSampling::applyCartesianNoise(const Eigen::VectorXd& reference_joint_pose)
{
  using namespace Eigen;
  using namespace moveit::core;
  using namespace stomp_moveit::utils;

  const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_);

  for(auto d = 0u; d < raw_noise_.size(); d++)
  {
    raw_noise_(d) = stddev_[d]*(*goal_rand_generator_)();
  }

  state_->setJointGroupPositions(joint_group,reference_joint_pose);
  Eigen::Affine3d tool_pose = state_->getFrameTransform(tool_link_);

  auto& n = raw_noise_;
  return tool_pose * Translation3d(Vector3d(n(0),n(1),n(2)))*AngleAxisd(n(3),Vector3d::UnitX())*AngleAxisd(n(4),Vector3d::UnitY())*AngleAxisd(n(5),Vector3d::UnitZ());
}

bool CartesianConstraintsSampling::inTolerance(const Eigen::Affine3d& noisy_pose, const Eigen::Affine3d& initial_pose)
{
  const auto diff = initial_pose.inverse() * noisy_pose;
  const Eigen::AngleAxisd rv(diff.rotation());
  return fabs(diff.translation().norm()) <= translation_tolerance_ && fabs(rv.angle()) <= rotation_tolerance_;
}

} /* namespace noise_generators */
} /* namespace stomp_moveit */
