#include <math.h>
#include <stomp_plugins/cost_functions/cartesian_distance.h>
#include <XmlRpcException.h>
#include <pluginlib/class_list_macros.h>
#include <ros/console.h>

PLUGINLIB_EXPORT_CLASS(stomp_moveit::cost_functions::CartesianDistance,stomp_moveit::cost_functions::StompCostFunction);

namespace stomp_moveit
{
namespace cost_functions
{

CartesianDistance::CartesianDistance():
    name_("CartesianDistance")
{
  // TODO Auto-generated constructor stub

}

CartesianDistance::~CartesianDistance()
{
  // TODO Auto-generated destructor stub
}

bool CartesianDistance::initialize(moveit::core::RobotModelConstPtr robot_model_ptr,
                        const std::string& group_name,XmlRpc::XmlRpcValue& config)
{
  group_name_ = group_name;
  robot_model_ = robot_model_ptr;

  return configure(config);
}

bool CartesianDistance::configure(const XmlRpc::XmlRpcValue& config)
{
  using namespace XmlRpc;

  try
  {
    auto get_double_param = [](const XmlRpcValue &params, const std::string &name) -> double
    {
      XmlRpcValue xml_value =  params[name];
      if (!xml_value.valid())
      {
        throw XmlRpc::XmlRpcException("Failed to load parameter [" + name + "]");
      }
      double value = static_cast<double>(xml_value);
      if (value < 0)
      {
        throw XmlRpc::XmlRpcException("The parameter [" + name + "] must be positive");
      }
      return value;
    };

    position_cost_weight_ = get_double_param(config, "position_cost_weight");
    orientation_cost_weight_ = get_double_param(config, "orientation_cost_weight");
    translation_tolerance_ = get_double_param(config, "translation_tolerance");
    rotation_tolerance_ = get_double_param(config, "rotation_tolerance");

    // Obtain tool link
    tool_link_ = static_cast<std::string>(config["link_id"]);
    if (tool_link_.empty())
    {
      ROS_ERROR("%s the 'link_id' parameter is empty",getName().c_str());
      return false;
    }

    // total weight
    cost_weight_ = position_cost_weight_ + orientation_cost_weight_;

  }
  catch(XmlRpc::XmlRpcException& e)
  {
    ROS_ERROR("%s failed to load parameters, %s",getName().c_str(),e.getMessage().c_str());
    return false;
  }

  return true;
}

bool CartesianDistance::setMotionPlanRequest(const planning_scene::PlanningSceneConstPtr& planning_scene,
                 const moveit_msgs::MotionPlanRequest &req,
                 const stomp_core::StompConfiguration &config,
                 moveit_msgs::MoveItErrorCodes& error_code)
{
  using namespace moveit::core;

  // Clear initial trajectory from previous task
  initial_trajectory_.clear();

  const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_name_);
  const int num_joints = joint_group->getActiveJointModels().size();
  state_.reset(new RobotState(robot_model_));
  robotStateMsgToRobotState(req.start_state,*state_);

  const std::vector<moveit_msgs::Constraints>& seed = req.trajectory_constraints.constraints;
  if(seed.empty())
  {
    ROS_ERROR("A seed trajectory was not provided");
    error_code.val = error_code.INVALID_GOAL_CONSTRAINTS;
    return false;
  }

  initial_trajectory_.resize(seed.size());
  for (size_t i = 0; i < seed.size(); ++i) {
    // Test the first point to ensure that it has all of the joints required
    auto n = seed.at(i).joint_constraints.size();
    if (n != num_joints)
    {
      ROS_ERROR("Seed trajectory index %lu does not have %lu constraints (has %lu instead).", i, num_joints, n);
      error_code.val = error_code.INVALID_GOAL_CONSTRAINTS;
      initial_trajectory_.clear();
      return false;
    }

    std::vector<double> jp(num_joints);
    for (size_t j = 0; j < num_joints; j++)
    {
      jp.at(j) = seed.at(i).joint_constraints.at(j).position;
    }
      state_->setJointGroupPositions(joint_group,jp);
      initial_trajectory_.at(i) = state_->getFrameTransform(tool_link_);
  }

  return true;
}

bool CartesianDistance::computeCosts(const Eigen::MatrixXd& parameters,
                          std::size_t start_timestep,
                          std::size_t num_timesteps,
                          int iteration_number,
                          int rollout_number,
                          Eigen::VectorXd& costs,
                          bool& validity)
{

  using namespace Eigen;
  using namespace moveit::core;

  const JointModelGroup* joint_group = robot_model_->getJointModelGroup(group_name_);

  if (initial_trajectory_.size() != parameters.cols()) {
      ROS_ERROR("Size of initial trajectory (%d) and stomp parameters (%d) does not match", initial_trajectory_.size(), parameters.cols());
      return false;
  }

  // Set default states
  costs.resize(parameters.cols());
  costs.setConstant(0.0);
  validity = true;

  // Calculate cost for each waypoint
  for(auto i = 0u; i < parameters.cols();i++)
  {
    // Obtain cartesian positions
    state_->setJointGroupPositions(joint_group,parameters.col(i));
    const auto tool_pose = state_->getFrameTransform(tool_link_);
    const auto &initial_pose = initial_trajectory_.at(i);

    // Compute diff matrix
    const auto diff = initial_pose.inverse() * tool_pose;
    const Eigen::AngleAxisd rv(diff.rotation());

    // Compute absolute error
    const double translation_error = diff.translation().norm();
    const double rotation_error = fabs(rv.angle());

    // Scale error
    const double translation_error_scaled = translation_error / translation_tolerance_;
    const double rotation_error_scaled = rotation_error / rotation_tolerance_;

    // Compute cost and validity
    costs(i) = translation_error_scaled*position_cost_weight_ + rotation_error_scaled * orientation_cost_weight_;
    validity = validity && translation_error <= translation_tolerance_ && rotation_error <= rotation_tolerance_;

    if (!validity)
    {
     ROS_DEBUG("Out of tolerance. translation error: %f, rotation error: %f", translation_error, rotation_error);
    }
  }

  return true;
}

void CartesianDistance::done(bool success,int total_iterations,double final_cost,const Eigen::MatrixXd& parameters)
{

}

} /* namespace cost_functions */
} /* namespace stomp_moveit */
