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
  state_ = std::make_shared<moveit::core::RobotState>(robot_model_);

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
    if(!state_->knowsFrameTransform(tool_link_))
    {
      ROS_ERROR("Frame '%s' is not part of the model",tool_link_.c_str());
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

  bool found_goal = false;

  // extracting goal joint values
  for(const auto& gc : req.goal_constraints) {

    // check joint constraints first
    if (!gc.joint_constraints.empty()) {

      // copying goal values into state
      for (auto j = 0u; j < gc.joint_constraints.size(); j++) {
        auto jc = gc.joint_constraints[j];
        state_->setVariablePosition(jc.joint_name, jc.position);
      }

      // extract FK and break loop if goal is found
      goal_ = state_->getFrameTransform(tool_link_);
      found_goal = true;
      break;
    }

    // check cartesian goal constraint
    if (!gc.position_constraints.empty() && !gc.orientation_constraints.empty()) {
        const auto pc = gc.position_constraints[0];
        const auto oc = gc.orientation_constraints[0];

        // assembling goal pose
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        geometry_msgs::Point p = pc.constraint_region.primitive_poses[0].position;
        tf::quaternionMsgToEigen(oc.orientation, q);
        goal_ = Eigen::Affine3d::Identity() * Eigen::Translation3d(Eigen::Vector3d(p.x, p.y, p.z)) * q;

        // verify target frame
        std::string frame_id = pc.header.frame_id;
        if(!state_->knowsFrameTransform(frame_id))
        {
            ROS_ERROR("Frame '%s' is not part of the model",frame_id.c_str());
            return false;
        }

        // transforming goal pose if target frame is differ
        std::string target_frame = robot_model_->getModelFrame();
        if(!frame_id.empty() && target_frame != frame_id)
        {
            Eigen::Affine3d root_to_frame = state_->getFrameTransform(frame_id);
            Eigen::Affine3d root_to_target = state_->getFrameTransform(target_frame);
            goal_ = (root_to_target.inverse()) * root_to_frame * goal_;
        }

        // break loop if goal is found
        found_goal = true;
        break;
    }
  }

  // extract start state
  robotStateMsgToRobotState(req.start_state,*state_);
  start_ = state_->getFrameTransform(tool_link_);
  ROS_ERROR_COND(!found_goal, "%s Unable to obtain goal state", getName().c_str());

  return found_goal;
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

  // Set default states
  costs.resize(parameters.cols());
  costs.setConstant(0.0);
  validity = true;

  // rotation part
  Eigen::Quaterniond start_quaternion(start_.rotation());
  Eigen::Quaterniond goal_quaternion(goal_.rotation());

  // Calculate cost for each waypoint
  for(auto i = 0u; i < parameters.cols();i++)
  {
    // Calculate ideal cartesian position
    double percentage = (double)i / (double)(parameters.cols()-1);
    Eigen::Isometry3d ideal_pose(start_quaternion.slerp(percentage, goal_quaternion));
      ideal_pose.translation() = percentage * goal_.translation() + (1 - percentage) * start_.translation();

    // Get cartesian position from input parameters
    state_->setJointGroupPositions(joint_group,parameters.col(i));
    const auto current_pose = state_->getFrameTransform(tool_link_);

    // Compute diff matrix
    const auto diff = ideal_pose.inverse() * current_pose;
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
    ROS_DEBUG_COND(!validity, "Out of tolerance. translation error: %f, rotation error: %f", translation_error, rotation_error);
  }
  return true;
}

void CartesianDistance::done(bool success,int total_iterations,double final_cost,const Eigen::MatrixXd& parameters)
{

}

} /* namespace cost_functions */
} /* namespace stomp_moveit */
