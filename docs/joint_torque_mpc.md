# Direct joint-torque MPC execution

`flexiv-control` exposes `JointTorqueTrajectory` for controllers whose sampled
action is physical joint effort rather than a joint-position or velocity
target. It is intentionally separate from `JointTrajectory`: callers cannot
accidentally reinterpret torque as position.

Hardware execution is fail closed unless all of the following hold:

- the robot configuration explicitly sets `allow_joint_torque: true`;
- the server runs at exactly 1 kHz;
- Flexiv RDK exposes `RT_JOINT_TORQUE` and `StreamJointTorque`;
- live `RobotInfo.tau_max` exists and has one positive value per joint;
- the requested `max_joint_torque_scale` does not exceed the active profile;
- every initial/waypoint torque is within the scaled live limits;
- the normal lease, E-stop, fault, wrench, cancellation, and soft-limit gates
  remain active.

The RDK call enables nonlinear dynamics compensation and firmware soft limits.
Waypoint endpoints are linearly connected at 1 kHz because Flexiv requires
smooth continuous torque commands. Gripper force events are synchronized with
the same segment boundaries and remain physical Newton commands, not width
targets.

This API supplies an execution mechanism, not a planner objective. Sampling a
torque and penalizing squared torque in a cost function are independent design
choices.
