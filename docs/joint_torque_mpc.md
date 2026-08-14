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

Keep the planner's dimensionless semantic coordinate separate from the wire
action. For ActAhead's convention, `latent=+1` means maximum closing force,
`latent=-1` maximum opening force, and decoding happens exactly once:

```python
target = JointGripperForceTarget.from_signed_effort_latent(
    latent,
    force_limit=80.0,  # deployment value; still checked against live limits
)
```

The resulting target stores and transmits only `force` in Newtons. It does not
transmit the latent or a desired width. Do not use
`GripperCommand.from_signed_action` for this path: that legacy positional API
uses the opposite sign convention (`+1` means open) and decodes to metres.

This matches the public Flexiv RDK semantics: `Gripper.Grasp(force)` is direct
force control, with positive force closing and negative force opening. Admission
is always checked against the connected gripper's live `min_force` and
`max_force`; the controller never assumes that a particular GN01 firmware
supports the full signed range.

This API supplies an execution mechanism, not a planner objective. Sampling a
torque and penalizing squared torque in a cost function are independent design
choices.

On success, `ExecutionResult.log` acknowledges the physical command endpoints
as `acknowledged_ending_joint_torque_nm` and
`acknowledged_ending_gripper_force_n`. `ExecutionResult.final_state` remains
the measured end state, including gripper width and measured force; callers
must not reconstruct measured aperture from the semantic latent.
