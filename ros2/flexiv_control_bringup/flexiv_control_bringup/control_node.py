"""flexiv_control ROS 2 bridge node.

Wraps a :class:`flexiv_control.Robot` and exposes it over ROS 2 so teams already
living in ROS 2 (e.g. an existing MoveIt Servo / ros2_control setup) can drive
the arm through the same action contract used everywhere else in this project.

ROS is intentionally an **optional overlay** here: the Python core needs no ROS
workspace at all. Use this node only if you want ROS topics/services/actions.

Interfaces
----------
Publishes
  ~/robot_state   flexiv_control_msgs/RobotState   (at ``state_rate`` Hz)
  ~/joint_states  sensor_msgs/JointState           (for RViz / standard tooling)
Services
  ~/set_mode      flexiv_control_msgs/SetMode
  ~/home          std_srvs/Trigger
  ~/stop          std_srvs/Trigger
Action
  ~/execute_cartesian_chunk  flexiv_control_msgs/ExecuteCartesianChunk
Subscribes
  ~/delta_twist_cmds  geometry_msgs/TwistStamped
      MoveIt-Servo-compatible Cartesian jog: each twist is applied as a short
      ``servo_cartesian_delta`` over one publish period. This makes the node a
      drop-in target for an existing SpaceMouse/MoveIt-Servo teleop pipeline.

Parameters
----------
  backend (str, "fake")        : flexiv_control backend ("fake" or "flexiv_rdk")
  robot_serial (str, "")       : Rizon serial (required for the real backend)
  control_hz (float, 100.0)    : control-loop rate handed to flexiv_control
  safety_profile (str, "tabletop_safe")
  state_rate (float, 50.0)     : RobotState / JointState publish rate (Hz)
  joint_names (str[], [])      : names for JointState (defaults to joint_1..N)

Community project, NOT affiliated with Flexiv Robotics.
"""
from __future__ import annotations

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from flexiv_control_msgs.action import ExecuteCartesianChunk
from flexiv_control_msgs.msg import RobotState as RobotStateMsg
from flexiv_control_msgs.srv import SetMode

from flexiv_control import (
    CartesianChunk,
    CartesianWaypoint,
    GripperCommand,
    Robot,
    RobotConfig,
)


class FlexivControlNode(Node):
    def __init__(self) -> None:
        super().__init__("flexiv_control_node")

        self.declare_parameter("backend", "fake")
        self.declare_parameter("robot_serial", "")
        self.declare_parameter("control_hz", 100.0)
        self.declare_parameter("safety_profile", "tabletop_safe")
        self.declare_parameter("state_rate", 50.0)
        self.declare_parameter("joint_names", [""])
        # ~/delta_twist_cmds semantics. "unitless" matches a MoveIt-Servo-style
        # joystick pipeline (values in [-1, 1] scaled by twist_scale_*, the
        # same convention as flexiv_ros2's Servo config); "speed_units" treats
        # the twist directly as m/s and rad/s.
        self.declare_parameter("twist_in_type", "unitless")
        self.declare_parameter("twist_scale_linear", 0.4)     # m/s at full deflection
        self.declare_parameter("twist_scale_rotational", 0.8) # rad/s at full deflection
        self.declare_parameter("twist_max_age", 0.25)         # s; 0 disables the check

        backend = self.get_parameter("backend").value
        serial = self.get_parameter("robot_serial").value
        control_hz = float(self.get_parameter("control_hz").value)
        profile = self.get_parameter("safety_profile").value

        self.get_logger().info(
            f"starting flexiv_control bridge: backend={backend} profile={profile}"
        )
        self.robot = Robot(
            RobotConfig(
                backend=backend,
                robot_sn=serial or "",
                control_hz=control_hz,
                default_safety_profile=profile,
            )
        )
        self.robot.connect()
        self.robot.acquire_lease("ros2_bridge")
        # A sensible default so jogging works immediately.
        self.robot.start_cartesian_impedance()

        n = int(self.robot.get_state().q.shape[0])
        jn = list(self.get_parameter("joint_names").value)
        self.joint_names = (
            jn if jn and jn[0] else [f"joint_{i + 1}" for i in range(n)]
        )

        cb = ReentrantCallbackGroup()
        self.state_pub = self.create_publisher(RobotStateMsg, "~/robot_state", 10)
        self.js_pub = self.create_publisher(JointState, "~/joint_states", 10)

        state_rate = float(self.get_parameter("state_rate").value)
        self.create_timer(1.0 / max(state_rate, 1.0), self._publish_state, callback_group=cb)

        self.create_subscription(
            TwistStamped, "~/delta_twist_cmds", self._on_twist, 10, callback_group=cb
        )
        self.create_service(SetMode, "~/set_mode", self._on_set_mode, callback_group=cb)
        self.create_service(Trigger, "~/home", self._on_home, callback_group=cb)
        self.create_service(Trigger, "~/stop", self._on_stop, callback_group=cb)

        self._action = ActionServer(
            self,
            ExecuteCartesianChunk,
            "~/execute_cartesian_chunk",
            execute_callback=self._execute_chunk,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            cancel_callback=lambda _g: CancelResponse.ACCEPT,
            callback_group=cb,
        )
        self.twist_in_type = str(self.get_parameter("twist_in_type").value)
        if self.twist_in_type not in ("unitless", "speed_units"):
            raise ValueError(f"twist_in_type must be 'unitless' or 'speed_units', got {self.twist_in_type!r}")
        self.twist_scale_linear = float(self.get_parameter("twist_scale_linear").value)
        self.twist_scale_rotational = float(self.get_parameter("twist_scale_rotational").value)
        self.twist_max_age = float(self.get_parameter("twist_max_age").value)

        self.get_logger().info("flexiv_control bridge ready")

    # -- publishers ----------------------------------------------------------
    def _publish_state(self) -> None:
        s = self.robot.get_state()
        now = self.get_clock().now().to_msg()

        msg = RobotStateMsg()
        msg.header.stamp = now
        msg.header.frame_id = "base"
        msg.q = s.q.tolist()
        msg.dq = s.dq.tolist()
        msg.tcp_pose = s.tcp_pose.tolist()
        msg.tcp_vel = s.tcp_vel.tolist()
        msg.wrench = s.wrench.tolist()
        msg.gripper_width = float(s.gripper_width)
        msg.gripper_is_moving = bool(s.gripper_is_moving)
        msg.control_mode = getattr(s.control_mode, "name", str(s.control_mode))
        msg.safety_status = getattr(s.safety_status, "name", str(s.safety_status))
        self.state_pub.publish(msg)

        js = JointState()
        js.header.stamp = now
        js.name = self.joint_names
        js.position = s.q.tolist()
        js.velocity = s.dq.tolist()
        self.js_pub.publish(js)

    # -- jog (MoveIt-Servo-compatible) --------------------------------------
    def _on_twist(self, msg: TwistStamped) -> None:
        # Drop stale commands: replaying an old twist after a hiccup would jerk
        # the arm. A zero stamp is allowed (some publishers do not stamp).
        if self.twist_max_age > 0.0 and (msg.header.stamp.sec or msg.header.stamp.nanosec):
            age = (self.get_clock().now() - rclpy.time.Time.from_msg(msg.header.stamp)).nanoseconds * 1e-9
            if age > self.twist_max_age:
                self.get_logger().warn(
                    f"Ignoring stale twist command ({age:.2f}s old)", throttle_duration_sec=2.0
                )
                return

        lin = msg.twist.linear
        ang = msg.twist.angular
        v = [lin.x, lin.y, lin.z, ang.x, ang.y, ang.z]
        if self.twist_in_type == "unitless":
            # Joystick convention: clip to [-1, 1], then scale to real speeds.
            v = [max(-1.0, min(1.0, x)) for x in v]
            v = [x * self.twist_scale_linear for x in v[:3]] + [
                x * self.twist_scale_rotational for x in v[3:]
            ]

        # Integrate the velocity over one control dt to get a per-tick delta.
        dt = self.robot.dt
        delta = [x * dt for x in v]
        if any(abs(d) > 1e-9 for d in delta):
            try:
                self.robot.servo_cartesian_delta(delta, duration=dt)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"jog rejected: {e}")

    # -- services ------------------------------------------------------------
    def _on_set_mode(self, req: SetMode.Request, resp: SetMode.Response):
        # The facade exposes only impedance starts, so dispatch the documented
        # ControlMode names explicitly and reject anything we cannot honour,
        # rather than silently collapsing every request to impedance.
        try:
            name = req.mode.strip().upper()
            realtime = bool(req.realtime) or name.startswith("RT_")
            if name in ("NRT_JOINT_IMPEDANCE", "RT_JOINT_IMPEDANCE"):
                self.robot.start_joint_impedance(realtime=realtime)
            elif name in ("NRT_CARTESIAN_MOTION_FORCE", "RT_CARTESIAN_MOTION_FORCE"):
                self.robot.start_cartesian_impedance(realtime=realtime)
            else:
                resp.ok = False
                resp.message = (
                    f"unsupported mode '{req.mode}'; this bridge can start only "
                    "NRT/RT_JOINT_IMPEDANCE or NRT/RT_CARTESIAN_MOTION_FORCE"
                )
                return resp
            resp.ok = True
            resp.message = f"mode set ({name}, realtime={realtime})"
        except Exception as e:  # noqa: BLE001
            resp.ok = False
            resp.message = str(e)
        return resp

    def _on_home(self, _req, resp: Trigger.Response):
        try:
            self.robot.home()
            resp.success = True
            resp.message = "homed"
        except Exception as e:  # noqa: BLE001
            resp.success = False
            resp.message = str(e)
        return resp

    def _on_stop(self, _req, resp: Trigger.Response):
        self.robot.stop()
        resp.success = True
        resp.message = "stopped"
        return resp

    # -- action --------------------------------------------------------------
    def _execute_chunk(self, goal_handle):
        g = goal_handle.request
        if g.safety_profile:
            self.robot.set_safety_profile(g.safety_profile)

        waypoints = []
        for w in g.waypoints:
            quat = None if w.hold_orientation else list(w.quaternion)
            grip = None
            if w.has_gripper and w.gripper >= 0.0:
                # Canonical mapping (see CartesianChunk.from_waypoint_array): normalised
                # [0,1] -> width in metres over the ~80 mm stroke; 1 = open.
                grip = GripperCommand(width=float(min(max(w.gripper, 0.0), 1.0)) * 0.08, force=20.0)
            kw = {"position": list(w.position), "quaternion": quat, "gripper": grip}
            if w.n_frames > 0:
                kw["n_frames"] = int(w.n_frames)
            else:
                kw["duration"] = float(w.duration)
            waypoints.append(CartesianWaypoint(**kw))

        chunk = CartesianChunk(waypoints=waypoints, frame=g.frame or "base")

        # Execute. The backend runs the chunk synchronously (atomic; no per-tick
        # hook), so we cannot interrupt mid-chunk. Honour an accepted cancel by
        # stopping and reporting CANCELED instead of falsely returning SUCCEEDED.
        result_msg = ExecuteCartesianChunk.Result()
        res = self.robot.execute_cartesian_chunk(chunk)

        if goal_handle.is_cancel_requested:
            self.robot.stop()
            goal_handle.canceled()
            result_msg.success = False
            result_msg.stop_reason = "canceled"
            return result_msg

        # The chunk is atomic, so there is no progressive per-tick feedback;
        # publish one terminal feedback with the final pose.
        fb = ExecuteCartesianChunk.Feedback()
        fb.tick = 1
        fb.current_pose = res.final_state.tcp_pose.tolist()
        goal_handle.publish_feedback(fb)

        goal_handle.succeed()
        result_msg.success = bool(res.success)
        result_msg.stop_reason = str(res.stop_reason)
        result_msg.path_tracking_error = float(res.path_tracking_error)
        result_msg.max_tcp_speed = float(res.max_tcp_speed)
        result_msg.max_wrench = float(res.max_wrench)
        result_msg.clipped = bool(res.clipped)
        return result_msg

    def destroy_node(self) -> bool:
        try:
            self.robot.stop()
            self.robot.disconnect()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FlexivControlNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
