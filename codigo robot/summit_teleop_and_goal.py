#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Tele-operación + envío de PoseStamped para Robotnik Summit XL-HL.
Compatible con ROS 1 y Python 2.7.

Teclas:
  w : adelante                 d : giro derecha in-place
  z : marcha atrás             a : giro izquierda in-place
  e : adelante-derecha (arco)  s : STOP
  q : adelante-izquierda       g : goal  (g x y yaw  ó pulsa g y escribe valores)
  x : salir
"""

from __future__ import print_function

import math
import sys
import rospy
import tf.transformations as tft
from geometry_msgs.msg import Twist, PoseStamped, Quaternion


# ─────────── utilidades PoseStamped ───────────
def yaw_deg_to_quat(yaw_deg):
    """Convierte yaw en grados a quaternion (x,y,z,w)."""
    q = tft.quaternion_from_euler(0.0, 0.0, math.radians(yaw_deg))
    return Quaternion(*q)


def make_pose_stamped(x, y, yaw_deg, frame_id="map"):
    """Genera un PoseStamped sellado con la hora actual."""
    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.orientation = yaw_deg_to_quat(yaw_deg)
    return pose


# ─────────── clase principal ───────────
class SummitTeleop(object):
    def __init__(self):
        rospy.init_node("summit_teleop_and_goal_py27", anonymous=True)

        self.pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.pub_goal = rospy.Publisher("/move_base_simple/goal",
                                        PoseStamped, queue_size=1)

        # Velocidades; se pueden sobreescribir con _linear_speed / _angular_speed
        self.v_lin = rospy.get_param("~linear_speed", 0.3)   # m/s
        self.v_ang = rospy.get_param("~angular_speed", 0.7)  # rad/s

        rospy.loginfo("Tele-op listo · v=%.2f m/s · ω=%.2f rad/s",
                      self.v_lin, self.v_ang)

    # ――― publicaciones Twist ―――
    def _publish_for(self, lin_x, ang_z, duration):
        twist = Twist()
        twist.linear.x = lin_x
        twist.angular.z = ang_z

        rate = rospy.Rate(10)  # 10 Hz
        end_time = rospy.Time.now() + rospy.Duration(duration)
        while rospy.Time.now() < end_time and not rospy.is_shutdown():
            self.pub_cmd.publish(twist)
            rate.sleep()
        self.stop()

    def stop(self):
        self.pub_cmd.publish(Twist())

    # ――― movimientos ―――
    def forward(self, t=2.0):
        self._publish_for(self.v_lin, 0.0, t)

    def backward(self, t=2.0):
        self._publish_for(-self.v_lin, 0.0, t)

    def forward_right(self, angle_deg=30, t=2.0):
        self._publish_for(self.v_lin, -math.radians(angle_deg) / t, t)

    def forward_left(self, angle_deg=30, t=2.0):
        self._publish_for(self.v_lin,  math.radians(angle_deg) / t, t)

    def spin_right(self, t=2.0):
        self._publish_for(0.0, -self.v_ang, t)

    def spin_left(self, t=2.0):
        self._publish_for(0.0,  self.v_ang, t)

    # ――― envío de goal ―――
    def send_goal(self, x, y, yaw_deg):
        goal = make_pose_stamped(x, y, yaw_deg)
        self.pub_goal.publish(goal)
        rospy.loginfo("Goal → (%.2f, %.2f, %.1f°) frame=%s",
                      x, y, yaw_deg, goal.header.frame_id)


# ─────────── interfaz de consola ───────────
MENU = """
w : adelante        d : giro derecha
z : marcha atrás    a : giro izquierda
e : fwd-derecha     q : fwd-izquierda
s : STOP            g : goal
x : salir
"""


def main():
    teleop = SummitTeleop()
    print(MENU)

    while not rospy.is_shutdown():
        try:
            line = raw_input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0].lower()

        # ――― tele-op ―――
        if cmd == "w":
            teleop.forward()
        elif cmd == "z":
            teleop.backward()
        elif cmd == "e":
            teleop.forward_right()
        elif cmd == "q":
            teleop.forward_left()
        elif cmd == "d":
            teleop.spin_right()
        elif cmd == "a":
            teleop.spin_left()
        elif cmd == "s":
            teleop.stop()

        # ――― goal ―――
        elif cmd == "g":
            try:
                if len(tokens) == 4:
                    x, y, yaw = map(float, tokens[1:4])
                else:
                    vals = raw_input("x y yaw_deg > ").strip().split()
                    x = float(vals[0]) if len(vals) > 0 else 0.0
                    y = float(vals[1]) if len(vals) > 1 else 0.0
                    yaw = float(vals[2]) if len(vals) > 2 else 0.0
                teleop.send_goal(x, y, yaw)
            except (ValueError, IndexError):
                print("⚠️  Valores numéricos incorrectos.")

        # ――― salir ―――
        elif cmd == "x":
            teleop.stop()
            break
        else:
            print(MENU)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")
