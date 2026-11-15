#!/usr/bin/env python3

import select
import sys
import termios
import time
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


HELP = """
CS7001 Assignment 3 Teleop

W : increase forward speed
X : increase backward speed
S : stop all motion

A : turn left while held
D : turn right while held

Q : quit

Click this terminal before using the keys.
"""


class Assignment3Teleop(Node):

    LINEAR_STEP = 0.02
    MAX_FORWARD_SPEED = 0.17
    MAX_REVERSE_SPEED = 0.12

    ANGULAR_SPEED = 0.45

    def __init__(self):
        super().__init__("assignment3_teleop")

        self.pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.linear = 0.0
        self.angular = 0.0

        self.last_turn_key = 0.0

        # Terminal keyboards do not provide true key-release events.
        # Repeated A/D presses while held refresh this timer.
        self.turn_release_timeout = 0.20

    def publish(self):
        msg = Twist()
        msg.linear.x = float(self.linear)
        msg.angular.z = float(self.angular)
        self.pub.publish(msg)

    def stop(self):
        self.linear = 0.0
        self.angular = 0.0
        self.publish()

    def handle_key(self, key):
        now = time.monotonic()

        if key == "w":
            self.linear += self.LINEAR_STEP
            self.linear = min(
                self.linear,
                self.MAX_FORWARD_SPEED,
            )

            print(
                f"FORWARD  {self.linear:.2f} m/s"
            )

        elif key == "x":
            self.linear -= self.LINEAR_STEP
            self.linear = max(
                self.linear,
                -self.MAX_REVERSE_SPEED,
            )

            print(
                f"REVERSE  {self.linear:.2f} m/s"
            )

        elif key == "s":
            self.linear = 0.0
            self.angular = 0.0
            print("STOP")

        elif key == "a":
            self.angular = self.ANGULAR_SPEED
            self.last_turn_key = now

        elif key == "d":
            self.angular = -self.ANGULAR_SPEED
            self.last_turn_key = now

        elif key == "q":
            self.stop()
            return False

        return True


def main(args=None):
    rclpy.init(args=args)

    node = Assignment3Teleop()

    old_settings = termios.tcgetattr(sys.stdin)

    print(HELP)
    print(
        "Limits: "
        "forward<=0.17 m/s, "
        "reverse<=0.12 m/s, "
        "linear step=0.02 m/s, "
        "turn=0.45 rad/s"
    )
    print("\nTeleop ready.\n")

    running = True

    try:
        tty.setcbreak(sys.stdin.fileno())

        while rclpy.ok() and running:

            ready, _, _ = select.select(
                [sys.stdin],
                [],
                [],
                0.02,
            )

            if ready:
                key = sys.stdin.read(1).lower()
                running = node.handle_key(key)

            # A/D release approximation.
            if node.angular != 0.0:
                if (
                    time.monotonic() - node.last_turn_key
                    > node.turn_release_timeout
                ):
                    node.angular = 0.0

            # Keep publishing the current commanded velocity.
            node.publish()

            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

    except KeyboardInterrupt:
        pass

    finally:
        node.stop()

        termios.tcsetattr(
            sys.stdin,
            termios.TCSADRAIN,
            old_settings,
        )

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
