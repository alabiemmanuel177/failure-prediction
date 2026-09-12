#!/usr/bin/env python3
"""Log (wall_unix_seconds, simulation_seconds) pairs from /clock at 10 Hz for capture alignment."""
import sys, time
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock

class Logger(Node):
    def __init__(self, path):
        super().__init__("demo_clock_wall_logger")
        self.out = open(path, "w"); self.out.write("wall_s,sim_s\n"); self.last = 0.0
        self.create_subscription(Clock, "/clock", self.cb, 10)
    def cb(self, msg):
        now = time.time()
        if now - self.last >= 0.1:
            self.last = now
            self.out.write(f"{now:.3f},{msg.clock.sec + msg.clock.nanosec / 1e9:.3f}\n"); self.out.flush()

rclpy.init(); node = Logger(sys.argv[1])
try: rclpy.spin(node)
except KeyboardInterrupt: pass
