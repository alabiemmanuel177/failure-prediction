import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROS_PACKAGE = ROOT / "ros_ws" / "src" / "failure_experiment"
if str(ROS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(ROS_PACKAGE))

