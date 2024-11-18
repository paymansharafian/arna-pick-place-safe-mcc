
import rospy
from std_msgs.msg import *
from geometry_msgs.msg import *
from sympy import Point3D

publishers = {}

def data_to_message(data):
    if isinstance(data, str):
        return String(data)
    elif isinstance(data, int):
        return Int32(data)
    elif isinstance(data, float):
        return Float32(data)
    elif isinstance(data, list):
        return Float32MultiArray(data=data)
    elif isinstance(data, dict):
        msg = String()
        msg.data = str(data)
        return msg
    elif isinstance(data, Point3D):
        msg = Point()
        msg.x = data.x
        msg.y = data.y
        msg.z = data.z
        return
    elif isinstance(data, tuple) and len(data) == 3:
        msg = Point()
        msg.x = data[0]
        msg.y = data[1]
        msg.z = data[2]
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")


def publish(topic, message):
    if topic not in publishers:
        publishers[topic] = rospy.Publisher(topic, message.__class__, queue_size=10)
    publishers[topic].publish(message)