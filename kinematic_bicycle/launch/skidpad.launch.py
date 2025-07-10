from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(get_package_share_directory("kinematic_bicycle"), "config", "params.yaml")
    return LaunchDescription([
        Node(
            package='kinematic_bicycle',
            namespace='',
            executable='path_gen',
            name='main',
            parameters=[
                # {'pathTopic': '/path'},
                # {'conesTopic': '/cones'},
                # {'stateTopic': '/state'}
                config
            ]
        )
    ])