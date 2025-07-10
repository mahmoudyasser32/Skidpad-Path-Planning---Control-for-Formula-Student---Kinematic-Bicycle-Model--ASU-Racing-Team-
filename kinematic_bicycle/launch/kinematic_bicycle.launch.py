from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory('kinematic_bicycle')) # also tried without realpath
    rviz_path=base_path+'/bicycle.rviz'
    config = os.path.join(get_package_share_directory('kinematic_bicycle'), 'config', 'params.yaml')
    return LaunchDescription([
        Node(
            package='kinematic_bicycle',
            namespace='car',
            executable='kinematic_bicycle',
            name='main'
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            output='screen', 
            arguments=['-d'+str(rviz_path)]
        ),
        Node(
            package='kinematic_bicycle',
            namespace='',
            executable='send_cones',
            name='main'
        ),
        Node(
            package='kinematic_bicycle',
            namespace='',
            executable='path_gen',
            name='main',
            parameters=[
                {'pathTopic': '/path'},
                {'conesTopic': '/cones'},
                {'stateTopic': '/state'}
                #config
            ]
        ),
        Node(
            package='kinematic_bicycle',
            namespace='',
            executable='controller',
            name='main'
        )
    ])