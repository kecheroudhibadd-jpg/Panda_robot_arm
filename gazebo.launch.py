import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    urdf_file = '/home/kisrane/panda_ws/fr3.urdf'

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # تشغيل Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', '-r', 'empty.sdf'],
            output='screen'
        ),

        # robot_state_publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description,
                        'use_sim_time': True}]
        ),

        # spawn الروبوت في Gazebo
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=[
                '-name', 'fr3',
                '-file', '/home/kisrane/panda_ws/fr3.urdf',
                '-z', '0.5',
            ],
            output='screen'
        ),
    ])
