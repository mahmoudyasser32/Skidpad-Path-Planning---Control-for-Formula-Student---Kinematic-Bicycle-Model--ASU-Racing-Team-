
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry, Path
import math
import matplotlib.pyplot as plt
import numpy as np
import time

#####################################################################
####          Skeleton code for controller node                  ####
####    Feel free to change/ add functions as you wish           ####
####    You can also change the arguments of the functions       ####
####    Note: Don't alter the node name                          ####
#####################################################################

class Controller(Node):
    def __init__(self):
        super().__init__('controller')
        self.publisher_steering = self.create_publisher(Float32, 'steer', 10)
        self.publisher_throttle = self.create_publisher(Float32, 'throttle', 10)
        self.subscription_state = self.create_subscription(Odometry, 'state', self.stateCallback, 10)
        self.subscription_path = self.create_subscription(Path, 'path', self.pathCallback, 10)
        self.state = Odometry()
        self.waypoints = Path()
        self.targetIndex = 0
        self.targetPoint = [0.0, 0.0]

        self.max_steering_angle = 35.0 * math.pi / 180.0
        self.Bicycle_length = 2.5
        self.s_0 = 0.0
        self.lookahead_distance = 3.0

        self.target_speed = 1.0
        self.max_speed = 25.0
        self.time_step = 0.0
        self.time_start = time.time()
        self.kp = 0.5# Proportional gain
        self.ki = 0.0 # Integral gain
        self.kd = 0.7  # Derivative gain
        self.max_throttle = 1.0
        self.last_error = 0.0
        self.error_sum = 0.0
        self.min_index = 0
        self.i=0

        #x=[]
        #y=[]
        #self.i=0
        #plt.ion()
        #self.fig1 = plt.figure()
        #ax = self.fig1.add_subplot(111)
        #self.line1, = ax.plot(x,y, 'b-')
        #ax.set_xlim(0, 1000)
        #ax.set_ylim(0, 10)
        pass

    def stateCallback(self, state:Odometry):
        self.state = state
        if len(self.waypoints.poses) == 0:
            return
        else:
            self.purePursuit()
            self.pidController()
            pass
        pass

    def pathCallback(self, path:Path):
        self.min_index = 0
        self.targetIndex = 0
        self.target_speed = 1.0
        self.waypoints.poses = []
        self.waypoints = path
        #self.get_logger().info(f'Path received with {len(self.waypoints.poses)} waypoints')
        #self.destroy_subscription(self.subscription_path)
        self.min_index=0
        x=[]
        y=[]
        for i in range(0,len(self.waypoints.poses)):
            x.append(self.waypoints.poses[i].pose.position.x)
            y.append(self.waypoints.poses[i].pose.position.y)
        # if self.i == 0:
        #     plt.ion()
        #     self.fig = plt.figure()
        #     ax = self.fig.add_subplot(111)
        #     self.line1, = ax.plot(x,y, 'b-')
        #     self.line2, = ax.plot(self.state.pose.pose.position.x,self.state.pose.pose.position.y , 'g*')
        #     self.line3, = ax.plot(self.state.pose.pose.position.x,self.state.pose.pose.position.y ,'r-')
        #     ax.set_xlim(-30, 30)
        #     ax.set_ylim(-10, 40)
        #     # ax.set_xlim(0, 100)
        #     # ax.set_ylim(-30, 40)
        # self.line1.set_ydata(y)
        # self.line1.set_xdata(x)
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        # self.i+=1
        pass

    def pidController(self):
        # Calculate the time step
        self.time_step = time.time() - self.time_start
        self.time_start = time.time()
        # Calculate the error
        error = self.target_speed - self.state.twist.twist.linear.x

        # Calculate the error sum
        self.error_sum += error * self.time_step
        if time.time()%10 == 0:
            self.error_sum = 0.0
        
        # Calculate the error difference
        error_diff = (error - self.last_error) / self.time_step

        # Calculate the throttle
        throttle_cmd = self.kp * error + self.ki * self.error_sum + self.kd * error_diff

        # Publish throttle command
        throttle = Float32()
        throttle.data = max(-1.0, min(1.0, throttle_cmd))
        self.publisher_throttle.publish(throttle)

        # Update the last error
        self.last_error = error

        #self.line1.set_ydata(np.append(self.line1.get_ydata(),self.state.twist.twist.linear.x))
        #self.line1.set_xdata(np.append(self.line1.get_xdata(),self.i))
        #self.i+=1
        #self.fig.canvas.draw()
        #self.fig.canvas.flush_events()
        pass

    def searchTargetPoint(self):
        # Find the closest waypoint
        min_dist = float('inf')
        #self.get_logger().info(f'self.min_index is {self.min_index} ')
        if self.min_index == 0:
            for i in range(self.min_index,len(self.waypoints.poses)):
                dist = math.sqrt((self.waypoints.poses[i].pose.position.x - self.state.pose.pose.position.x)**2 + (self.waypoints.poses[i].pose.position.y - self.state.pose.pose.position.y)**2)
                if dist < min_dist:
                    min_dist = dist
                    self.min_index = i

        # Find the target point
        # self.get_logger().info(f'self.min_index is {self.min_index} ')
        for i in range(self.min_index,len(self.waypoints.poses)):
            dist = math.sqrt((self.waypoints.poses[i].pose.position.x - self.state.pose.pose.position.x)**2 + (self.waypoints.poses[i].pose.position.y - self.state.pose.pose.position.y)**2)
            if dist > self.lookahead_distance:
                self.targetIndex = i
                self.min_index = i
                break
        # self.get_logger().info(f'len(self.waypoints.poses)is {len(self.waypoints.poses)} ')
        # self.get_logger().info(f'self.targetIndex is {self.targetIndex} ')
        # Calculate the target point
        self.targetPoint[0] = self.waypoints.poses[self.targetIndex].pose.position.x
        self.targetPoint[1] = self.waypoints.poses[self.targetIndex].pose.position.y

    def purePursuit(self):
        # Search the target point
        self.searchTargetPoint()
        # Calculate the steering angle
        seta = math.atan2(2 * (self.state.pose.pose.orientation.w * self.state.pose.pose.orientation.z + self.state.pose.pose.orientation.x * self.state.pose.pose.orientation.y), 1 - 2 * (self.state.pose.pose.orientation.y**2 + self.state.pose.pose.orientation.z**2))
        alpha = math.atan2(self.targetPoint[1] - self.state.pose.pose.position.y, self.targetPoint[0] - self.state.pose.pose.position.x)-seta
        delta = math.atan2(2 * self.Bicycle_length * math.sin(alpha), self.lookahead_distance)

        # Publish steering command
        steering = Float32()
        steering.data = max(-self.max_steering_angle, min(self.max_steering_angle, delta))* 180/math.pi

        if self.targetIndex > len(self.waypoints.poses) - 3:
            self.target_speed = 0.0
        self.publisher_steering.publish(steering)


        # self.line2.set_ydata(self.targetPoint[1])
        # self.line2.set_xdata(self.targetPoint[0])
        # self.line3.set_ydata(np.append(self.line3.get_ydata(),self.state.pose.pose.position.y))
        # self.line3.set_xdata(np.append(self.line3.get_xdata(),self.state.pose.pose.position.x))
        # self.fig.canvas.draw()
        # self.fig.canvas.flush_events()
        pass

def main(args=None):
    rclpy.init(args=args)
    controller = Controller()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()
