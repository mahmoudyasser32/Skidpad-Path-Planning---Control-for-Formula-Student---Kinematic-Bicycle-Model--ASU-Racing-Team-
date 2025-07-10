import rclpy
from rclpy.node import Node
import math
import numpy as np
from nav_msgs.msg import Odometry
from asurt_msgs.msg import LandmarkArray
from asurt_msgs.msg import Landmark


class SendCones(Node):
    def __init__(self):
        super().__init__('ConesPublisher')
        self.publisher_ = self.create_publisher(LandmarkArray, 'cones', 10)
        self.subscription_state = self.create_subscription(Odometry, 'state', self.stateCallback, 10)
        self.subscriptions
        self.cone_positions = [
            [0.0   , 1.5   ,  2], 
            [5.0   , 1.5   ,  2], 
            [13.45 , 1.5   ,  3], 
            [16.55 , 1.5   ,  3], 
            [25.0  , 1.5   ,  2], 
            [30.0  , 1.5   ,  2], 
            [35.0  , 1.5   ,  2], 
            [40.0  , 1.5   ,  2], 
            [40.0  , 0.5   ,  2], 
            [40.0  , -0.5    ,  2], 
            [0.0   , -1.5    ,  2], 
            [5.0   , -1.5    ,  2], 
            [13.45 , -1.5    ,  3], 
            [16.55 , -1.5    ,  3], 
            [25.0  , -1.5    ,  2], 
            [30.0  , -1.5    ,  2], 
            [35.0  , -1.5    ,  2], 
            [40.0  , -1.5    ,  2], 
            [15.0  , 1.5   ,  0], 
            [17.918, 2.08  ,  0], 
            [20.392, 3.733 ,  0], 
            [22.045, 6.207 ,  0], 
            [22.625, 9.125 ,  0], 
            [22.045, 12.043,  0], 
            [20.392, 14.517,  0], 
            [17.918, 16.17 ,  0], 
            [15.0  , 16.75 ,  0], 
            [12.082, 16.17 ,  0], 
            [9.608 , 14.517,  0], 
            [7.955 , 12.043,  0], 
            [7.375 , 9.125 ,  0],# 
            [7.955 , 6.207 ,  0], 
            [9.608 , 3.733 ,  0], 
            [12.082, 2.08  ,  0],  
            [17.918, -16.17  ,  1], 
            [20.392, -14.517 ,  1], 
            [22.045, -12.043 ,  1], 
            [22.625, -9.125  ,  1], 
            [22.045, -6.207  ,  1], 
            [20.392, -3.733  ,  1], 
            [17.918, -2.08   ,  1], 
            [15.0  , -1.5    ,  1], 
            [12.082, -2.08   ,  1], 
            [9.608 , -3.733  ,  1], 
            [7.955 , -6.207  ,  1], 
            [7.375 , -9.125  ,  1],
            [7.955 , -12.043 ,  1], 
            [9.608 , -14.517 ,  1], 
            [12.082, -16.17  ,  1], 
            [15.0  , -16.75  ,  1], 
            [22.399, 1.5   ,  1],
            [24.633, 4.642 ,  1], 
            [25.598, 8.373 ,  1], 
            [25.169, 12.204,  1], 
            [23.401, 15.629,  1], 
            [20.528, 18.199,  1], 
            [16.927, 19.574,  1], 
            [13.073, 19.574,  1], 
            [9.472 , 18.199,  1], 
            [6.599 , 15.629,  1], 
            [4.831 , 12.204,  1], 
            [4.402 , 8.373 ,  1],# 
            [5.367 , 4.642 ,  1], 
            [7.601 , 1.5   ,  1], 
            [7.601 , -1.5    ,  0], 
            [5.367 , -4.642  ,  0], 
            [4.402 , -8.373  ,  0], 
            [4.831 , -12.204 ,  0], 
            [6.599 , -15.629 ,  0], 
            [9.472 , -18.199 ,  0],
            [13.073, -19.574 ,  0], 
            [16.927, -19.574 ,  0], 
            [20.528, -18.199 ,  0], 
            [23.401, -15.629 ,  0], 
            [25.169, -12.204 ,  0], 
            [25.598, -8.373  ,  0], 
            [24.633, -4.642  ,  0], 
            [22.399, -1.5    ,  0]
            ]
        self.state=Odometry()
    def stateCallback(self, state:Odometry):
        self.state = state
        conePosition = LandmarkArray()
        nearCones = self.get_cones_near_pos([self.state.pose.pose.position.x,self.state.pose.pose.position.y])
        for i in nearCones:
            cone =Landmark()
            cone.position.x = float(i[0])
            cone.position.y = float(i[1])
            cone.type = int(i[2])
            conePosition.landmarks.append(cone)
        self.publisher_.publish(conePosition)
    def get_cones_near_pos(self,pos):
        cones_near_pos = []
        for i in self.cone_positions:
            if math.sqrt((pos[0]-i[0])**2+(pos[1]-i[1])**2) < 6 :
                # rd=np.random.rand()
                # if rd>0.9 and i[2]!=3:
                #     i=self.errorcolor(i)

                i=self.errorposition(i)
                cones_near_pos.append(i)
        return cones_near_pos   
    def generate_normal_distribution_around_point(self,point, threshold, std_dev):
        """
        Generates a random point (x, y) from a normal distribution centered around a given point,
        while ensuring that the farthest point from the given point has a distance equal to the specified threshold.

        Args:
            point: The (x, y) coordinates around which to generate the distribution.
            threshold: The desired threshold distance from the given point.
            std_dev: The desired standard deviation of the distribution.

        Returns:
            Tuple: A random (x, y) coordinates drawn from the normal distribution within the specified range around the given point.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.normal(loc=threshold, scale=std_dev)

        x = point[0] + distance * np.cos(angle)
        y = point[1] + distance * np.sin(angle)

        return [x, y, point[2]]
    def errorposition(self,point):
        point1=point
        threshold_distance = 0
        std_deviation = 0.1/3

        point1 = self.generate_normal_distribution_around_point(point1, threshold_distance, std_deviation)
        return point1  
    def errorcolor(self,point):
        new_point=point
        rd1=np.random.randint(0,100)
        if rd1>95:
            rd2=np.random.randint(0,100)
            if rd2<25:
                new_point[2]=4
            elif rd2<50:
                new_point[2]=0
            elif rd2<75:
               new_point[2]=1
            else:
                new_point[2]=2
        return new_point

def main(args=None):
    rclpy.init(args=args)

    cones = SendCones()

    rclpy.spin(cones)

    cones.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()