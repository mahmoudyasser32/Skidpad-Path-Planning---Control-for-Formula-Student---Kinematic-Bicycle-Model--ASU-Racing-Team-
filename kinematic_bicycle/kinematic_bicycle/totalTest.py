# import matplotlib.pyplot as plt
from typing import Tuple
import math
import numpy as np
import matplotlib.pyplot as plt

from nav_msgs.msg import Odometry

class SendPath():
    def __init__(self) -> None:
        self.state=Odometry()
        self.count = 0
        self.pastPos = None
        self.origin = np.array([0,0])
        self.i=0
    def merge_points(self,
                     points:np.ndarray,
                     threshold:float)->np.ndarray:
        """
        Merges points that are close to each other based on a given threshold.

        Args:
            points (np.ndarray): Array of points.
            threshold (float): Maximum distance between points to merge.

        Returns:
            np.ndarray: Merged points array.
        """
        n = len(points)

        # Create a boolean array to keep track of merged points
        merged = np.zeros(n, dtype=bool)

        # Initialize merged points array with zeros
        merged_points = np.zeros((0, points.shape[1]), dtype=points.dtype)

        # Iterate over each point
        for i in range(n):
            # Skip if the current point has already been merged
            if merged[i]:
                continue

            # Add the current point to the merged points array
            merged_points = np.vstack((merged_points, points[i]))

            # Get the color of the current point
            color = points[i, 2]

            # Compare the current point with the remaining points
            for j in range(i+1, n):
                # Skip if the point has already been merged
                if merged[j]:
                    continue

                # Calculate the Euclidean distance between the points
                distance = np.linalg.norm(points[i, :2] - points[j, :2])

                # Merge the points if they are close enough
                if distance <= threshold:
                    merged[j] = True
                    # Update the color of the merged point with the maximum color
                    color = max(color, points[j, 2])

            # Update the color of the merged point with the maximum color
            merged_points[-1, 2] = color

        return merged_points     
    def conesClassification(self,cones: np.ndarray)->Tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:
        """
        Classifies cones based on their position relative to the robot's pose.

        Returns:
            Tuple: A tuple containing the classified cones arrays in the following order:
            - rightBlueCones: Array of blue cones on the right side of the robot.
            - leftBlueCones: Array of blue cones on the left side of the robot.
            - rightYellowCones: Array of yellow cones on the right side of the robot.
            - leftYellowCones: Array of yellow cones on the left side of the robot.
            - orangeCones: Array of orange cones.
            - bigOrange: Array of large orange cones.
            - unknownCones: Array of unknown cones.
        """
        # filter_outliers cones based on position and color
                
        rightBlueCones = np.array(cones[cones[:, 2] == 0][cones[cones[:, 2] == 0][:, 0] > 0])
        leftBlueCones = np.array(cones[cones[:, 2] == 0][cones[cones[:, 2] == 0][:, 0] <= 0])
        rightYellowCones = np.array(cones[cones[:, 2] == 1][cones[cones[:, 2] == 1][:, 0] > 0])
        leftYellowCones = np.array(cones[cones[:, 2] == 1][cones[cones[:, 2] == 1][:, 0] <= 0])
        orangeCones = np.array(cones[cones[:, 2] == 2])
        bigOrange = np.array(cones[cones[:, 2] == 3])
        unknownCones = np.array(cones[cones[:, 2] == 4])

        # Sort cones based on distance from robot's pose
        rightBlueCones = rightBlueCones[np.argsort((rightBlueCones[:, 0] - self.state.pose.pose.position.x)**2 + (rightBlueCones[:, 1] - self.state.pose.pose.position.y)**2)]
        leftBlueCones = leftBlueCones[np.argsort((leftBlueCones[:, 0] - self.state.pose.pose.position.x)**2 + (leftBlueCones[:, 1] - self.state.pose.pose.position.y)**2)]
        rightYellowCones = rightYellowCones[np.argsort((rightYellowCones[:, 0] - self.state.pose.pose.position.x)**2 + (rightYellowCones[:, 1] - self.state.pose.pose.position.y)**2)]
        leftYellowCones = leftYellowCones[np.argsort((leftYellowCones[:, 0] - self.state.pose.pose.position.x)**2 + (leftYellowCones[:, 1] - self.state.pose.pose.position.y)**2)]
        orangeCones = orangeCones[np.argsort((orangeCones[:, 0] - self.state.pose.pose.position.x)**2 + (orangeCones[:, 1] - self.state.pose.pose.position.y)**2)]
        bigOrange = bigOrange[np.argsort((bigOrange[:, 0] - self.state.pose.pose.position.x)**2 + (bigOrange[:, 1] - self.state.pose.pose.position.y)**2)]
        unknownCones = unknownCones[np.argsort((unknownCones[:, 0] - self.state.pose.pose.position.x)**2 + (unknownCones[:, 1] - self.state.pose.pose.position.y)**2)]

        return rightBlueCones, leftBlueCones, rightYellowCones, leftYellowCones, orangeCones, bigOrange ,unknownCones
    def findOrangeNodes(self,Orangeconesmap: np.ndarray)->np.ndarray:
        """
        Finds orange nodes based on specific conditions from the given orange cones map.

        Args:
            Orangeconesmap (numpy.ndarray): Array containing nodes information.

        Returns:
            numpy.ndarray: Array of orange nodes that satisfy the conditions.
        """
        # Initialize an empty array to store orange nodes
        OrangeNodes = np.zeros((0, 3))
        
        for cone1 in Orangeconesmap:
            lowestDist = float('inf')
            nearestCone = np.zeros((0,))

            # Find the nearest cone to the current cone
            for cone2 in Orangeconesmap:
                if not np.array_equal(cone1, cone2):
                    dist = math.sqrt((cone1[0] - cone2[0]) ** 2 + (cone1[1] - cone2[1]) ** 2)
                    if dist < lowestDist:
                        lowestDist = dist
                        nearestCone = cone2
            # Calculate the average position between the current cone and the nearest cone
            if not np.array_equal(nearestCone, np.zeros((0,))):
                avg_x = round((cone1[0] + nearestCone[0]) / 2, 2)
                avg_y = round((cone1[1] + nearestCone[1]) / 2, 2)
                new_node = np.array([[avg_x, avg_y, cone1[2]]])

                # Add the new node to the orange nodes array if it does not already exist
                if not np.any(np.all(OrangeNodes[:, :2] == new_node[:, :2], axis=1)):
                    OrangeNodes = np.concatenate((OrangeNodes, new_node), axis=0)
        
        return OrangeNodes
    def dist (self,seta1:float,seta2:float)->float:
        """
        Calculates the distance between two angles.

        Args:
            seta1 (float): First angle.
            seta2 (float): Second angle.

        Returns:
            float: Distance between the two angles.
        """
        return abs(seta1 - seta2)
    def linePath(self,OrangeNodes: np.ndarray, pos: np.ndarray)->np.ndarray:
        """
        Generates a line path based on orange nodes and a given position.

        Args:
            orangeNodes (np.ndarray): Array of orange nodes.
            pos (list): Current position.

        Returns:
            np.ndarray: Array representing the line path.
        """
        # Initialize variables
        x = OrangeNodes[:, 0]
        y = OrangeNodes[:, 1]
        noNeedIndex = np.array([], dtype=int)
        path = np.zeros((0, 2))

        # Check if there are no orange nodes
        if OrangeNodes.shape[0] < 1:
            return path
        
        # filter_outliers orange nodes based on position
        for i in range(len(OrangeNodes)):
            if OrangeNodes[i, 1] < pos[1]:
                noNeedIndex = np.append(noNeedIndex, i)
        noNeedIndex = np.flip(noNeedIndex)
        OrangeNodes = np.delete(OrangeNodes, noNeedIndex, axis=0)

        # Find the end point of the path
        if self.count < 4 and self.origin[1] != 0:
            end = self.origin[1]
        else:
            end = OrangeNodes[:, 1].max()
            
        # Generate the path
        try:
            A=np.c_[x,np.ones_like(x)]
            a,b = np.linalg.solve(A.T@A,A.T@y)
        except:
            a = 0
            b = 0
            
        y = np.linspace(float(pos[1]), end, int(abs(pos[1]-end) + 1) * 5)

        if round(a,1) != 0.0:
            path = np.column_stack(((y - b) / a, y))
        else:
            path = np.column_stack((np.full_like(y, pos[0]), y))
        return path
    def fitCircle(self,points: np.ndarray)->Tuple[float,float,float]:
        """
        Fits a circle to the given points using a least squares method.

        Args:
            points (np.ndarray): Array of points.

        Returns:
            Tuple[float, float, float]: x-coordinate of the center, y-coordinate of the center, and radius of the circle.
        """
        # Fit a circle to the given points
        A=np.c_[-2*points[:,0],-2*points[:,1],np.ones_like(points[:,0])]
        # Solve the linear system of equations
        x0,y0,b = np.linalg.solve(A.T@A,A.T@(-points[:,0]**2-points[:,1]**2))
        # Calculate the radius of the circle
        r=np.sqrt(x0**2+y0**2-b)

        # get deviation of each point from the circle
        for point in points:
            deviation = math.sqrt((point[0]-x0)**2 + (point[1]-y0)**2) - r
            if deviation > 0.05:
                points = np.delete(points, np.where((points == point).all(axis=1)), axis=0)
                x0,y0,r = self.fitCircle(points)

        return x0,y0,r
    def meanCircles(self,outerCones: np.ndarray, innerCones: np.ndarray)->Tuple[float,float,float]:
        """
        Calculates the mean circle from outer and inner cones.

        Args:
            outer_cones (np.ndarray): Array of outer cones.
            inner_cones (np.ndarray): Array of inner cones.

        Returns:
            Tuple[float, float, np.ndarray]: x-coordinate of the center mean, y-coordinate of the center mean,
            and array of mean radii.
        """
        # Initialize arrays
        # # x0 = np.empty(0)
        # # y0 = np.empty(0)
        r1 = np.empty(0)
        r0 = np.empty(0)
        x0 = np.empty(2)
        y0 = np.empty(2)

        # Fit circles to the given cones
        x0[0], y0[0], r1 = self.fitCircle(outerCones)
        x0[1], y0[1], r0 = self.fitCircle(innerCones)

        # # # Calculate the center and radius of outer circle that connect each three cones
        # # for i in range(len(outerCones) - 2):
        # #     x, y, r = self.circle(np.array([outerCones[i], outerCones[i + 1], outerCones[i + 2]]))
        # #     x0 = np.append(x0, x)
        # #     y0 = np.append(y0, y)
        # #     r1 = np.append(r1, r)

        # # # Calculate the center and radius of inner circle that connect each three cones
        # # for i in range(len(innerCones) - 2):
        # #     x, y, r = self.circle(np.array([innerCones[i], innerCones[i + 1], innerCones[i + 2]]))
        # #     x0 = np.append(x0, x)
        # #     y0 = np.append(y0, y)
        # #     r0 = np.append(r0, r)

        # # # Filter outliers
        # # x0 = self.filter_outliers(x0)
        # # y0 = self.filter_outliers(y0)
        # # r1 = self.filter_outliers(r1)
        # # r0 = self.filter_outliers(r0)

        # Calculate the mean of the circle
        x0_mean = np.mean(x0)
        y0_mean = np.mean(y0)
        # # r1_mean = np.mean(r1)
        # # r0_mean = np.mean(r0)
        

        # #reduisMean = (r1_mean + r0_mean) / 2
        reduisMean = (r1 + r0) / 2

        return x0_mean, y0_mean, reduisMean
    def circlePath(self,outerCones: np.ndarray, innerCones: np.ndarray, pos: np.ndarray=np.array([0, 0]))->np.ndarray:
        """
        Generates a circular path based on outer and inner cones.

        Args:
            outerCones (np.ndarray): Array of outer cones.
            innerCones (np.ndarray): Array of inner cones.
            pos (np.ndarray, optional): Position array. Defaults to np.array([0, 0]).

        Returns:
            np.ndarray: Circular path represented by an array of points.
        """
        # Initialize variables
        path = np.empty((0, 2))
        x = np.empty(0)
        y1 = np.empty(0)
        y2 = np.empty(0)

        # Calculate the mean circle
        x0, y0, Rm = self.meanCircles(outerCones, innerCones)

        if np.array_equal(pos, [0, 0]) and outerCones[0, 2] == 0:
            pos = np.array([x0 - Rm, y0])
        elif np.array_equal(pos, [0, 0]) and outerCones[0, 2] == 1:
            pos = np.array([x0 + Rm, y0])

        # Calculate the start angle
        if round((pos[0] - x0), 3) == 0:
            if pos[1] > y0:
                start = math.pi / 2
            else:
                start = -math.pi / 2
        else:
            start = math.atan(round((pos[1] - y0) / (pos[0] - x0), 3))

        # Generate the circular path
        if outerCones[0, 2] == 0:
            if round((pos[0] - x0), 3) < 0 and round((pos[1] - y0), 3) < 0:
                start = start - math.pi
            elif round((pos[0] - x0), 3) < 0 and round((pos[1] - y0), 3) >= 0:
                start = start + math.pi

            seta = np.linspace(start, -math.pi, int(self.dist(start, -math.pi) + 1) * 5)
            for i in seta:
                x = np.append(x, Rm * np.cos(i) + x0)
                if i < 0:
                    y2 = np.append(y2, -math.sqrt(round(Rm ** 2 - (x[-1] - x0) ** 2, 3)) + y0)
                else:
                    y1 = np.append(y1, math.sqrt(round(Rm ** 2 - (x[-1] - x0) ** 2, 3)) + y0)

        else:
            if start < 0:
                start = start + 2 * math.pi
            if round((pos[0] - x0), 3) < 0 and round((pos[1] - y0), 3) < 0:
                start = start + math.pi
            elif round((pos[0] - x0), 3) < 0 and round((pos[1] - y0), 3) >= 0:
                start = start - math.pi

            seta = np.linspace(start, 2 * math.pi, int(self.dist(start, 2 * math.pi) + 1) * 5)
            for i in seta:
                x = np.append(x, Rm * np.cos(i) + x0)
                if i > math.pi:
                    y2 = np.append(y2, -math.sqrt(round(Rm ** 2 - (x[-1] - x0) ** 2, 3)) + y0)
                else:
                    y1 = np.append(y1, math.sqrt(round(Rm ** 2 - (x[-1] - x0) ** 2, 3)) + y0)

        # Generate the path
        path = np.vstack((np.column_stack((x[:len(y1)], y1)), np.column_stack((x[len(y1):], y2))))
        return path
    def counter(self, pos: np.ndarray, bigOrange: np.ndarray)->None:
        """
        Counts the number of orange nodes passed in and updates internal state.

        Args:
            pos (np.array): Current position [x, y].
            bigOrange (np.array): Array of big orange nodes.

        Returns:
            None
        """
        # get the orange nodes
        counter_OrangeNodes = self.findOrangeNodes(bigOrange)
        
        # Check if the robot has passed an orange node and update the counter and origin
        if np.array_equal(self.origin, np.array([0, 0])):
            if len(counter_OrangeNodes) < 1:
                return None
            else:
                # Update the origin if two orange nodes are being seen
                if len(counter_OrangeNodes) == 2:
                    self.origin = np.array([(counter_OrangeNodes[0][0] + counter_OrangeNodes[1][0]) / 2, (counter_OrangeNodes[0][1] + counter_OrangeNodes[1][1]) / 2])
        # Check if the robot has passed origin
        else:
            if (pos[1] > self.origin[1]) and (self.pastPos[1] < self.origin[1]) and (round(pos[0], 0) < round(self.origin[0] + 1, 0) and round(pos[0], 0) > round(self.origin[0] - 1, 0)):
                self.count += 1
                print(self.count)
        return None     
    def path(self, 
             rightBlueCones: np.ndarray, 
             leftBlueCones: np.ndarray, 
             rightYellowCones: np.ndarray, 
             leftYellowCones: np.ndarray, 
             orangeCones: np.ndarray,
             bigOrange: np.ndarray,
             unknownCones: np.ndarray)-> np.ndarray :
        """
        Generates a path based on the given cones and current state.

        Args:
            rightBlueCones (np.array): Right blue cones.
            leftBlueCones (np.array): Left blue cones.
            rightYellowCones (np.array): Right yellow cones.
            leftYellowCones (np.array): Left yellow cones.
            orangeCones (np.array): Orange cones.
            bigOrange (np.array): Big orange cones.
            unknownCones (np.array): Unknown cones.

        Returns:
            np.array: Generated path as a NumPy array.
        """
        # Initialize variables
        path = np.empty((0, 2))  # Initialize an empty NumPy array
        # Get the current position
        pos = np.array([self.state.pose.pose.position.x, self.state.pose.pose.position.y])
        # sum all orange cones(big and small)
        orange = np.concatenate((orangeCones, bigOrange), axis=0)
        # Sort orange cones based on distance from robot's pose
        orange = orange[np.lexsort(((orange[:, 1] - pos[1]) ** 2 + (orange[:, 0] - pos[0]) ** 2,))]  
        # Find orange nodes
        OrangeNodes = self.findOrangeNodes(orange)     
        # update the counter if there are more than 2 big orange cones
        if len(bigOrange) > 2:
            self.counter(pos, bigOrange)

        # Generate the path based on the current count
        if self.count < 1:
            path = self.linePath(OrangeNodes, pos)
            if len(rightBlueCones) >= 3 and len(rightYellowCones) >= 3:
                path = np.concatenate((path, 
                                       self.circlePath(rightBlueCones, rightYellowCones)),
                                        axis=0)
        elif self.count < 2:
            path = self.circlePath(rightBlueCones, rightYellowCones, pos)
            path = np.concatenate((path, self.circlePath(rightBlueCones, rightYellowCones)), axis=0)   
        elif self.count < 3:
            path = self.circlePath(rightBlueCones, rightYellowCones, pos)
            if len(leftYellowCones) >= 3 and len(leftBlueCones) >= 3:
                path = np.concatenate((path, 
                                       self.circlePath(leftYellowCones, leftBlueCones)), 
                                       axis=0)
        elif self.count < 4:
            path = self.circlePath(leftYellowCones, leftBlueCones, pos)
            path = np.concatenate((path, self.circlePath(leftYellowCones, leftBlueCones)), axis=0)
        elif self.count < 5:
            path = self.circlePath(leftYellowCones, leftBlueCones, pos)
            if len(OrangeNodes) > 0 and pos[1] < self.origin[1]:
                path = np.concatenate((path, self.linePath(OrangeNodes, path[-1])), axis=0)
        elif self.count >= 5:
            path = np.concatenate((path, self.linePath(OrangeNodes, pos)), axis=0)      
        
        
        if self.i == 0:
            plt.ion()
            self.fig = plt.figure()
            ax = self.fig.add_subplot(111)
            self.line1, = ax.plot([x[0]for x in leftBlueCones],[x[0]for x in leftBlueCones], 'bo')
            self.line2, = ax.plot([x[0]for x in rightBlueCones],[x[0]for x in rightBlueCones] , 'bo')
            self.line3, = ax.plot([x[0]for x in rightYellowCones],[x[0]for x in rightYellowCones] ,'yo')
            self.line4, = ax.plot([x[0]for x in leftYellowCones],[x[0]for x in leftYellowCones] ,'yo')
            self.line5, = ax.plot([x[0]for x in orangeCones],[x[0]for x in orangeCones] ,'go')
            self.line6, = ax.plot([x[0]for x in bigOrange],[x[0]for x in bigOrange] ,'go')
            self.line7, = ax.plot([pos[0]],[pos[1]] ,'gs')
            self.line8, = ax.plot([x[0]for x in path],[x[0]for x in path] ,'r-')
            ax.set_xlim(-25, 25)
            ax.set_ylim(0, 30)
            self.i+=1
        self.line1.set_ydata([x[1]for x in leftBlueCones])
        self.line1.set_xdata([x[0]for x in leftBlueCones])
        self.line2.set_ydata([x[1]for x in rightBlueCones])
        self.line2.set_xdata([x[0]for x in rightBlueCones])
        self.line3.set_ydata([x[1]for x in rightYellowCones])
        self.line3.set_xdata([x[0]for x in rightYellowCones])
        self.line4.set_ydata([x[1]for x in leftYellowCones])
        self.line4.set_xdata([x[0]for x in leftYellowCones])
        self.line5.set_ydata([x[1]for x in orangeCones])
        self.line5.set_xdata([x[0]for x in orangeCones])
        self.line6.set_ydata([x[1]for x in bigOrange])
        self.line6.set_xdata([x[0]for x in bigOrange])
        self.line7.set_ydata([pos[1]])
        self.line7.set_xdata([pos[0]])
        self.line8.set_ydata([x[1]for x in path])
        self.line8.set_xdata([x[0]for x in path])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        return path
    def getPath(self,state,cones):
        """
        Retrieves the generated path.

        Returns:
            np.array: Generated path as a NumPy array.
        """
         # Store the previous position if the current state is not None
        if self.state is not None:
            self.pastPos = np.array([self.state.pose.pose.position.x,
                                     self.state.pose.pose.position.y])
        self.state=state

        # Merge cones that are close to each other
        cones=self.merge_points(cones, 0.2)
        
        # Classify cones based on their position and color
        (rightBlueCones,
         leftBlueCones,
         rightYellowCones,
         leftYellowCones,
         orangeCones,
         bigOrange,
         unknownCones) = self.conesClassification(cones)
        # Generate the path
        path = self.path(rightBlueCones,
                         leftBlueCones,
                         rightYellowCones,
                         leftYellowCones,
                         orangeCones,
                         bigOrange,unknownCones)
        return path

cone_positions = [
            [-1.5, 0.0, 2], 
            [-1.5, 5.0, 2], 
            [-1.5, 13.45, 3], 
            [-1.5, 16.55, 3], 
            [-1.5, 25.0, 2], 
            [-1.5, 30.0, 2], 
            [-1.5, 35.0, 2], 
            [-1.5, 40.0, 2], 
            [-0.5, 40.0, 2], 
            [0.5, 40.0, 2], 
            [1.5, 0.0, 2], 
            [1.5, 5.0, 2], 
            [1.5, 13.45, 3], 
            [1.5, 16.55, 3], 
            [1.5, 25.0, 2], 
            [1.5, 30.0, 2], 
            [1.5, 35.0, 2], 
            [1.5, 40.0, 2], 
            [-1.5, 15.0, 0], 
            [-2.08, 17.918, 0], 
            [-3.733, 20.392, 0], 
            [-6.207, 22.045, 0], 
            [-9.125, 22.625, 0], 
            [-12.043, 22.045, 0], 
            [-14.517, 20.392, 0], 
            [-16.17, 17.918, 0], 
            [-16.75, 15.0, 0], 
            [-16.17, 12.082, 0], 
            [-14.517, 9.608, 0], 
            [-12.043, 7.955, 0], 
            [-9.125, 7.375, 0],# 
            [-6.207, 7.955, 0], 
            [-3.733, 9.608, 0], 
            [-2.08, 12.082, 0],  
            [16.17, 17.918, 1], 
            [14.517, 20.392, 1], 
            [12.043, 22.045, 1], 
            [9.125, 22.625, 1], 
            [6.207, 22.045, 1], 
            [3.733, 20.392, 1], 
            [2.08, 17.918, 1], 
            [1.5, 15.0, 1], 
            [2.08, 12.082, 1], 
            [3.733, 9.608, 1], 
            [6.207, 7.955, 1], 
            [9.125, 7.375, 1],# 
            [12.043, 7.955, 1], 
            [14.517, 9.608, 1], 
            [16.17, 12.082, 1], 
            [16.75, 15.0, 1], 
            [-1.5, 22.399, 1],
            [-4.642, 24.633, 1], 
            [-8.373, 25.598, 1], 
            [-12.204, 25.169, 1], 
            [-15.629, 23.401, 1], 
            [-18.199, 20.528, 1], 
            [-19.574, 16.927, 1], 
            [-19.574, 13.073, 1], 
            [-18.199, 9.472, 1], 
            [-15.629, 6.599, 1], 
            [-12.204, 4.831, 1], 
            [-8.373, 4.402, 1],# 
            [-4.642, 5.367, 1], 
            [-1.5, 7.601, 1], 
            [1.5, 7.601, 0], 
            [4.642, 5.367, 0], 
            [8.373, 4.402, 0], 
            [12.204, 4.831, 0], 
            [15.629, 6.599, 0],# 
            [18.199, 9.472, 0],# 
            [19.574, 13.073, 0], 
            [19.574, 16.927, 0], 
            [18.199, 20.528, 0], 
            [15.629, 23.401, 0], 
            [12.204, 25.169, 0], 
            [8.373, 25.598, 0], 
            [4.642, 24.633, 0], 
            [1.5, 22.399, 0]]

state=Odometry()
state.pose.pose.position.x = 0.0
state.pose.pose.position.y = 0.0
conePositions=[]
for i in cone_positions:
            if math.sqrt((state.pose.pose.position.x-i[0])**2+(state.pose.pose.position.y-i[1])**2) < 10 :
                conePositions.append(i)
pathGen=SendPath()
path = pathGen.getPath(state,np.array(conePositions))
print(path)