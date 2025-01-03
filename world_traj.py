import numpy as np
from math import sqrt
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix
from .graph_search import graph_search
from scipy.sparse import lil_matrix as sparse_matrix

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.1, 0.1, 0.1])
        self.margin = 0.2

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.

        # STUDENT CODE HERE
        # self.points = self.simplyfy_traj(self.path)
        self.epsilon = 0.1 # 0.25
        self.midpoint_thresh = 0.5    
        self.points = self.getPoints(self.path)
        # midpoints = []
        # for i in range(len(self.points) - 1):
        #    midpoints.append(self.points[i])
        #    if np.linalg.norm(self.points[i] - self.points[i+1]) > self.midpoint_thresh:
        #        midpoints.append((self.points[i] + self.points[i+1]) / 2)
        # midpoints.append(self.points[-1])
        # self.points = np.array(midpoints)     
        self.traj_len = len(self.points)

        # We will pre-compute the unit direction vector and distance between the different waypoints
        self.Pi_1 = self.points[1:self.traj_len]
        self.Pi = self.points[:self.traj_len - 1]

        unit_vec_numerator = self.Pi_1 - self.Pi
        unit_vec_denominator = np.linalg.norm(unit_vec_numerator, axis=1).reshape(-1,1)
        self.segment_dis = unit_vec_denominator # setting the variable with the distance between the waypoints
        self.segment_unit = unit_vec_numerator / unit_vec_denominator

        self.t0 = 0
        self.v_const = 1.6
        self.time_scalling_start = 2

        self.ti = self.segment_dis / self.v_const
        self.ti[0] = self.ti[0] * self.time_scalling_start
        self.ti[-1] = self.ti[-1] * self.time_scalling_start
        coeff = np.sqrt(1.65 / self.ti)
        self.ti = self.ti * coeff
        # self.ti = self.ti.clip(0.25, np.inf) 

        self.T = np.vstack((np.zeros(1), np.cumsum(self.ti, axis=0))).flatten()
        self.N = len(self.points)
        self.M = 8 * (self.N - 1)

 
        self.total_time = np.sum(self.ti)

        self.init_snap()   

    def init_snap(self):
        self.precomputed_sub_matrices = self.precompute_sub_matrices()
        
        self.B = lil_matrix((self.M, 3))
        iterator_b = np.arange(self.N - 1)
        self.B[8 * iterator_b + 3] = self.points[iterator_b]
        self.B[8 * iterator_b + 4] = self.points[iterator_b + 1]
        self.B = self.B.tocsc()

        self.A = lil_matrix((self.M, self.M))
        iterator_a = np.arange(len(self.ti)-1)
        self.A[[0, 1, 2], [6, 5, 4]] = [1, 2, 6]
        self.A[5 + 8 * iterator_a, 14 + 8 * iterator_a] = -1
        self.A[6 + 8 * iterator_a, 13 + 8 * iterator_a] = -2
        self.A[7 + 8 * iterator_a, 12 + 8 * iterator_a] = -6
        self.A[8 + 8 * iterator_a, 11 + 8 * iterator_a] = -24
        self.A[9 + 8 * iterator_a, 10 + 8 * iterator_a] = -120
        self.A[10 + 8 * iterator_a, 9 + 8 * iterator_a] = -720          

    def update(self, t):
        """
        PRIMARY METHOD
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        x_matrix = self.min_snap(t)
        # x_matrix = self.constant_vel(t)

        # STUDENT CODE END
        x = x_matrix[0]
        x_dot = x_matrix[1]
        x_ddot = x_matrix[2]
        x_dddot = x_matrix[3]
        x_ddddot = x_matrix[4]
        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
    
    def constant_vel(self, t):
        x_matrix = np.zeros((5,3))
        index = -1
        for i in range(0, self.T.shape[0] - 1):
            if self.T[i] <= t and t < self.T[i + 1]:
                index = i
                break
        
        # Finding the velcoity and position of the quadrotor
        if index >= 0:
            x_matrix[1] = self.v_const * self.segment_unit[index]
            x_matrix[0] = x_matrix[1] * (t - self.T[index]) + self.Pi[index]
        else:
            # This is hover position of the robot
            x_matrix[1] = np.zeros(3)
            x_matrix[0] = self.points[-1]

        return x_matrix
        
    def min_snap(self, t):
        x_matrix = np.zeros((5,3))

        A = self.A
        iter = 0
        for ti in self.ti:
            ti = ti[0]
            sub = self.precomputed_sub_matrices[ti]
            if iter != len(self.ti) - 1:
                A[8*iter + 3:8*iter + 11, 8*iter:8*iter + 8] = sub
            else:
                A[8*iter + 3:8*iter + 11, 8*iter:8*iter + 8] = sub[:5, :]
            iter += 1

        A = A.tocsc()
        C = spsolve(A, self.B).toarray()

        if t > self.total_time:
            x_matrix[0] = self.points[-1]
        else:
            id = np.where(np.sign(self.T - t) > 0)[0][0] - 1

            t_diff = t - self.T[id]
            t_arr = self.compute_poly_matrix(t_diff)

            x_matrix[:5] = t_arr @ C[8 * id:8 * id + 8, :]

        return x_matrix
        
    def precompute_sub_matrices(self):
        precomputed_sub_matrices = {}
        for ti in self.ti:
            ti = ti[0]
            ti2 = ti**2
            ti3 = ti**3
            ti4 = ti**4
            ti5 = ti**5
            ti6 = ti**6
            ti7 = ti**7
            sub = np.array([[0           , 0          , 0          , 0          , 0         , 0        , 0    , 1],
                            [ti7       , ti6      , ti5      , ti4      , ti3     , ti2    , ti   , 1],
                            [7 * ti6   , 6 * ti5  , 5 * ti4  , 4 * ti3  , 3 * ti2 , 2 * ti   , 1    , 0],
                            [42 * ti5  , 30 * ti4 , 20 * ti3 , 12 * ti2 , 6 * ti    , 2        , 0    , 0],
                            [210 * ti4 , 120 * ti3, 60 * ti2 , 24 * ti    , 6         , 0        , 0    , 0],
                            [840 * ti3 , 360 * ti2, 120 * ti   , 24         , 0         , 0        , 0    , 0],
                            [2520 * ti2, 720 * ti   , 120        , 0          , 0         , 0        , 0    , 0],
                            [5040 * ti   , 720        , 0          , 0          , 0         , 0        , 0    , 0]])
            precomputed_sub_matrices[ti] = sub
        return precomputed_sub_matrices

    def compute_poly_matrix(self, t_diff):
        return np.array([
                    [t_diff**7, t_diff**6, t_diff**5, t_diff**4, t_diff**3, t_diff**2, t_diff, 1],
                    [7 * t_diff**6, 6 * t_diff**5, 5 * t_diff**4, 4 * t_diff**3, 3 * t_diff**2, 2 * t_diff, 1, 0],
                    [42 * t_diff**5, 30 * t_diff**4, 20 * t_diff**3, 12 * t_diff**2, 6 * t_diff, 2, 0, 0],
                    [210 * t_diff**4, 120 * t_diff**3, 60 * t_diff**2, 24 * t_diff, 6, 0, 0, 0],
                    [840 * t_diff**3, 360 * t_diff**2, 120 * t_diff, 24, 0, 0, 0, 0]
                ])

    def getPoints(self, points):
        if len(points) <= 2:
            return points
        
        max_dist = 0
        max_index = 0
        start = points[0]
        goal = points[-1]
        for i in range(1, len(points) - 1):
            d = 0
            if np.all(start == goal):
                d = np.linalg.norm(points[i] - start)
            else:
                d = np.abs(np.linalg.norm(np.cross(goal - start, start - points[i]))) / np.linalg.norm(goal - start)
            if d > max_dist:
                max_index = i
                max_dist = d

        if max_dist > self.epsilon:
            rec_updated_points1 = self.getPoints(points[:max_index + 1])
            rec_updated_points2 = self.getPoints(points[max_index:])

            return np.vstack((rec_updated_points1[:-1], rec_updated_points2))  
        else:
            return np.vstack((start, goal))    


    def simplyfy_traj(self, points):
        simplified_points = np.array(points)
        indices_to_keep = [0]
        prev_i = 0
        for i in range(1, len(simplified_points) - 1):
            direction = np.linalg.norm(np.cross(simplified_points[i - 1] - simplified_points[i], simplified_points[i] - simplified_points[i + 1]))
            if direction > 0:
                indices_to_keep.append(i)
            elif direction == 0:
                dist = np.linalg.norm(simplified_points[indices_to_keep[-1]] - simplified_points[i])
                if dist > 2:
                    indices_to_keep.append(i)

        indices_to_keep.append(len(simplified_points) - 1)
        simplified_points = simplified_points[indices_to_keep]

        return simplified_points
