import numpy as np
from scipy.spatial.transform import Rotation

class SE3Control(object):
    """

    """
    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.arm_length      = quad_params['arm_length'] # meters
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s
        self.k_thrust        = quad_params['k_thrust'] # N/(rad/s)**2
        self.k_drag          = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz])) # kg*m^2
        self.g = 9.81 # m/s^2

        # STUDENT CODE HERE
        self.gamma = self.k_drag / self.k_thrust
        # Linear - x,y,z
        # 15, 15, 3
        # 20, 20, 180
        self.K_d = np.diag(np.array([4, 4, 7]))
        self.K_p = np.diag(np.array([4.5, 4.5, 11]))

        # Rotation - roll, pitch, yaw
        # 80, 80, 14
        # 2000, 2000, 80
        self.K_w = np.diag(np.array([50., 50., 15.]))
        self.K_r = np.diag(np.array([200., 200., 85.]))


    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        # Solve for commanded acceleration
        r_ddot_des = flat_output.get("x_ddot") - self.K_d @ (state.get("v") - flat_output.get("x_dot")) - self.K_p @ (state.get("x") - flat_output.get("x"))
        # print("commanded acceleration: ", r_ddot_des)
        
        # Solve for F_des
        F_des = self.mass * r_ddot_des + np.array([0, 0, self.mass * self.g])
        # print("F_dex: ", F_des)

        # Get phi, theta, and psi from quaternion
        r = Rotation.from_quat(state.get("q"))
        # psi, theta, phi = r.as_euler('zyx', degrees=False)
        R = r.as_matrix()
        # print(R)
        
        # Solve for u1
        b3 = R @ np.array([0, 0, 1])
        u1 = b3 @ F_des

        # Solve b3_des
        b3_des = F_des / np.linalg.norm(F_des)

        # Solve for a_yaw
        a_yaw = np.array([np.cos(flat_output.get("yaw")), np.sin(flat_output.get("yaw")), 0])

        # Solve for b2_des
        b2_des = np.cross(b3_des, a_yaw)
        b2_des /= np.linalg.norm(b2_des)

        # Get R_des
        R_des = np.array([np.cross(b2_des, b3_des), b2_des, b3_des]).T

        # Set quaternion
        cmd_q = Rotation.as_quat(Rotation.from_matrix(R_des))

        # Get error
        e_R = .5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R[2, 1], e_R[0, 2], e_R[1, 0]])

        #  print("Error: ", e_R)
        # Solve for u2
        u2 = self.inertia @ (-self.K_r @ e_R - self.K_w @ state.get("w"))

        # Solve sytem of equations to get force of each motor
        b = np.append(u2, u1)
        # print(b)
        A = np.array([[0.0, self.arm_length, 0.0, -self.arm_length],
                      [-self.arm_length, 0.0, self.arm_length, 0.0],
                      [self.gamma, -self.gamma, self.gamma, -self.gamma],
                      [1.0, 1.0, 1.0, 1.0]])
        # print(A)
        forces = np.linalg.solve(A, b)
        # print(forces)
        # Set negative forces to 0
        forces[forces < 0] = 0
        # print("Forces:", forces)

        # Solve for angular velocity of motors
        omegas = np.sqrt(forces / self.k_thrust)

        # Solve for moments
        moments = self.k_drag * np.square(omegas)

        # Set control inputs
        cmd_motor_speeds = omegas
        cmd_moment = moments
        cmd_thrust = u1

        control_input = {'cmd_motor_speeds':cmd_motor_speeds,
                         'cmd_thrust':cmd_thrust,
                         'cmd_moment':cmd_moment,
                         'cmd_q':cmd_q}
        return control_input
