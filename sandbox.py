import inspect
import json
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import time

from flightsim.animate import animate
from flightsim.axes3ds import Axes3Ds
from flightsim.crazyflie_params import quad_params
from flightsim.simulate import Quadrotor, simulate, ExitStatus
from flightsim.world import World

from proj1_4.code.occupancy_map import OccupancyMap
from proj1_4.code.se3_control import SE3Control
from proj1_4.code.world_traj import WorldTraj

# read in real trajectory here
# ../proj1_4_all/lab4_2/quadcopter_response_map_3_cv_vel_1_6_states.csv
real_traj = np.genfromtxt('../proj1_4_all/lab4_2/quadcopter_response_map_2_cv_run4_1_states.csv', delimiter=',')
real_traj_x = real_traj[:-10, 1:4] # xyz
real_traj_v = real_traj[:-10, 4:7] # xyz_dot
start = real_traj_x[0, :]
vicon_time = real_traj[:-450, 0]

# Improve figure display on high DPI screens.
# mpl.rcParams['figure.dpi'] = 200

# Choose a test example file. You should write your own example files too!
# filename = '../util/test_window.json'
# filename = '../util/test_maze.json'
# filename = '../util/maze_2022_1.json'
# filename = '../util/test_over_under.json'
filename = '../util/test_lab_2.json'

# Load the test example.
file = Path(inspect.getsourcefile(lambda:0)).parent.resolve() / '..' / 'util' / filename
world = World.from_file(file)          # World boundary and obstacles.
# start  = world.world['start']          # Start point, shape=(3,)
goal   = world.world['goal']           # Goal point, shape=(3,)

# This object defines the quadrotor dynamical model and should not be changed.
quadrotor = Quadrotor(quad_params)
robot_radius = 0.25

# Your SE3Control object (from project 1-1).
my_se3_control = SE3Control(quad_params)

# Your MapTraj object. This behaves like the trajectory function you wrote in
# project 1-1, except instead of giving it waypoints you give it the world,
# start, and goal.
planning_start_time = time.time()
my_world_traj = WorldTraj(world, start, goal)
planning_end_time = time.time()

# Help debug issues you may encounter with your choice of resolution and margin
# by plotting the occupancy grid after inflation by margin. THIS IS VERY SLOW!!
# fig = plt.figure('world')
# ax = Axes3Ds(fig)
# world.draw(ax)
# fig = plt.figure('occupancy grid')
# ax = Axes3Ds(fig)
# resolution = SET YOUR RESOLUTION HERE
# margin = SET YOUR MARGIN HERE
# oc = OccupancyMap(world, resolution, margin)
# oc.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=10, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=10, markeredgewidth=3, markerfacecolor='none')
# plt.show()

# Set simulation parameters.
t_final = 60
initial_state = {'x': start,
                 'v': (0, 0, 0),
                 'q': (0, 0, 0, 1), # [i,j,k,w]
                 'w': (0, 0, 0)}

# Perform simulation.
#
# This function performs the numerical simulation.  It returns arrays reporting
# the quadrotor state, the control outputs calculated by your controller, and
# the flat outputs calculated by you trajectory.

print()
print('Simulate.')
(sim_time, state, control, flat, exit) = simulate(initial_state,
                                              quadrotor,
                                              my_se3_control,
                                              my_world_traj,
                                              t_final)
print(exit.value)

# Print results.
#
# Only goal reached, collision test, and flight time are used for grading.

collision_pts = world.path_collisions(state['x'], robot_radius)

stopped_at_goal = (exit == ExitStatus.COMPLETE) and np.linalg.norm(state['x'][-1] - goal) <= 0.05
no_collision = collision_pts.size == 0
flight_time = sim_time[-1]
flight_distance = np.sum(np.linalg.norm(np.diff(state['x'], axis=0),axis=1))
planning_time = planning_end_time - planning_start_time

print()
print(f"Results:")
print(f"  No Collision:    {'pass' if no_collision else 'FAIL'}")
print(f"  Stopped at Goal: {'pass' if stopped_at_goal else 'FAIL'}")
print(f"  Flight time:     {flight_time:.1f} seconds")
print(f"  Flight distance: {flight_distance:.1f} meters")
print(f"  Planning time:   {planning_time:.1f} seconds")
if not no_collision:
    print()
    print(f"  The robot collided at location {collision_pts[0]}!")

# Plot Results
#
# You will need to make plots to debug your quadrotor.
# Here are some example of plots that may be useful.

# my_world_traj.path = None
# my_world_traj.points = None

# Visualize the original dense path from A*, your sparse waypoints, and the
# smooth trajectory.
fig = plt.figure('Lab Maze 3')
ax = Axes3Ds(fig)
world.draw(ax)
ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
ax.plot(real_traj_x[:,0], real_traj_x[:,1], real_traj_x[:,2], 'bo', markersize=4, markeredgewidth=3, markerfacecolor='none')
if hasattr(my_world_traj, 'path'):
    if my_world_traj.path is not None:
        world.draw_line(ax, my_world_traj.path, color='red', linewidth=1)
else:
    print("Have you set \'self.path\' in WorldTraj.__init__?")
if hasattr(my_world_traj, 'points'):
    if my_world_traj.points is not None:
        world.draw_points(ax, my_world_traj.points, color='purple', markersize=8)
else:
    print("Have you set \'self.points\' in WorldTraj.__init__?")
world.draw_line(ax, flat['x'], color='black', linewidth=2)
ax.legend(handles=[
    Line2D([], [], color='red', linewidth=1, label='Dense A* Path'),
    Line2D([], [], color='purple', linestyle='', marker='.', markersize=8, label='Sparse Waypoints'),
    Line2D([], [], color='black', linewidth=2, label='Trajectory'),
    Line2D([], [], color='blue', linestyle='', marker='.', markersize=8, label='Flight')],
    loc='upper right')

# Position and Velocity vs. Time
# print(sim_time[-1] / len(sim_time)) # 500hz
# print(vicon_time[-1] / len(vicon_time)) # 100hz

# downsample sim_time from 500hz to 100hz
sim_time = sim_time[::5]
# -1.8
offset = -1.8
vicon_time += offset

# print(sim_time[-1])

# align start and end with sim_time
if (vicon_time[0] < sim_time[0]):
    sim_start = 0
    vicon_start = np.where(vicon_time >= sim_time[0])[0][0]
else:
    sim_start = np.where(sim_time >= vicon_time[0])[0][0]
    vicon_start = 0
sim_time = sim_time[sim_start:]
vicon_time = vicon_time[vicon_start:]
real_traj_x = real_traj_x[vicon_start:]
real_traj_v = real_traj_v[vicon_start:]
if (vicon_time[-1] > sim_time[-1]):
    sim_end = len(sim_time)
    vicon_end = np.where(vicon_time <= sim_time[-1])[0][-1]
else:
    sim_end = np.where(sim_time <= vicon_time[-1])[0][-1]
    vicon_end = len(vicon_time)
sim_time = sim_time[:sim_end]
vicon_time = vicon_time[:vicon_end]
real_traj_x = real_traj_x[:vicon_end]
real_traj_v = real_traj_v[:vicon_end]

(fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Position vs Time')
x = state['x']
x_des = flat['x']
x_des = x_des[::5]
x_des = x_des[sim_start:sim_end]
ax = axes[0]
ax.plot(sim_time, x_des[:,0], 'r', sim_time, x_des[:,1], 'g', sim_time, x_des[:,2], 'b')
# ax.plot(vicon_time, np.ones(len(vicon_time)), 'b')
ax.plot(vicon_time, real_traj_x[:,0], 'r.',    vicon_time, real_traj_x[:,1], 'g.',    vicon_time, real_traj_x[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('position, m')
ax.grid('major')
ax.set_title('Position')
# ax.set_xlabel('time, s')
v = state['v']
v_des = flat['x_dot']
v_des = v_des[::5]
ax = axes[1]
ax.plot(sim_time, v_des[:,0], 'r', sim_time, v_des[:,1], 'g', sim_time, v_des[:,2], 'b')
ax.plot(vicon_time, real_traj_v[:,0], 'r.',    vicon_time, real_traj_v[:,1], 'g.',    vicon_time, real_traj_v[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('velocity, m/s')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Velocity')
plt.show()

x = state['x']
x_ddot_des = flat['x_ddot']
x_ddot_des = x_ddot_des[::5]
x_dddot_des = flat['x_dddot']
x_dddot_des = x_dddot_des[::5]
x_ddddot_des = flat['x_ddddot']
x_ddddot_des = x_ddddot_des[::5]

real_traj_v = np.diff(real_traj_x, axis=0) / 0.01
real_traj_v = np.clip(real_traj_v, -2, 2)
smoothing_window = 20
real_traj_v[:, 0] = np.convolve(real_traj_v[:, 0], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_v[:, 1] = np.convolve(real_traj_v[:, 1], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_v[:, 2] = np.convolve(real_traj_v[:, 2], np.ones(smoothing_window)/smoothing_window, mode='same')

real_traj_ddot = np.diff(real_traj_v, axis=0) / 0.01
real_traj_ddot = np.clip(real_traj_ddot, -2, 2)

real_traj_ddot[:, 0] = np.convolve(real_traj_ddot[:, 0], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_ddot[:, 1] = np.convolve(real_traj_ddot[:, 1], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_ddot[:, 2] = np.convolve(real_traj_ddot[:, 2], np.ones(smoothing_window)/smoothing_window, mode='same')

real_traj_dddot = np.diff(real_traj_ddot, axis=0) / 0.01
real_traj_dddot = np.clip(real_traj_dddot, -5, 5)
real_traj_dddot[:, 0] = np.convolve(real_traj_dddot[:, 0], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_dddot[:, 1] = np.convolve(real_traj_dddot[:, 1], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_dddot[:, 2] = np.convolve(real_traj_dddot[:, 2], np.ones(smoothing_window)/smoothing_window, mode='same')

real_traj_ddddot = np.diff(real_traj_dddot, axis=0) / 0.01
real_traj_ddddot = np.clip(real_traj_ddddot, -50, 50)
real_traj_ddddot[:, 0] = np.convolve(real_traj_ddddot[:, 0], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_ddddot[:, 1] = np.convolve(real_traj_ddddot[:, 1], np.ones(smoothing_window)/smoothing_window, mode='same')
real_traj_ddddot[:, 2] = np.convolve(real_traj_ddddot[:, 2], np.ones(smoothing_window)/smoothing_window, mode='same')

(fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Position vs Time')
ax = axes[0]
# ax.plot(sim_time, x_ddot_des[:,0], 'r', sim_time, x_ddot_des[:,1], 'g', sim_time, x_ddot_des[:,2], 'b')
ax.plot(vicon_time[2:], real_traj_ddot[:,0], 'r.',    vicon_time[2:], real_traj_ddot[:,1], 'g.',    vicon_time[2:], real_traj_ddot[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('acceleration, $m/s^2$')
ax.grid('major')
ax.set_title('Acceleration')

ax = axes[1]
# ax.plot(sim_time, x_dddot_des[:,0], 'r', sim_time, x_dddot_des[:,1], 'g', sim_time, x_dddot_des[:,2], 'b')
ax.plot(vicon_time[3:], real_traj_dddot[:,0], 'r.',    vicon_time[3:], real_traj_dddot[:,1], 'g.',    vicon_time[3:], real_traj_dddot[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('jerk, $m/s^3$')
ax.grid('major')
ax.set_title('Jerk')

ax = axes[2]
# ax.plot(sim_time, x_ddddot_des[:,0], 'r', sim_time, x_ddddot_des[:,1], 'g', sim_time, x_ddddot_des[:,2], 'b')
ax.plot(vicon_time[4:], real_traj_ddddot[:,0], 'r.',    vicon_time[4:], real_traj_ddddot[:,1], 'g.',    vicon_time[4:], real_traj_ddddot[:,2], 'b.')
ax.legend(('x', 'y', 'z'), loc='upper right')
ax.set_ylabel('snap, $m/s^4$')
ax.set_xlabel('time, s')
ax.grid('major')
ax.set_title('Snap')
plt.show()

x_des_sampled = np.zeros((100, 3))
for i in range(100):
    x_des_sampled[i] = x_des[i*len(x_des)//100]
real_traj_x_sampled = np.zeros((100, 3))
for i in range(100):
    real_traj_x_sampled[i] = real_traj_x[i*len(real_traj_x)//100]
# compute error between the two
error_x = np.sum(np.abs(x_des_sampled[:,0] - real_traj_x_sampled[:,0]))/100
error_y = np.sum(np.abs(x_des_sampled[:,1] - real_traj_x_sampled[:,1]))/100
error_z = np.sum(np.abs(x_des_sampled[:,2] - real_traj_x_sampled[:,2]))/100
print(f"Error in x: {error_x}")
print(f"Error in y: {error_y}")
print(f"Error in z: {error_z}")

# # Orientation and Angular Velocity vs. Time
# (fig, axes) = plt.subplots(nrows=2, ncols=1, sharex=True, num='Orientation vs Time')
# q_des = control['cmd_q']
# q = state['q']
# ax = axes[0]
# ax.plot(sim_time, q_des[:,0], 'r', sim_time, q_des[:,1], 'g', sim_time, q_des[:,2], 'b', sim_time, q_des[:,3], 'k')
# ax.plot(sim_time, q[:,0], 'r.',    sim_time, q[:,1], 'g.',    sim_time, q[:,2], 'b.',    sim_time, q[:,3],     'k.')
# ax.legend(('i', 'j', 'k', 'w'), loc='upper right')
# ax.set_ylabel('quaternion')
# ax.set_xlabel('time, s')
# ax.grid('major')
# w = state['w']
# ax = axes[1]
# ax.plot(sim_time, w[:,0], 'r.', sim_time, w[:,1], 'g.', sim_time, w[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('angular velocity, rad/s')
# ax.set_xlabel('time, s')
# ax.grid('major')

# # Commands vs. Time
# (fig, axes) = plt.subplots(nrows=3, ncols=1, sharex=True, num='Commands vs Time')
# s = control['cmd_motor_speeds']
# ax = axes[0]
# ax.plot(sim_time, s[:,0], 'r.', sim_time, s[:,1], 'g.', sim_time, s[:,2], 'b.', sim_time, s[:,3], 'k.')
# ax.legend(('1', '2', '3', '4'), loc='upper right')
# ax.set_ylabel('motor speeds, rad/s')
# ax.grid('major')
# ax.set_title('Commands')
# M = control['cmd_moment']
# ax = axes[1]
# ax.plot(sim_time, M[:,0], 'r.', sim_time, M[:,1], 'g.', sim_time, M[:,2], 'b.')
# ax.legend(('x', 'y', 'z'), loc='upper right')
# ax.set_ylabel('moment, N*m')
# ax.grid('major')
# T = control['cmd_thrust']
# ax = axes[2]
# ax.plot(sim_time, T, 'k.')
# ax.set_ylabel('thrust, N')
# ax.set_xlabel('time, s')
# ax.grid('major')

# # 3D Paths
# fig = plt.figure('3D Path')
# ax = Axes3Ds(fig)
# world.draw(ax)
# ax.plot([start[0]], [start[1]], [start[2]], 'go', markersize=16, markeredgewidth=3, markerfacecolor='none')
# ax.plot( [goal[0]],  [goal[1]],  [goal[2]], 'ro', markersize=16, markeredgewidth=3, markerfacecolor='none')
# world.draw_line(ax, flat['x'], color='black', linewidth=2)
# world.draw_points(ax, state['x'], color='blue', markersize=4)
# if collision_pts.size > 0:
#     ax.plot(collision_pts[0,[0]], collision_pts[0,[1]], collision_pts[0,[2]], 'rx', markersize=36, markeredgewidth=4)
# ax.legend(handles=[
#     Line2D([], [], color='black', linewidth=2, label='Trajectory'),
#     Line2D([], [], color='blue', linestyle='', marker='.', markersize=4, label='Flight')],
#     loc='upper right')


# Animation (Slow)
#
# Instead of viewing the animation live, you may provide a .mp4 filename to save.

# R = Rotation.from_quat(state['q']).as_matrix()
# ani = animate(sim_time, state['x'], R, world=world, filename=None)

plt.show()
