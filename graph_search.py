from heapq import heappush, heappop  # Recommended.
import numpy as np

from flightsim.world import World

# from .occupancy_map import OccupancyMap # Recommended.
from .occupancy_map import OccupancyMap

def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """
    if astar == True:
        print("A* ALGO CHOSEN")
        occ_map = OccupancyMap(world, resolution, margin)
        # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
        start_index = tuple(occ_map.metric_to_index(start))  # This is a tuple of XYZ Index
        goal_index = tuple(occ_map.metric_to_index(goal))  # This is a tuple of XYZ Index

        # def heuristic(current_node):
        #     # Max metric is the fastest between euclidean and manhattan
        #     dx = abs(goal_index[0] - current_node[0])
        #     dy = abs(goal_index[1] - current_node[1])
        #     dz = abs(goal_index[2] - current_node[2])
        #     return max(dx, dy, dz)
        def heuristic(current_node):
            # Euclidean: Number of node expanded drop, but calculation is more complex than max metric
            return np.linalg.norm(np.array(current_node)-np.array(goal_index)) * 1.2


        prio_queue = []  # Prio for Astar
        dist = {start_index: 0}  # Dist of starting point is 0
        parent = {}  # Parent node for node in SHORTEST PATH ONLY
        explored = set()
        nodes_expanded = 0

        # Action in + or - direction in XYZ

        # More action leads to better path but way more expansion especially since Djikstra is BFS
        actions = [
            (-1, 0, 0), (1, 0, 0),  # X Dir
            (0, -1, 0), (0, 1, 0),  # Y Dir
            (0, 0, -1), (0, 0, 1),  # Z Dir
            (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),  # XY Dir
            (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),  # XZ Dir
            (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),  # YZ Dir
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),  # XYZ Dir
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)  # XYZ Dir
        ]

        # Little action saves more time and reduce expansion, but might not solve
        # actions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        # Add start node to prio_que
        heappush(prio_queue, (heuristic(start_index),0, start_index))

        # Start ASTAR
        while prio_queue:  # While the queue is not empty, keep exploring
            _, current_dist, current_index = heappop(prio_queue)  # Explore

            # If arrived
            if current_index == goal_index:
                path = [goal]  # path starting from goal
                while current_index != start_index:  # Work backwards
                    current_index = parent[current_index]  # Find the parent
                    path.append(occ_map.index_to_metric_center(current_index))  # Append the parent
                path.reverse()  # When path is completed, reverse so the path start from beginning
                path[0] = start
                path = np.array(path)  # Sandbox take array as an input
                return path, nodes_expanded

            # Check if current node is already explored
            if current_index in explored:
                continue

            # Mark current node as explored
            explored.add(current_index)

            # Expansion
            for delta in actions:  # Loop over all action
                neighbor = (current_index[0] + delta[0], current_index[1] + delta[1], current_index[2] + delta[2])  # Manually created to avoid looping
                nodes_expanded += 1

                # If not valid, then skip
                if occ_map.is_occupied_index(neighbor) or not occ_map.is_valid_index(neighbor) or neighbor in explored:
                    continue

                # Calculate the Dist
                # neighbor_dist = current_dist + abs(neighbor[0] - current_index[0]) + abs(neighbor[1] - current_index[1]) + abs(neighbor[2] - current_index[2]) # Can do with for loop but its slower, do it manually to parallelized
                neighbor_dist = current_dist + np.linalg.norm(np.array(neighbor)-np.array(current_index)) # Can do with for loop but its slower, do it manually to parallelized

                # When it is not explored yet (not in dist) and value is smaller than prev dist
                if neighbor not in dist or neighbor_dist < dist[neighbor]:
                    dist[neighbor] = neighbor_dist  # Update the dist dict
                    parent[neighbor] = current_index  # Update the parent (Where it came from previously)
                    heappush(prio_queue, (heuristic(neighbor)+neighbor_dist, neighbor_dist, neighbor))
        # Return none when we did not find any path
        return None, 0

    else:
        print("DJIKSTRA ALGO CHOSEN")
        # While not required, we have provided an occupancy map you may use or modify.
        occ_map = OccupancyMap(world, resolution, margin)
        # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
        start_index = tuple(occ_map.metric_to_index(start)) # This is a tuple of XYZ Index
        goal_index = tuple(occ_map.metric_to_index(goal)) # This is a tuple of XYZ Index
        prio_queue = [] # Prio for Dijkstra
        dist = {start_index: 0} # Dist of starting point is 0
        parent = {} # Parent node for node in SHORTEST PATH ONLY
        explored = set()
        nodes_expanded = 0

        # Action in + or - direction in XYZ

        # More action leads to better path but way more expansion especially since Djikstra is BFS
        actions = [
            (-1, 0, 0), (1, 0, 0),  # X Dir
            (0, -1, 0), (0, 1, 0),  # Y Dir
            (0, 0, -1), (0, 0, 1),  # Z Dir
            (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0), # XY Dir
            (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1), # XZ Dir
            (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),  # YZ Dir
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),  # XYZ Dir
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1)  # XYZ Dir
        ]

        # Little action saves more time and reduce expansion, but might not solve
        # actions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        # Add start node to prio_que
        heappush(prio_queue,(0, start_index))

        #Start Dijkstra
        while prio_queue: # While the queue is not empty, keep exploring
            current_dist, current_index = heappop(prio_queue) # Explore

            # If arrived
            if current_index==goal_index:
                path = [goal] # path starting from goal
                while current_index != start_index: # Work backwards
                    current_index = parent[current_index] # Find the parent
                    path.append(occ_map.index_to_metric_center(current_index)) # Append the parent
                path.reverse() # When path is completed, reverse so the path start from beginning
                path[0] = start
                path = np.array(path) # Sandbox take array as an input
                return path, nodes_expanded

            # Check if current node is already explored
            if current_index in explored:
                continue

            # Mark current node as explored
            explored.add(current_index)
            # Expansion
            for delta in actions:  # Loop over all action
                neighbor = (current_index[0] + delta[0], current_index[1] + delta[1], current_index[2] + delta[2])  # Manually created to avoid looping
                nodes_expanded += 1
                # If not valid, then skip
                if occ_map.is_occupied_index(neighbor) or not occ_map.is_valid_index(neighbor) or neighbor in explored:
                    continue

                # Calculate the Dist
                neighbor_dist = current_dist + abs(neighbor[0] - current_index[0]) + abs(neighbor[1] - current_index[1]) + abs(neighbor[2] - current_index[2]) # Can do with for loop but its slower, do it manually to parallelized

                # When it is not explored yet (not in dist) and value is smaller than prev dist
                if neighbor not in dist or neighbor_dist <dist[neighbor]:
                    dist[neighbor] = neighbor_dist # Update the dist dict
                    parent[neighbor] = current_index # Update the parent (Where it came from previously)
                    heappush(prio_queue,(neighbor_dist,neighbor))
        # Return none when we did not find any path
        return None, 0
