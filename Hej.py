import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
import time


def read_coordinate_file(filename):
    with open(filename, mode="r") as file:
        sample_coord = file.readline()
        a = []
        while sample_coord:
            res = [float(line) for line in sample_coord.strip('{}\n').split(',')]
            a.append(res)
            sample_coord = file.readline()
    b = np.array(a)
    R = 1
    x = R * np.pi / 180 * b[:, 1]
    y = R * np.log(np.tan(np.pi / 4 + np.pi / 360 * b[:, 0]))
    coord_list = np.array([x, y]).T
    return coord_list


def plot_points(coord_list, indices, path):

    # PLOT CITIES
    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=5, c="r")
    plt.gca().set_aspect('equal')

    # PLOT AVAILABLE CONNECTIONS
    city_pair_coord = []
    for i, j in indices:
        city_pair_coord.append([[coord_list[i, 0], coord_list[i, 1]], [coord_list[j, 0], coord_list[j, 1]]])
    lc = LineCollection(city_pair_coord, linewidth=0.4, colors="gray")
    fig = plt.subplot()
    fig.add_collection(lc)

    # PLOT SHORTEST PATH
    path_coord_x = []
    path_coord_y = []
    for city in path:
        path_coord_x.append(coord_list[city, 0])
        path_coord_y.append(coord_list[city, 1])
    plt.plot(path_coord_x, path_coord_y, linewidth=1, c="blue")
    plt.show()

# Adjust inner for-loop
def construct_graph_connections(coord_list, radius):
    pair_indices = []
    distances = []
    for i, city in enumerate(coord_list):
        dxdy = coord_list - city
        tot_distances = np.sqrt(np.square(dxdy[i + 1:, 0]) + np.square(dxdy[i + 1:, 1]))
        for j, distance in enumerate(tot_distances):
            if distance <= radius:
                pair_indices.append([i, i + 1 + j])
                distances.append(distance)
    return np.array(pair_indices), np.array(distances)


def construct_graph(indices, distance, N):
    graph = csr_matrix((distance, indices.T), shape=(N, N))
    return graph


def find_shortest_path(graph, start_node, end_node):
    distance, predecessor = shortest_path(graph, directed=False, return_predecessors=True, indices=start_node)
    start_end_dist = distance[end_node]
    a = end_node
    path = [end_node]
    while predecessor[a] != start_node:
        a = predecessor[a]
        path.append(a)
    path.append(start_node)
    path.reverse()
    return path, start_end_dist


def construct_fast_graph_connections(coord_list, radius):
    Tree = cKDTree(coord_list)
    possible_cities = Tree.query_ball_point(coord_list, radius)
    indices=[]
    distances=[]
    for i, element in enumerate(possible_cities):
        for j in element:
            if i < j:
                indices.append([i, j])

                dxdy = coord_list[i] - coord_list[j]

                distances.append(np.sqrt(np.square(dxdy[0]) + np.square(dxdy[1])))

    return np.array(indices), np.array(distances)



# To have in the end
# print("\nInput filename and radius: ")
# print("For example 'SampleCoordinates 0.08' and press enter.")
# Inputs

# GIVEN DATA
radius = 0.08
start_node = 0
end_node = 5

# Call functions
start = time.time()
coord_list = read_coordinate_file('SampleCoordinates.txt')
N = len(coord_list)
end = time.time()
print('Time to finish function: "read_coordinate_file"', end - start)


start = time.time()
indices, distance = construct_graph_connections(coord_list, radius)
end = time.time()
print('Time to finish function: "construct_graph_connections"', end - start)


start = time.time()
indices, distance = construct_fast_graph_connections(coord_list, radius)
end = time.time()
print('Time to finish function: "construct_graph_connections"', end - start)


start = time.time()
graph = construct_graph(indices, distance, N)
end = time.time()
print('Time to finish function: "construct_graph"', end - start)

start = time.time()
path, start_end_dist = find_shortest_path(graph, start_node, end_node)
end = time.time()
print('Time to finish function: "find_shortest_path"', end - start)

start = time.time()
plot_points(coord_list, indices, path)
end = time.time()
print('Time to finish function: "plot_points"', end - start)

print(indices, distance)






"""
while True:
    try:
        print("\nInput > ", end="")
        filename, r = input().split()
        calculate(read_coordinate_file(filename + ".txt"), float(r))
        break
    except (FileNotFoundError, ValueError) as error_message:
        print(error_message)
"""