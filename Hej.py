import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
import time


def which_file(filename):
    radius = start_node = end_node = None
    if filename == 'SampleCoordinates':
        radius = 0.08
        start_node = 0
        end_node = 5
    elif filename == 'HungaryCities':
        radius = 0.005
        start_node = 311
        end_node = 702
    elif filename == 'GermanyCities':
        radius = 0.0025
        start_node = 1573
        end_node = 10584
    return radius, start_node, end_node


def read_coordinate_file(filename):
    with open(filename + ".txt", mode='r') as file:
        sample_coord = file.readline()
        a = []
        while sample_coord:
            # Do we need for loop here
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


while True:
    print("\nEnter the name of the file.")
    print("For example 'SampleCoordinates' and press enter.")
    print("Input > ", end="")
    filename = input()
    radius, start_node, end_node = which_file(filename)
    if radius is not None:
        break
    print("Could not find given filename")


while True:

    # Call functions
    start_1 = time.time()
    coord_list = read_coordinate_file(filename)
    N = len(coord_list)
    end_1 = time.time()

    print("Choose function:")
    print("1. Fast")
    print("2. Normal")
    print("Selection > ", end="")
    selection = input()
    if selection.isnumeric() and (int(selection) == 1 or int(selection) == 2):
        break
    print('Incorrect input, try alternatives 1 or 2\n')

if int(selection) == 1:
    print('Time to finish function: "read_coordinate_file"', end_1 - start_1)
    start_2 = time.time()
    indices, distance = construct_fast_graph_connections(coord_list, radius)
    end_2 = time.time()
    print('Time to finish function: "construct_fast_graph_connections"', end_2 - start_2)
elif int(selection) == 2:
    print('Time to finish function: "read_coordinate_file"', end_1 - start_1)
    start_2 = time.time()
    indices, distance = construct_graph_connections(coord_list, radius)
    end_2 = time.time()
    print('Time to finish function: "construct_graph_connections"', end_2 - start_2)

start_3 = time.time()
graph = construct_graph(indices, distance, N)
end_3 = time.time()
print('Time to finish function: "construct_graph"', end_3 - start_3)

start_4 = time.time()
path, start_end_dist = find_shortest_path(graph, start_node, end_node)
end_4 = time.time()
print('Time to finish function: "find_shortest_path"', end_4 - start_4)

start_5 = time.time()
plot_points(coord_list, indices, path)
end_5 = time.time()
print('Time to finish function: "plot_points"', end_5 - start_5)