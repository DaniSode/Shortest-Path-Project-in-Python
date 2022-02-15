# Assignment_1_code
# Authors: Felix Mare, Daniel Soderqvist

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
import time
from tabulate import tabulate


def which_file(filename):

    """Takes the name of a coordinate file and returns the corresponding radius, start and end city.

        :param filename: string of the name of the file containing the desired set of coordinates.
        :type filename: str

        :return: The corresponding values of radius, start and end city according to the assignment.
        :rtype: (float, int)
    """

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

    """Takes the name of a coordinate file and converts its content to a numpy array.

        :param filename: string of the name of the file containing the desired set of coordinates.
        :type filename: str

        :return: Numpy array with the coordinate pairs found in the file.
        :rtype: np.ndarray
    """

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

    """
    Takes an array of coordinates, a list of city pairs and a list with the shortest route between 2 cities.
    Then plots all cities, routes between the cities and the shortest route.

        :param coord_list: Array containing the desired set of coordinates.
        :type coord_list: np.ndarray
        :param indices: Array with city pairs that are within a certain radius from each other.
        :type indices: np.ndarray
        :param path: A sequence of cities ensuring the shortest possible distance from start to end.
        :type path: list of int


        :return: Figure showing possible routes between all cities and shortest path from start to end.
        :rtype: None
    """

    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=5, c="r")
    plt.gca().set_aspect('equal')

    city_pair_coord = []
    for i, j in indices:
        city_pair_coord.append([[coord_list[i, 0], coord_list[i, 1]], [coord_list[j, 0], coord_list[j, 1]]])
    lc = LineCollection(city_pair_coord, linewidth=0.4, colors="gray")
    fig = plt.subplot()
    fig.add_collection(lc)

    path_coord_x = []
    path_coord_y = []
    for city in path:
        path_coord_x.append(coord_list[city, 0])
        path_coord_y.append(coord_list[city, 1])
    plt.plot(path_coord_x, path_coord_y, linewidth=1, c="blue")
    plt.title('Shortest path')



def construct_graph_connections(coord_list, radius):

    """Calculates the distance between cities and returns city pairs that are within a given radius along with the
    corresponding distance.

    :param coord_list: Array containing a set of coordinates.
    :type coord_list: np.ndarray
    :param radius: The maximum allowed distance between cities.
    :type radius: float

    :return: Array of available city pairs and array with their distances.
    :rtype: (np.ndarray, np.ndarray)
    """

    pair_indices = []
    distances = []
    for i, city in enumerate(coord_list):
        tot_distances = np.hypot(coord_list[i+1:, 0]-city[0], coord_list[i+1:, 1]-city[1])
        for j, distance in enumerate(tot_distances):
            if distance <= radius:
                pair_indices.append([i, i + 1 + j])
                distances.append(distance)

    return np.array(pair_indices), np.array(distances)


def construct_graph(indices, distance, N):

    """Constructs a sparse row matrix containing city pairs and the distance between them.

    :param indices: Array with city pairs that are within a certain radius from each other.
    :type indices: np.ndarray
    :param distance: Array with distances between city pairs
    :type distance: np.ndarray
    :param N: Integer with number of cities
    :type N: int

    :return: Compressed sparse row matrix satisfying the relationship a[row_ind[k], col_ind[k]] = data[k]
    :rtype: csr_matrix
    """

    graph = csr_matrix((distance, indices.T), shape=(N, N))

    return graph


def find_shortest_path(graph, start_node, end_node):

    """Calculates the shortest path between two specified cities.

    :param graph: Compressed sparse row matrix containing city pairs and the distance between them.
    :type graph: csr_matrix
    :param start_node: The city where the path starts.
    :type start_node: int
    :param end_node: The city where the path ends.
    :type end_node: int

    :return: A sequence of cities ensuring the shortest possible distance along with the total distance.
    :rtype: (list of int, float)
    """

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

    """Returns city pairs that are within a given radius along with the corresponding distance using a faster method.

    :param coord_list: Array containing a set of coordinates.
    :type coord_list: np.ndarray
    :param radius: The maximum allowed distance between cities.
    :type radius: float
    :return: Array of available city pairs and array of distances.
    :rtype: (np.ndarray, np.ndarray)
    """

    Tree = cKDTree(coord_list)
    possible_cities = Tree.query_ball_point(coord_list, radius)
    indices=[]
    city_1 = []
    city_2 = []
    for i, element in enumerate(possible_cities):
        for j in element:
            if i < j:
                indices.append([i, j])
                city_1.append(coord_list[i])
                city_2.append(coord_list[j])
    city_1 = np.array(city_1)
    city_2 = np.array(city_2)
    distances = np.hypot(city_1[:, 0]-city_2[:, 0], city_1[:, 1]-city_2[:, 1])

    return np.array(indices), np.array(distances)

# Creating a menu where the input is which file to run and whether to run the fast or slow function when constructing graph connections.
# Printing time for each function and showing plot.

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
start_6 = time.time()
table = []
if int(selection) == 1:
    start_2 = time.time()
    indices, distance = construct_fast_graph_connections(coord_list, radius)
    end_2 = time.time()
    table.append('read_coordinate_file')
    table.append('construct_fast_graph_connections')
elif int(selection) == 2:
    start_2 = time.time()
    indices, distance = construct_graph_connections(coord_list, radius)
    end_2 = time.time()
    table.append('read_coordinate_file')
    table.append('construct_graph_connections')

start_3 = time.time()
graph = construct_graph(indices, distance, N)
end_3 = time.time()
table.append('construct_graph')

start_4 = time.time()
path, start_end_dist = find_shortest_path(graph, start_node, end_node)
end_4 = time.time()
table.append('find_shortest_path')

start_5 = time.time()
plot_points(coord_list, indices, path)
end_5 = time.time()
table.append('plot_points')

print('\nThe shortest path from city:', start_node,'to', end_node,'is through cities:',path)
print('The total distance is:', start_end_dist)
end_6 = time.time()
print('\nThe time to finish each and one of the functions is listed in the table below:')
time_total = [[round(end_1 - start_1, 3)],
              [round(end_2 - start_2, 3)],
              [round(end_3 - start_3, 3)],
              [round(end_4 - start_4, 3)],
              [round(end_5 - start_5, 3)],
              [round(end_6 - start_6, 3)]]
table.append('Running the entire program')
dictionary = list(zip(table, time_total))
headers = ["function", "time (s)"]
print(tabulate(dictionary, headers=headers, tablefmt="pretty"))

plt.show()