import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial import cKDTree
import time


##### WHICH FILENAME #####
def which_file(filename):
    """Take the name of a coordinate file and return the corresponding radius, start and end city.

        :param filename: string of the name of the file containing the desired set of coordinates.
        :type filename: str

        :return: The corresponding values of radius, start and end city according to the assignment.
        :rtype: (float, int)
    """

    # A function taking the name of the file as input
    # and returns the corresponding radius, start city and end city.

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
    """Take the name of a coordinate file and convert its content to a numpy array.

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


##### TASK 2, 5 and 7 #####
def plot_points(coord_list, indices, path):
    """Takes an array of coordinates, a list of possible city pairs and a list with the shortest route between 2 cities.
    Then plots all cities, possible routes from each city and the shortest route.

        :param coord_list: string of the name of the file containing the desired set of coordinates.
        :type coord_list: np.ndarray
        :param indices:
        :type indices:
        :param path:


        :return: Numpy array with the coordinate pairs found in the file.
        :rtype: np.ndarray
    """
    # A function taking inputs as coord_list (returned from task 1), indices (defined in the
    # functions construct_graph_connections or construct_fast_graph_connections) and
    # path (defined in function construct_graph). The function returns a merged plot of all
    # sub-tasks

    ##### Task 2 #####

    # The cities are ploted with the scatter command and taking the coordinates of each
    # city as input showing all cities as dots.

    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=5, c="r")
    plt.gca().set_aspect('equal')

    ##### TASK 5 #####

    # The available connections between the cities were ploted using a for loop giving the
    # connected cities the right coordinates using coord_list (all coordinates) and
    # indices (all possible connections). The coordinates of each connection were appended
    # to a list and the list were converted to a LineCollection which in turn were ploted
    # showing all possible connections as grey lines.

    city_pair_coord = []

    for i, j in indices:
        city_pair_coord.append([[coord_list[i, 0], coord_list[i, 1]], [coord_list[j, 0], coord_list[j, 1]]])

    lc = LineCollection(city_pair_coord, linewidth=0.4, colors="gray")
    fig = plt.subplot()
    fig.add_collection(lc)

    ##### TASK 7 #####

    # With path (number of cities describing the shortest path between start and end city)
    # the corresponding coordinates were linked to each city in path using a for loop. The
    # coordinates were then ploted showing the shortest path as a blue line.

    path_coord_x = []
    path_coord_y = []

    for city in path:
        path_coord_x.append(coord_list[city, 0])
        path_coord_y.append(coord_list[city, 1])

    plt.plot(path_coord_x, path_coord_y, linewidth=1, c="blue")
    plt.show()


##### TASK 3 #####
def construct_graph_connections(coord_list, radius):

    # A function taking coord_list (coordinates of all cities) and radius (the maximum possible
    # distance between two cities) and returns indices (all possible connections between cities)
    # and distances (the corresponding distances between the cities). The function are using a
    # nested for loop firstly to with enumerate give all cities a specific number than by look
    # at city to city and with the inner loop check against all other connections to see if the
    # distance is less than the given maximum radius. All possible connections are saved to
    # indices and distances. By using 'i+1:' in the calculations of tot_distances the loop just
    # consider all possible connections forward on which exclude duplicates like 1-2 and 2-1.

    pair_indices = []
    distances = []

    for i, city in enumerate(coord_list):
        tot_distances = np.hypot(coord_list[i+1:, 0]-city[0], coord_list[i+1:, 1]-city[1])
        for j, distance in enumerate(tot_distances):
            if distance <= radius:
                pair_indices.append([i, i + 1 + j])
                distances.append(distance)

    return np.array(pair_indices), np.array(distances)
"""
"""
##### TASK 4 #####
def construct_graph(indices, distance, N):

    # A function taking indices (possible connections), distances (corresponding distances)
    # and N (total amount of cities) and returning graph (sparse matrix of indices and distances)

    graph = csr_matrix((distance, indices.T), shape=(N, N))

    return graph


##### TASK 6 #####
def find_shortest_path(graph, start_node, end_node):

    #

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


##### TASK 9 #####
def construct_fast_graph_connections(coord_list, radius):

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

if int(selection) == 1:
    print('\nTime to finish function: "read_coordinate_file"', end_1 - start_1, 's')
    start_2 = time.time()
    indices, distance = construct_fast_graph_connections(coord_list, radius)
    end_2 = time.time()
    print('Time to finish function: "construct_fast_graph_connections"', end_2 - start_2, 's')
elif int(selection) == 2:
    print('\nTime to finish function: "read_coordinate_file"', end_1 - start_1, 's')
    start_2 = time.time()
    indices, distance = construct_graph_connections(coord_list, radius)
    end_2 = time.time()
    print('Time to finish function: "construct_graph_connections"', end_2 - start_2, 's')

start_3 = time.time()
graph = construct_graph(indices, distance, N)
end_3 = time.time()
print('Time to finish function: "construct_graph"', end_3 - start_3, 's')

start_4 = time.time()
path, start_end_dist = find_shortest_path(graph, start_node, end_node)
end_4 = time.time()
print('Time to finish function: "find_shortest_path"', end_4 - start_4, 's')

start_5 = time.time()
plot_points(coord_list, indices, path)
end_5 = time.time()
print('Time to finish function: "plot_points"', end_5 - start_5, 's')

print('\nThe shortest path from city:', start_node,'to', end_node,'is through cities:',path)
print('The total distance is:', start_end_dist)

print(read_coordinate_file.__doc__)
print(which_file.__doc__)