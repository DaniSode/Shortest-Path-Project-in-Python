import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
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


def plot_points(coord_list, indices, seq):
    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=5, c="r")
    plt.gca().set_aspect(1.05)  # height to width ratio 1.5
    city_pair_coord = []
    for i, j in indices:
        city_pair_coord.append([[coord_list[i, 0], coord_list[i, 1]], [coord_list[j, 0], coord_list[j, 1]]])
    hej = LineCollection(city_pair_coord, linewidth=0.4, colors="gray")
    fig = plt.subplot()
    fig.add_collection(hej)

    troll_x = []
    troll_y = []
    for o in seq:
        troll_x.append(coord_list[o, 0])
        troll_y.append(coord_list[o, 1])
    plt.plot(troll_x, troll_y, linewidth=1, c="blue")
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
    distance, predecessor = shortest_path(csgraph = graph, directed = False, return_predecessors = True, indices = start_node)

    start_end_dist = distance[end_node]



    a = end_node

    seq = [end_node]

    while predecessor[a] != start_node:


        a = predecessor[a]

        seq.append(a)
    seq.append(start_node)

    seq.reverse()


    return seq, start_end_dist

# To have in the end
# print("\nInput filename and radius: ")
# print("For example 'SampleCoordinates 0.08' and press enter.")
# Inputs



radius = 0.005
#N = 7 #length(SampleCoordinates)
start_node = 311
end_node = 702

# Call functions
start = time.time()
coord_list = read_coordinate_file('HungaryCities.txt')
N = len(coord_list)
end = time.time()
print('Time to finish function: "read_coordinate_file"', end - start)

start = time.time()
indices, distance = construct_graph_connections(coord_list, radius)
end = time.time()
print('Time to finish function: "construct_graph_connections"', end - start)

start = time.time()
graph = construct_graph(indices, distance, N)
end = time.time()
print('Time to finish function: "construct_graph"', end - start)



start = time.time()
seq, start_end_dist = find_shortest_path(graph, start_node, end_node)
end = time.time()
print('Time to finish function: "find_shortest_path"', end - start)

start = time.time()
plot_points(coord_list, indices, seq)
end = time.time()
print('Time to finish function: "plot_points"', end - start)


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