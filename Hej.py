import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import csr_matrix
import time

#test
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


def plot_points(coord_list, indices):
    plt.scatter(coord_list[:, 0], coord_list[:, 1], s=5, c="r")
    plt.gca().set_aspect(1.05)  # height to width ratio 1.5
    city_pair_coord = []
    for i, j in indices:
        city_pair_coord.append([[coord_list[i, 0], coord_list[i, 1]], [coord_list[j, 0], coord_list[j, 1]]])
    hej = LineCollection(city_pair_coord, linewidth=0.4, colors="gray")
    fig = plt.subplot()
    fig.add_collection(hej)
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


# To have in the end
# print("\nInput filename and radius: ")
# print("For example 'SampleCoordinates 0.08' and press enter.")
# Inputs

radius = 0.0025
N = 7 #length(SampleCoordinates)

# Call functions
start = time.time()
coord_list = read_coordinate_file('GermanyCities.txt')
end = time.time()
print('Time to finish function: "read_coordinate_file"', end - start)

start = time.time()
indices, distance = construct_graph_connections(coord_list, radius)
end = time.time()
print('Time to finish function: "construct_graph_connections"', end - start)

start = time.time()
#construct_graph(indices, distance, N)
end = time.time()
print('Time to finish function: "construct_graph"', end - start)

start = time.time()
plot_points(coord_list, indices)
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