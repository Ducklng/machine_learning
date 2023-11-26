from random import randint
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph

MAX_EDGE_WEIGHT = 15


def generate_weights(nodes_count: int) -> list[list[int]]:
    weights: list[list[int]] = [
        [-1 for _ in range(nodes_count)]
        for _ in range(nodes_count)
    ]
    for i in range(nodes_count):
        for j in range(nodes_count):
            if i != j:
                weights[i][j] = weights[j][i] = randint(1, MAX_EDGE_WEIGHT + 1)
    return weights


def find_min_edge(weights: list[list[int]]) -> tuple[int, int, int]:
    min_weight = weights[0][1]
    min_weight_nodes = (0, 1)
    for i in range(len(weights)):
        for j in range(len(weights)):
            if i < j:
                if weights[i][j] < min_weight:
                    min_weight = weights[i][j]
                    min_weight_nodes = (i, j)
    return min_weight_nodes[0], min_weight_nodes[1], min_weight


def main():
    nodes_count = 7
    clusters_count = 3

    # случайным образом заполнить матрицу весов в будущем графе и вывести его в виде рисунка
    weights = generate_weights(nodes_count)
    g = Graph()
    g.add_nodes_from(range(nodes_count))
    for i in range(nodes_count):
        for j in range(nodes_count):
            if i < j:
                g.add_edge(i, j, weight=weights[i][j])
    draw(g)

    # найти минимальное остовное дерево и вывести его в виде рисунка
    min_spanning_tree_edges: list[tuple[int, int]] = []
    visited_nodes: set[int] = set()
    not_visited_nodes: set[int] = set(range(nodes_count))
    min_edge = find_min_edge(weights)
    min_spanning_tree_edges.append((min_edge[0], min_edge[1]))
    for node in [min_edge[0], min_edge[1]]:
        visited_nodes.add(node)
        not_visited_nodes.remove(node)
    while len(not_visited_nodes) > 0:
        min_weight = MAX_EDGE_WEIGHT + 1
        min_weight_edge_nodes = (-1, -1)
        for visited_node in visited_nodes:
            for (neighbor_node, weight) in enumerate(weights[visited_node]):
                if neighbor_node not in visited_nodes and \
                        neighbor_node != visited_node and \
                        weight < min_weight:
                    min_weight = weight
                    min_weight_edge_nodes = (visited_node, neighbor_node)
        min_spanning_tree_edges.append(min_weight_edge_nodes)
        visited_nodes.add(min_weight_edge_nodes[1])
        not_visited_nodes.remove(min_weight_edge_nodes[1])
    min_spanning_tree = Graph()
    min_spanning_tree.add_nodes_from(visited_nodes)
    for u, v in min_spanning_tree_edges:
        min_spanning_tree.add_edge(u, v, weight=weights[u][v])
    draw(min_spanning_tree)

    # разбить на кластеры и вывести итоговое множество в виде списка: номер кластера - точки
    if clusters_count == 1:
        return

    sorted_min_spanning_tree_edges = sorted(min_spanning_tree_edges, key=lambda uv: weights[uv[0]][uv[1]])
    clusters_edges = sorted_min_spanning_tree_edges[:-(clusters_count - 1)]
    clusters: dict[int, list[int]] = dict()
    cluster_num = 0
    visited_clusters_nodes: set[int] = set()
    for node in range(nodes_count):
        if node in visited_clusters_nodes:
            continue

        cluster_num += 1
        neighbor_nodes = recursively_find_all_neighbors(node, clusters_edges)
        cluster_nodes = [node] + neighbor_nodes
        clusters[cluster_num] = cluster_nodes
        for cluster_node in cluster_nodes:
            visited_clusters_nodes.add(cluster_node)

    for (key, value) in clusters.items():
        print(f"Кластер {key} содержит ноды: {sorted(value)}")


def recursively_find_all_neighbors(for_node: int,
                                   in_clusters_edges: list[tuple[int, int]],
                                   excluding_nodes: Optional[list[int]] = None) -> list[int]:
    if excluding_nodes is None:
        excluding_nodes = []

    neighbors: list[int] = []
    for u, v in in_clusters_edges:
        if for_node == u and v not in excluding_nodes:
            neighbors.append(v)
        elif for_node == v and u not in excluding_nodes:
            neighbors.append(u)
    if len(neighbors) == 0:
        return []

    excluding_nodes.append(for_node)
    result = neighbors.copy()
    for neighbor in neighbors:
        result += recursively_find_all_neighbors(neighbor, in_clusters_edges, excluding_nodes)
    return result


def draw(g: Graph):
    pos = nx.spring_layout(g)
    nx.draw_networkx(g, pos)
    labels = nx.get_edge_attributes(g, "weight")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    plt.show()


if __name__ == "__main__":
    main()
