from Node import Node
from Edge import Edge
from Cluster import Cluster
from steady_state import steady_state
import numpy as np
from typing import List
from scipy import sparse


class Graph:

    def __init__(self, nodes: List[Node], edges: List[Edge], states = None):

        self._nodes = nodes
        self._node_dict = self._create_dict()
        self._node_edges = edges

        self._node_adjacency = self._nodes_transition()
        self._node_states = steady_state(self._node_adjacency) if states is None else np.array(states)
        self._node_joint = self._joint(self._node_adjacency, self._node_states)

        self.update_neighborhood()

        self._clusters = [Cluster(i, [node]) for i, node in enumerate(self._nodes)]
        self._cluster_edges = [Edge(self._node_dict[edge.source], self._node_dict[edge.target], edge.weight) for edge in self._node_edges]
        for i, node in enumerate(self._nodes):
            node.cluster = i

        self._cluster_adjacency = self._node_adjacency
        self._cluster_states = self._node_states
        self._cluster_joint = self._node_joint

    @staticmethod
    def _node_uniqueness(nodes):

        if len(nodes) != len(set([node.name for node in nodes])):
            raise ValueError("There was a duplicate in provided nodes.")

    @staticmethod
    def _change_cluster(node, cluster):

        node.cluster = cluster

        return node

    def _create_dict(self):

        self._node_uniqueness(self._nodes)

        return {node.name: i for i, node in enumerate(self._nodes)}

    def _nodes_transition(self):

        n = self.no_of_nodes()
        edge_matrix = np.array([[edge.source, edge.target, edge.weight] for edge in self._node_edges])
        dangling_nodes = set(range(n)).difference(set(edge_matrix[:, 0]))
        if len(dangling_nodes) > 0:
            edge_matrix = np.vstack((edge_matrix, np.array([[node, node, 1] for node in dangling_nodes])))

        matrix = sparse.csr_matrix((edge_matrix[:, 2], (edge_matrix[:, 0], edge_matrix[:, 1])), shape=(n, n))
        del edge_matrix

        return matrix.multiply(1 / matrix.sum(axis=1))

    def update_neighborhood(self):

        nzr, nzc = self._node_adjacency.nonzero()
        for i in range(self.no_of_nodes()):
            self._nodes[i].neighbours = [self._nodes[neighbour] for neighbour in set(np.concatenate((nzr[nzc == i],
                                                                                                     nzc[nzr == i])))]

    def _clusters_transition(self):

        n = self.no_of_clusters()
        matrix = np.zeros((n, n))
        for edge in self._cluster_edges:
            matrix[edge.source, edge.target] += edge.weight

        return matrix

    @staticmethod
    def _joint(adjacency, states):

        return adjacency.multiply(sparse.coo_matrix(states).T)

    def no_of_nodes(self):

        return len(self._nodes)

    def no_of_clusters(self):

        return len(self._clusters)

    @staticmethod
    def neighbours(node):

        return node.neighbours

    def list_nodes(self):

        return list(map(lambda node: node.name, self._nodes))

    def get_node(self, node):

        return self._nodes[self._node_dict[node]]

    def get_cluster(self, node):

        return self._clusters[node.cluster]

    def _partial_objectivity(self, cluster):

        try:
            inside = self._cluster_joint[cluster, cluster] * np.log(self._cluster_adjacency[cluster, cluster] / self._cluster_states[cluster])
        except RuntimeWarning:
            inside = 0

        try:
            outside = (self._cluster_states[cluster] - self._cluster_joint[cluster, cluster]) * np.log((1 - self._cluster_adjacency[cluster, cluster]) / (1 - self._cluster_states[cluster]))
        except RuntimeWarning:
            outside = 0

        return inside + outside

    def objectivity(self):

        return max(0, sum(np.vectorize(self._partial_objectivity)(np.arange(self.no_of_clusters()))))

    def recalculate(self):

        self._cluster_adjacency = self._clusters_transition()
        self._cluster_states = np.array([sum(self._node_states[[self._node_dict[node] for node in cluster.nodes]]) for cluster in self._clusters])
        self._cluster_joint = self._joint(self._cluster_adjacency, self._cluster_states)

        return self

    def __str__(self):

        return f'{[(node.name, node.cluster) for node in self._nodes]}'

    def __repr__(self):

        return f'{[(node.name, node.cluster) for node in self._nodes]}'

