class Node:

    def __init__(self, name, cluster=None, neighbours=None):

        self._name = name
        self._cluster = cluster
        self._neighbours = [] if neighbours is None else neighbours

    @property
    def name(self):

        return self._name

    @property
    def cluster(self):

        return self._cluster

    @cluster.setter
    def cluster(self, cluster):

        self._cluster = cluster

    @property
    def neighbours(self):

        return self._neighbours

    @neighbours.setter
    def neighbours(self, nodes):

        self._neighbours = nodes

    def add_neighbour(self, node):

        self._neighbours += [node]

    def remove_neighbour(self, node):

        self._neighbours.remove(node)

    def __str__(self):

        return f'{self.name}'

    def __repr__(self):

        return f'{self.name}'
