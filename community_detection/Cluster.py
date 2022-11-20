class Cluster:

    def __init__(self, name, nodes):

        self._name = name
        self._nodes = nodes
        self._nodes_names = [node.name for node in self._nodes]

    @property
    def name(self):

        return self._name

    @property
    def nodes(self):

        return self._nodes_names

    @nodes.setter
    def nodes(self, new_nodes):

        self._nodes = new_nodes
        self._nodes_names = [node.name for node in self._nodes]

    def append_node(self, node):

        self._nodes += [node]
        self._nodes_names += [node.name]

    def remove_node(self, node):

        self._nodes.remove(node)
        self._nodes_names.remove(node.name)

    def __str__(self):

        return f'{self.name}, {self.nodes}'

    def __repr__(self):

        return f'{self.name}, {self.nodes}'
