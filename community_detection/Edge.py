class Edge:

    def __init__(self, source, target, weight: float = 1):

        self._source = source
        self._target = target
        self._weight = weight

    @property
    def source(self):

        return self._source

    @property
    def target(self):

        return self._target

    @property
    def weight(self):

        return self._weight
