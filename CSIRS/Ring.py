import numpy as np


class ConcentricRings:
    def __init__(self, *args):
        if len(args) == 0:
            self.seed_point = None
            self.rings = []
        elif len(args) == 1 and isinstance(args[0], Ring):
            self.seed_point = None
            self.rings = [args[0]]
        elif len(args) == 1 and isinstance(args[0], np.ndarray):
            self.rings = []
            self.seed_point = args[0]

    def __getitem__(self, *args):
        if isinstance(args[0], (int, slice)):
            return self.rings[args[0]]
        elif isinstance(args[0][0], (int, slice)) and isinstance(args[0][1], slice):
            return np.array([ring[args[0][1]] for ring in self.rings[args[0][0]]]).reshape((-1, 3))
        elif isinstance(args[0][0], int) and isinstance(args[0][1], int):
            return self.rings[args[0][0]].points[args[0][1]]
        return [self.rings[i] for i in args[0]]

    def __len__(self):
        return len(self.rings)

    def getFaces(self, *args):
        if isinstance(args[0][0], slice) and isinstance(args[0][1], slice):
            return np.array([ring.faces[args[0][1]] for ring in self.rings[args[0][0]]]).reshape((-1, len(self.rings[0].faces[args[0][1]])))
        elif isinstance(args[0][0], int) and isinstance(args[0][1], (int, slice)):
            return self.rings[args[0][0]].faces[args[0][1]]
        return None

    def getNonNaNFacesIdx(self):
        indices = []
        for ring in self.rings:
            indices.extend(ring.faces[ring.getNonNan()])
        return indices

    def getNonNaNPoints(self):
        points = np.empty((0, 3))
        for ring in self.rings:
            points = np.vstack((points, ring.points[ring.getNonNan()]))
        return points

    def first_valid_rings(self, valid_number=1):
        i = 0
        while i < valid_number and i < len(self.rings):
            if len(self.rings[i].getNonNan()) != len(self.rings[i]):
                return False
            i += 1
        return True

    def addRing(self, *args):
        if len(args) == 1 and isinstance(args[0], Ring):
            self.rings.append(args[0])
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            self.rings.append(Ring(args[0], args[1]))

    def getRingsNumber(self):
        return len(self.rings)

    def getElementsNumber(self):
        """
        :return: number of elements in a ring
        """
        if self.rings:
            elements = 0
            for ring in self.rings:
                elements += ring.getElementsNumber()
            return elements
        return 0


class Ring:
    def __init__(self, *args):
        if len(args) == 0:
            self.points = []
            self.faces = []
        elif len(args) == 2 and isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
            self.points = np.array(args[0])
            self.faces = np.array(args[1])

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            return self.points[item]
        return [self.points[i] for i in item]

    def __len__(self):
        return len(self.points)

    def getElementsNumber(self):
        return np.count_nonzero(np.any(~np.isnan(self.points), axis=1))

    def getNonNan(self):
        return list(np.where(np.any(~np.isnan(self.points), axis=1) == True)[0])
