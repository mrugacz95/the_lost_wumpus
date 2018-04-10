import numpy as np
from action import Action

from world import World


class Agent:
    """
        Srodowisko sklada sie z:
        - prawdopodobienstwa wykonania poprawnego ruchu (p),
        - prawdopodobienstwa wyczucia jamy gdy stoi sie w niej (pj),
        - prawdopodobienstwa wyczucia jamy gdy stoi sie poza nia (pn)
    """
    def __init__(self, p, pj, pn, height, width, areaMap):
        self.p = p
        self.pj = pj
        self.pn = pn
        self.height = height
        self.width = width
        self.map = np.zeros((self.height, self.width), dtype='U1')
        for idx, row in enumerate(areaMap):
            self.map[idx, :] = list(areaMap[idx])

        self.exit = np.where(self.map == World.EXIT)
        self.caves = self.map == World.CAVE
        self.empty = np.invert(self.caves)

        self.hist = np.full(self.map.shape, 1)
        self.hist[self.exit] = 0
        self._update_hist()

        def calculate_distances():
            dist = np.zeros(self.map.shape, dtype=float)
            w, h = self.width, self.height

            def distance(a, b):
                x1, y1 = a
                x2, y2 = b
                return min(abs(x1 - x2), w - abs(x1 - x2)) + min(abs(y1 - y2), h - abs(y1 - y2))

            for idx, val in np.ndenumerate(dist):
                dist[idx] = distance(idx, self.exit)
            dist[self.exit] = 1  # to not divide by zero
            return dist

        self.distances_to_exit = 1 / (calculate_distances())
        self.distances_to_exit[self.exit] = width * height
        self.certainty = np.zeros((4,) + self.map.shape, dtype=float)
        self.result = np.zeros(4, dtype=float)

    def normalize_hist(self, hist):
        return hist / np.sum(hist)

    def _update_hist(self):
        self.hist = self.normalize_hist(self.hist)

    def sense_hist(self, sensor, hist):
        hist[self.caves] *= (self.pj if sensor else (1 - self.pj))
        hist[self.empty] *= (self.pn if sensor else (1 - self.pn))
        return hist

    def sense(self, sensor):
        self.hist = self.sense_hist(sensor, self.hist)
        self._update_hist()

    def simulate_move(self, direction):
        tmp_hist = self.hist.copy()
        roll, axis = {
            Action.LEFT: (-1, 1),
            Action.RIGHT: (1, 1),
            Action.DOWN: (1, 0),
            Action.UP: (-1, 0)
        }.get(direction)

        tmp_hist = np.roll(tmp_hist, roll, axis)
        mistake_prob = (1 - self.p) / 4
        tmp_hist += (np.roll(tmp_hist, 1, axis=0) * mistake_prob
                     + np.roll(tmp_hist, -1, axis=0) * mistake_prob
                     + np.roll(tmp_hist, 1, axis=1) * mistake_prob
                     + np.roll(tmp_hist, -1, axis=1) * mistake_prob
                     + tmp_hist * self.p)
        tmp_hist = self.normalize_hist(tmp_hist)
        return tmp_hist

    def calculate_certainty(self, histogram):
        return np.sum(histogram * self.distances_to_exit)

    def move(self):
        self.hist[self.exit] = 0

        moves = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

        for idx, action in enumerate(moves):
            self.certainty[idx] = self.simulate_move(action)
            self.result[idx] = self.calculate_certainty(self.certainty[idx])
        max = int(np.argmax(self.result))
        dir = moves[max]
        self.hist = self.simulate_move(moves[max])
        return dir

    def histogram(self):
        return self.hist.copy()
