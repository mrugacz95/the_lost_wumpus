# Przykladowy agent do zadania 'zagubiony Wumpus'. Agent porusza sie losowo.

import random
import numpy as np
from action import Action

# nie zmieniac nazwy klasy
from world import World


class Agent:
    # nie zmieniac naglowka konstruktora, tutaj agent dostaje wszystkie informacje o srodowisku
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
        self.indices = [pos for pos, val in np.ndenumerate(self.map)]
        self.tiled_map = np.tile(self.map, (3, 3))
        self.tiled_exits = np.argwhere(self.tiled_map == World.EXIT)
        self.tiled_caves = np.argwhere(self.tiled_map == World.CAVE)
        self.caves = self.map == World.CAVE
        self.empty = np.invert(self.caves)
        self.hist = np.full(self.map.shape, 1 / (self.width * self.height))
        # self.hist = np.full((self.width, self.height), 0.01)
        # self.hist[0, self.height - 1] = 0.2
        self._update_hist()
        return

    def _update_hist(self):
        self.hist = self.hist / np.sum(self.hist)

    def sense(self, sensor):
        self.hist[self.caves] *= (self.pj if sensor else (1 - self.pj))
        self.hist[self.empty] *= (self.pn if sensor else (1 - self.pn))
        self._update_hist()

    def move(self):
        max_pos = np.argmax(self.hist)
        y, x = np.unravel_index(max_pos, self.hist.shape)
        pos = y + self.height, x + self.width

        # pos = self.indices[np.random.choice(len(self.indices),
        #                                     p=self.hist.flatten())]  # select pos based on possibility
        # pos = pos[0] + self.height, pos[1] + self.width

        def find_closest(indices):
            def distance(a, b):
                return abs(a[0] - b[0]) + abs(a[1] - b[1])

            to_ret = None
            closest_dist = np.inf

            for idx in indices:
                if idx[0] == pos[0] and idx[1] == pos[1]:
                    continue
                d = distance(idx, pos)
                if d < closest_dist or to_ret is None:
                    to_ret = idx
                    closest_dist = d
            return to_ret

        closest = find_closest(self.tiled_exits)
        # make move
        min_pos = min(pos[0], closest[0]),min(pos[1], closest[1])
        max_pos = max(pos[0], closest[0]),max(pos[1], closest[1])

        condition = np.logical_and(
            np.logical_and(self.tiled_caves[:,0] >= min_pos[0], self.tiled_caves[:,0] <= max_pos[0]),
            np.logical_and(self.tiled_caves[:,1] >= min_pos[1], self.tiled_caves[:,1] <= max_pos[1]),
        )

        caves = self.tiled_caves[np.where(condition)]
        if len(caves) != 0:
            closest_cave = find_closest(caves)
            if closest_cave is not None:
                closest = closest_cave

        if closest[0] > pos[0]:
            dir = Action.DOWN
        elif closest[0] < pos[0]:
            dir = Action.UP
        elif closest[1] > pos[1]:
            dir = Action.RIGHT
        elif closest[1] < pos[1]:
            dir = Action.LEFT
        else:
            dir = random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

        for exit in self.tiled_exits:
            if pos in exit: # standing on exit
                dir = random.choice([Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT])

        roll, axis = {
            Action.LEFT: (-1, 1),
            Action.RIGHT: (1, 1),
            Action.DOWN: (1, 0),
            Action.UP: (-1, 0)
        }.get(dir)

        self.hist = np.roll(self.hist, roll, axis)
        mistake_prob = (1 - self.p) / 4
        self.hist += (np.roll(self.hist, 1, axis=0) * mistake_prob
                      + np.roll(self.hist, -1, axis=0) * mistake_prob
                      + np.roll(self.hist, 1, axis=1) * mistake_prob
                      + np.roll(self.hist, -1, axis=1) * mistake_prob
                      + self.hist * self.p)

        self._update_hist()
        return dir

    def histogram(self):
        return self.hist.copy()
