import pygame
from pygame import draw
from pygame.rect import Rect

from world import World
from copy import deepcopy
import numpy as np


class GUI(object):
    """Okienko wizualizacji zachowania agenta w srodowisku."""

    __MARGIN = 4
    """Margines dla etykiet i przyciskow."""

    __ACTION_LABEL_TEXT = 'Last action : {}'
    """Tresc etykiety opisujacej ostatnia wybrana akcje."""

    __MOTION_LABEL_TEXT = 'Last motion (dy, dx): {}'
    """Tresc etykiety opisujacej ostatnio wykonane przemieszczenie."""

    __SENSOR_LABEL_TEXT = 'Sensor state: {}'
    """Tresc etykiety opisujacej stan sensora jam."""

    __NSTEPS_LABEL_TEXT = 'Steps counter: {}'
    """Tresc etykiety opisujacej liczbe wykonanych ruchow."""

    __DENORM_LABEL_TEXT = 'Denormalize'
    """Tresc etykiety checkboxa okreslajacego czy histogram ma zostac zdenormalizowany."""

    __NORMALIZED_LABEL_TEXT = '{:.2f}'

    def __init__(self, agent_factory, environment, size):
        """Inicjalizuje obiekt okna wizualizera."""
        pygame.init()
        self.box_size = size

        self.agent_factory = agent_factory
        self.env = environment
        self.env.reset(self.agent_factory)
        if not self.env.is_completed():
            self.env.step_sense()

        self.draw_width = self.env.width * self.box_size
        self.draw_height = self.env.height * self.box_size

        self.screen = pygame.display.set_mode((self.draw_width, self.draw_height))
        self.font = pygame.font.SysFont("monospace", 15)
        return

    def draw_labels(self):
        """Wymusza aktualizacje zawartosci okna."""

        self.action_label = self.font.render(GUI.__ACTION_LABEL_TEXT.format(self.env.agent_last_action), 1,
                                             (0, 0, 0))
        self.motion_label = self.font.render(GUI.__MOTION_LABEL_TEXT.format(self.env.agent_last_motion), 1,
                                             (0, 0, 0))
        self.sensor_label = self.font.render(GUI.__SENSOR_LABEL_TEXT.format(self.env.agent_sensor), 1,
                                             (0, 0, 0))
        self.nsteps_label = self.font.render(GUI.__NSTEPS_LABEL_TEXT.format(self.env.agent_steps_counter), 1,
                                             (0, 0, 0))

        y = 1
        label_height = self.font.get_height()

        self.screen.blit(self.action_label, (GUI.__MARGIN, y))
        y += label_height + GUI.__MARGIN

        self.screen.blit(self.motion_label, (GUI.__MARGIN, y))
        y += label_height + GUI.__MARGIN

        self.screen.blit(self.sensor_label, (GUI.__MARGIN, y))
        y += label_height + GUI.__MARGIN

        self.screen.blit(self.nsteps_label, (GUI.__MARGIN, y))

    def step(self):
        """Akcja wykonywana po wcisnieciu przycisku Step"""

        self.env.step_move()
        if not self.env.is_completed():
            self.env.step_sense()
        self.draw()

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.draw_map()
        self.draw_labels()
        pygame.display.flip()

    def reset(self):
        """Akcja wykonywana po wcisnieciu przycisku Reset"""

        self.env.reset(self.agent_factory)
        if not self.env.is_completed():
            self.env.step_sense()

    def __gradient(self, val):
        """Dostarcza gradient kolorow dla wartosci histogramu."""

        r = 2 * (1 - val) if val > 0.5 else 1
        g = 2 * val if val < 0.5 else 1
        return r * 255, g * 255, 0

    def __denormalize_histogram(self, histogram):
        # histogram = deepcopy(histogram)
        # denominator = max(max(line) for line in histogram)
        # if denominator != 0:
        #     for y in range(self.env.height):
        #         for x in range(self.env.width):
        #             histogram[y][x] /= denominator
        histogram = (histogram - np.min(histogram)) / np.max(histogram)
        return histogram

    def draw_map(self):
        """Rysuje mape srodowiska i wiedzy agenta."""

        histogram = self.env.agent.histogram()
        norm_histogram = self.__denormalize_histogram(self.env.agent.histogram())
        # if (self.denorm_chbox.get_active()):
        #     histogram = self.__denormalize_histogram(histogram)

        for y in range(self.env.height):
            for x in range(self.env.width):
                draw.rect(self.screen, self.__gradient(norm_histogram[y][x]), Rect(x * self.box_size + 1,
                                                                                   y * self.box_size + 1,
                                                                                   self.box_size - 2,
                                                                                   self.box_size - 2), 0)
                prob = self.font.render(GUI.__NORMALIZED_LABEL_TEXT.format(histogram[y][x]), 1,
                                        (0, 20, 80))
                self.screen.blit(prob, (x * self.box_size + 1, y * self.box_size + 1))
        map = self.env.map

        for y in range(self.env.height):
            for x in range(self.env.width):

                if map[y][x] == World.CAVE:
                    draw.rect(self.screen, (120, 120, 120), Rect(x * self.box_size + self.box_size / 4,
                                                                 y * self.box_size + self.box_size / 4,
                                                                 self.box_size / 2,
                                                                 self.box_size / 2), 0)

                if map[y][x] == World.EXIT:
                    draw.line(self.screen, (0, 0, 0), (x * self.box_size, y * self.box_size),
                              (x * self.box_size + self.box_size, y * self.box_size + self.box_size))
                    draw.line(self.screen, (0, 0, 0), (x * self.box_size + self.box_size, y * self.box_size),
                              (x * self.box_size, y * self.box_size + self.box_size))

        xc = int(self.env.agent_x * self.box_size + (self.box_size / 2))
        yc = int(self.env.agent_y * self.box_size + (self.box_size / 2))

        draw.circle(self.screen, (0, 20, 200), (xc, yc), int(self.box_size / 3))

        yc, xc = np.unravel_index(np.argmax(histogram), histogram.shape)
        draw.circle(self.screen, (20, 20, 100),
                    (int(xc * self.box_size + self.box_size / 2),
                     int(yc * self.box_size + self.box_size / 2)),
                    int(self.box_size / 3), 1)

    def main(self):
        done = False
        last_tick = 0
        while not done:
            while last_tick + 1 > pygame.time.get_ticks() // 1000:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        break
                    elif event.type == pygame.key:
                        print('key')
                        {
                            pygame.K_r: lambda: self.reset()
                        }.get(event.key, lambda: None)()
            last_tick += 1
            self.step()


def visualise(agent_factory, environment, size):
    """Tworzy i wyswietla okienko wizualizacji."""
    GUI(agent_factory, environment, size).main()
