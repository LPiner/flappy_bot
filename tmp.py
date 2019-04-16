from selenium import webdriver
from flappy_bot.models.game import Game
from flappy_bot.types.keys import Keys
from flappy_bot.models.image_processor import ImageProcessor
import cv2
import time
import matplotlib.pyplot as plt
import numpy
from matplotlib.pyplot import plot, draw, show
from structlog import get_logger
from typing import List


logger = get_logger()



game = Game()
game.start_game()
#count = 1
while True:
    start_time = time.time()
    screen = game._grab_screen()

    # Here are the areas that will kill us if we touch it.
    danger_points, danger_mask = ImageProcessor.find_by_color(
        cv2.blur(screen, (9, 9)),
        upper_bound=numpy.array([70, 250, 250]),
        lower_bound=numpy.array([40, 115, 115])
    )

    danger_points = danger_points.reshape((-1, 2))
    # Array needs to be flipped from x, y to y, x
    danger_points = numpy.flip(danger_points, (1,))

    # sets are stupid fast.
    danger_points = set(tuple(map(tuple, danger_points)))


    def get_pipe_location() -> (int, int):
        t = time.time()
        width, height = danger_mask.shape[::-1]
        pipe_start_x_pos = None
        for y in range(25, 175, 10):
            if pipe_start_x_pos:
                break
            for x in range(30, width, 5):
                if (y, x) in danger_points:
                    #screen[y][x] = (100, 0, 0)
                    pipe_start_x_pos = x
                    break

        if not pipe_start_x_pos:
            raise Exception

        y_pos: List[int] = []
        for y in range(25, 175, 2):
            if (y, pipe_start_x_pos + 5) in danger_points:
                screen[y][pipe_start_x_pos + 5] = (150, 0, 0)
                y_pos.append(y)

        for c, i in enumerate(y_pos):
            if i == y_pos[-1]:
                raise Exception

            if y_pos[c+1] - i > 40:
                return i, y_pos[c+1]
        logger.debug("[get_pipe_location] Runtime.", runtime=time.time() - t)

    try:
        print(get_pipe_location())
    except Exception:
        pass


    plt.cla()
    plt.imshow(screen)
    plt.pause(0.00001)
    logger.debug("[__main__] Time for loop.", loop_time=time.time() - start_time)


#ImageProcessor.find_match_with_features(screen, bird)
#ImageProcessor._show_orb_keypoints(image=bird)
"""
27
Took 0.0009980201721191406seconds to find feature.
28
Took 0.0010137557983398438seconds to find feature.
29
Took 0.0009982585906982422seconds to find feature.
30
Took 0.002001523971557617seconds to find feature.
31
Took 0.0010004043579101562seconds to find feature.
32
Took 0.004002809524536133seconds to find feature.
33
Took 0.002001523971557617seconds to find feature.
34
Took 0.0020017623901367188seconds to find feature.
35
Took 0.002132415771484375seconds to find feature.
"""