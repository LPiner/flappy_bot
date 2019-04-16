from selenium import webdriver
from flappy_bot.models.game import Game
from flappy_bot.types.keys import Keys
from flappy_bot.models.image_processor import ImageProcessor
import cv2
import time
import matplotlib.pyplot as plt
import numpy
from matplotlib.pyplot import plot, draw, show



game = Game()
game.start_game()
#count = 1
while True:
    game._grab_screen()


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