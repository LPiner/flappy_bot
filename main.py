from selenium import webdriver
from flappy_bot.models.game import Game
from typing import List
from flappy_bot.types.keys import Keys
from flappy_bot.models.image_processor import ImageProcessor
import cv2
import time
import matplotlib.pyplot as plt
import numpy
from matplotlib.pyplot import plot, draw, show
from structlog import get_logger

logger = get_logger(__name__)


if __name__ == "__main__":
    game = Game()
    game.start_game()
    game.input(Keys.SPACE)
    screen_history = []

    restart_button = cv2.imread("img/restart.png", cv2.IMREAD_GRAYSCALE)

    count = 0
    last_screen = None
    last_position = None
    last_screen_shot = time.time()
    last_jump_at = time.time()
    while True:
        # Start our browser and the game.
        start_time = time.time()

        #Get the screen shot of the game.
        screen = game._grab_screen()
        grey_scale_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        save_screen = False

        # Check for the game over button.
        restart_button_search = ImageProcessor.find_match_with_template(grey_scale_screen, restart_button)
        game_over = False if restart_button_search[0].size == 0 else True

        if game_over:
            for idx, c in enumerate(screen_history):
                cv2.imwrite(f"tmp/{idx}_screen.png", c)
            # ded
            game.input(Keys.SPACE)
            game.input(Keys.SPACE)
            continue


        # Here are the areas that will kill us if we touch it.
        danger_points, danger_mask = ImageProcessor.find_by_color(
            cv2.blur(screen, (9, 9)),
            upper_bound=numpy.array([75, 255, 255]),
            lower_bound=numpy.array([35, 90, 0])
        )

        # Find colors that only the bird is going to have.
        bird_points, red_bird_mask = ImageProcessor.find_by_color(
            screen,
            upper_bound=numpy.array([15, 255, 255]),
            lower_bound=numpy.array([0, 100, 100])
        )

        # merge our findings together.

        # If we dont find the bird then move to the next screen
        if bird_points is None or bird_points.size == 0:
            continue


        # This should be the center point of the bird.
        bird_center = numpy.average(bird_points, axis=0)[0]

        bird_width = 30
        bird_height = 15

        #screen = ImageProcessor.draw_circle(image=screen, center=(int(bird_center[0]), int(bird_center[1])), radius=int(bird_width/2))
        bird_x, bird_y = int(bird_center[0]), int(bird_center[1])

        lowest = bird_y

        # Figure out what is below our bird
        lower_hitbox = []
        lower_hitbox_y_range = range(int(bird_y), int(bird_y + bird_height / 2) + 13)
        lower_hitbot_x_range = range(int(bird_x - bird_width / 2 - 5), int(bird_x + bird_width / 2 + 5))
        for y in lower_hitbox_y_range:
            for x in lower_hitbot_x_range:
                lower_hitbox.append((y, x))
            if y > lowest:
                lowest = y

        lower_hitbox = set(lower_hitbox)

        upper_hitbox = []
        upper_hitbox_y_range = range(int(bird_y) - 30, int(bird_y+bird_height/2))
        upper_hitbox_x_range = range(int(bird_x-bird_width/2), int(bird_x + bird_width/2) + 5)
        for y in upper_hitbox_y_range:
            for x in upper_hitbox_x_range:
                upper_hitbox.append((y, x))

        upper_hitbox = set(upper_hitbox)

        # Visually draw our lower hitbox.
        screen = cv2.rectangle(screen, (list(lower_hitbot_x_range)[0], list(lower_hitbox_y_range)[0]), (list(lower_hitbot_x_range)[-1], list(lower_hitbox_y_range)[-1]), (0, 0, 255), 1)
        screen = cv2.rectangle(screen, (list(upper_hitbox_x_range)[0], list(upper_hitbox_y_range)[0]), (list(upper_hitbox_x_range)[-1], list(upper_hitbox_y_range)[-1]), (0, 0, 255), 1)
        danger_mask = cv2.rectangle(danger_mask, (list(lower_hitbot_x_range)[0], list(lower_hitbox_y_range)[0]), (list(lower_hitbot_x_range)[-1], list(lower_hitbox_y_range)[-1]), (0, 0, 255), 1)
        danger_mask = cv2.rectangle(danger_mask, (list(upper_hitbox_x_range)[0], list(upper_hitbox_y_range)[0]), (list(upper_hitbox_x_range)[-1], list(upper_hitbox_y_range)[-1]), (0, 0, 255), 1)

        bird_y = lowest

        should_jump = False
        force_no_jump = False
        danger_points = danger_points.reshape((-1, 2))
        # Array needs to be flipped from x, y to y, x
        danger_points = numpy.flip(danger_points, (1,))
        # sets are stupid fast.
        danger_points = set(tuple(map(tuple, danger_points)))



        def get_pipe_location() -> (int, int):
            t = time.time()
            width, height = danger_mask.shape[::-1]
            pipe_start_x_pos = None
            for y in range(10, 175, 10):
                if pipe_start_x_pos:
                    break
                for x in range(40, width, 5):
                    if (y, x) in danger_points:
                        # screen[y][x] = (100, 0, 0)
                        pipe_start_x_pos = x
                        break

            if not pipe_start_x_pos:
                raise Exception

            y_pos: List[int] = []
            for y in range(10, 175, 2):
                if (y, pipe_start_x_pos + 5) in danger_points:
                    screen[y][pipe_start_x_pos + 5] = (150, 0, 0)
                    y_pos.append(y)

            for c, i in enumerate(y_pos):
                if i == y_pos[-1]:
                    raise Exception

                if y_pos[c + 1] - i > 40:
                    logger.debug("[get_pipe_location] Runtime.", runtime=time.time() - t, y1=i, y2=y_pos[c + 1])
                    return i, y_pos[c + 1]

        floor = 120
        try:
            top, bottom = get_pipe_location()
            floor = bottom
        except Exception as e:
            print(e)
            pass

        target_height = floor
        screen = ImageProcessor.draw_line_on_image(screen, (50, target_height), (50+15, target_height))

        # if the x jumps forward too much then we know the detection has fucked up

        if last_screen is None:
            last_screen = screen

        """
        if emergency_jump and bird_y < last_position:
            pass
        elif bird_y < target_height and bird_y < last_position:
            # we are below the floor but are currently rising, take no action
            pass
        """
        # Bird takes about .25 seconds to hit the top of its jump.
        # This prevents jump spam
        required_delay = .22

        if len(screen_history) > 30:
            screen_history.pop(1)

        screen_history.append(screen)
        cv2.imwrite(f"tmp/{count}_screen.png", screen)
        if time.time() - last_jump_at > required_delay:
            lower_hitbox_triggered = False
            upper_hitbox_triggered = False
            for point in danger_points:
                if point in upper_hitbox:
                    upper_hitbox_triggered = True
                if point in lower_hitbox:
                    lower_hitbox_triggered = True
                    break

            if lower_hitbox_triggered:
                game.input(Keys.SPACE)
                last_jump_at = time.time()
            elif upper_hitbox_triggered:
                pass
            elif bird_y > target_height:
                game.input(Keys.SPACE)
                last_jump_at = time.time()

        logger.debug("[__main__] Time for loop.", loop_time=time.time() - start_time, current_count=count)
        count += 1



"""
2019-04-16 15:08.59 [__main__] Time for loop.      current_count=317 loop_time=0.039034366607666016
2019-04-16 15:08.59 [_grab_screen] Exec time       runtime=0.019009828567504883
2019-04-16 15:08.59 [find_match_with_template]     runtime=0.0
2019-04-16 15:08.59 [find_match_with_template]     runtime=0.0010068416595458984
2019-04-16 15:08.59 [find_by_color]                runtime=0.0
2019-04-16 15:08.59 [find_by_color]                runtime=0.0010006427764892578
2019-04-16 15:08.59 [__main__] Time for loop.      current_count=318 loop_time=0.02901768684387207

"""