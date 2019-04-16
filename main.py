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

logger = get_logger(__name__)

bird = cv2.imread("img/bird/bird_neutral.png", cv2.IMREAD_GRAYSCALE)

if __name__ == "__main__":
    game = Game()
    game.start_game()
    game.input(Keys.SPACE)

    pipe_top = cv2.imread("img/top_pipe.png", cv2.IMREAD_GRAYSCALE)
    pipe_bottom = cv2.imread("img/bottom_pipe.png", cv2.IMREAD_GRAYSCALE)
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


        # Get all pipes that are on the screen.
        pipe_tops = ImageProcessor.find_match_with_template(grey_scale_screen, pipe_top)

        # Check for the game over button.
        restart_button_search = ImageProcessor.find_match_with_template(grey_scale_screen, restart_button)
        game_over = False if restart_button_search[0].size == 0 else True

        if game_over:
            # ded
            game.input(Keys.SPACE)
            game.input(Keys.SPACE)
            continue


        # Here are the areas that will kill us if we touch it.
        danger_points, danger_mask = ImageProcessor.find_by_color(
            cv2.blur(screen, (9, 9)),
            upper_bound=numpy.array([70, 250, 250]),
            lower_bound=numpy.array([40, 115, 115])
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
        lower_hitbox_y_range = range(int(bird_y), int(bird_y + bird_height / 2) + 15)
        lower_hitbot_x_range = range(int(bird_x - bird_width / 2), int(bird_x + bird_width / 2 + 10))
        for y in lower_hitbox_y_range:
            for x in lower_hitbot_x_range:
                lower_hitbox.append([y, x])
            if y > lowest:
                lowest = y

        upper_hitbox = []
        upper_hitbox_y_range = range(int(bird_y) - 35, int(bird_y+bird_height/2))
        upper_hitbox_x_range = range(int(bird_x-bird_width/2) - 5, int(bird_x + bird_width/2 + 10))
        for y in upper_hitbox_y_range:
            for x in upper_hitbox_x_range:
                upper_hitbox.append([y, x])

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
        # convert this to a python list cause I have legit no fucking idea whats going on.
        danger_points = danger_points.tolist()

        # so this is really dumb but by checking that the x, y values exist in the range
        # allowed by the hitbox we can confirm that the point could be a collision before
        # actually checking. This method is way faster.
        for point in danger_points:
            if point[0] in upper_hitbox_y_range and point[1] in upper_hitbox_x_range:
                if point in upper_hitbox:
                    force_no_jump = True
            elif point[0] in lower_hitbox_y_range and point[1] in lower_hitbot_x_range:
                if point in lower_hitbox:
                    should_jump = True

        floor = 70
        w, h = pipe_top.shape[::-1]
        for (x, y) in zip(pipe_tops[1], pipe_tops[0]):
            floor = y
            break

        target_height = floor + 70
        screen = ImageProcessor.draw_line_on_image(screen, (50, target_height), (50+15, target_height))
        #floor -= 75
        #print(f"Setting floor at {target_height}")

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
        required_delay = .2
        if should_jump or bird_y > target_height:
            cv2.imwrite(f"tmp/{count}_screen.png", screen)
            cv2.imwrite(f"tmp/{count}_screen_mask.png", danger_mask)
            if force_no_jump:
                logger.debug("[__main__] Not jumping due to obstacle above.")
            elif time.time() - last_jump_at < required_delay:
                logger.debug("[__main__] Not jumping due to time check.", last_jump_at=last_jump_at, current_wait_time=time.time()-last_jump_at, required_delay=required_delay)
            else:
                logger.debug("[__main__] Jumping")
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