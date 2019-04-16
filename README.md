## About
A fun learning project using numpy and opencv to automate the web based flappy bird game, [found here](https://flappybird.io/).
This is a very raw project and should not be used for any other purpose than learning. There is also a non-zero chance that it may not work on your computer, due to window positions not lining up, etc.

## How it works
There are a few pieces of required data that we need to make this work.
1. The position of the bird.
2. The position of all the obstacles(pipes).

## Important
You must install opencv on your system outside of the the pipenv.
```
pip3 install opencv-contrib-python --upgrade
```

geckodriver should also be in your path, you can download it from [here](https://github.com/mozilla/geckodriver/releases) and drop it into the project root if you wish.