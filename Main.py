'''
Created on Feb. 28, 2022

@author: Phil Snelgrove
'''
import pygame as pg
import os

VALUE_LIMIT = 1000000
ITERATION_LIMIT = 10
main_dir = os.path.split(os.path.abspath(__file__))[0]

def fibonacci(count, old, new, seq):
    newer = old + new
    if newer < VALUE_LIMIT:
        count += 1
        seq.append(newer)
        fibonacci(count, new, newer, seq)
    else:
        seq.append(f"(The next item would be {newer})")
        seq.append(f"{count} steps total.")
    return seq

def blit_text(surface, text, pos, font, color=pg.Color('black')):
    words = [word.split(' ') for word in text.splitlines()]  # 2D array where each row is a list of words.
    space = font.size(' ')[0]  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in words:
        for word in line:
            word_surface = font.render(word, 0, color)
            word_width, word_height = word_surface.get_size()
            if x + word_width >= max_width:
                x = pos[0]  # Reset the x.
                y += word_height  # Start on new row.
            surface.blit(word_surface, (x, y))
            x += word_width + space
        x = pos[0]  # Reset the x.
        y += word_height  # Start on new row.

def main():
    # Initialize pygame
    if pg.get_sdl_version()[0] == 2:
        pg.mixer.pre_init(44100, 32, 2, 1024)
    pg.init()
    white = (255, 255, 255)
    X = 1024
    Y = 764
    screen = pg.display.set_mode((X, Y), pg.RESIZABLE)
    # print(f"Fibonacci sequence up to {VALUE_LIMIT}:", ', '.join(str(e) for e in fibonacci(1, 0, 1, [1])))
    pg.display.set_caption(f"Fibonacci sequence up to {VALUE_LIMIT}:")
    font = pg.font.Font('freesansbold.ttf', 32)
    text = ', '.join(str(e) for e in fibonacci(1, 0, 1, [1]))
    while True:
        screen.fill(white)
        blit_text(screen, text, (20, 20), font)
        # get input
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return
        pg.display.update()

if __name__ == '__main__':
    main()
    pg.quit()