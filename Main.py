'''
Created on Feb. 28, 2022
GNU License

@author: Phil Snelgrove
'''
import pygame as pg
import os
import Display.Screen as ds

VALUE_LIMIT = 10000000
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

def main():
    # Initialize pygame
    if pg.get_sdl_version()[0] == 2:
        pg.mixer.pre_init(44100, 32, 2, 1024)
    pg.init()
    X = 1024
    Y = 768
    screen = pg.display.set_mode((X, Y), pg.RESIZABLE)
    # print(f"Fibonacci sequence up to {VALUE_LIMIT}:", ', '.join(str(e) for e in fibonacci(1, 0, 1, [1])))
    pg.display.set_caption(f"Fibonacci sequence up to {VALUE_LIMIT}:")
    font = pg.font.Font('C:\\WINDOWS\\Fonts\\cour.ttf', 32)
    text = ', '.join(str(e) for e in fibonacci(1, 0, 1, [1]))
    while True:
        screen.fill(pg.Color('black'))
        ds.blit_text(screen, text, (20, 20), font)
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