import pygame
from collections import deque


highlighted_node = deque([*jack_nodes, *cops_nodes, *path_nodes])
curr_id = 700
highlighted_node.rotate(-curr_id+1)


pygame.init()
scale_down = 4
SCREEN_WIDTH = 1434
SCREEN_HEIGHT = 965
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
screen.fill((90, 90, 90))

over = False
while not over:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            over = True
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                highlighted_node.rotate(-1)
                print(highlighted_node[0].id)
            if event.key == pygame.K_LSHIFT:
                highlighted_node.rotate(1)
                print(highlighted_node[0].id)
    for node in reversed(all_nodes):
        node.draw(screen)
    highlighted_node[0].draw(screen, set_color=(0, 255, 0))
    for node in [*highlighted_node[0].connected_cops, *highlighted_node[0].connected_jack]:
        node.draw(screen, set_color=(255, 255, 255))
    pygame.display.flip()
