import pygame
import sys
import numpy as np
from PIL import Image
import os
import re

pygame.init()

WINDOW_SIZE = 280
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BRUSH_RADIUS = 8

SAVE_DIR = "painted_numbers"
os.makedirs(SAVE_DIR, exist_ok=True)

screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 40)) 
pygame.display.set_caption("Draw your number")
screen.fill(BLACK)

font = pygame.font.SysFont(None, 36)
input_active = False
user_input = ""

# Find the next available filename for a given digit label
def get_next_filename(digit_label):
    files = os.listdir(SAVE_DIR)
    pattern = re.compile(rf"{digit_label}_(\d+)\.png")
    numbers = [int(match.group(1)) for f in files if (match := pattern.match(f))]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(SAVE_DIR, f"{digit_label}_{next_number}.png")

# Save the drawn image to a file
def save_image(digit):
    data = pygame.surfarray.array3d(screen.subsurface((0, 0, WINDOW_SIZE, WINDOW_SIZE)))
    image = np.transpose(data, (1, 0, 2))
    image = Image.fromarray(image)
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image = Image.eval(image, lambda x: 255 - x) # Invert colors if necessary
    filename = get_next_filename(digit)
    image.save(filename)
    print(f"Picture Saved: {filename}")

# Draw the text input box at the bottom
def draw_input_box():
    pygame.draw.rect(screen, BLACK, (0, WINDOW_SIZE, WINDOW_SIZE, 40))
    txt_surface = font.render(f"Enter digit (0-9): {user_input}", True, WHITE)
    screen.blit(txt_surface, (10, WINDOW_SIZE + 5))

while True:
    screen.fill(BLACK, (0, WINDOW_SIZE, WINDOW_SIZE, 40))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if pygame.mouse.get_pressed()[0] and not input_active:
            pos = pygame.mouse.get_pos()
            if pos[1] < WINDOW_SIZE:
                pygame.draw.circle(screen, WHITE, pos, BRUSH_RADIUS)

        if event.type == pygame.KEYDOWN:
            if input_active:
                if event.key == pygame.K_RETURN:
                    if user_input.isdigit() and 0 <= int(user_input) <= 9:
                        save_image(user_input)
                    else:
                        print("Invalid input. Please enter a digit from 0 to 9.")
                    input_active = False
                    user_input = ""
                    screen.fill(BLACK)
                elif event.key == pygame.K_BACKSPACE:
                    user_input = user_input[:-1]
                elif event.unicode.isdigit() and len(user_input) < 1:
                    user_input += event.unicode
            else:
                if event.key == pygame.K_s: # 's' to save
                    input_active = True
                    user_input = ""
                elif event.key == pygame.K_c: # 'c' to clear
                    screen.fill(BLACK)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    if input_active:
        draw_input_box()

    pygame.display.update()