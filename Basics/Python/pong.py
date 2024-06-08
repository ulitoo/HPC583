import pygame
import sys

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
BALL_RADIUS = 15
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PADDLE_SPEED = 6
BALL_SPEED_X, BALL_SPEED_Y = 4, 4
FONT_SIZE = 36

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Pong')

# Font setup
font = pygame.font.Font(None, FONT_SIZE)

# Paddle class
class Paddle:
    def __init__(self, x):
        self.rect = pygame.Rect(x, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
    
    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect)
    
    def move(self, y):
        if self.rect.top + y >= 0 and self.rect.bottom + y <= HEIGHT:
            self.rect.y += y

# Ball class
class Ball:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH // 2 - BALL_RADIUS // 2, HEIGHT // 2 - BALL_RADIUS // 2, BALL_RADIUS, BALL_RADIUS)
        self.speed_x = BALL_SPEED_X
        self.speed_y = BALL_SPEED_Y
    
    def draw(self):
        pygame.draw.ellipse(screen, WHITE, self.rect)
    
    def move(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.top <= 0 or self.rect.bottom >= HEIGHT:
            self.speed_y = -self.speed_y
        
        if self.rect.left <= 0 or self.rect.right >= WIDTH:
            self.reset_position()

    def reset_position(self):
        self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.speed_x = -self.speed_x

# Main game function
def main():
    clock = pygame.time.Clock()
    player1 = Paddle(30)
    player2 = Paddle(WIDTH - 40)
    ball = Ball()
    score1, score2 = 0, 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            player1.move(-PADDLE_SPEED)
        if keys[pygame.K_s]:
            player1.move(PADDLE_SPEED)
        if keys[pygame.K_UP]:
            player2.move(-PADDLE_SPEED)
        if keys[pygame.K_DOWN]:
            player2.move(PADDLE_SPEED)

        ball.move()

        if ball.rect.colliderect(player1.rect) or ball.rect.colliderect(player2.rect):
            ball.speed_x = -ball.speed_x
        
        if ball.rect.left <= 0:
            score2 += 1
            ball.reset_position()

        if ball.rect.right >= WIDTH:
            score1 += 1
            ball.reset_position()

        screen.fill(BLACK)
        player1.draw()
        player2.draw()
        ball.draw()

        # Draw scores
        score_text = font.render(f"{score1} - {score2}", True, WHITE)
        screen.blit(score_text, (WIDTH // 2 - score_text.get_width() // 2, 20))

        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
