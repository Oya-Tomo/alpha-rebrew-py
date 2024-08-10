import sys
import argparse
import random
import numpy as np
import torch
import pygame
from pygame.locals import *

from bitboard import Board, Stone, flip
from config import MCTSConfig
from mcts import MCT
from models import PVNet

parser = argparse.ArgumentParser(description="Othello AI with AlphaZero")
parser.add_argument("--sim", type=int, default=800, help="number of mcts simulation")
parser.add_argument(
    "-da", "--dirichlet_alpha", type=float, default=10.0, help="dirichlet noise alpha"
)
parser.add_argument(
    "-df", "--dirichlet_frac", type=float, default=0.01, help="dirichlet noise fraction"
)
args = parser.parse_args()


class Predictor:
    def __init__(self):
        # estimator
        checkpoint = torch.load("checkpoint/model_75.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PVNet().to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()
        self.config = MCTSConfig(
            simulation=args.sim,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_frac=args.dirichlet_frac,
            c_base=19652,
            c_init=1.25,
        )
        self.mct = MCT(
            self.model,
            self.config,
        )

    def predict(self, board: Board, color: Stone) -> int:
        actions = board.get_actions(color)
        policy = self.mct.search(
            board,
            color,
        )
        policy = np.array(policy)
        while True:
            action = random.choice(np.where(policy == policy.max())[0])
            if action in actions:
                break
            policy[action] = 0
        return int(action)


pygame.init()

FONT_NAME = pygame.font.get_default_font()
FONTS = {
    20: pygame.font.Font(FONT_NAME, 20),
    30: pygame.font.Font(FONT_NAME, 30),
    40: pygame.font.Font(FONT_NAME, 40),
    50: pygame.font.Font(FONT_NAME, 50),
    60: pygame.font.Font(FONT_NAME, 60),
    70: pygame.font.Font(FONT_NAME, 70),
    80: pygame.font.Font(FONT_NAME, 80),
    90: pygame.font.Font(FONT_NAME, 90),
}

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (10, 150, 10)
GREEN_SHADOW = (0, 50, 0)
BLUE = (0, 0, 150)
YELLOW = (255, 200, 0)


class OthelloGame:
    def __init__(self) -> None:
        # estimator
        self.predictor = Predictor()

        self.board = Board()
        self.player = Stone.BLACK
        self.turn = Stone.BLACK

        # bind objects
        self.cursor = None
        self.click = False

        # scenes
        self.screen = pygame.display.set_mode((1000, 850))
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.scenes = [
            self.splash_screen,
            self.config_screen,
            self.ready_screen,
            self.render_game,
            self.render_result,
        ]

    def bind_exit(self, event: pygame.event.Event) -> None:
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                pygame.quit()
                sys.exit()

    def bind_click(self, event: pygame.event.Event) -> None:
        self.cursor = pygame.mouse.get_pos()
        if event.type == MOUSEBUTTONDOWN:
            self.click = True
        else:
            self.click = False

    def run(self) -> None:
        self.scenes[0]()
        while True:
            for i in range(1, len(self.scenes)):
                self.scenes[i]()

    # utils functions
    def time_to_frame(self, sec: float) -> int:
        return int(sec * self.fps)

    def get_anchor_pos(self, x: float, y: float) -> tuple[int, int]:
        width = self.screen.get_width()
        height = self.screen.get_height()
        return int(width * x), int(height * y)

    def splash_screen(self) -> None:
        title = FONTS[80].render("Othello AI", True, (255, 255, 255))
        title_rect = title.get_rect()
        title_rect.center = self.get_anchor_pos(0.5, 0.4)

        for frame in range(self.time_to_frame(3)):
            if frame < self.time_to_frame(1):
                title.set_alpha(frame * 255 // self.time_to_frame(1))
            elif frame < self.time_to_frame(2):
                title.set_alpha(255)
            else:
                title.set_alpha(
                    255 - (frame - self.time_to_frame(2)) * 255 // self.time_to_frame(1)
                )

            self.screen.fill(BLACK)
            self.screen.blit(
                title,
                title_rect,
            )

            pygame.display.update()
            self.clock.tick(self.fps)

            for event in pygame.event.get():
                self.bind_exit(event)
                self.bind_click(event)

    def config_screen(self) -> None:
        select_text = FONTS[60].render("Select your stone", True, WHITE)
        select_rect = select_text.get_rect()
        select_rect.center = self.get_anchor_pos(0.5, 0.15)

        quit_text = FONTS[40].render("Quit : [ESC]", True, YELLOW)
        quit_rect = quit_text.get_rect()
        quit_rect.center = self.get_anchor_pos(0.5, 0.85)

        black_stone_rect = pygame.Rect(0, 0, 350, 400)
        black_stone_rect.center = self.get_anchor_pos(0.25, 0.5)

        black_stone = pygame.Rect(0, 0, 200, 200)
        black_stone.center = self.get_anchor_pos(0.25, 0.5)

        black_stone_shadow = pygame.Rect(0, 0, 200, 200)
        black_stone_shadow.center = self.get_anchor_pos(0.26, 0.51)

        white_stone_rect = pygame.Rect(0, 0, 350, 400)
        white_stone_rect.center = self.get_anchor_pos(0.75, 0.5)

        white_stone = pygame.Rect(0, 0, 200, 200)
        white_stone.center = self.get_anchor_pos(0.75, 0.5)

        white_stone_shadow = pygame.Rect(0, 0, 200, 200)
        white_stone_shadow.center = self.get_anchor_pos(0.76, 0.51)

        while True:
            self.render_background()

            self.screen.blit(select_text, select_rect)
            self.screen.blit(quit_text, quit_rect)

            pygame.draw.rect(self.screen, GREEN, black_stone_rect, border_radius=20)
            pygame.draw.rect(self.screen, GREEN, white_stone_rect, border_radius=20)

            pygame.draw.ellipse(self.screen, GREEN_SHADOW, black_stone_shadow)
            pygame.draw.ellipse(self.screen, GREEN_SHADOW, white_stone_shadow)

            pygame.draw.ellipse(self.screen, BLACK, black_stone)
            pygame.draw.ellipse(self.screen, WHITE, white_stone)

            pygame.display.update()
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                self.bind_exit(event)
                self.bind_click(event)

            if self.click:
                if black_stone.collidepoint(self.cursor):
                    self.player = Stone.BLACK
                    break
                elif white_stone.collidepoint(self.cursor):
                    self.player = Stone.WHITE
                    break
            else:
                if black_stone.collidepoint(self.cursor):
                    black_stone_shadow.center = self.get_anchor_pos(0.255, 0.505)
                elif white_stone.collidepoint(self.cursor):
                    white_stone_shadow.center = self.get_anchor_pos(0.755, 0.505)
                else:
                    black_stone_shadow.center = self.get_anchor_pos(0.26, 0.51)
                    white_stone_shadow.center = self.get_anchor_pos(0.76, 0.51)

    def ready_screen(self) -> None:
        self.board = Board()
        for frame in range(self.time_to_frame(5)):
            if frame < self.time_to_frame(2):
                text = FONTS[60].render("Ready?", True, YELLOW)
            elif frame < self.time_to_frame(3):
                text = FONTS[60].render("3", True, YELLOW)
            elif frame < self.time_to_frame(4):
                text = FONTS[60].render("2", True, YELLOW)
            else:
                text = FONTS[60].render("1", True, YELLOW)
            text_rect = text.get_rect()
            text_rect.center = self.get_anchor_pos(0.5, 0.5)

            self.render_background()
            self.render_board()

            self.screen.blit(
                text,
                text_rect,
            )
            pygame.display.update()
            self.clock.tick(self.fps)

            for event in pygame.event.get():
                self.bind_exit(event)
                self.bind_click(event)

    def render_game(self) -> None:
        while not self.board.is_over():
            if self.turn == self.player:
                self.render_player_turn()
            else:
                self.render_ai_turn()

    def render_player_turn(self) -> None:
        actions = self.board.get_actions(self.turn)

        if actions == [64]:
            self.turn = flip(self.turn)
            return

        while True:
            self.render_background()
            pos = self.render_board()

            pygame.display.update()
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                self.bind_exit(event)
                self.bind_click(event)

            if self.click and pos is not None:
                x = pos[0]
                y = pos[1]
                action = y * 8 + x
                if action in actions:
                    self.board.act(self.turn, action)
                    self.turn = flip(self.turn)
                    break

    def render_ai_turn(self) -> None:
        self.render_background()
        self.render_board()
        pygame.display.update()

        action = self.predictor.predict(self.board, self.turn)
        self.board.act(self.turn, action)
        self.turn = flip(self.turn)

    def render_result(self) -> None:
        black, white, empty = self.board.get_count()
        if black > white:
            text = FONTS[60].render("Black Win!", True, YELLOW)
        elif black < white:
            text = FONTS[60].render("White Win!", True, YELLOW)
        else:
            text = FONTS[60].render("Draw!", True, YELLOW)

        text_rect = text.get_rect()
        text_rect.center = self.get_anchor_pos(0.5, 0.45)

        count_text = FONTS[40].render(
            f"Black: {black}, White: {white}, Empty: {empty}", True, YELLOW
        )
        count_rect = count_text.get_rect()
        count_rect.center = self.get_anchor_pos(0.5, 0.55)

        next_text = FONTS[40].render("Next Game : [SPACE]", True, YELLOW)
        next_rect = next_text.get_rect()
        next_rect.center = self.get_anchor_pos(0.5, 0.7)

        while True:
            self.render_background()
            self.render_board()

            self.screen.blit(text, text_rect)
            self.screen.blit(count_text, count_rect)
            self.screen.blit(next_text, next_rect)

            pygame.display.update()
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                self.bind_exit(event)
                self.bind_click(event)
                if event.type == KEYDOWN and event.key == K_SPACE:
                    return

    def render_background(self):
        box = 30
        width = self.screen.get_width()
        height = self.screen.get_height()
        x_count = width // box + 1
        y_count = height // box + 1

        self.screen.fill(BLACK)
        for y in range(y_count):
            for x in range(x_count):
                if (x + y) % 2 == 0:
                    pygame.draw.rect(
                        self.screen,
                        BLUE,
                        (x * box, y * box, box, box),
                    )

    def render_board(self) -> tuple[int, int] | None:
        board = self.render_board_surface()
        board_rect = board.get_rect()
        board_rect.center = self.get_anchor_pos(0.5, 0.5)
        self.screen.blit(board, board_rect)

        if board_rect.collidepoint(self.cursor):
            x, y = self.cursor
            x = (x - board_rect.left) // 100
            y = (y - board_rect.top) // 100
            return x, y
        return None

    def render_board_surface(self) -> pygame.Surface:
        box_size = 100
        stone_size = 80

        surface = pygame.Surface((box_size * 8, box_size * 8))
        surface.fill(GREEN)

        board = self.board.get_board()

        for y in range(0, 8):
            for x in range(0, 8):
                pygame.draw.rect(
                    surface,
                    BLACK,
                    (x * box_size, y * box_size, box_size, box_size),
                    width=2,
                )
                stone_pos = (
                    x * box_size + 10,
                    y * box_size + 10,
                    stone_size,
                    stone_size,
                )
                shodow_pos = (
                    x * box_size + 14,
                    y * box_size + 14,
                    stone_size,
                    stone_size,
                )
                if board[y][x] == Stone.BLACK:
                    pygame.draw.ellipse(surface, GREEN_SHADOW, shodow_pos)
                    pygame.draw.ellipse(surface, BLACK, stone_pos)
                elif board[y][x] == Stone.WHITE:
                    pygame.draw.ellipse(surface, GREEN_SHADOW, shodow_pos)
                    pygame.draw.ellipse(surface, WHITE, stone_pos)

        return surface


if __name__ == "__main__":
    game = OthelloGame()
    game.run()
