from agents.agent import Agent
from store import register_agent
import numpy as np
import time
from copy import deepcopy
from helpers import get_valid_moves, execute_move, check_endgame


@register_agent("alpha_beta_test")
class StudentAgent(Agent):
    """
    Pure alpha-beta search with evaluation = disc difference.
    Added only for experimentation.
    """

    def __init__(self):
        super().__init__()
        self.name = "alpha_beta_test"
        self.time_limit = 2.0
        self.max_depth = 6

    def evaluate_board(self, chess_board, player, opponent):
        player_pieces = np.sum(chess_board == player)
        opponent_pieces = np.sum(chess_board == opponent)
        return player_pieces - opponent_pieces

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        best_move = None
        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None

        depth = 1
        while True:
            if time.time() - start_time >= self.time_limit:
                break
            value, move = self.minimax(
                chess_board,
                depth,
                float("-inf"),
                float("inf"),
                True,
                player,
                opponent,
                start_time,
            )
            if move is not None:
                best_move = move
            depth += 1
            if depth > self.max_depth:
                break
        return best_move or valid_moves[0]

    def minimax(
        self,
        board,
        depth,
        alpha,
        beta,
        maximizing,
        player,
        opponent,
        start_time,
    ):
        if time.time() - start_time >= self.time_limit:
            return self.evaluate_board(board, player, opponent), None

        is_end, p1, p2 = check_endgame(board)
        if depth == 0 or is_end:
            if is_end:
                if player == 1:
                    score = p1 - p2
                else:
                    score = p2 - p1
                return score, None
            return self.evaluate_board(board, player, opponent), None

        current = player if maximizing else opponent
        valid_moves = get_valid_moves(board, current)
        if not valid_moves:
            return self.minimax(
                board,
                depth - 1,
                alpha,
                beta,
                not maximizing,
                player,
                opponent,
                start_time,
            )

        best_move = None
        ordered_moves = self.order_moves(board, valid_moves, current, opponent)

        if maximizing:
            value = float("-inf")
            for move in ordered_moves:
                next_board = deepcopy(board)
                execute_move(next_board, move, current)
                eval_score, _ = self.minimax(
                    next_board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    player,
                    opponent,
                    start_time,
                )
                if eval_score > value:
                    value = eval_score
                    best_move = move
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value, best_move
        else:
            value = float("inf")
            for move in ordered_moves:
                next_board = deepcopy(board)
                execute_move(next_board, move, current)
                eval_score, _ = self.minimax(
                    next_board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    player,
                    opponent,
                    start_time,
                )
                if eval_score < value:
                    value = eval_score
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    def order_moves(self, board, moves, player, opponent):
        scored = []
        for move in moves:
            next_board = deepcopy(board)
            execute_move(next_board, move, player)
            score = self.evaluate_board(next_board, player, opponent)
            scored.append((score, move))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

