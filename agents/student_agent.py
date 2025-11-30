# Student agent: Ataxx iterative deepening minimax with alpha-beta pruning.
from agents.agent import Agent
from store import register_agent

import numpy as np
import time

from helpers import (
    check_endgame,
    get_valid_moves,
    MoveCoordinates,
    count_disc_count_change,
    get_directions,
)


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Tournament-ready Ataxx agent using iterative deepening alpha-beta search
    with dynamic heuristics and in-place move/undo to maximize depth under
    the COMP424 time constraints.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.time_limit = 1.95  # stay safely below the 2s tournament cap
        self.max_depth_reached = 0

        # Geometry caches, initialized lazily per board size
        self.edges = None
        self.corners = None
        self.x_squares = None
        self.c_squares = None

        # Direction cache for frontier calculations
        self.frontier_dirs = get_directions()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def step(self, chess_board, player, opponent):
        self._ensure_geometry(chess_board.shape[0])
        self.max_depth_reached = 0

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None

        start_time = time.time()
        best_move = valid_moves[0]
        depth = 1
        board_copy = chess_board.copy()

        while True:
            if time.time() - start_time >= self.time_limit:
                break
            try:
                value, move = self._alpha_beta_root(
                    board_copy,
                    depth,
                    float("-inf"),
                    float("inf"),
                    player,
                    opponent,
                    start_time,
                )
                if move is not None:
                    best_move = move
            except TimeoutError:
                break
            depth += 1

        return best_move

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _ensure_geometry(self, size):
        if self.edges is not None:
            return

        edges = set()
        for i in range(size):
            edges.add((0, i))
            edges.add((size - 1, i))
            edges.add((i, 0))
            edges.add((i, size - 1))
        self.edges = edges

        self.corners = {
            (0, 0),
            (0, size - 1),
            (size - 1, 0),
            (size - 1, size - 1),
        }

        self.x_squares = {
            (1, 1),
            (1, size - 2),
            (size - 2, 1),
            (size - 2, size - 2),
        }

        self.c_squares = {
            (0, 1),
            (1, 0),
            (0, size - 2),
            (1, size - 1),
            (size - 1, 1),
            (size - 2, 0),
            (size - 1, size - 2),
            (size - 2, size - 1),
        }

    # ------------------------------------------------------------------
    # Alpha-beta core
    # ------------------------------------------------------------------
    def _alpha_beta_root(
        self,
        board,
        depth,
        alpha,
        beta,
        player,
        opponent,
        start_time,
    ):
        if time.time() - start_time >= self.time_limit:
            raise TimeoutError

        best_value = float("-inf")
        best_move = None

        moves = get_valid_moves(board, player)
        if not moves:
            score = self._evaluate(board, player, opponent)
            return score, None

        ordered = self._order_moves(moves, board, player)

        for move in ordered:
            if time.time() - start_time >= self.time_limit:
                raise TimeoutError

            changes = self._execute_move_inplace(board, move, player)

            if depth > self.max_depth_reached:
                self.max_depth_reached = depth

            value = self._minimax(
                board,
                depth - 1,
                alpha,
                beta,
                maximizing_player=False,
                root_player=player,
                opponent=opponent,
                start_time=start_time,
            )

            self._undo_move(board, changes)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_value, best_move

    def _minimax(
        self,
        board,
        depth,
        alpha,
        beta,
        maximizing_player,
        root_player,
        opponent,
        start_time,
    ):
        if time.time() - start_time >= self.time_limit:
            raise TimeoutError

        if depth > self.max_depth_reached:
            self.max_depth_reached = depth

        is_end, p1, p2 = check_endgame(board)
        if is_end:
            return p1 - p2 if root_player == 1 else p2 - p1

        if depth == 0:
            return self._evaluate(board, root_player, opponent)

        current_player = root_player if maximizing_player else opponent
        moves = get_valid_moves(board, current_player)

        if not moves:
            return self._minimax(
                board,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                root_player,
                opponent,
                start_time,
            )

        ordered = self._order_moves(moves, board, current_player)

        if maximizing_player:
            value = float("-inf")
            for move in ordered:
                if time.time() - start_time >= self.time_limit:
                    raise TimeoutError
                changes = self._execute_move_inplace(board, move, current_player)
                value = max(
                    value,
                    self._minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        root_player,
                        opponent,
                        start_time,
                    ),
                )
                self._undo_move(board, changes)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value

        else:
            value = float("inf")
            for move in ordered:
                if time.time() - start_time >= self.time_limit:
                    raise TimeoutError
                changes = self._execute_move_inplace(board, move, current_player)
                value = min(
                    value,
                    self._minimax(
                        board,
                        depth - 1,
                        alpha,
                        beta,
                        True,
                        root_player,
                        opponent,
                        start_time,
                    ),
                )
                self._undo_move(board, changes)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    # ------------------------------------------------------------------
    # Move ordering & execution helpers
    # ------------------------------------------------------------------
    def _order_moves(self, moves, board, player):
        opponent = 1 if player == 2 else 2
        center = board.shape[0] // 2

        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        mobility_diff = player_moves - opponent_moves

        def score(move_coords: MoveCoordinates):
            dest = move_coords.get_dest()
            if dest in self.corners:
                return 10000
            if dest in self.x_squares:
                return -10000
            if dest in self.c_squares:
                return -5000

            row_bias = dest[0] if player == 1 else (board.shape[0] - 1 - dest[0])
            center_bonus = -(
                abs(dest[0] - center) + abs(dest[1] - center)
            )
            edge_bonus = 10 if dest in self.edges else 0
            return mobility_diff * 10 + row_bias * 5 + center_bonus + edge_bonus

        return sorted(moves, key=score, reverse=True)

    def _execute_move_inplace(self, board, move_coords: MoveCoordinates, player: int):
        changes = []
        opponent = 1 if player == 2 else 2
        size = board.shape[0]

        r_dest, c_dest = move_coords.get_dest()
        r_src, c_src = move_coords.get_src()

        changes.append((r_dest, c_dest, board[r_dest, c_dest]))
        board[r_dest, c_dest] = player

        for dr, dc in get_directions():
            nr, nc = r_dest + dr, c_dest + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == opponent:
                changes.append((nr, nc, board[nr, nc]))
                board[nr, nc] = player

        if abs(r_dest - r_src) == 2 or abs(c_dest - c_src) == 2:
            changes.append((r_src, c_src, board[r_src, c_src]))
            board[r_src, c_src] = 0

        return changes

    def _undo_move(self, board, changes):
        for r, c, old_val in reversed(changes):
            board[r, c] = old_val

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, board, player, opponent):
        size = board.shape[0]
        total_cells = size * size
        filled = np.count_nonzero(board)
        empty = total_cells - filled
        progress = filled / total_cells

        if progress < 0.30:
            DISC_DIFF_WEIGHT = 0
            MOBILITY_WEIGHT = 100
            CORNER_WEIGHT = 75
            PARITY_WEIGHT = 0
            CAPTURE_WEIGHT = 0
            STABILITY_WEIGHT = 7
        elif progress < 0.66:
            DISC_DIFF_WEIGHT = 7
            MOBILITY_WEIGHT = 180
            CORNER_WEIGHT = 250
            PARITY_WEIGHT = 0
            CAPTURE_WEIGHT = 5
            STABILITY_WEIGHT = 100
        elif self.max_depth_reached >= empty:
            DISC_DIFF_WEIGHT = 100
            MOBILITY_WEIGHT = 0
            CORNER_WEIGHT = 0
            PARITY_WEIGHT = 0
            CAPTURE_WEIGHT = 0
            STABILITY_WEIGHT = 0
        else:
            DISC_DIFF_WEIGHT = 15
            MOBILITY_WEIGHT = 100
            CORNER_WEIGHT = 120
            PARITY_WEIGHT = 3
            CAPTURE_WEIGHT = 25
            STABILITY_WEIGHT = 35

        player_discs = np.count_nonzero(board == player)
        opponent_discs = np.count_nonzero(board == opponent)
        disc_diff_score = DISC_DIFF_WEIGHT * (
            player_discs - opponent_discs
        ) / (player_discs + opponent_discs + 1)

        player_moves = get_valid_moves(board, player)
        opponent_moves = get_valid_moves(board, opponent)
        mobility_score = MOBILITY_WEIGHT * (
            len(player_moves) - len(opponent_moves)
        ) / (len(player_moves) + len(opponent_moves) + 1)

        player_corners = sum(1 for pos in self.corners if board[pos] == player)
        opponent_corners = sum(1 for pos in self.corners if board[pos] == opponent)
        corner_score = CORNER_WEIGHT * (player_corners - opponent_corners)

        player_x = sum(1 for pos in self.x_squares if board[pos] == player)
        opponent_x = sum(1 for pos in self.x_squares if board[pos] == opponent)
        x_penalty = -20 * player_x + 7 * opponent_x

        player_c = sum(1 for pos in self.c_squares if board[pos] == player)
        opponent_c = sum(1 for pos in self.c_squares if board[pos] == opponent)
        c_penalty = -10 * player_c + 4 * opponent_c

        parity_score = PARITY_WEIGHT * (1 if empty % 2 == 1 else -1)

        player_captures = sum(
            count_disc_count_change(board, mv, player) for mv in player_moves
        )
        opponent_captures = sum(
            count_disc_count_change(board, mv, opponent) for mv in opponent_moves
        )
        capture_score = CAPTURE_WEIGHT * (
            player_captures - opponent_captures
        ) / (player_captures + opponent_captures + 1)

        player_stable = sum(1 for pos in self.edges if board[pos] == player)
        opponent_stable = sum(1 for pos in self.edges if board[pos] == opponent)
        stability_score = STABILITY_WEIGHT * (player_stable - opponent_stable)

        frontier_score = self._frontier_delta(board, player, opponent)

        return (
            disc_diff_score
            + mobility_score
            + corner_score
            + x_penalty
            + c_penalty
            + parity_score
            + capture_score
            + stability_score
            + frontier_score
        )

    def _frontier_delta(self, board, player, opponent):
        return self._count_frontier(board, opponent) - self._count_frontier(board, player)

    def _count_frontier(self, board, player):
        positions = np.argwhere(board == player)
        frontier = 0
        size = board.shape[0]
        for r, c in positions:
            for dr, dc in self.frontier_dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                    frontier += 1
                    break
        return frontier

