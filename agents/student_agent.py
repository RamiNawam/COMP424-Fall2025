from agents.agent import Agent
from store import register_agent
import numpy as np
import time
from helpers import execute_move, check_endgame, get_valid_moves


@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.time_limit = 1.95      # time limit per move, keeping buffer
        self.max_depth = 20         # max depth for ID search

        # Opponent behavior tracking
        self.prev_board = None
        self.opponent_is_greedy = None
        self.greedy_match_count = 0
        self.total_observed_moves = 0

 
    # Greedy evaluation used to guess opponent style

    def greedy_evaluate(self, board, color, opponent):
        # piece difference count
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count

        # corner bonus calc
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5

        # mobility-like penalty
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves

        return score_diff + corner_bonus + mobility_penalty


    # Predict what greedy agent would play

    def predict_greedy_move(self, board, color, opponent):
        moves = get_valid_moves(board, color)
        if not moves:
            return None

        best_move = None
        best_score = float("-inf")

        # try each move and evaluate greedily
        for move in moves:
            board_copy = board.copy()
            execute_move(board_copy, move, color)
            score = self.greedy_evaluate(board_copy, color, opponent)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


    # Try to guess if opponent is greedy based on past few moves

    def update_opponent_model(self, prev_board, current_board, player, opponent):
        if prev_board is None:
            return

        opp_color = opponent
        opp_moves = get_valid_moves(prev_board, opp_color)
        if not opp_moves:
            return

        # try to reconstruct which move they took
        actual_move = None
        for move in opp_moves:
            test_board = prev_board.copy()
            execute_move(test_board, move, opp_color)
            if np.array_equal(test_board, current_board):
                actual_move = move
                break

        if actual_move is None:
            return

        # compare to greedy predicted move
        greedy_move = self.predict_greedy_move(prev_board, opp_color, player)
        if greedy_move is None:
            return

        # basic ratio update
        if actual_move.get_dest() == greedy_move.get_dest():
            self.greedy_match_count += 1
        self.total_observed_moves += 1

        # decide after a few samples
        min_samples = 5
        if self.total_observed_moves >= min_samples:
            ratio = self.greedy_match_count / float(self.total_observed_moves)
            if ratio >= 0.9:
                self.opponent_is_greedy = True
            elif ratio <= 0.5:
                self.opponent_is_greedy = False




    def evaluate_board(self, board, root_player, opponent):
        n = board.shape[0]

        # count pieces
        player_count = np.count_nonzero(board == root_player)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count

        # strong corner weighting
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus_player = sum(1 for (i, j) in corners if board[i, j] == root_player)
        corner_bonus_opp = sum(1 for (i, j) in corners if board[i, j] == opponent)
        corner_term = 30 * (corner_bonus_player - corner_bonus_opp)

        # X-square penalty calc
        x_square_penalty = 0
        x_squares = set()
        for corner in corners:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    x_sq = (corner[0] + dr, corner[1] + dc)
                    if 0 <= x_sq[0] < n and 0 <= x_sq[1] < n and x_sq not in corners:
                        x_squares.add(x_sq)

        # apply penalty if sitting on vulnerable X-squares
        for x_sq in x_squares:
            if board[x_sq[0], x_sq[1]] == root_player:
                for corner in corners:
                    if abs(x_sq[0] - corner[0]) <= 1 and abs(x_sq[1] - corner[1]) <= 1:
                        if board[corner[0], corner[1]] == 0:
                            x_square_penalty -= 8
            elif board[x_sq[0], x_sq[1]] == opponent:
                for corner in corners:
                    if abs(x_sq[0] - corner[0]) <= 1 and abs(x_sq[1] - corner[1]) <= 1:
                        if board[corner[0], corner[1]] == 0:
                            x_square_penalty += 8

        # mobility calculation
        player_moves = len(get_valid_moves(board, root_player))
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_diff = player_moves - opp_moves
        mobility_term = 4.0 * mobility_diff


        if opp_moves == 0 and player_moves > 0:
            mobility_term += 40
        elif player_moves == 0 and opp_moves > 0:
            mobility_term -= 40

        # very rough stability estimate
        def quick_stability(color):
            stable = 0
            for r in range(n):
                for c in range(n):
                    if board[r, c] != color:
                        continue
                    friendly = 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < n and 0 <= cc < n and board[rr, cc] == color:
                                friendly += 1
                    stable += friendly
            return stable

        player_stability = quick_stability(root_player)
        opp_stability = quick_stability(opponent)
        stability_term = 0.6 * (player_stability - opp_stability)

        # minor edge penalties
        edge_penalty = 0
        for r in range(n):
            for c in range(n):
                if (r == 0 or r == n - 1 or c == 0 or c == n - 1) and (r, c) not in corners:
                    if board[r, c] == root_player:
                        edge_penalty -= 1
                    elif board[r, c] == opponent:
                        edge_penalty += 1

        # game phase weighting
        empty_count = np.count_nonzero(board == 0)
        total_cells = n * n
        empties_ratio = empty_count / total_cells

        # adjust weights based on phase
        if empties_ratio > 0.5:
            w_piece = 0.8
            w_corner = 1.0
            w_mobility = 1.2
            w_stability = 0.4
            w_edge = 0.3
        elif empties_ratio > 0.25:
            w_piece = 1.5
            w_corner = 1.0
            w_mobility = 1.0
            w_stability = 0.5
            w_edge = 0.2
        else:
            w_piece = 12.0
            w_corner = 0.7
            w_mobility = 0.4
            w_stability = 0.3
            w_edge = 0.1

        # combine everything
        value = (
            w_piece * score_diff
            + w_corner * corner_term
            + w_mobility * mobility_term
            + w_stability * stability_term
            + x_square_penalty
            + w_edge * edge_penalty
        )

        return value


    # Move ordering to speed up search

    def order_moves(self, board, moves, current_player, root_player, opponent):
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_set = set(corners)

        move_scores = []
        for move in moves:
            score = 0
            dest = move.get_dest()

            # high priority on corners
            if dest in corner_set:
                score += 3000

            # estimate quick captures
            captures = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = dest[0] + dr, dest[1] + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        if board[rr, cc] == opponent:
                            captures += 1

            # slight penalty for edge moves that aren't corners
            if dest not in corner_set and (dest[0] == 0 or dest[0] == n - 1 or dest[1] == 0 or dest[1] == n - 1):
                score -= 20

            score += captures * 200
            move_scores.append((score, move))

        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in move_scores]

    # Core minimax search w/ alpha-beta
    def minimax(self, board, depth, alpha, beta,
                current_player, maximizing_player,
                root_player, opponent, start_time, is_opponent_greedy):

        # basic time guard
        if time.time() - start_time > self.time_limit:
            return self.evaluate_board(board, root_player, opponent), None

        is_end, p1, p2 = check_endgame(board)
        if depth == 0 or is_end:
            # terminal eval
            if is_end:
                if root_player == 1:
                    return (p1 - p2), None
                else:
                    return (p2 - p1), None
            return self.evaluate_board(board, root_player, opponent), None

        moves = get_valid_moves(board, current_player)
        if not moves:
            # skip turn case
            next_player = opponent if current_player == root_player else root_player
            return self.minimax(board, depth - 1,
                                alpha, beta,
                                next_player, not maximizing_player,
                                root_player, opponent,
                                start_time, is_opponent_greedy)

        # maximizing our  turn
        if maximizing_player:
            best_val = float("-inf")
            best_move = None
            ordered = self.order_moves(board, moves, current_player, root_player, opponent)
            for move in ordered:
                if time.time() - start_time > self.time_limit:
                    return self.evaluate_board(board, root_player, opponent), best_move

                new_board = board.copy()
                execute_move(new_board, move, current_player)

                next_player = opponent if current_player == root_player else root_player
                val, _ = self.minimax(new_board, depth - 1,
                                      alpha, beta,
                                      next_player, False,
                                      root_player, opponent,
                                      start_time, is_opponent_greedy)

                if val > best_val:
                    best_val = val
                    best_move = move

                alpha = max(alpha, val)
                if beta <= alpha:
                    break

            return best_val, best_move

        # minimizing (opponent turn)
        else:
            # shortcut: simulate only greedy move if detected
            if is_opponent_greedy and depth > 1:
                greedy_move = self.predict_greedy_move(board, current_player, root_player)
                if greedy_move is not None:
                    new_board = board.copy()
                    execute_move(new_board, greedy_move, current_player)
                    next_player = root_player
                    val, _ = self.minimax(new_board, depth - 1,
                                          alpha, beta,
                                          next_player, True,
                                          root_player, opponent,
                                          start_time, is_opponent_greedy)
                    return val, greedy_move

            # otherwise normal minimax
            best_val = float("inf")
            best_move = None
            ordered = self.order_moves(board, moves, current_player, root_player, opponent)
            for move in ordered:
                if time.time() - start_time > self.time_limit:
                    return self.evaluate_board(board, root_player, opponent), best_move

                new_board = board.copy()
                execute_move(new_board, move, current_player)

                next_player = root_player
                val, _ = self.minimax(new_board, depth - 1,
                                      alpha, beta,
                                      next_player, True,
                                      root_player, opponent,
                                      start_time, is_opponent_greedy)

                if val < best_val:
                    best_val = val
                    best_move = move

                beta = min(beta, val)
                if beta <= alpha:
                    break

            return best_val, best_move

    # Iterative deepening 
    def iterative_deepening(self, board, player, opponent, is_opponent_greedy):
        start = time.time()
        best_move = None
        best_val = float("-inf")

        moves = get_valid_moves(board, player)
        if len(moves) == 1:
            return moves[0]

        # keep increasing depth until time nearly up
        for depth in range(1, self.max_depth + 1):
            elapsed = time.time() - start
            if elapsed > self.time_limit * 0.98:
                break

            try:
                val, move = self.minimax(board, depth,
                                         float("-inf"), float("inf"),
                                         player, True,
                                         player, opponent,
                                         start, is_opponent_greedy)
                if move is not None:
                    best_move = move
                    best_val = val

                # optional cutoff if win seems guaranteed
                if val > 1000:
                    break

                elapsed = time.time() - start
                if elapsed > self.time_limit * 0.92:
                    break

            except Exception:
                break

        return best_move

    # Main agent step each turn
    def step(self, board, player, opponent):
        start = time.time()

        # update model based on opponent last move
        if self.prev_board is not None:
            self.update_opponent_model(self.prev_board, board, player, opponent)

        # default assumption until enough samples
        is_greedy = self.opponent_is_greedy if self.opponent_is_greedy is not None else False

        moves = get_valid_moves(board, player)
        if not moves:
            return None

        best_move = self.iterative_deepening(board, player, opponent, is_greedy)
        if best_move is None:
            best_move = moves[0]

        # store board after our move
        board_copy = board.copy()
        execute_move(board_copy, best_move, player)
        self.prev_board = board_copy

        print(f"My AI turn time: {time.time() - start:.4f}s")
        return best_move
