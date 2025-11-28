# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import execute_move, check_endgame, get_valid_moves, MoveCoordinates, count_disc_count_change

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super().__init__()
        self.name = "StudentAgent"
        self.time_limit = 1.95  # Leave small buffer
        self.max_depth = 20  # Increased for deeper search

    # ----------------------------------------------------------------------
    # GREEDY AGENT EVALUATION (for opponent modeling)
    # ----------------------------------------------------------------------
    def greedy_evaluate(self, board, color, opponent):
        """
        Replicate the greedy agent's evaluation function exactly.
        Used for opponent modeling - predicting what greedy will do.
        """
        player_count = np.count_nonzero(board == color)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count
        
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus = sum(1 for (i, j) in corners if board[i, j] == color) * 5
        
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_penalty = -opp_moves
        
        return score_diff + corner_bonus + mobility_penalty

    # ----------------------------------------------------------------------
    # PREDICT GREEDY MOVE (opponent modeling)
    # ----------------------------------------------------------------------
    def predict_greedy_move(self, board, color, opponent):
        """
        Predict what the greedy agent would do - pick move that maximizes greedy evaluation.
        This is the KEY to beating greedy - we model its behavior, not optimal play.
        """
        moves = get_valid_moves(board, color)
        if not moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in moves:
            board_copy = deepcopy(board)
            execute_move(board_copy, move, color)
            score = self.greedy_evaluate(board_copy, color, opponent)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

    # ----------------------------------------------------------------------
    # LIGHTWEIGHT EVALUATION (fast for deeper search)
    # ----------------------------------------------------------------------
    def evaluate_board(self, board, root_player, opponent):
        """
        Lightweight but comprehensive evaluation:
        - Piece difference (phase-weighted)
        - Corner control (critical)
        - Mobility (heavily weighted)
        - X-square penalties (avoid giving corners)
        - Stability (pieces with friendly neighbors)
        - Edge penalties (edges are vulnerable)
        """
        n = board.shape[0]

        # --- basic counts (fast) ---
        player_count = np.count_nonzero(board == root_player)
        opp_count = np.count_nonzero(board == opponent)
        score_diff = player_count - opp_count

        # --- corners: CRITICAL ---
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_bonus_player = sum(1 for (i, j) in corners if board[i, j] == root_player)
        corner_bonus_opp = sum(1 for (i, j) in corners if board[i, j] == opponent)
        corner_term = 30 * (corner_bonus_player - corner_bonus_opp)

        # --- X-squares: penalize giving opponent corner access ---
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
        
        for x_sq in x_squares:
            if board[x_sq[0], x_sq[1]] == root_player:
                # We're on X-square - check if we're blocking our own corner
                for corner in corners:
                    if abs(x_sq[0] - corner[0]) <= 1 and abs(x_sq[1] - corner[1]) <= 1:
                        if board[corner[0], corner[1]] == 0:
                            x_square_penalty -= 8  # Bad: blocking our corner
            elif board[x_sq[0], x_sq[1]] == opponent:
                # Opponent on X-square - check if they're blocking their corner
                for corner in corners:
                    if abs(x_sq[0] - corner[0]) <= 1 and abs(x_sq[1] - corner[1]) <= 1:
                        if board[corner[0], corner[1]] == 0:
                            x_square_penalty += 8  # Good: opponent blocking their corner

        # --- mobility: HEAVILY weighted ---
        player_moves = len(get_valid_moves(board, root_player))
        opp_moves = len(get_valid_moves(board, opponent))
        mobility_diff = player_moves - opp_moves
        mobility_term = 4.0 * mobility_diff  # Increased from 3.0
        
        # Pass bonuses/penalties
        if opp_moves == 0 and player_moves > 0:
            mobility_term += 40  # Opponent must pass
        elif player_moves == 0 and opp_moves > 0:
            mobility_term -= 40  # We must pass

        # --- stability: pieces with friendly neighbors are safer ---
        def quick_stability(color):
            stable = 0
            for r in range(n):
                for c in range(n):
                    if board[r, c] != color:
                        continue
                    # Count friendly neighbors (fast check)
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

        # --- edge penalties: edges are vulnerable (except corners) ---
        edge_penalty = 0
        for r in range(n):
            for c in range(n):
                if (r == 0 or r == n-1 or c == 0 or c == n-1) and (r, c) not in corners:
                    if board[r, c] == root_player:
                        edge_penalty -= 1  # Our piece on edge (vulnerable)
                    elif board[r, c] == opponent:
                        edge_penalty += 1  # Opponent piece on edge (good for us)

        # --- game phase ---
        empty_count = np.count_nonzero(board == 0)
        total_cells = n * n
        empties_ratio = empty_count / total_cells

        # Phase-dependent weights
        if empties_ratio > 0.5:  # Early game
            w_piece = 0.8
            w_corner = 1.0
            w_mobility = 1.2  # Mobility very important early
            w_stability = 0.4
            w_edge = 0.3
        elif empties_ratio > 0.25:  # Mid game
            w_piece = 1.5
            w_corner = 1.0
            w_mobility = 1.0
            w_stability = 0.5
            w_edge = 0.2
        else:  # Late game
            w_piece = 12.0  # Piece count critical
            w_corner = 0.7
            w_mobility = 0.4
            w_stability = 0.3
            w_edge = 0.1

        value = (
            w_piece * score_diff +
            w_corner * corner_term +
            w_mobility * mobility_term +
            w_stability * stability_term +
            x_square_penalty +
            w_edge * edge_penalty
        )

        return value

    # ----------------------------------------------------------------------
    # ULTRA-FAST MOVE ORDERING
    # ----------------------------------------------------------------------
    def order_moves(self, board, moves, current_player, root_player, opponent):
        """
        Ultra-fast move ordering without expensive operations.
        Prioritizes: corners > captures > edge avoidance
        """
        n = board.shape[0]
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        corner_set = set(corners)
        
        # Pre-compute captures for all moves (fast)
        move_scores = []
        for move in moves:
            score = 0
            dest = move.get_dest()
            
            # Corner moves are highest priority
            if dest in corner_set:
                score += 3000
            
            # Count captures (fast approximation - check adjacent)
            captures = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = dest[0] + dr, dest[1] + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        if board[rr, cc] == opponent:
                            captures += 1
            
            # Single-tile moves duplicate (count as +1)
            src = move.get_src()
            is_jump = (abs(dest[0] - src[0]) == 2) or (abs(dest[1] - src[1]) == 2)
            if not is_jump:
                captures += 1
            
            score += captures * 200
            
            # Slight penalty for edge moves (except corners)
            if dest not in corner_set and (dest[0] == 0 or dest[0] == n-1 or dest[1] == 0 or dest[1] == n-1):
                score -= 20
            
            move_scores.append((score, move))
        
        # Sort by score (highest first)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in move_scores]

    # ----------------------------------------------------------------------
    # MINIMAX WITH OPPONENT MODELING (greedy simulation)
    # ----------------------------------------------------------------------
    def minimax(self, board, depth, alpha, beta,
                current_player, maximizing_player,
                root_player, opponent, start_time, is_opponent_greedy=True):
        """
        Minimax with opponent modeling:
        - When maximizing: we play optimally (minimax)
        - When minimizing: opponent plays greedily (we predict greedy moves)
        """

        # TIMEOUT
        if time.time() - start_time > self.time_limit:
            return self.evaluate_board(board, root_player, opponent), None

        # TERMINAL / DEPTH
        is_end, p1, p2 = check_endgame(board)
        if depth == 0 or is_end:
            if is_end:
                if root_player == 1:
                    return (p1 - p2), None
                else:
                    return (p2 - p1), None
            return self.evaluate_board(board, root_player, opponent), None

        moves = get_valid_moves(board, current_player)

        # NO MOVES → PASS TURN
        if not moves:
            next_player = opponent if current_player == root_player else root_player
            return self.minimax(board, depth - 1, alpha, beta,
                                next_player, not maximizing_player,
                                root_player, opponent, start_time, is_opponent_greedy)

        # ------------------------------------------------------------------
        # MAXIMIZING (our turn - play optimally)
        # ------------------------------------------------------------------
        if maximizing_player:
            best_val = float("-inf")
            best_move = None

            ordered = self.order_moves(board, moves, current_player, root_player, opponent)

            for move in ordered:
                if time.time() - start_time > self.time_limit:
                    return self.evaluate_board(board, root_player, opponent), best_move

                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)

                next_player = opponent if current_player == root_player else root_player

                val, _ = self.minimax(new_board, depth - 1, alpha, beta,
                                      next_player, False,
                                      root_player, opponent, start_time, is_opponent_greedy)

                if val > best_val:
                    best_val = val
                    best_move = move

                alpha = max(alpha, val)
                if beta <= alpha:
                    break

            return best_val, best_move

        # ------------------------------------------------------------------
        # MINIMIZING (opponent's turn - model greedy behavior)
        # ------------------------------------------------------------------
        else:
            # KEY INSIGHT: If opponent is greedy, predict its move instead of minimax
            if is_opponent_greedy and depth > 1:
                # Predict what greedy would do
                greedy_move = self.predict_greedy_move(board, current_player, root_player)
                if greedy_move is not None:
                    new_board = deepcopy(board)
                    execute_move(new_board, greedy_move, current_player)
                    next_player = root_player
                    val, _ = self.minimax(new_board, depth - 1, alpha, beta,
                                          next_player, True,
                                          root_player, opponent, start_time, is_opponent_greedy)
                    return val, greedy_move
            
            # Fallback: if no greedy move or depth is shallow, use minimax
            best_val = float("inf")
            best_move = None

            ordered = self.order_moves(board, moves, current_player, root_player, opponent)

            for move in ordered:
                if time.time() - start_time > self.time_limit:
                    return self.evaluate_board(board, root_player, opponent), best_move

                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)

                next_player = root_player

                val, _ = self.minimax(new_board, depth - 1, alpha, beta,
                                      next_player, True,
                                      root_player, opponent, start_time, is_opponent_greedy)

                if val < best_val:
                    best_val = val
                    best_move = move

                beta = min(beta, val)
                if beta <= alpha:
                    break

            return best_val, best_move

    # ----------------------------------------------------------------------
    # ITERATIVE DEEPENING WITH TIME MANAGEMENT
    # ----------------------------------------------------------------------
    def iterative_deepening(self, board, player, opponent):
        start = time.time()
        best_move = None
        best_val = float("-inf")
        
        moves = get_valid_moves(board, player)
        if len(moves) == 1:
            return moves[0]

        for depth in range(1, self.max_depth + 1):
            elapsed = time.time() - start
            if elapsed > self.time_limit * 0.98:
                break

            try:
                val, move = self.minimax(board, depth,
                                         float("-inf"), float("inf"),
                                         player, True,
                                         player, opponent,
                                         start, is_opponent_greedy=True)
                if move is not None:
                    best_move = move
                    best_val = val
                if val > 1000:  # Clear win
                    break
                
                elapsed = time.time() - start
                if elapsed > self.time_limit * 0.92:
                    break

            except (TimeoutError, Exception):
                break

        return best_move

    # ----------------------------------------------------------------------
    # MAIN STEP
    # ----------------------------------------------------------------------
    def step(self, board, player, opponent):
        start = time.time()

        moves = get_valid_moves(board, player)
        if not moves:
            return None

        best_move = self.iterative_deepening(board, player, opponent)

        if best_move is None:
            best_move = moves[0]

        print(f"My AI turn time: {time.time() - start:.4f}s")
        return best_move
