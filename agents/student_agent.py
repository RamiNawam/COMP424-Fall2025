# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates, count_disc_count_change

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  An intelligent Ataxx agent using minimax with alpha-beta pruning.
  Implements iterative deepening, evaluation function with game-specific heuristics,
  and move ordering for optimal performance within time constraints.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
    # Time limit for iterative deepening (2 seconds max as mentioned in comments)
    self.time_limit = 2.0
    # Maximum depth for search - reduced for better time management
    self.max_depth = 4  # Focus on quality over depth

  def evaluate_board(self, chess_board, player, opponent):
    """
    Optimized evaluation function for Ataxx based on greedy agent analysis.
    Uses balanced weights similar to the winning greedy approach.
    """
    # Count pieces for each player
    player_pieces = np.sum(chess_board == player)
    opponent_pieces = np.sum(chess_board == opponent)

    # Basic piece difference (primary factor but not over-weighted)
    piece_score = player_pieces - opponent_pieces

    # Corner control (very important in Ataxx - matches greedy agent approach)
    board_size = chess_board.shape[0]
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    corner_score = 0
    for corner in corners:
      if chess_board[corner[0], corner[1]] == player:
        corner_score += 8  # Strong bonus for corners (slightly less than greedy's 5x3=15)
      elif chess_board[corner[0], corner[1]] == opponent:
        corner_score -= 8  # Strong penalty for opponent corners

    # Mobility (opponent mobility penalty - matches greedy approach)
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    mobility_penalty = -opponent_moves * 3  # Stronger penalty than greedy's -1x

    # Edge control bonus (second-row pieces are valuable)
    edge_bonus = 0
    for i in range(board_size):
      for j in range(board_size):
        if chess_board[i, j] == player:
          # Bonus for being near edges (defensive position)
          if i == 0 or i == board_size-1 or j == 0 or j == board_size-1:
            edge_bonus += 1
        elif chess_board[i, j] == opponent:
          if i == 0 or i == board_size-1 or j == 0 or j == board_size-1:
            edge_bonus -= 1

    # Combine scores with balanced weights (learned from greedy agent)
    total_score = piece_score * 3 + corner_score + mobility_penalty + edge_bonus

    return total_score

  def minimax_alpha_beta(self, chess_board, depth, alpha, beta, maximizing_player, player, opponent, start_time):
    """
    Minimax algorithm with alpha-beta pruning for Ataxx.
    Returns (best_value, best_move) where best_value is the evaluation score
    and best_move is the optimal MoveCoordinates object.
    """
    # Check time limit
    if time.time() - start_time > self.time_limit:
      return (self.evaluate_board(chess_board, player if maximizing_player else opponent,
                                 opponent if maximizing_player else player), None)

    # Terminal node or max depth reached
    is_endgame, p1_score, p2_score = check_endgame(chess_board)
    if depth == 0 or is_endgame:
      if is_endgame:
        # Game over - calculate final score
        if maximizing_player:
          if player == 1:
            return (p1_score - p2_score, None)
          else:
            return (p2_score - p1_score, None)
        else:
          if player == 1:
            return (p2_score - p1_score, None)
          else:
            return (p1_score - p2_score, None)
      else:
        # Depth limit reached - use evaluation function
        return (self.evaluate_board(chess_board, player if maximizing_player else opponent,
                                   opponent if maximizing_player else player), None)

    current_player = player if maximizing_player else opponent
    valid_moves = get_valid_moves(chess_board, current_player)

    # No valid moves - pass turn
    if not valid_moves:
      return self.minimax_alpha_beta(chess_board, depth - 1, alpha, beta,
                                   not maximizing_player, player, opponent, start_time)

    if maximizing_player:
      max_eval = float('-inf')
      best_move = None

      # Order moves (simple heuristic: prefer moves that capture more pieces)
      ordered_moves = self.order_moves(chess_board, valid_moves, current_player, opponent)

      for move_coords in ordered_moves:
        # Create a deep copy for simulation
        board_copy = deepcopy(chess_board)
        execute_move(board_copy, move_coords, current_player)

        eval_score, _ = self.minimax_alpha_beta(board_copy, depth - 1, alpha, beta,
                                              False, player, opponent, start_time)

        if eval_score > max_eval:
          max_eval = eval_score
          best_move = move_coords

        alpha = max(alpha, eval_score)
        if beta <= alpha:
          break  # Alpha-beta pruning

      return (max_eval, best_move)
    else:
      min_eval = float('inf')
      best_move = None

      # Order moves for minimizing player too
      ordered_moves = self.order_moves(chess_board, valid_moves, current_player, opponent)

      for move_coords in ordered_moves:
        # Create a deep copy for simulation
        board_copy = deepcopy(chess_board)
        execute_move(board_copy, move_coords, current_player)

        eval_score, _ = self.minimax_alpha_beta(board_copy, depth - 1, alpha, beta,
                                              True, player, opponent, start_time)

        if eval_score < min_eval:
          min_eval = eval_score
          best_move = move_coords

        beta = min(beta, eval_score)
        if beta <= alpha:
          break  # Alpha-beta pruning

      return (min_eval, best_move)

  def order_moves(self, chess_board, valid_moves, player, opponent):
    """
    Order moves using evaluation function for better alpha-beta pruning.
    Simulates each move and evaluates the resulting position.
    """
    move_scores = []

    for move_coords in valid_moves:
      # Simulate the move
      board_copy = deepcopy(chess_board)
      execute_move(board_copy, move_coords, player)

      # Evaluate the resulting position
      eval_score = self.evaluate_board(board_copy, player, opponent)
      move_scores.append((eval_score, move_coords))

    # Sort by evaluation score (descending) - best moves first
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move_coords for _, move_coords in move_scores]

  def iterative_deepening_search(self, chess_board, player, opponent):
    """
    Optimized iterative deepening with better time management.
    Focus on quality at shallower depths rather than deep but rushed search.
    """
    start_time = time.time()
    best_move = None

    # Get valid moves first
    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
      return None

    # If few moves available, we can afford deeper search
    num_moves = len(valid_moves)
    if num_moves <= 10:
      max_depth = min(4, self.max_depth)  # Allow deeper search for critical positions
    else:
      max_depth = min(2, self.max_depth)  # Stick to shallow search for complex positions

    # Start with depth 1 and increase
    for depth in range(1, max_depth + 1):
      try:
        # More aggressive time management
        time_spent = time.time() - start_time
        if time_spent > self.time_limit * 0.7:  # Use 70% of time
          break

        # Run minimax for this depth
        eval_score, move = self.minimax_alpha_beta(chess_board, depth, float('-inf'), float('inf'),
                                        True, player, opponent, start_time)

        if move is not None:
          best_move = move

        # Emergency break if we're running out of time
        if time.time() - start_time > self.time_limit * 0.9:
          break

      except Exception as e:
        # Silently handle errors and use best move found so far
        break

    return best_move

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    Uses optimized minimax with fallback to greedy evaluation.
    """

    start_time = time.time()

    # Get valid moves
    valid_moves = get_valid_moves(chess_board, player)

    # If no valid moves, return a random move (though this shouldn't happen in normal play)
    if not valid_moves:
      time_taken = time.time() - start_time
      print(f"My AI's turn took {time_taken:.3f} seconds.")
      return random_move(chess_board, player)

    # Try iterative deepening search first
    best_move = self.iterative_deepening_search(chess_board, player, opponent)

    # If search failed or timed out, use greedy evaluation of all moves
    if best_move is None:
      print("Using optimized greedy fallback...")
      best_move = self.greedy_best_move(chess_board, valid_moves, player, opponent)

    time_taken = time.time() - start_time
    print(f"My AI's turn took {time_taken:.3f} seconds.")

    return best_move

  def greedy_best_move(self, chess_board, valid_moves, player, opponent):
    """
    Fallback method: evaluate all moves greedily (like the winning agent).
    Returns the best move according to evaluation function.
    """
    best_move = None
    best_score = float('-inf')

    for move_coords in valid_moves:
      # Simulate the move
      board_copy = deepcopy(chess_board)
      execute_move(board_copy, move_coords, player)

      # Evaluate the resulting position
      move_score = self.evaluate_board(board_copy, player, opponent)

      if move_score > best_score:
        best_score = move_score
        best_move = move_coords

    return best_move

