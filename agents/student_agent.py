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
    # Maximum depth for search
    self.max_depth = 10

  def evaluate_board(self, chess_board, player, opponent):
    """
    Evaluation function for the Ataxx board state.
    Returns a score where positive values favor the maximizing player (current player).
    Higher scores indicate better positions for the player.
    """
    # Count pieces for each player
    player_pieces = np.sum(chess_board == player)
    opponent_pieces = np.sum(chess_board == opponent)

    # Basic piece difference (this is the most important factor)
    score = player_pieces - opponent_pieces

    # Bonus for controlling corners (very important in Ataxx)
    board_size = chess_board.shape[0]
    corners = [(0, 0), (0, board_size-1), (board_size-1, 0), (board_size-1, board_size-1)]
    corner_bonus = 0
    for corner in corners:
      if chess_board[corner[0], corner[1]] == player:
        corner_bonus += 10  # Strong bonus for controlling corners
      elif chess_board[corner[0], corner[1]] == opponent:
        corner_bonus -= 10  # Strong penalty if opponent controls corners

    # Mobility bonus (number of available moves) - less important than pieces
    player_moves = len(get_valid_moves(chess_board, player))
    opponent_moves = len(get_valid_moves(chess_board, opponent))
    mobility_score = (player_moves - opponent_moves) * 2

    # Combine scores with weights
    total_score = score * 10 + corner_bonus + mobility_score

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
    Order moves to improve alpha-beta pruning efficiency in Ataxx.
    Prioritize moves that capture more opponent pieces (duplication + jump moves).
    """
    move_scores = []

    for move_coords in valid_moves:
      # Simulate the move to see how many pieces we gain
      board_copy = deepcopy(chess_board)
      try:
        pieces_gained = count_disc_count_change(board_copy, move_coords, player)
        move_scores.append((pieces_gained, move_coords))
      except:
        # If simulation fails, give neutral score
        move_scores.append((0, move_coords))

    # Sort by pieces gained (descending) - best moves first
    move_scores.sort(key=lambda x: x[0], reverse=True)
    return [move_coords for _, move_coords in move_scores]

  def iterative_deepening_search(self, chess_board, player, opponent):
    """
    Perform iterative deepening search for Ataxx with time limit.
    Gradually increases search depth until time runs out, then returns best move found.
    """
    start_time = time.time()
    best_move = None

    # Get valid moves first
    valid_moves = get_valid_moves(chess_board, player)
    if not valid_moves:
      return None

    # Start with depth 1 and increase
    for depth in range(1, self.max_depth + 1):
      try:
        # Check if we have time for this depth
        if time.time() - start_time > self.time_limit * 0.5:  # Use 50% of time limit
          break

        # Run minimax for this depth
        eval_score, move = self.minimax_alpha_beta(chess_board, depth, float('-inf'), float('inf'),
                                        True, player, opponent, start_time)

        if move is not None:
          best_move = move
          # Debug output for first few depths
          if depth <= 3:
            print(f"Depth {depth}: eval={eval_score}, move=({move.get_dest()})")

        # Check time after each depth
        if time.time() - start_time > self.time_limit:
          break

      except Exception as e:
        print(f"Error at depth {depth}: {e}")
        break

    return best_move

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the Ataxx board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's pieces (Blue),
      and 2 represents Player 2's pieces (Brown), 3 represents obstacles.
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a MoveCoordinates object specifying the source and destination
    of your move. Use functions in helpers to determine valid moves and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    start_time = time.time()

    # Get valid moves
    valid_moves = get_valid_moves(chess_board, player)

    # If no valid moves, return a random move (though this shouldn't happen in normal play)
    if not valid_moves:
      print("No valid moves available, using random move")
      time_taken = time.time() - start_time
      print(f"My AI's turn took {time_taken:.3f} seconds.")
      return random_move(chess_board, player)

    # Use iterative deepening search to find the best move
    best_move = self.iterative_deepening_search(chess_board, player, opponent)

    # If search failed or timed out, fall back to random move
    if best_move is None:
      print("Search failed or timed out, using random move")
      time_taken = time.time() - start_time
      print(f"My AI's turn took {time_taken:.3f} seconds.")
      return random_move(chess_board, player)

    time_taken = time.time() - start_time
    print(f"My AI's turn took {time_taken:.3f} seconds.")

    # Return the MoveCoordinates object directly
    return best_move

