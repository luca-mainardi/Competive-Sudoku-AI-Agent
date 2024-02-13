"""Competitive Sudoku AI.

Adapted from /naive_player/sudokuai.py

Changes:
A1: basic MiniMax implementation.
"""

import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
import competitive_sudoku.sudokuai
from team33_A1.utils import is_possible, is_illegal, region_range, next_player


class SudokuAI(competitive_sudoku.sudokuai.SudokuAI):
    """
    Sudoku AI that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
        self.transposition_table = []

    def minimax(
        self,
        game_state: GameState,
        move: Move,
        moves: list[Move],
        current_player: int,
        maximizing_player: int,
        depth: int,
        *,
        alpha: int = float("-inf"),
        beta: int = float("inf"),
    ) -> int:
        """Returns the score of the best move.

        The score is the number of empty squares on the board after the move.

        Searches until some depth.
        Uses alpha beta pruning to avoid searching branches which cannot lead to better results.
        """
        # TODO store intermittent state between iterative deeping steps (transposition table)

        old_score = game_state.scores[current_player]

        # Apply move and find new possible moves
        game_state.board.put(move.i, move.j, move.value)
        game_state.scores[current_player] += self.score_move(game_state, move)

        # TODO edit moves in place for efficiency, calculate if newly illegal instead of checking entire board
        # TODO use different data structure for moves, e.g. binary matrix
        new_moves = [
            m for m in moves if not is_illegal(move=move, for_state=game_state)
        ]

        # switches current player (more efficient than storing all moves in game state)
        current_player = next_player(current_player)

        if (
            depth == 0 or len(new_moves) == 0
        ):  # Game is finished or maximum depth is reached
            # evaluate with high value when the maximising player is winning
            best_value = (
                game_state.scores[maximizing_player]
                - game_state.scores[next_player(maximizing_player)]
            )
        else:
            const_function = max if maximizing_player == current_player else min
            best_value = float("-inf")

            for try_move in new_moves:
                # Recurse and find value up to some depth
                value = self.minimax(
                    game_state,
                    try_move,
                    new_moves,
                    current_player,
                    maximizing_player,
                    depth - 1,
                    alpha=alpha,
                    beta=beta,
                )
                best_value = const_function(best_value, value)
                # Alpha beta pruning, do not search branches which cannot lead to better results
                alpha = const_function(alpha, best_value)
                if beta <= alpha:
                    break

        # Undo move
        game_state.board.put(move.i, move.j, 0)
        current_player = next_player(current_player)
        game_state.scores[current_player] = old_score

        # Recursion result
        return best_value

    @staticmethod
    def score_move(game_state: GameState, move: Move) -> int:
        """
        Check if a move completes any regions and returns the score earned
        """
        # TODO can make this faster with sums of rows, columns and regions
        row_complete = col_complete = block_complete = True

        # Check if completed a row
        for col in range(game_state.board.board_width()):
            if game_state.board.get(move.i, col) == SudokuBoard.empty:
                row_complete = False
                break

        # Check if completed a column
        for row in range(game_state.board.board_height()):
            if game_state.board.get(row, move.j) == SudokuBoard.empty:
                col_complete = False
                break

        # Check if completed a block
        for row, col in region_range(row=move.i, col=move.j, board=game_state.board):
            if game_state.board.get(row, col) == SudokuBoard.empty:
                block_complete = False
                break

        # Return score by move
        regions_complete = int(row_complete) + int(col_complete) + int(block_complete)
        return {
            0: 0,
            1: 1,
            2: 3,
            3: 7,
        }[regions_complete]

    def compute_best_move(self, game_state: GameState) -> None:
        board_size = game_state.board.board_height()
        # TODO save state between moves, not allowed for A1
        # TODO create unitttests
        # TODO also actively avoid taboo moves (to avoid loss of move)?

        # Generate possible moves
        initial_moves = [
            Move(i, j, value)
            for i in range(board_size)
            for j in range(board_size)
            for value in range(1, board_size + 1)
            if is_possible(move=Move(i, j, value), for_state=game_state)
        ]

        # Shuffle moves to be less predictable
        random.shuffle(initial_moves)

        # Propose a certain move (initial, avoid timeout)
        self.propose_move(initial_moves[0])

        # evaluate different moves based on minimax
        # TODO track lime limit
        best_move: tuple[Move, float] | None = None
        for depth_limit in range(1, len(initial_moves), 1):
            # evaluate different moves based on minimax
            for move in initial_moves:
                player_index = (
                    game_state.current_player() - 1
                )  # player_index is 0 or 1 (self or opponent)

                value = self.minimax(
                    game_state,
                    move,
                    initial_moves,
                    player_index,  # Current player is self
                    player_index,  # Maximising own score
                    depth_limit,
                )

                if best_move is None or value > best_move[1]:
                    best_move = (move, value)
                    # Update proposed move (best so far, avoid timeout while find a better move)
                    self.propose_move(best_move[0])
