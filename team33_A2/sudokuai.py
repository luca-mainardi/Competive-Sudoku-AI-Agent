"""Competitive Sudoku AI.

Adapted from /naive_player/sudokuai.py

A1: iterative deepening minimax search with alpha beta pruning.
A2: heuristic search.
"""

import os
from random import shuffle

# from numpy import full

# Import types and libraries
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from competitive_sudoku.sudokuai import SudokuAI

from .utils import block_range  # Game specific logic
from .utils import block_index, calculate_move_score, is_illegal, next_player


class SudokuAI(SudokuAI):
    """
    Sudoku AI agent that computes a move for a given sudoku configuration.
    """

    def __init__(self):
        super().__init__()
        self.transposition_table = []

    def update_legal(self, game_state: GameState, move: Move, moves: dict) -> list:
        """Update which moves are legal in the recursive minimax search.

        :param move: The move to be evaluated
        :param moves: A dictionary containing the inital set of moves, whether they are still legal and other properties of the moves.
            initial: list of initally legal moves, used as subset to avoid iterating over all moves
            legal: numpy array of shape (board_size, board_size, board_size + 1) where legal[i, j, k] is True if Move(i, j, k) is legal
            count: Counter for legal moves, avoid repeated iteration
        :return: A list of moves that were invalidated by the move.
        """
        _moves_invalidated = []
        for row in range(game_state.board.board_height()):
            # move invalidates another move if not already illegal, avoid double counting
            if moves["legal"][row, move.j, move.value]:
                _moves_invalidated.append(Move(row, move.j, move.value))
                moves["legal"][row, move.j, move.value] = False  # Set legal status
        for column in range(game_state.board.board_width()):
            if moves["legal"][move.i, column, move.value]:
                _moves_invalidated.append(Move(move.i, column, move.value))
                moves["legal"][move.i, column, move.value] = False
        for row, col in block_range(row=move.i, col=move.j, board=game_state.board):
            if moves["legal"][row, col, move.value]:
                _moves_invalidated.append(Move(row, col, move.value))
                moves["legal"][row, col, move.value] = False
        moves["count"] -= len(_moves_invalidated)
        return _moves_invalidated

    def minimax(
            self,
            game_state: GameState,
            move: Move,
            moves: dict,
            current_player: int,
            maximizing_player: int,
            depth: int,
            *,
            alpha: float = float("-inf"),
            beta: float = float("inf"),
    ) -> float:
        """Returns the score of a given move.

        Minimax search that considers the perspectives of two players.
        Searches until some depth before returning the score.
        Uses alpha beta pruning to avoid searching branches which cannot lead to better results.

        :param game_state: The current game state. Describes the board, scores, taboo moves and move history.
        :param move: The move to be evaluated
        :param moves: A dictionary containing the inital set of moves, whether they are still legal and other properties of the moves.
            initial: list of initally legal moves, used as subset to avoid iterating over all moves
            legal: numpy array of shape (board_size, board_size, board_size + 1) where legal[i, j, k] is True if Move(i, j, k) is legal
            count: Counter for the number of legal moves.
            free: shows per region (row, col or block) what the number of free squares is, used in heuristics.
        :param current_player: The player who's turn it is. Will be the same throughout the turn. (0 or 1 for first or second player)
        :param maximizing_player: The player who's score is to be maximised. (0 or 1 for first or second player)
        :param depth: The maximum depth to search before returning the score.
        :param alpha: The highest so far value for alpha beta pruning. (initially -inf)
        :param beta: The lowest so far value for alpha beta pruning. (initially inf)
        :return: The score of the move (higher is better for maximizing player, lower is better for minimizing player)
        """
        block_indices = block_index(move.i, move.j, game_state.board)

        # Apply move, update resulting scores
        # Update legal moves and count newly invalidated moves
        game_state.board.put(move.i, move.j, move.value)
        _score_achieved = calculate_move_score(game_state, move)
        game_state.scores[current_player] += _score_achieved
        _moves_invalidated = self.update_legal(game_state, move, moves)
        # switches perspective to other player
        current_player = next_player(current_player)

        # Update properties used in heuristics
        moves["free"]["row"][move.i] -= 1
        moves["free"]["col"][move.j] -= 1
        moves["free"]["block"][block_indices[0]][block_indices[1]] -= 1

        # Search until game is finished or maximum depth is reached
        if depth == 0 or moves["count"] == 0:
            # evaluate the current board
            best_value = self.evaluate_state(
                maximizing_player, current_player, game_state, moves["free"]
            )
        else:
            if maximizing_player == current_player:  # maximising player
                best_value = float("-inf")
                for try_move in moves["initial"]:
                    if not moves["legal"][try_move.i, try_move.j, try_move.value]:
                        continue
                    # Recurse and find value up to some depth
                    value = self.minimax(
                        game_state,
                        try_move,
                        moves,
                        current_player,
                        maximizing_player,
                        depth - 1,
                        alpha=alpha,
                        beta=beta,
                        )
                    best_value = max(best_value, value)
                    alpha = max(alpha, best_value)
                    if beta < alpha:
                        break
            else:  # minimising player
                best_value = float("inf")
                for try_move in moves["initial"]:
                    if not moves["legal"][try_move.i, try_move.j, try_move.value]:
                        continue
                    # Recurse and find value up to some depth
                    value = self.minimax(
                        game_state,
                        try_move,
                        moves,
                        current_player,
                        maximizing_player,
                        depth - 1,
                        alpha=alpha,
                        beta=beta,
                        )
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                    if beta < alpha:
                        break

        # Undo move and its effects
        current_player = next_player(current_player)
        moves["free"]["row"][move.i] += 1
        moves["free"]["col"][move.j] += 1
        moves["free"]["block"][block_indices[0]][block_indices[1]] += 1
        moves["count"] += len(_moves_invalidated)
        for inv_move in _moves_invalidated:
            moves["legal"][inv_move.i, inv_move.j, inv_move.value] = True
        game_state.scores[current_player] -= _score_achieved
        game_state.board.put(move.i, move.j, 0)

        # Recursion result
        return best_value

    def evaluate_state(
            self,
            maximizing_player: int,
            current_player: int,
            game_state: GameState,
            free: dict,
    ) -> float:
        """Heuristic evaluation of the current game state.

        Is used by minimax to evaluate the current game state which is most often not a complete game.
        Base score is the difference between the scores of the two players since maximizing this will result in a win.
        Additional heuristic contributions are made for early game where score is often 0.

        :param maximizing_player: The player who's score is to be maximised. (0 or 1 for first or second player)
        :param game_state: The current game state. Describes the board, scores, taboo moves and move history.
        :param free: The number of free squares per region.
        :return: The score of the game state (higher is better for maximizing player, and vice versa)
        """
        score = (
                game_state.scores[maximizing_player]
                - game_state.scores[next_player(maximizing_player)]
        )

        if not os.environ.get("not_prefer_more_empty"):
            # Scale to avoid this additional heuristic dominating the score
            # Will result in an early game strategy avoiding a filled field.
            # Positive contribution, our player will thus prefer less filled fields.
            if current_player == maximizing_player:
                score += 0.1 * self.prefer_empty_regions(game_state, free)

        return score

    def prefer_empty_regions(self, game_state: GameState, free: dict):
        """Heuristic for evaluating a state.

        Prefer less filled out regions by counting the number of free moves
        Normalising using the the maximum number of squares in a region

        :param maximizing_player: The player who's score is to be maximised. (0 or 1 for first or second player)
        :param game_state: The current game state. Describes the board, scores, taboo moves and move history.
        :param free: The number of free squares per region.
        :return: The heuristic score, will be 1 for a completely empty field.
        """
        board_size = game_state.board.board_height()
        num_blocks = board_size / game_state.board.region_height()
        score = sum([count / board_size for count in free["row"]]) / board_size / 3
        score += sum([count / board_size for count in free["col"]]) / board_size / 3
        score += (
                sum(
                    sum([count / board_size for count in block]) / num_blocks
                    for block in free["block"]
                )
                / num_blocks
                / 3
        )
        return score

    def find_initial_moves(self, game_state: GameState) -> list[Move]:
        """
        Find all possible moves for a given state. This is copy of method used in A1 used for benchmarking purposes.
        @param game_state: GameState
        @return: list of moves
        """
        board_size = game_state.board.board_width()

        # Generate possible moves
        initial_moves = [
            Move(i, j, value)
            for i in range(board_size)
            for j in range(board_size)
            for value in range(1, board_size + 1)
            if not is_illegal(move=Move(i, j, value), state=game_state)
        ]

        # Shuffle moves to be less predictable
        shuffle(initial_moves)

        return initial_moves

    def find_initial_moves_heuristics(self, state: GameState) -> list[Move]:
        """
        Find all possible moves for a given state and order them by priority. The priority is determined by the
        number of possible moves for a cell. Cells with fewer possible moves are prioritised because they are
        more likely to result in a completed row, column, or block.
        Any cell with 2 possible moves is *de*-prioritised, because that means the opponent could complete a row, column
        or block next turn.
        @param state: GameState
        @return: ordered list of moves
        """
        size = state.board.board_width()

        # Store moves in a dictionary with the number of possible moves for that cell as key
        priority_dict = dict([(key, []) for key in range(0, size + 1)])

        for i in range(size):
            for j in range(size):
                possible_moves_for_cell = []
                for value in range(1, size + 1):
                    move_candidate = Move(i, j, value)
                    if not is_illegal(move=move_candidate, state=state):
                        possible_moves_for_cell.append(move_candidate)

                priority_dict[len(possible_moves_for_cell)].extend(possible_moves_for_cell)

        key_order = sorted(priority_dict.keys())

        # Prioritise cells that have 2 possible moves,
        # because the opponent could complete a row, column or block
        if len(key_order) > 2:
            key_order.pop(2)
            key_order.append(2)

        # Return list of moves in order of search priority
        return [move for key in key_order for move in priority_dict[key]]

    def compute_best_move(self, game_state: GameState) -> None:
        """Computes the best move for the agent and proposes it.

        Will initially propose a random move, then evaluate different moves based on minimax search.
        Since the turn time is not known it will propose the best move found so far by iteratively deepening the search.

        :param game_state: The current game state. Describes the board, scores and move history.
        """
        board_size = game_state.board.board_height()
        num_blocks = board_size // game_state.board.region_height()

        # print(f"Avoid 2 moves: {not os.environ.get('not_avoid_2_moves')}")
        # print(f"Prefer empty regions: {not os.environ.get('not_prefer_more_empty')}")

        # Generate possible moves
        initial_moves = (
            self.find_initial_moves_heuristics(game_state)
            if not os.environ.get("not_avoid_2_moves")
            else self.find_initial_moves(game_state)
        )

        # Move cache
        moves = {}

        # List of initially legal moves, used as subset to avoid iterating over all moves
        moves["initial"] = initial_moves

        # Propose a random move (initial, avoid timeout)
        self.propose_move(moves["initial"][0])

        # Avoid repeated regeneration of legal moves by tracking their status
        from numpy import full
        moves["legal"] = full(
            shape=(board_size, board_size, board_size + 1),
            dtype=bool,
            fill_value=False,
        )
        moves["count"] = len(moves["initial"])
        for move in moves["initial"]:
            moves["legal"][move.i, move.j, move.value] = True

        # Track some properties of the game to be used in statistical search
        # Count how many free squares there are per region
        # TODO do this above when generating moves?
        moves["free"] = {
            "row": [
                len(set(((m.i, m.j) for m in moves["initial"] if m.i == y)))
                for y in range(board_size)
            ],
            "col": [
                len(set(((m.i, m.j) for m in moves["initial"] if m.j == x)))
                for x in range(board_size)
            ],
            "block": [
                [
                    len(
                        set(
                            (
                                (m.i, m.j)
                                for m in moves["initial"]
                                if block_index(m.i, m.j, game_state.board) == (i, j)
                            )
                        )
                    )
                    for j in range(num_blocks)
                ]
                for i in range(num_blocks)
            ],
        }

        # Iteratively increase the search depth of minimax
        best_move: tuple[Move, float] | None = None
        for depth_limit in range(1, len(moves["initial"]), 1):
            # evaluate different moves based on minimax
            for move in moves["initial"]:
                # player_index is 0 or 1 (first or second player)
                player_index = game_state.current_player() - 1

                value = self.minimax(
                    game_state,
                    move,
                    moves,
                    player_index,  # Current player is self
                    player_index,  # Maximising own score
                    depth_limit,
                )

                if best_move is None or value > best_move[1]:
                    best_move = (move, value)
                    # Update proposed move (best so far, avoid timeout while find a better move)
                    self.propose_move(best_move[0])
                    # print(move.i, move.j, value)
