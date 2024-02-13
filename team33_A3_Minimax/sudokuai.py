"""
Competitive Sudoku AI.

Adapted from /naive_player/sudokuai.py

A1: iterative deepening minimax search with alpha beta pruning.
A2: heuristic search.
A3: heuristic search with unsolvable moves list.

"""

from random import shuffle

# Import types and libraries
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from competitive_sudoku.sudokuai import SudokuAI

from .utils import block_range, StateMatrixT, SavedDataT, is_unsolvable  # Game specific logic
from .utils import (
    block_index,
    calculate_filling_rate,
    calculate_move_score,
    is_illegal,
    next_player,
)


class SudokuAI(SudokuAI):
    """
    Sudoku AI agent that computes a move for a given sudoku configuration.
    """

    def minimax(
        self,
        *,
        game_state: GameState,
        move: Move,
        state_matrix: StateMatrixT,
        current_player: int,
        maximizing_player: int,
        depth: int,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:
        """
        Returns the score of a given move.

        Minimax search that considers the perspectives of two players.
        Searches until some depth before returning the score.
        Uses alpha beta pruning to avoid searching branches which cannot lead to better results.

        :param game_state: The current game state. Describes the board, scores, unsolvable moves and move history.
        :param move: The move to be evaluated
        :param state_matrix: State matrix containing the initial set of moves and whether they are still legal
        :param current_player: The player who's turn it is. Will be the same throughout the turn. (0 or 1 for first or second player)
        :param maximizing_player: The player who's score is to be maximised. (0 or 1 for first or second player)
        :param depth: The maximum depth to search before returning the score.
        :param alpha: The highest so far value for alpha beta pruning. (initially -inf)
        :param beta: The lowest so far value for alpha beta pruning. (initially inf)
        :return: The score of the move (higher is better for maximizing player, lower is better for minimizing player)
        """

        # Apply move, update resulting scores
        # Update legal moves and count newly invalidated moves
        game_state.board.put(move.i, move.j, move.value)
        _score_achieved = calculate_move_score(game_state, move)
        game_state.scores[current_player] += _score_achieved
        _moves_invalidated = self.update_legal(game_state, move, state_matrix)

        # Switches perspective to other player
        current_player = next_player(current_player)

        # Search until game is finished or maximum depth is reached
        if depth == 0 or state_matrix["legal_count"] == 0:
            # evaluate the current board
            best_value = self.evaluate_state(
                maximizing_player,
                game_state,
            )
        else:
            if maximizing_player == current_player:  # maximising player
                best_value = float("-inf")
                for try_move in state_matrix["initial"]:
                    if not state_matrix["legal"][try_move.i][try_move.j][try_move.value]:
                        continue
                    # Recurse and find value up to some depth
                    value = self.minimax(
                        game_state=game_state,
                        move=try_move,
                        state_matrix=state_matrix,
                        current_player=current_player,
                        maximizing_player=maximizing_player,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                    )
                    best_value = max(best_value, value)
                    alpha = max(alpha, best_value)
                    if beta < alpha:
                        break
            else:  # minimising player
                best_value = float("inf")
                for try_move in state_matrix["initial"]:
                    if not state_matrix["legal"][try_move.i][try_move.j][try_move.value]:
                        continue
                    # Recurse and find value up to some depth
                    value = self.minimax(
                        game_state=game_state,
                        move=try_move,
                        state_matrix=state_matrix,
                        current_player=current_player,
                        maximizing_player=maximizing_player,
                        depth=depth - 1,
                        alpha=alpha,
                        beta=beta,
                    )
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                    if beta < alpha:
                        break
        # Undo move and its effects
        current_player = next_player(current_player)
        state_matrix["legal_count"] += len(_moves_invalidated)
        for inv_move in _moves_invalidated:
            state_matrix["legal"][inv_move.i][inv_move.j][inv_move.value] = True
        game_state.scores[current_player] -= _score_achieved
        game_state.board.put(move.i, move.j, 0)

        # Recursion result
        return best_value

    def evaluate_state(self, maximizing_player: int, game_state: GameState) -> float:
        """
        Calculates the score of a given game state.

        :param maximizing_player: The player whose score is to be maximised. (0 or 1 for first or second player)
        :param game_state: The current game state.
        :return: The score of the game state. (higher is better for maximizing player, and vice versa)
        """
        return game_state.scores[maximizing_player] - game_state.scores[next_player(maximizing_player)]

    def update_legal(self, game_state: GameState, move: Move, moves: StateMatrixT) -> list:
        """
        Update which moves are legal in the recursive minimax search.

        :param game_state: The current game state. Describes the board, scores, taboo moves and move history.
        :param move: The move to be evaluated
        :param moves: State matrix containing the initial set of moves and whether they are still legal
        :return: A list of moves that were invalidated by the move.
        """

        _moves_invalidated = []
        for row in range(game_state.board.board_height()):
            # move invalidates another move if not already illegal, avoid double counting
            if moves["legal"][row][move.j][move.value]:
                _moves_invalidated.append(Move(row, move.j, move.value))
                moves["legal"][row][move.j][move.value] = False  # Set legal status

        for column in range(game_state.board.board_width()):
            if moves["legal"][move.i][column][move.value]:
                _moves_invalidated.append(Move(move.i, column, move.value))
                moves["legal"][move.i][column][move.value] = False

        for row, col in block_range(row=move.i, col=move.j, board=game_state.board):
            if moves["legal"][row][col][move.value]:
                _moves_invalidated.append(Move(row, col, move.value))
                moves["legal"][row][col][move.value] = False
        moves["legal_count"] -= len(_moves_invalidated)
        return _moves_invalidated

    def find_initial_moves(self, game_state: GameState) -> list[Move]:
        """
        Find all possible moves for a given state.

        :param game_state: The current game state.
        :return: The list of legal initial moves.
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

    def calculate_free_cells(self, game_state: GameState) -> dict:
        """
        Calculate how many free cells there are in each region of the sudoku (row, column, block), given a game state.

        It is used to assign each move a priority level.

        :return: A dictionary containing the number of free cells, for each region.
            row: list containing a value for each row of the sudoku, corresponding to the number of free cells in the row
            col: list containing a value for each column of the sudoku, corresponding to the number of free cells in the column
            block: bidimensional list containing a value for each block of the sudoku, corresponding to the number of free cells in the block
        """
        board_size = game_state.board.board_height()

        free_cells = {
            "row": [],
            "col": [],
            "block": [],
        }
        # Calculate free cells in rows
        for y in range(board_size):
            empty_count = sum(
                1
                for x in range(board_size)
                if game_state.board.get(y, x) == SudokuBoard.empty
            )

            free_cells["row"].append(empty_count)

        # Calculate free cells in columns
        for x in range(board_size):
            empty_count = sum(
                1
                for y in range(board_size)
                if game_state.board.get(y, x) == SudokuBoard.empty
            )

            free_cells["col"].append(empty_count)

        # Calculate free cells in blocks
        for block_i in range(board_size // game_state.board.region_height()):
            empty_list = []
            for block_j in range(board_size // game_state.board.region_width()):
                empty_count = 0
                for row, col in block_range(
                    row=block_i * game_state.board.region_height(),
                    col=block_j * game_state.board.region_width(),
                    board=game_state.board,
                ):
                    if game_state.board.get(row, col) == SudokuBoard.empty:
                        empty_count += 1
                empty_list.append(empty_count)
            free_cells["block"].append(empty_list)

        return free_cells

    def order_moves(
        self, board: SudokuBoard, moves_list: list[Move], free_cells: dict
    ) -> tuple[list[Move], list[Move]]:
        """
        Sort a list of moves by their priority. Priority is based on the number of regions the move allows to complete.
        Moves that complete three regions have top priority and are therefore placed at the top of the list,
        followed by those that complete two regions and those that complete one.
        Moves that do not complete any region have no priority, and are returned in a separate list.

        :param board: The current board.
        :param moves_list: List of moves that have to be sorted.
        :param free_cells: dictionary containing the number of free cells for each region. It's used to assign priority to moves.

        :return: Two lists, one with the moves with priority, sorted, the other with the moves without priority.
        """

        # Assign priority to initial moves based on how many regions each move completes
        priority = {
            0: [],  # None priority
            1: [],
            2: [],
            3: [],
        }

        for move in moves_list:
            row = move.i
            col = move.j
            block_i, block_j = block_index(row, col, board)

            # Count how many regions the move completes
            # sum counts the number of True values in the list
            regions_to_be_completed = sum([
                free_cells["row"][row] == 1,
                free_cells["col"][col] == 1,
                free_cells["block"][block_i][block_j] == 1
            ])

            priority[regions_to_be_completed].append(move)

        return priority[3] + priority[2] + priority[1], priority[0]

    def initialize_stored_data(self, lower_limit: float, upper_limit: float) -> SavedDataT:
        """
        Initialize stored data.
        :param lower_limit: lower limit of the range of filling rate
                            values in which the list of moves that make Sudoku unsolvable is calculated.
        :param upper_limit: upper limit of the range of filling rate
        :return: initial values for the dictionary containing the stored data.
        """
        return {
            "unsolvable_moves": [],
            "check_unsolvable_range": (
                lower_limit,
                upper_limit,
            ),  # Check if a move makes the sudoku unsolvable only after
                # the board is 20% complete and before is 85% complete
            "unsolvable_list_building_started": False,
            "unsolvable_list_building_finished": False,
        }

    def calculate_unsolvable_moves(
        self, game_state: GameState, moves: list[Move], data: SavedDataT
    ) -> None:
        """
        Calculate the list of unsolvable moves, simulating each move in the list moves and checking
        if the new gamestate represents a Sudoku impossible to solve.

        :moves: list of possible unsolvable moves.
        :data: SavedDataT dictionary containing the list of unsolvable moves.
        """
        for move in moves:
            game_state.board.put(move.i, move.j, move.value)
            if is_unsolvable(game_state.board):
                data["unsolvable_moves"].append(move)
            game_state.board.put(move.i, move.j, SudokuBoard.empty)

    def update_unsolvable_moves(self, data: dict, moves: StateMatrixT) -> None:
        """
        Update the list of unsolvable moves,
        checking if they are still present in the list of legal moves
        calculated at the beginning of each turn.

        :data: dictionary containing the list of unsolvable moves
        :moves: list of legal moves
        """
        updated_moves = []
        for move in data["unsolvable_moves"]:
            if move in moves["initial"]:
                updated_moves.append(move)
                moves["initial"].remove(move)

        data["unsolvable_moves"] = updated_moves

    def compute_best_move(self, game_state: GameState) -> None:
        """
        Computes the best move for the agent and proposes it.

        It initially proposes a random move, then sorts the moves based on their priority and proposes the one with the highest priority.
        It then searches for the best move with minimax search.

        Since the turn time is not known it proposes the best move found so far by iteratively deepening the search, starting with depth 2.
        There is no need to search with depth 1, as moves are already sorted by priority.

        If the best move found so far leads to a negative result (calculated by the evaluation function),
        it proposes a move that makes the sudoku impossible to solve, in order to pass the turn without entering new values.

        The list of moves that make Sudoku impossible is calculated when the board is 20% full,
        and is saved by the game engine so that it is also available in the next rounds.
        At each turn the list must be updated, to ensure that all the moves on it are still legal.
        If the time limit of the turn does not allow the list of moves that invalidate the sudoku to be calculated,
        the agent tries to calculate it again when the board is 30% full.
        If even then the time available is not enough, it waits for the board to be 40% full, and so on.
        When the list is empty (all moves have been used or are no longer legal), it is recalculated

        :param game_state: The current game state. Describes the board, scores and move history.
        """
        board_size = game_state.board.board_height()

        # Generate possible moves
        initial_moves = self.find_initial_moves(game_state)

        # Propose a random move (initial, avoid timeout)
        self.propose_move(initial_moves[0])

        # Order initial moves by priority
        free_cells = self.calculate_free_cells(game_state)
        priority_moves, non_priority_moves = self.order_moves(game_state.board, initial_moves, free_cells)

        # Initial moves are sorted by priority | set(initial_moves) == set(priority_moves + non_priority_moves)
        initial_moves = priority_moves + non_priority_moves

        # Propose move with the highest priority (initial, avoid timeout)
        self.propose_move(initial_moves[0])

        # Initialize legal moves matrix with all moves being illegal
        legal_moves = [
            [[False for v in range(board_size + 1)] for col in range(board_size)]
            for row in range(board_size)
        ]

        # Mark initial moves as legal in the matrix
        for move in initial_moves:
            legal_moves[move.i][move.j][move.value] = True

        # Initialize state matrix
        state_matrix: StateMatrixT = {
            "initial": initial_moves,
            "legal_count": len(initial_moves),
            "legal": legal_moves,
        }

        data: SavedDataT = self.load()
        if data is None:  # if no data is saved, initialize it with default values
            data = self.initialize_stored_data(0.2, 0.85)
            self.save(data)

        # If the previous attempt to build unsolvable moves list failed,
        # then increase the lower limit of the filling range by 10% (this number was chosen arbitrarily)
        if (
            data["unsolvable_list_building_started"] is True
            and data["unsolvable_list_building_finished"] is False
        ):
            data["check_unsolvable_range"] = (
                data["check_unsolvable_range"][0] + 0.1,
                data["check_unsolvable_range"][1],
            )
            data["unsolvable_list_building_started"] = False

            self.save(data)

        filling_rate = calculate_filling_rate(free_cells)
        # First turn or unsolvable move list is empty (all unsolvable moves have been used or are not legal anymore)
        if (
            len(data["unsolvable_moves"]) == 0
            and filling_rate > data["check_unsolvable_range"][0]
            and filling_rate < data["check_unsolvable_range"][1]
        ):
            data["unsolvable_list_building_started"] = True
            self.save(data)

            # Only moves that not complete any regions can invalidate the game
            self.calculate_unsolvable_moves(game_state, non_priority_moves, data)

            data["unsolvable_list_building_finished"] = True
            self.save(data)

        # Check which unsolvable moves can still be used and remove them from moves
        if len(data["unsolvable_moves"]) > 0:
            self.update_unsolvable_moves(data, state_matrix)
            self.save(data)

        # Iteratively increase the search depth of minimax
        best_move: tuple[Move, float] | None = None
        for depth_limit in range(2, len(state_matrix["initial"]), 1):
            # Evaluate different moves based on minimax
            for move in state_matrix["initial"]:
                # Player_index is 0 or 1 (first or second player)
                player_index = game_state.current_player() - 1

                value = self.minimax(
                    game_state=game_state,
                    move=move,
                    state_matrix=state_matrix,
                    current_player=player_index,  # Current player is self
                    maximizing_player=player_index,  # Maximising own score
                    depth=depth_limit,
                )

                if best_move is None or value > best_move[1]:
                    best_move = (move, value)
                    # If every explored move leads to a negative score, propose a unsolvable move
                    if best_move[1] < 0 and len(data["unsolvable_moves"]) > 0:
                        self.propose_move(data["unsolvable_moves"][0])
                    else:
                        self.propose_move(best_move[0])
