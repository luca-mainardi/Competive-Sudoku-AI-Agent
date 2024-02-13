""" 
Module containing helper functions for the sudoku game.

These functions are used for game specific logic, such as checking if a move is legal or calculating the score of a move.
They do not contain any strategic logic specific to the AI agent.
"""

from typing import Iterator, TypedDict

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard


class StateMatrixT(TypedDict):
    """
    Type definition for the state matrix.

    :param initial: List of moves that were initially on the board
    :param legal: Array of shape (board_size, board_size, board_size + 1)
                  where legal[i, j, k] is True if Move(i, j, k) is legal
    :param legal_count: count of legal moves
    """
    initial: list[Move]
    legal: list[list[list[bool]]]
    legal_count: int


class SavedDataT(TypedDict):
    """
    Type definition for the saved data.

    :param unsolvable_moves: List of moves that make the board unsolvable
    :param check_unsolvable_range: When to calculate the unsolvable moves
    :param unsolvable_list_building_started: True if the unsolvable list building has started
    :param unsolvable_list_building_finished: True if the unsolvable list building has finished
    """

    unsolvable_moves: list[Move]
    check_unsolvable_range: tuple[float, float]
    unsolvable_list_building_started: bool
    unsolvable_list_building_finished: bool


def next_player(current_player: int) -> int:
    """
    Returns the next player.

    :param current_player: the current player.
    :return: the next player.
    """
    return (current_player + 1) % 2


def block_index(row: int, col: int, board) -> tuple[int, int]:
    """
    Transform in which block a certain coordinate is.

    Enumerates blocks in the same way as coordinates, top left is (0, 0).

    :param row: a row index inside the block.
    :param col: a column index inside the block.
    :param board: the board to which the indices belong.
    :return: the indices of the block in the board (vertical, horizontal).
    """
    return (
        row // board.region_height(),  # floor division
        col // board.region_width(),
    )


def block_range(*, row: int, col: int, board: SudokuBoard) -> Iterator[tuple[int, int]]:
    """
    Return the range of indices in a block.

    :param row: a row index inside the block.
    :param col: a column index inside the block.
    :param board: the board to which the indices belong.
    :return: an iterator of indices in the block.
    """
    region_width, region_height = board.region_width(), board.region_height()
    block_indices = block_index(row, col, board)

    region_start = {
        "x": block_indices[1] * region_width,
        "y": block_indices[0] * region_height,
    }

    region_end = {
        "x": (block_indices[1] + 1) * region_width,
        "y": (block_indices[0] + 1) * region_height,
    }

    for row in range(region_start["y"], region_end["y"]):
        for col in range(region_start["x"], region_end["x"]):
            yield row, col


def is_illegal(*, move: Move, state: GameState) -> bool:
    """
    Returns whether a move is illegal.

    A move is illegal if it puts a duplicate value in a position, block, row or column.
    Additionally, moves should not be 'taboo', meaning that they make the board unsolvable.
    Illegal moves are not allowed and will result in a loss.

    :param move: The move to be checked.
    :param state: The current state of the game.
    :return: Whether the move is illegal (True) or not (False).
    """

    # Check if square is empty
    if state.board.get(move.i, move.j) != SudokuBoard.empty:
        return True

    # Check if move is taboo
    if move in state.taboo_moves:
        return True

    # Check duplicate value in row
    if any(
            state.board.get(row, move.j) == move.value
            for row in range(state.board.board_height())
    ):
        return True

    # Check duplicate value in column
    if any(
            state.board.get(move.i, col) == move.value
            for col in range(state.board.board_width())
    ):
        return True

    # Lastly check for duplicate values in the region
    return any(
        state.board.get(row, col) == move.value
        for row, col in block_range(row=move.i, col=move.j, board=state.board)
    )


def calculate_move_score(game_state: GameState, move: Move) -> int:
    """
    Check if a move completes any regions and returns the score earned.

    Static method, uses less memoy sinds it does not need to be instantiated.

    :param game_state: The current game state. Describes the board, scores and move history.
    :param move: The move to be evaluated
    :return: The score earned by the move (0, 1, 3 or 7)
    """
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
    for row, col in block_range(row=move.i, col=move.j, board=game_state.board):
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


def calculate_filling_rate(free_cells: dict) -> float:
    """
    Calculate the percentage of occupied cells on the sudoku board.

    :param free_cells: A dictionary containing the number of free cells, for each region.
        row: list containing a value for each row of the sudoku, corresponding to the number of free cells in the row
        col: list containing a value for each column of the sudoku, corresponding to the number of free cells in the column
        block: bidimensional list containing a value for each block of the sudoku, corresponding to the number of free cells in the block
    """
    num_cells = len(free_cells["row"]) * len(free_cells["col"])
    num_empty_cells = 0
    for empty_in_row in free_cells["row"]:
        num_empty_cells += empty_in_row

    return (num_cells - num_empty_cells) / num_cells


def is_unsolvable(board: SudokuBoard) -> bool:
    """
    Determines whether a sudoku is impossible to solve, given a game state.
    It is used to create the unsolvable move list.

    :return: True if the sudoku does not have any solutions, False otherwise.

    """
    board_size = board.board_height()
    num_blocks = board_size // board.region_width()

    # Consider rows of each block
    for row_index in range(board_size):
        for block_ind in range(num_blocks):
            # List containing the values not present in the condidered row of the block
            missing_values = [val for val in range(1, board_size + 1)]

            block_row_index = row_index // board.region_height()
            # List containing the indexes of the rows in the block that are not considered
            missing_row_indexes = [
                index
                for index in range(
                    block_row_index * board.region_height(),
                    block_row_index * board.region_height() + board.region_height(),
                    )
                if index != row_index
            ]
            # List containing the indexes of the columns without values in the considered row of the block
            missing_col_indexes = [
                index
                for index in range(
                    block_ind * board.region_width(),
                    block_ind * board.region_width() + board.region_width(),
                    )
            ]
            # Build the lists of missing values and missing indexes
            for col_index in range(
                    block_ind * board.region_width(),
                    block_ind * board.region_width() + board.region_width(),
            ):
                value = board.get(row_index, col_index)
                if value != SudokuBoard.empty:
                    missing_values.remove(value)
                    missing_col_indexes.remove(col_index)

            # If the row of the block is empty, then it can't invalidate the sudoku
            if len(missing_values) == board.region_height():
                continue

            # Check the impossibility conditions of sudoku
            for missing_value in missing_values:
                # If the value missing in the block row is present in the same block,
                # then that value is not invalidating the sudoku
                found_in_block = False
                for row, col in block_range(
                        row=row_index,
                        col=block_ind * board.region_width(),
                        board=board,
                ):
                    if board.get(row, col) == missing_value:
                        found_in_block = True
                        break
                # Check the next missing value
                if found_in_block:
                    continue

                # Check invalidating values in rows (the missing value must be present in all missing rows)
                found_in_rows_count = 0
                for missing_row_index in missing_row_indexes:
                    for col in range(board_size):
                        if board.get(missing_row_index, col) == missing_value:
                            found_in_rows_count += 1
                # Check the next missing value
                if found_in_rows_count != len(missing_row_indexes):
                    continue

                # Check invalidating values in columns (the missing value must be present in all missing columns)
                found_in_cols_count = 0
                for missing_col_index in missing_col_indexes:
                    for row in range(board_size):
                        if board.get(row, missing_col_index) == missing_value:
                            found_in_cols_count += 1
                # Check the next missing value
                if found_in_cols_count != len(missing_col_indexes):
                    continue

                # If the value has been found in all rows and columns and is not in the block,
                # then the sudoku is impossible to solve
                return True

    # All the missing values have been checked and none of them is invalidating the sudoku
    return False
