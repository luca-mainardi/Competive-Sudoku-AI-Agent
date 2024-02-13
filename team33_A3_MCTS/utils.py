""" Module containing helper functions for the sudoku game.

These functions are used for game specific logic, such as checking if a move is legal or calculating the score of a move.
They do not contain any strategic logic specific to the AI agent.
"""

from typing import Iterator, Type, Literal
from competitive_sudoku.sudoku import SudokuBoard, GameState, Move


PlayerID: Type = Literal[1, 2]


def next_player(current_player: PlayerID) -> PlayerID:
    """Returns the next player.

    :param current_player: the current player.
    :return: the next player.
    """
    return (current_player + 1) % 2


def block_index(row: int, col: int, board) -> tuple[int, int]:
    """Transform in which block a certain coordinate is.

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
    """Return the range of indices in a block.

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
    """Check if a move completes any regions and returns the score earned.

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
