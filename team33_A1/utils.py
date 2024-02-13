from typing import Iterator

from competitive_sudoku.sudoku import SudokuBoard, GameState, Move


def next_player(current_player: int) -> int:
    """Returns the next player."""
    return (current_player + 1) % 2


def region_range(
    *, row: int, col: int, board: SudokuBoard
) -> Iterator[tuple[int, int]]:
    """Return the range of cells in a region."""
    # TODO: Preprocess regions for faster lookup
    region_width, region_height = board.region_width(), board.region_height()

    region_start = {
        "x": (col // region_width) * region_width,
        "y": (row // region_height) * region_height,
    }

    region_end = {
        "x": (col // region_width + 1) * region_width,
        "y": (row // region_height + 1) * region_height,
    }

    for row in range(region_start["y"], region_end["y"]):
        for col in range(region_start["x"], region_end["x"]):
            yield row, col


def is_illegal(*, move: Move, for_state: GameState) -> bool:
    """
    Returns whether a move is illegal.

    A move is illegal if it puts a duplicate value in a region, row or column.
    Illegal moves are not allowed and will result in a loss.
    """

    # Check if square is empty
    if for_state.board.get(move.i, move.j) != SudokuBoard.empty:
        return True

    # Check if move is in taboo list
    # Idea: Hashmap would be faster (constant time lookup)
    if move in for_state.taboo_moves:
        return True

    # Check duplicate value in row
    if any(
        for_state.board.get(row, move.j) == move.value
        for row in range(for_state.board.board_height())
    ):
        return True

    # Check duplicate value in column
    if any(
        for_state.board.get(move.i, col) == move.value
        for col in range(for_state.board.board_width())
    ):
        return True

    # Lastly check values in the region
    return any(
        for_state.board.get(row, col) == move.value
        for row, col in region_range(row=move.i, col=move.j, board=for_state.board)
    )


def is_possible(*, move: Move, for_state: GameState):
    """Returns which moves are possible.

    All returned moves are on empty squares and not in the taboo list.
    Taboo moves are moves that would result in an unsolvable board and would thus be rejected.
    When making such a move, it will be added to the taboo list.
    Illegal moves are also removed.
    """
    return move not in for_state.taboo_moves and not is_illegal(
        for_state=for_state, move=move
    )
