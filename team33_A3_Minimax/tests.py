import unittest

from competitive_sudoku.sudoku import SudokuBoard, GameState, Move, TabooMove
from team33_A2.utils import block_range, is_illegal, calculate_move_score


class RegionRange(unittest.TestCase):
    def test_symmetrical_region(self):
        """Test that the region is correct when the region is symmetrical for board with width 4 and height 4."""

        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for_region = block_range(row=0, col=0, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

        for_region = block_range(row=1, col=1, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

        expected = [(2, 2), (2, 3), (3, 2), (3, 3)]
        for_region = block_range(row=2, col=2, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

        for_region = block_range(row=3, col=3, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

    def test_asymmetrical_region_region(self):
        """Test that the region is correct when the region is asymmetrical for board with width 4 and height 4."""

        expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        for_region = block_range(row=0, col=0, board=SudokuBoard(3, 2))
        self.assertEqual(expected, list(for_region))

        for_region = block_range(row=2, col=1, board=SudokuBoard(3, 2))
        self.assertEqual(expected, list(for_region))

        for_region = block_range(row=4, col=3, board=SudokuBoard(3, 2))
        expected = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)]
        self.assertEqual(expected, list(for_region))


class TestIsIllegal(unittest.TestCase):
    def setUp(self):
        self.board = SudokuBoard()
        self.state = GameState(self.board, self.board, [], [], [])

    def test_move_on_empty_square_is_not_illegal(self):
        """Test that a move on an empty square is not illegal."""
        move = Move(0, 0, 1)
        self.assertFalse(is_illegal(move=move, state=self.state))

    def test_move_on_filled_square_is_illegal(self):
        """Test that a move on a filled square is illegal."""
        self.board.put(0, 0, 1)
        move = Move(0, 0, 2)
        self.assertTrue(is_illegal(move=move, state=self.state))

    def test_move_in_taboo_list_is_illegal(self):
        """Test that a move in the taboo list is illegal."""
        move = Move(0, 0, 1)
        self.state.taboo_moves.append(TabooMove(0, 0, 1))
        self.assertTrue(is_illegal(move=move, state=self.state))

    def test_duplicate_value_in_row_is_illegal(self):
        """Test that a move with a duplicate value in the row is illegal."""
        self.board.put(0, 1, 1)
        move = Move(0, 0, 1)
        self.assertTrue(is_illegal(move=move, state=self.state))

    def test_duplicate_value_in_column_is_illegal(self):
        """Test that a move with a duplicate value in the column is illegal."""
        self.board.put(1, 0, 1)
        move = Move(0, 0, 1)
        self.assertTrue(is_illegal(move=move, state=self.state))

    def test_duplicate_value_in_region_is_illegal(self):
        """Test that a move with a duplicate value in the region is illegal."""
        self.board.put(1, 1, 1)
        move = Move(0, 0, 1)
        self.assertTrue(is_illegal(move=move, state=self.state))


class CalculateMoveScoreTests(unittest.TestCase):
    def setUp(self):
        self.board = SudokuBoard()
        self.state = GameState(self.board, self.board, [], [], [])

    def test_move_completes_no_regions(self):
        """Complete no regions."""
        move = (0, 0, 1)
        self.assertEqual(
            calculate_move_score(game_state=self.state, move=Move(*move)), 0
        )

    def test_move_completes_one_region(self):
        """Complete one row."""
        for i in range(1, self.board.board_width()):
            self.board.put(0, i, i)
        move = (0, 0, 9)
        self.board.put(*move)
        self.assertEqual(
            calculate_move_score(game_state=self.state, move=Move(*move)), 1
        )

    def test_move_completes_two_regions(self):
        """Complete one row and one column."""
        for i in range(1, self.board.board_width()):
            self.board.put(0, i, 1)
            self.board.put(i, 0, 1)

        move = (0, 0, 9)
        self.board.put(*move)
        self.assertEqual(
            calculate_move_score(game_state=self.state, move=Move(*move)), 3
        )

    def test_move_completes_three_regions(self):
        """Complete one row, one column and one block."""
        for i in range(1, self.board.board_width()):
            self.board.put(0, i, 1)
            self.board.put(i, 0, 1)

        for i, j in block_range(row=0, col=0, board=self.board):
            if i != 0 and j != 0:
                self.board.put(i, j, 1)

        move = (0, 0, 9)
        self.board.put(*move)
        self.assertEqual(
            calculate_move_score(game_state=self.state, move=Move(*move)), 7
        )


if __name__ == "__main__":
    unittest.main()
