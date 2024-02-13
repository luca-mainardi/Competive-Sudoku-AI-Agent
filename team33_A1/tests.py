import unittest

from competitive_sudoku.sudoku import SudokuBoard
from team33_A1.utils import region_range


class RegionRange(unittest.TestCase):
    def test_symmetrical_region(self):
        """Test that the region is correct when the region is symmetrical for board with width 4 and height 4."""

        expected = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for_region = region_range(row=0, col=0, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

        for_region = region_range(row=1, col=1, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

        expected = [(2, 2), (2, 3), (3, 2), (3, 3)]
        for_region = region_range(row=2, col=2, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

        for_region = region_range(row=3, col=3, board=SudokuBoard(2, 2))
        self.assertEqual(expected, list(for_region))

    def test_asymmetrical_region_region(self):
        """Test that the region is correct when the region is asymmetrical for board with width 4 and height 4."""

        expected = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        for_region = region_range(row=0, col=0, board=SudokuBoard(3, 2))
        self.assertEqual(expected, list(for_region))

        for_region = region_range(row=2, col=1, board=SudokuBoard(3, 2))
        self.assertEqual(expected, list(for_region))

        for_region = region_range(row=4, col=3, board=SudokuBoard(3, 2))
        expected = [(3, 2), (3, 3), (4, 2), (4, 3), (5, 2), (5, 3)]
        self.assertEqual(expected, list(for_region))


if __name__ == "__main__":
    unittest.main()
