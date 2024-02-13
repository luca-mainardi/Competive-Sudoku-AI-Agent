"""Competitive Sudoku AI.

Adapted from /naive_player/sudokuai.py

A1: iterative deepening minimax search with alpha beta pruning.
A2: heuristic search.
A3: different improvements and a second strategy.
"""

from typing import Literal, TypedDict, TypeVar, Type, Union

from competitive_sudoku.sudoku import GameState, Move, SudokuBoard
from competitive_sudoku.sudokuai import SudokuAI
from random import shuffle, choice
from copy import deepcopy
from math import log

DEBUG = False

if DEBUG:
    from graphviz import Digraph
    import uuid


# Game specific logic not related to strategy stored in utils.py
from .utils import (
    block_range,
    calculate_move_score,
    is_illegal,
    next_player,
    PlayerID,
)

# Hyperparameters
MCTS_EXPLORATION = 2


def build_graph(
    node: "MCGTNode", max_player: PlayerID, graph=None, node_name=None, depth_limit=None
):
    """Visualise the MCTS game tree using graphviz"""
    if node_name is None:
        node_name = uuid.uuid4().hex

    if graph is None:
        graph = Digraph()
        graph.node(name=node_name, label=str(node).replace(";", "\n"))

    for child in node.children:
        child_name = uuid.uuid4().hex

        graph.node(
            name=child_name,
            label=str(child).replace(";", "\n"),
            color="green" if child.player == max_player else "black",
        )
        graph.edge(tail_name=node_name, head_name=child_name)

        if depth_limit is None or child.depth < depth_limit:
            build_graph(child, max_player, graph, child_name, depth_limit)

    return graph


class StateMatrixT(TypedDict):
    """Type of state matrix used in MCTS"""

    initial: list[Move]
    legal: list[list[list[bool]]]
    legal_moves_count: int


class MCGTNode:
    """Monte Carlo Game Tree Node"""

    def __init__(
        self,
        parent: Union["MCGTNode", None],
        move: Move,
        depth: int,
        player: PlayerID,
        visited=0,
        score=0,
    ):
        """MC Game tree node
        :param parent: parent node, allows backtracking
        :param move: move that was made to reach this node
        :param depth: depth of this node in the tree
        :param visited: number of times this node has been visited
        :param score: score of the game at this node
        """
        self.move = move
        self.visited = visited
        self.score = score
        self.parent = parent
        self.children: list[MCGTNode] = []  # Children are added in expansion phase
        self.depth = depth
        self.player: PlayerID = player

    @property
    def is_leaf(self):
        return not self.children

    @property
    def average_score(self):
        return self.score / self.visited if self.visited > 0 else 0

    @property
    def ucb(self):
        """Upper Confidence Bound of this node"""
        if self.visited == 0:
            return float("inf")

        return self.average_score + MCTS_EXPLORATION * (
            (log(self.parent.visited) / self.visited) ** 0.5
        )

    def __str__(self):
        return f"{str(self.move)}; n = {self.visited}; q = {self.score}"


class MCGameTree:
    def __init__(
        self,
        root: MCGTNode,
        game_state: GameState,
        state_matrix: StateMatrixT,
        max_player: PlayerID,
    ):
        """Monte Carlo Game Tree

        :param root: root node of the tree
        :param game_state: current game state
        :param state_matrix: state matrix used to track legal moves
        :param max_player: player to maximize score for
        """
        self.root = root
        self.maximizing_player = max_player
        self.game_state = game_state
        self.state_matrix = state_matrix

    def selection(self, node: MCGTNode) -> MCGTNode:
        """Select a leaf node to expand.

        Recursively select the child with the highest UCB until a leaf node is reached.

        :param node: node to start selection from.
        :return: leaf node.
        """
        if node.is_leaf:
            return node

        # Remove illegal moves (can occur because of reloading the tree)
        node.children = [
            child for child in node.children
            if self.state_matrix["legal"][child.move.i][child.move.j][child.move.value]
        ]

        # Select child with highest UCB
        return self.selection(max(node.children, key=lambda child: child.ucb))

    def expansion(self, node: MCGTNode) -> MCGTNode:
        """Expand a leaf node.

        A node is expanded when it is selected twice.
        Expansion adds all legal moves in the current game state as children of the node.

        :param node: node to expand.
        :return: node to simulate from.
        """
        # Apply all moves from node to the root (excluding)
        self.apply_move_on_node(node)

        #  If v is a terminal state of the game, move to Backpropagation phase
        if self.state_matrix["legal_moves_count"] == 0:
            return node

        # If n(v) = 0, move to Simulation phase with node v
        if node.visited == 0:
            return node

        # If n(v) > 0, add new states reached from legal moves in v
        # TODO use array of legal moves instead of matrix and inital moves?
        for move in self.state_matrix["initial"]:
            if self.state_matrix["legal"][move.i][move.j][move.value]:
                node.children.append(
                    MCGTNode(node, move, node.depth + 1, next_player(node.player))
                )

        return choice(node.children)

    def simulation(self, node: MCGTNode) -> int:
        """Simulate a game from a node.

        Simulate a game from a node by applying random moves until the game ends.

        :param node: node to simulate from.
        :return: score of the game. 1 when the maximizing player wins, 0 otherwise.
        """

        self.apply_move_on_node(node)
        player = next_player(node.player)
        while self.state_matrix["legal_moves_count"] > 0:  # until game ends
            # Select a random move
            # TODO use array of legal moves instead of matrix and inital moves?
            move = None
            while move is None:
                move = choice(self.state_matrix["initial"])
                if not self.state_matrix["legal"][move.i][move.j][move.value]:
                    move = None
            # Apply move
            self.apply_move(player, move)
            player = next_player(player)

        return (
            1
            if self.game_state.scores[self.maximizing_player]
            > self.game_state.scores[next_player(self.maximizing_player)]
            else 0
        )

    def backpropagation(self, node: MCGTNode | None, score: int) -> None:
        """Backpropagation the score from a node to the root

        :param node: node to start backpropagation from
        :param score: score to backpropagation
        """
        if node is None:
            return

        node.visited += 1
        node.score += score if node.player == self.maximizing_player else -score
        self.backpropagation(node.parent, score)

    def find_best_child(self) -> Move:
        """Find the best child of the root node

        Is based on the best average score of the children of the root node.

        :return: best move found so far
        """
        best_child = None
        best_avg_score = float("-inf")
        for child in self.root.children:
            if child.average_score > best_avg_score:
                best_avg_score = child.average_score
                best_child = child

        if best_child is None: # Happens in the first iteration
            return None

        return best_child.move

    def iterate(self) -> Move:
        """Perform one iteration of MCTS.

        Runs all phases of MCTS once.
        Selection, expansion, simulation and backpropagation.

        :return: best move found so far as defined in `find_best_child`.
        """
        # TODO optimise
        selected_leaf = self.selection(self.root)
        selected_leaf = self.expansion(selected_leaf)

        score = self.simulation(selected_leaf)
        self.backpropagation(selected_leaf, score)

        return self.find_best_child()

    def apply_move_on_node(self, node: MCGTNode) -> None:
        """Recursively apply moves to the game state"""
        # If node is root, no moves to apply (root is a dummy node)
        if node is self.root:
            return

        # If node parent is not a root, apply parent's moves first
        if node.parent is not None:
            self.apply_move_on_node(node.parent)

        self.apply_move(node.player, node.move)

    def apply_move(self, player: PlayerID, move: Move):
        self.game_state.board.put(move.i, move.j, move.value)
        self._apply_move_update_state_matrix(move)
        self.game_state.scores[player] += calculate_move_score(self.game_state, move)

    def _apply_move_update_state_matrix(self, move: Move):
        """Update the state matrix when a move is made"""
        # Update which moves are legal
        changes = 0
        for idx in range(self.game_state.board.N):
            # move invalidates another move if not already illegal, avoids double counting
            if self.state_matrix["legal"][idx][move.j][move.value]:
                self.state_matrix["legal"][idx][move.j][move.value] = False
                changes += 1

            if self.state_matrix["legal"][move.i][idx][move.value]:
                self.state_matrix["legal"][move.i][idx][move.value] = False
                changes += 1

        for row, col in block_range(
            row=move.i, col=move.j, board=self.game_state.board
        ):
            if self.state_matrix["legal"][row][col][move.value]:
                self.state_matrix["legal"][row][col][move.value] = False
                changes += 1

        self.state_matrix["legal_moves_count"] -= changes


class SudokuAI(SudokuAI):
    """
    Sudoku AI agent that computes a move for a given sudoku configuration.
    """

    def compute_best_move(self, game_state: GameState) -> None:
        """Computes the best move for the agent and proposes it.

        Will initially propose a random move, then evaluate different moves based on minimax search.
        Since the turn time is not known it will propose the best move found so far by iteratively deepening the search.

        :param game_state: The current game state. Describes the board, scores and move history.
        """
        board_size = game_state.board.board_height()
        num_blocks = board_size // game_state.board.region_height()

        # TODO also save game state matrix?
        # player_index is 0 or 1 (first or second player)
        player_index = game_state.current_player() - 1

        # List of initially legal moves, used as subset to avoid iterating over all moves
        initial_moves = [
            Move(i, j, value)
            for i in range(board_size)
            for j in range(board_size)
            for value in range(1, board_size + 1)
            if not is_illegal(move=Move(i, j, value), state=game_state)
        ]
        # Propose a random move (initial, avoid timeout)
        self.propose_move(choice(initial_moves))

        # Avoid repeated regeneration of legal moves by tracking their status
        legal_moves = [
            [[False for _ in range(board_size + 1)] for _ in range(board_size)]
            for _ in range(board_size)
        ]
        for move in initial_moves:
            legal_moves[move.i][move.j][move.value] = True

        state_matrix: StateMatrixT = {
            "initial": initial_moves,
            "legal": legal_moves,
            "legal_moves_count": len(initial_moves),
        }

        cache = self.load()
        if cache is None: # First turn
            # Create dummy root node for MCTS, the move "belongs" to the opponent
            dummy_root = MCGTNode(
                None, Move(-1, -1, -1), 0, next_player(player_index)
            )
            game_tree = MCGameTree(
                dummy_root,
                game_state,
                state_matrix,
                player_index,  # Maximizing player
            )
            cache = {}
            cache["tree"] = game_tree
            # cache["state_matrix"] = state_matrix
            self.save(cache)
        else: # Return with cached state
            game_tree = cache["tree"]
            # Go to new position in tree
            moves = game_state.moves[-2:] # Moves made since last turn (from argument)
            node = game_tree.root
            for move in moves:
                if move not in game_state.taboo_moves: # This move was indeed made since it is not taboo
                    # game_tree.apply_move(player_index, move) # Update state and state matrix
                    for child in node.children: # Move to child
                        if child.move == move:
                            node = child
                            node.parent = None
                            break
                    else: # Child not found
                        node = MCGTNode(None, move, node.depth + 1, next_player(node.player))
            game_tree.root = node
            self.save(cache)

        # Remove data not used in MCTS
        del game_state.taboo_moves
        del game_state.moves
        del game_state.initial_board

        # Keep improving the estimate of best move using MCTS until timeout
        while True:
            game_tree.state_matrix = deepcopy(state_matrix)
            game_tree.game_state = deepcopy(game_state)
            best_move = game_tree.iterate()

            if best_move is not None:
                self.propose_move(best_move)
                # Store game tree to reuse
                # TODO save only if timeout is close
                if game_tree.root.visited % 10 == 0:
                    self.save(cache)

            if DEBUG and game_tree.root.visited == 25:
                graph = build_graph(game_tree.root, player_index, depth_limit=5)
                graph.view()
                print("Graph built")
                exit()

