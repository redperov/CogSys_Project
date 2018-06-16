"""
Implementation of the Monte Carlo Tree Search algorithm.
"""
import numpy as np
from valid_action import PythonValidActions
from random import choice

NUM_OF_SIMULATIONS = 3
C = 1.4
valid_actions_getter = None


# class State(object):
#
#     def __init__(self, str_state):
#         """
#         Constructor.
#         :param str_state: string pddl representation of the state.
#         """
#         self._str_state = str_state


class Node(object):

    def __init__(self, str_state, parent):
        """
        Constructor.
        :param str_state: string pddl representation of the state.
        :param parent: parent state
        """
        self._state = str_state
        self._parent = parent
        self._children = []
        self._visit_count = 0
        self._win_score = 0

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_children(self):
        return self._children

    def get_visit_count(self):
        return self._visit_count

    def get_win_score(self):
        return self._win_score

    def get_all_possible_states(self):
        # TODO to understand how to use it, watch the tutorial of the algorithm
        pass


def monte_carlo_tree_search(root):

    for i in xrange(NUM_OF_SIMULATIONS):
        leaf = traverse(root) # leaf = unvisited node
        simulation_result = rollout(leaf)
        backpropagate(leaf, simulation_result)

    return best_child(root)


def traverse(node):

    while fully_expanded(node):
        node = best_uct(node)

    return pick_unvisited(node.get_children()) or node  # in case no children are present / node is terminal


def fully_expanded(node):
    """
    Checks if all the node's children are visited.
    :param node: node
    :return: boolean
    """
    for child in node.get_children():
        if child.get_visit_count() != 0:
            return False

    return True


def best_uct(node):

    choices_weights = [
        (c.get_win_score() / (c.get_win_score())) + C * np.sqrt((2 * np.log(node.get_win_score()) /
                                                                 (c.get_win_score())))
        for c in node.children]

    return node.children[np.argmax(choices_weights)]


def pick_unvisited(node):

    children = node.get_children()

    # TODO if using numpy array, change to .size()
    if len(children) == 0:
        return None

    for child in children:
        if children.get_visit_count() == 0:
            return child

    return None


def rollout(node):

    while non_terminal(node):
        node = rollout_policy(node)

    return result(node)


def non_terminal(node):
    # TODO implement, need to consider both available children and the max depth I want to check
    pass


def rollout_policy(node):

    return choice(node.children)


def result(node):
    # TODO implement, need to consider both available children and the max depth I want to check
    pass


def backpropagate(node, result):

    # Check if node is root.
    if node.get_parent() is None:
        return

    node.stats = update_stats(node, result)
    backpropagate(node.parent)


def update_stats(node, result):
    # TODO implement, need to consider both available children and the max depth I want to check
    pass


def best_child(node):

    # Pick the child with the highest number of visits.
    visit_count_list = [child.get_visit_count() for child in node.get_children()]
    return node.get_children()[np.argmax(visit_count_list)]
