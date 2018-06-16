"""
Implementation of the Monte Carlo Tree Search algorithm.
"""
import numpy as np
from valid_action import PythonValidActions

NUM_OF_SIMULATIONS = 3
C = 1.4
valid_actions_getter = PythonValidActions()


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

    return pick_unvisited(node.get_children()) or node # in case no children are present / node is terminal

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
        for c in node.children
    ]

    return node.children[np.argmax(choices_weights)]


def rollout(node):

    while non_terminal(node):
        node = rollout_policy(node)

    return result(node)

def rollout_policy(node):

    return pick_random(node.children)

def backpropagate(node, result):

   if is_root(node):
       return

   node.stats = update_stats(node, result)
   backpropagate(node.parent)

def best_child(node):

    pick child with highest number of visits
