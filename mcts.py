"""
Implementation of the Monte Carlo Tree Search algorithm.
"""
import numpy as np


class State(object):

    def __init__(self, str_state):
        """
        Constructor.
        :param str_state: string pddl representation of the state.
        """
        self._str_state = str_state
        self._visit_count = 0
        self._win_score = 0

    def get_all_posible_states(self):
        pass


class Node(object):

    def __init__(self, state, parent):
        """
        Constructor.
        :param state: state
        :param parent: parent state
        """
        self._state = state
        self._parent = parent
        self._children = []

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_children(self):
        return self._children

