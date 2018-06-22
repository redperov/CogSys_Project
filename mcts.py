"""
Implementation of the Monte Carlo Tree Search algorithm.
"""
import numpy as np
from valid_action import PythonValidActions
from random import choice

NUM_OF_SIMULATIONS = 3
C = 1.4
MAX_ROLLOUT_DEPTH = 3
valid_actions_getter = None
sim_services = None
sim_simulations_max_tries = 0
sim_max_simulation_depth = 0
sim_action_max_tries = 0


# class State(object):
#
#     def __init__(self, str_state):
#         """
#         Constructor.
#         :param str_state: string pddl representation of the state.
#         """
#         self._str_state = str_state


class Node(object):

    def __init__(self, str_state, parent, valid_actions=None, applied_action=None):
        """
        Constructor.
        :param str_state: string pddl representation of the state.
        :param parent: parent state
        """
        self._state = str_state
        self._parent = parent
        self._children = None
        self._valid_actions = valid_actions
        self._applied_action = applied_action
        self._visit_count = 0
        self._win_score = 0

    def get_state(self):
        return self._state

    def get_parent(self):
        return self._parent

    def get_children(self):

        # Check if the current state tried to create any children.
        if self._children is None:
            self._children = []

            # Create a child for every available action.
            for action in self.get_valid_actions():

                # The state on which the action will be applied.
                state = self._state

                # Used for checking whether the action succeeded.
                prev_state = self._state

                # Counts the number of times an action failed.
                counter = 0

                # TODO maybe there is a need for try except here
                while counter < sim_action_max_tries:

                    # Apply the action on the state.
                    sim_services.parser.apply_action_to_state(action, state)
                    self._children.append(Node(state, self, applied_action=action))
                    counter += 1

                    # Check if the state has changed.
                    # TODO comparison might not be working
                    if state != prev_state:
                        break

        return self._children

    def get_visit_count(self):
        return self._visit_count

    def increase_visit_count(self):
        self._visit_count += 1

    def add_win_score(self, score):
        self._win_score += score

    def get_win_score(self):
        return self._win_score

    # def get_all_possible_states(self):
    #     pass

    def get_valid_actions(self):

        if self._valid_actions is None:
            self._valid_actions = valid_actions_getter.get(self._state)

        return self._valid_actions

    def get_applied_action(self):
        return self._applied_action


def init_helper_objects(services, simulations_max_tries, max_simulation_depth, action_max_tries):
    global valid_actions_getter, sim_services, sim_simulations_max_tries, sim_max_simulation_depth, sim_action_max_tries
    sim_services = services
    valid_actions_getter = PythonValidActions(sim_services.parser, sim_services.perception)
    sim_simulations_max_tries = simulations_max_tries
    sim_max_simulation_depth = max_simulation_depth
    sim_action_max_tries = action_max_tries


def monte_carlo_tree_search(pddl_state, valid_actions):

    root = Node(pddl_state, None, valid_actions=valid_actions)

    for i in xrange(NUM_OF_SIMULATIONS):
        leaf = traverse(root)  # leaf = unvisited node
        simulation_result = rollout(leaf)
        back_propagate(leaf, simulation_result)

    return best_child(root)


def traverse(node):

    depth = 0

    while fully_expanded(node) and depth < sim_max_simulation_depth:
        node = best_uct(node)
        depth += 1

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

    curr_depth = 0
    while non_terminal(node) and curr_depth < MAX_ROLLOUT_DEPTH:
        node = rollout_policy(node)
        curr_depth += 1

    return get_result(node)


def non_terminal(node):
    # TODO implement the line below
    # if reached one of the goal states:
    #     return True

    if len(node.get_valid_actions()) == 0:
        return True

    return False


def rollout_policy(node):

    return choice(node.children)


def get_result(node):
    # TODO implement the line below
    # TODO maybe if stopped because reached max depth return x and if got to dead end return y?
    # if reached one of the goal states:
    #     return 1
    return 0


def back_propagate(node, result):

    # Check if node is root.
    if node.get_parent() is None:
        return

    update_stats(node, result)
    back_propagate(node.parent, result)


def update_stats(node, result):
    node.increase_visit_count()
    node.add_win_score(result)


def best_child(node):

    # Pick the child with the highest number of visits.
    visit_count_list = [child.get_visit_count() for child in node.get_children()]
    child = node.get_children()[np.argmax(visit_count_list)]

    return child.get_applied_action()
