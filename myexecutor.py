from pddlsim.local_simulator import LocalSimulator
from pddlsim.executors.executor import Executor
from pddlsim.planner import local
import sys


class MyExecutor(Executor):
    """
    My Executor.
    """
    def __init__(self):
        super(MyExecutor, self).__init__()
        self.services = None

    def initialize(self, services):
        self.services = services

    def next_action(self):

        # Check if reached all goals.
        if self.services.goal_tracking.reached_all_goals():
            return None

        # Get all the valid actions.
        options = self.services.valid_actions.get()


if __name__ == '__main__':

    domain_path = sys.argv[1]
    problem_path = sys.argv[2]

    print LocalSimulator(local).run(domain_path, problem_path, MyExecutor())
