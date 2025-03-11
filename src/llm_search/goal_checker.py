from llm_search.register import *
from llm_search.models import *
from llm_search.state import State

class GoalChecker(Register):
    registry = GOAL_CHECKER_REGISTRY
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abc.abstractmethod
    def is_goal(self, state:State) -> bool:
        raise NotImplementedError

class ModelBasedGoalChecker(GoalChecker, ModelBasedClass):
    def get_prompt(self, state):
        prompt = """Does the """