import numpy as np
from llm_search.models import *
from llm_search.successor_generator import *
from llm_search.state import State
from llm_search.state_evaluator import *
from llm_search.register import *
import heapq

class Solver(Register):
    registry = SOLVER_REGISTRY
    def __init__(self, successor_generator:SuccessorGenerator, state_evaluator:StateEvaluator, **kwargs):
        self._sucessor_generator = successor_generator
        self._state_evaluator = state_evaluator
        super().__init__(**kwargs)

    @abc.abstractmethod
    def solve(self, initial_state:State) -> State:
        raise NotImplementedError


class BeamSearchSolver(Solver):    
    def solve(self, initial_state:State) -> State:
        steps = self.__dict__.get("steps")
        states = [initial_state]
        heapq.heapify(states)
        for i in range(steps):
            s = heapq.heappop(states)
            successors = self._sucessor_generator.generate_successors(s)
            self._state_evaluator.evaluate_state_batch(successors)
            for succ in successors:
                heapq.heappush(states, succ)
        return heapq.heappop(states)
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["beam-search"]
    
class DepthFirstSearchSolver(Solver):
    def solve(self, state:State):
        steps = self.__dict__.get("steps")
        stack = [state]
        successors = self._sucessor_generator.generate_successors(state)
        if 

    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["depth-first-search", "dfs"]