import numpy as np
from llm_search.models import *
from llm_search.environments.environment import *
from llm_search.state import State
from llm_search.state_evaluator import *
from llm_search.register import *
import heapq

class Solver(Register):
    registry = SOLVER_REGISTRY
    def __init__(self, environment:Environment, state_evaluator:StateEvaluator, **kwargs):
        self._environment = environment
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
            successors = self._environment.expand(s)
            self._state_evaluator.evaluate_state_batch(successors)
            for succ in successors:
                heapq.heappush(states, succ)
        return heapq.heappop(states)
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["beam-search"]
    
class DepthFirstSearchSolver(Solver):
    def __init__(self, environment:Environment, state_evaluator:StateEvaluator, steps:int, symmetry_level:str, **kwargs):
        self.steps = steps
        self.symmetry_level = symmetry_level
        super().__init__(environment, state_evaluator, **kwargs)

    def check_symmetries(self, s1: State, s2: State) -> bool:
        if self.symmetry_level == "none":
            return s1._data == s2._data
        elif self.symmetry_level == "weak":
            return sorted(s1._data.split(' ')) == sorted(s2._data.split(' '))
        elif self.symmetry_level in ["medium", "strong"]:
            p_tokens = 0.5 if self.symmetry_level == "strong" else 0.75   
            assert hasattr(self._environment, "_model") and isinstance(self._environment._model, Model) and callable(self._environment._model.tokenize), "The successor generator does not have a valid _model attribute with a callable tokenize method."
            tokenized_s1 = sorted(self._environment._model.tokenize(s1._data))
            tokenized_s2 = sorted(self._environment._model.tokenize(s2._data))
            # Compute the multiset intersection count using two pointers over the sorted lists
            i = j = common = 0
            while i < len(tokenized_s1) and j < len(tokenized_s2):
                if tokenized_s1[i] == tokenized_s2[j]:
                    common += 1
                    i += 1
                    j += 1
                elif tokenized_s1[i] < tokenized_s2[j]:
                    i += 1
                else:
                    j += 1
            return common > len(tokenized_s1) * p_tokens
        else:
            raise ValueError(f"Invalid value for symmetry_level: {self.symmetry_level}")

    def dfs(self, state: State, steps: int, explored_states_by_depth: list[set[State]]) -> State | None:
        print(f"dfs({state._data}, {steps})")
        if steps == 0:
            return state if state._data == "24" else None
        
        for explored_state in explored_states_by_depth[steps]:
            print(f"Symmetry ({self.symmetry_level}) detected between states {state._data} and {explored_state._data}; prunning this branch.")
            if self.check_symmetries(state, explored_state):
                return None
    
        explored_states_by_depth[steps].add(state)
        print(f"Generating successors for state [{state._data}].")
        successors = self._environment.expand(state)
        for action, child in state._children.items():
            print(f"\t{state._data} ---[{action}]--> {child._data}")
        for succ in successors:
            result = self.dfs(succ, steps - 1, explored_states_by_depth)
            if result is not None:
                return result
        return None
            
    def solve(self, initial_state: State) -> State | None:
        states_explored_by_depth = [set() for _ in range(self.steps + 1)]
        return self.dfs(initial_state, self.steps, states_explored_by_depth)

    @classmethod
    def get_entries(cls) -> list[str]:
        return ["depth-first-search", "dfs"]