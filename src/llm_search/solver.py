import numpy as np
from llm_search.models import *
from llm_search.environments.environment import *
from llm_search.state import State
from llm_search.register import *
import heapq

class Solver(Register):
    registry = SOLVER_REGISTRY
    def __init__(self, environment:Environment, **kwargs):
        self._environment = environment
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
            print(f"Step {i}: {s._data}")
            successors:list[State] = self._environment.expand(s)
            self._environment.evaluate(successors)
            for succ in successors:
                heapq.heappush(states, succ)
        return heapq.heappop(states)
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["beam-search"]


def check_symmetries(environment: Environment, symmetry_level: str, s1: State, s2: State) -> bool:
    if symmetry_level == "none":
        return s1._data == s2._data
    elif symmetry_level == "weak":
        return sorted(s1._data.split(' ')) == sorted(s2._data.split(' '))
    elif symmetry_level in ["medium", "strong"]:
        p_tokens = 0.5 if symmetry_level == "strong" else 0.75
        assert hasattr(environment, "_model") and isinstance(environment._model, Model) and callable(environment._model.tokenize), "The successor generator does not have a valid _model attribute with a callable tokenize method."
        tokenized_s1 = sorted(environment._model.tokenize(s1._data))
        tokenized_s2 = sorted(environment._model.tokenize(s2._data))
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
        raise ValueError(f"Invalid value for symmetry_level: {symmetry_level}")


class DepthFirstSearchSolver(Solver):
    def __init__(self, environment:Environment, **kwargs):
        super().__init__(environment, **kwargs)

    def dfs(self, state: State):
        steps = self.__dict__.get("steps")
        symmetry_level = self.__dict__.get("symmetry_level")
        states_explored_by_depth = [set() for _ in range(self.steps + 1)]
        budget = self.__dict__.get("budget")
        stack = [(state, steps)]
        while stack:
            s, step = stack.pop()
            print(f"dfs(state={s._data}, steps={step}, budget={budget})")
            
            if self._environment.is_goal_state(s): 
                return s

            if step == 0 or budget == 0: 
                continue
            
            if any(check_symmetries(self._environment, symmetry_level, s, explored_state)
                for explored_state in states_explored_by_depth[step]):
                print(f"Symmetry detected [{symmetry_level}]: {s._data}")
                continue
            
            budget -= 1
            states_explored_by_depth[step].add(s)
            print(f"Expanding {s._data}")
            successors = self._environment.expand(s)
            for succ in successors:
                print(f"\t{s._data} ---[{s.get_action_to_child(succ)}]--> {succ._data}")
                stack.append((succ, step - 1))
        return None
            
    def solve(self, initial_state: State) -> State | None:
        steps = self.__dict__.get("steps")
        self.symmetry_level = self.__dict__.get("symmetry_level")
        states_explored_by_depth = [set() for _ in range(self.steps + 1)]
        return self.dfs(initial_state)

    @classmethod
    def get_entries(cls) -> list[str]:
        return ["depth-first-search", "dfs"]