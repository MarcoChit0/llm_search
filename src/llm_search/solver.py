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
    def solve(self) -> State:
        initial_state = self._environment.get_initial_state()
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
    if symmetry_level == "weak":
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
    def solve(self) -> State | None:
        initial_state = self._environment.get_initial_state()
        steps = self.__dict__.get("steps")
        symmetry_level = self.__dict__.get("symmetry_level")
        states_explored_by_depth = [set() for _ in range(self.steps + 1)]
        explored_states = set()
        budget = self.__dict__.get("budget")
        stack = [(initial_state, steps)]
        while stack:
            state, step = stack.pop()
            print(f"dfs(state={state._data}, steps={step}, budget={budget})")
            
            if self._environment.is_goal_state(state): 
                return state

            if step == 0 or budget == 0: 
                continue

            if state in explored_states:
                continue
            
            if symmetry_level and any(check_symmetries(self._environment, symmetry_level, state, explored_state)
                for explored_state in states_explored_by_depth[step]):
                print(f"Symmetry detected [{symmetry_level}]: {state._data}")
                continue
            
            budget -= 1
            explored_states.add(state)
            states_explored_by_depth[step].add(state)
            print(f"Expanding {state._data}")
            successors = self._environment.expand(state)
            for succ in successors:
                print(f"\t{state._data} ---[{state.get_action_to_child(succ)}]--> {succ._data}")
                stack.append((succ, step - 1))
        return None

    @classmethod
    def get_entries(cls) -> list[str]:
        return ["depth-first-search", "dfs"]