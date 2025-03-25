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
    def solve(self, **kwargs) -> State:
        raise NotImplementedError
    

class BeamSearchSolver(Solver):    
    def solve(self, **kwargs) -> State:
        log_file = kwargs.get("log_file", None)
        
        try:
            initial_state = self._environment.get_initial_state()
        except Exception as e:
            if log_file:
                log_file.write(f"Error in get_initial_state: {e}\n")
            raise ExpectedError("Error when retrieving initial state") from e
        
        steps = self.__dict__.get("steps")
        states = [initial_state]
        heapq.heapify(states)
        for i in range(steps):
            state = heapq.heappop(states)
            if log_file:
                log_file.write(f"beam-search(state={state}, step={i})\n")
            
            try:
                successors: list[State] = self._environment.expand(state)
            except Exception as e:
                if log_file:
                    log_file.write(f"Error in expand for {state}: {e}\n")
                raise ExpectedError(f"Error when expanding state {state} : {e}") from e
            
            try:
                self._environment.evaluate(successors)
            except Exception as e:
                if log_file:
                    log_file.write(f"Error in evaluate for successors from {state}: {e}\n")
                raise ExpectedError(f"Error during evaluation of successors : {e}") from e
            
            for succ in successors:
                heapq.heappush(states, succ)

        result = heapq.heappop(states)
        return result
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["beam-search"]


class DepthFirstSearchSolver(Solver):
    def solve(self, **kwargs) -> State | None:
        log_file = kwargs.get("log_file", None)
        try:
            initial_state = self._environment.get_initial_state()
        except Exception as e:
            if log_file:
                log_file.write(f"Error in get_initial_state: {e}\n")
            raise ExpectedError(f"Error when retrieving initial state : {e}") from e

        steps = self.__dict__.get("steps")
        symmetry_level = self.__dict__.get("symmetry_level")
        states_explored_by_depth = [set() for _ in range(steps + 1)]
        explored_states = set()
        stack = [(initial_state, steps)]
        
        while stack:
            state, step = stack.pop()
            if log_file:
                log_file.write(f"dfs(state={state}, steps={step})\n")
            
            try:
                if self._environment.is_goal_state(state):
                    return state
            except Exception as e:
                if log_file:
                    log_file.write(f"Error in is_goal_state for {state}: {e}\n")
                raise ExpectedError(f"Error when checking if state is goal [{state}] : {e}") from e

            if step == 0:
                continue

            if state in explored_states:
                continue

            if symmetry_level and any(
                state.is_symmetric(explored_state, symmetry_level, self._environment._model)
                for explored_state in states_explored_by_depth[step]
            ):
                if log_file:
                    log_file.write(f"Symmetry detected [{symmetry_level}]: {state}\n")
                continue

            try:
                successors = self._environment.expand(state)
            except Exception as e:
                if log_file:
                    log_file.write(f"Error in expand for {state}: {e}\n")
                raise ExpectedError(f"Error when expanding state [{state}] : {e}") from e

            states_explored_by_depth[step].add(state)
            explored_states.add(state)
            for succ in successors:
                if log_file:
                    log_file.write(f"\t{state} ---[{state.get_action_to_child(succ)}]--> {succ}\n")
                stack.append((succ, step - 1))
        return None

    @classmethod
    def get_entries(cls) -> list[str]:
        return ["depth-first-search", "dfs"]