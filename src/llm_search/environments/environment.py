from llm_search.state import State
import re
from collections import Counter
import pandas as pd
from llm_search.models import *
from llm_search.register import *

class Environment(Register):
    registry = ENVIRONMENT_REGISTRY
    def __init__(self, model, **kwargs):
        self.__dict__.update(kwargs)
        self._model = model
        self._available_actions:dict[State, list[str]] = {}
    
    
    def expand(self, state: State) -> list[State]:
        avaialble_actions = self.get_available_actions(state)
        successors = []
        for action in avaialble_actions:
            successor = self.apply_action(state, action)
            successors.append(successor)
        return successors

    def get_available_actions(self, state:State) -> list[str]:
        if state not in self._available_actions:
            actions = self.get_actions(state)
            available_actions = []
            for action in actions:
                if self.is_available_action(action):
                    available_actions.append(action)
            self._available_actions[state] = available_actions
        return self._available_actions[state]

    @abc.abstractmethod
    def get_task(self, index:int) -> State:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_model_response_correct(self, initial_state: State, final_state: State | None) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_actions(self, state:State) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, state: State, action: str) -> State:
        raise NotImplementedError

    @abc.abstractmethod
    def is_available_action(self, action:str) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def wrap_successor_generator_prompt(self) -> str:
        raise NotImplementedError