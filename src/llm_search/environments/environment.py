from llm_search.state import State
import re
from collections import Counter
import pandas as pd
from llm_search.models import *
from llm_search.register import *
import numpy as np

class Environment(Register):
    registry = ENVIRONMENT_REGISTRY
    def __init__(self, model, **kwargs):
        self.__dict__.update(kwargs)
        self._model = model
        self._available_actions:dict[State, list[str]] = {}
        
    @abc.abstractmethod
    def get_available_actions(self, state:State) -> list[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def apply_action(self, state:State, action:str) -> State:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_task(self, index:int) -> State:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_model_response_correct(self, initial_state: State, final_state: State | None) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate(self, x: State | list[State]) -> None:
        raise NotImplementedError
