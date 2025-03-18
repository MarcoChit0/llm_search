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
        
    @abc.abstractmethod
    def expand (self, state: State) -> list[State]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_task(self, index:int) -> State:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_model_response_correct(self, initial_state: State, final_state: State | None) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate(self, states: list[State]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_goal_state(self, state: State) -> bool:
        raise NotImplementedError