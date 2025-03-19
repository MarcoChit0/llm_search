from llm_search.state import State
import re
from collections import Counter
import pandas as pd
from llm_search.models import *
from llm_search.register import *
import numpy as np

class Task(abc.ABC):
    def __init__(self, problem:str, answer:str, index:int|None = None, **kwgars):
        self._problem = problem
        self._answer = answer
        self._index = index

class Environment(Register):
    registry = ENVIRONMENT_REGISTRY
    def __init__(self, model, **kwargs):
        self.__dict__.update(kwargs)
        self._model = model
        self._initial_state : State | None = None
        self._task : Task | None = None
        
    def reset(self):
        self._initial_state = None
        self._task = None
        self._model.reset()
    
    def initialize(self, **kwargs) -> None:
        self.reset()
        self._initialize(**kwargs)
        assert self._initial_state is not None, "The initial state was not set."
        assert self._task is not None, "The task was not set."
    
    @abc.abstractmethod
    def _initialize(self, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def expand (self, state: State) -> list[State]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_model_response_correct(self, task:Task, final_state: State | None) -> bool:
        raise NotImplementedError
    
    @abc.abstractmethod
    def evaluate(self, states: list[State]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def is_goal_state(self, state: State) -> bool:
        raise NotImplementedError
    
    def get_initial_state(self) -> State:
        return self._initial_state

    def get_task(self) -> Task:
        return self._task
    
    @abc.abstractmethod
    def save_results(self, final_state: State|None, file_pointer) -> None:
        raise NotImplementedError

import csv 
def dict_to_csv(data: dict, file_pointer) -> None:
    writer = csv.DictWriter(file_pointer, fieldnames=list(data.keys()))
    # If file is at the beginning, write header
    if file_pointer.tell() == 0:
        writer.writeheader()
    writer.writerow(data)