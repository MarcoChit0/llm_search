from llm_search.state import *
import re
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
        self._candidate_count : int = kwargs.get("candidate_count", 1)
        self._budget : int | None = kwargs.get("budget", None)
        self._used_budget = 0
        self._log_file = kwargs.get("log_file", None)
    
    def generate(self, prompt, **kwargs):
        if self._budget is not None:
            if self._used_budget > 0:
                candidate_count = kwargs.pop("candidate_count", self._candidate_count)
                candidate_count = min(self._used_budget, candidate_count)
                response = self._model.generate(prompt, candidate_count=candidate_count, **kwargs)
                self._used_budget -= candidate_count
            else:
                raise ExpectedError("Budget exceeded.")
        else:
            response = self._model.generate(prompt, **kwargs)
        return response
        
    def reset(self):
        self._initial_state = None
        self._task = None
        self._model.reset()
        self._used_budget = self._budget
    
    def initialize(self, **kwargs) -> None:
        self._log_file = kwargs.get("log_file", None)
        self.reset()
        try:
            self._initialize(**kwargs)
        except Exception as e:
            raise CriticalError(f"Initialize : {e}") from e
        if self._initial_state is None:
            raise CriticalError("Initialize : The initial state was not set.")
        if self._task is None:
            raise CriticalError("Initialize : The task was not set.")
    
    @abc.abstractmethod
    def _initialize(self, **kwargs) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def expand (self, state: State, **kwargs) -> list[State]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def is_model_response_correct(self, **kwargs) -> bool:
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
    def get_statistics(self, final_state: State|None, **kwargs) -> dict[str, object]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_columns(self) -> list[str]:
        raise NotImplementedError

import csv 
def dict_to_csv(data: dict, file_pointer) -> None:
    writer = csv.DictWriter(file_pointer, fieldnames=list(data.keys()))
    # If file is at the beginning, write header
    if file_pointer.tell() == 0:
        writer.writeheader()
    writer.writerow(data)