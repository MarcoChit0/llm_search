from llm_search.state import State
import re
from collections import Counter
import pandas as pd

class Environment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def get_task(self, index:int) -> State:
        raise NotImplementedError
    
    def is_model_response_correct(self, initial_state: State, final_state: State | None) -> bool:
        raise NotImplementedError