from __future__ import annotations
from typing import Dict
import abc
from llm_search.models import Model

class State(abc.ABC):
    def __init__(self, data : object, parent : State | None = None, action: str | None = None) -> None:
        self._parent = parent
        if self._parent is not None:
            self._parent.add_child(action, self)
        else:
            self._is_initial_state = True
        self._data : object = data
        self._value : float = float('inf')
        self._children : Dict[str, State] = {}
    
    def add_child(self, action:str, child:State):
        self._children[action] = child
    
    def __lt__(self, other:State) -> bool:
        return self._value < other._value

    def get_action_to_child(self, child:State) -> str:
        for action, c in self._children.items():
            if c == child:
                return action
        return None
    
    def __str__(self) -> str:
        return str(self._id)

    def print(self, log_file = None):
        data = []
        cur = self
        while cur is not None:
            parent = cur._parent
            if parent is not None:
                action = parent.get_action_to_child(cur)
                data.append([str(parent), action, str(cur)])
            cur = cur._parent
        data.append([str(parent), action, str(cur)])
        if log_file: 
            for parent, action, child in reversed(data):
                log_file.write(f"{parent} ---[{action}]--> {child}\n")
        else:
            for parent, action, child in reversed(data):
                print(f"{parent} ---[{action}]--> {child}")
    
    @property
    def _id(self):
        return id(self)
    
    @abc.abstractmethod
    def is_symmetric(self, other:State, symmetry_level:str, model: Model) -> bool:
        raise NotImplementedError

def tokens_similarity(tokens_list_a: list, tokens_list_b: list, threshold_ratio: float) -> bool:
    sorted_a = sorted(tokens_list_a)
    sorted_b = sorted(tokens_list_b)
    i = j = common_tokens = 0
    while i < len(sorted_a) and j < len(sorted_b):
        if sorted_a[i] == sorted_b[j]:
            common_tokens += 1
            i += 1
            j += 1
        elif sorted_a[i] < sorted_b[j]:
            i += 1
        else:
            j += 1
    return common_tokens > len(sorted_a) * threshold_ratio