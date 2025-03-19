from __future__ import annotations
from typing import Dict

class State:
    def __init__(self, data : str, parent : State | None = None, action: str | None = None) -> None:
        self._parent = parent
        if self._parent is not None:
            self._parent.add_child(action, self)
        else:
            self._is_initial_state = True
        self._data : str = data
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
    
    def print(self):
        data = {}
        current = self._parent
        prev = self
        while current is not None:
            action = current.get_action_to_child(prev)
            data[action] = prev._data
            prev = current
            current = current._parent
        data['@root'] = prev._data
        print(data)