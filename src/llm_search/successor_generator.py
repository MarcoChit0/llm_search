from llm_search.state import *
from typing import Dict, List
from llm_search.models import *
import re
import abc

class SuccessorGenerator(Register):
    registry = SUCCESSOR_GENERATOR_REGISTRY
    
    def __init__(self,**kwargs) -> None:
        self._available_actions:Dict[State, list[str]] = {}
        super().__init__(**kwargs)

    @abc.abstractmethod
    def get_actions(self, state:State) -> list[str]:
        raise NotImplementedError

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
    def is_available_action(self, action:str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, state: State, action: str) -> State:
        raise NotImplementedError

    def generate_successors(self, state: State) -> List[State]:
        avaialble_actions = self.get_available_actions(state)
        successors = []
        for action in avaialble_actions:
            successor = self.apply(state, action)
            successors.append(successor)
        return successors



class ProposeModelBasedSuccessorGenerator(SuccessorGenerator, ModelBasedClass):

    def get_actions(self, state:State) -> list[str]:
        response = self._model.generate_text(self.get_prompt(state), **self._text_generation_args)
        print(response)
        actions = []
        for r in response:
            actions.extend(r.split('\n'))
        return actions

    def apply(self, state: State, action: str) -> State:
        successor_data = action.split('left: ')[1].replace(')', '').strip()
        return State(successor_data, state, action)

    @classmethod
    def get_entries(cls) -> list[str]:
        return ["propose"]

    def get_prompt(self, state:State) -> str:
        propose_prompt = """Given a list of numbers, propose possible next steps using basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/). Each step must involve exactly two numbers from the list, and the result should replace those two numbers in a new list.

Rules:
- Only use basic arithmetic operations.
- Each operation should be displayed in the format:
  number [operation] number = result (left: updated list)
- List each possible next step on a separate line.

Example:

Input:  2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)

Now, generate the possible next steps for the following input:

Input: {input}
Possible next steps:"""
        return propose_prompt.format(input=state._data)
    
    def is_available_action(self, action:str) -> bool:
        return re.match(
            r'^\s*\d+(?:\.\d+)?\s*[\+\-\/\*]\s*\d+(?:\.\d+)?\s*=\s*\d+(?:\.\d+)?\s*\(left:\s*(?:\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*?)\)\s*$',
            action
        )