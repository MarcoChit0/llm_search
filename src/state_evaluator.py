from state import *
from models import *
import numpy as np
import abc

class StateEvaluator(abc.ABC):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    @abc.abstractmethod
    def evaluate_state_batch(self, state_batch: list[State]) -> None:
        raise NotImplementedError

class ModelBasedStateEvaluator(StateEvaluator):
    def __init__(self, model: Model, text_generation_args: dict, **kwargs):
        self._model: Model = model
        self._text_generation_args:dict = text_generation_args
        super().__init__(**kwargs)
    
    @abc.abstractmethod
    def get_evaluation_prompt(self, state:State) -> str:
        raise NotImplementedError

class VoteModelBasedStateEvaluator(ModelBasedStateEvaluator):
    def get_evaluation_prompt(self, state:State) -> str: 
        vote_prompt = '''Below, there are several candidate steps for the Input {input}. Please, vote for the most promising one to reach the target number 24, using only basic arithmetic operations (addition, subtraction, multiplication, and division). The response should only be the selected candidate step. Candidate steps:
{candidate_steps}
Vote:'''
        return vote_prompt.format(input=state._data, candidate_steps='\n'.join(list(state._children.keys())))

    '''
    The successor state whose action is the most voted by the model receives a value of 0. The remaining states maintain ther values as infinity.
    '''
    def evaluate_state_batch(self, state_batch:list[State]) -> None:
        state_batch[0].print()
        parent_state:State = state_batch[0]._parent
        if parent_state is None:
            raise ValueError("Missing the argument parent_state for vote evaluation.")
        voted_states = self._model.generate_text(self.get_evaluation_prompt(parent_state), **self._text_generation_args)
        states_batch_votes = {action:0 for action in parent_state._children.keys()}
        for voted_state in voted_states:
            if voted_state in states_batch_votes:
                states_batch_votes[voted_state] += 1
        max_votes = max(states_batch_votes.values())
        best_actions = [action for action, votes in states_batch_votes.items() if votes == max_votes]
        best_action = np.random.choice(best_actions)
        parent_state._children[best_action]._value = 0