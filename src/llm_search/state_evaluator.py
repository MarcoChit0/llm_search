from llm_search.state import *
from llm_search.models import Model
from llm_search.register import *
import numpy as np
import abc

class StateEvaluator(Register): 
    registry = STATE_EVALUATOR_REGISTRY

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
        vote_prompt = """Given a list of candidate steps, select the best one to move toward the target number 24 using basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).  

Rules:  
- Choose only one candidate step.  
- The response must contain **only** the selected step.  

Example:  

Input:  2 8 8 14  
Candidate steps:  
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)

Vote: 14 - 8 = 6 (left: 2 6 8)  

Now, select the best step for the following input:  

Input: {input}  
Candidate steps:  
{candidate_steps}  

Vote:"""
        return vote_prompt.format(input=state._data, candidate_steps='\n'.join(list(state._children.keys())))

    '''
    The successor state whose action is the most voted by the model receives a value of 0. The remaining states maintain ther values as infinity.
    '''
    def evaluate_state_batch(self, state_batch:list[State]) -> None:
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
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["vote"]