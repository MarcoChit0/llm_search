import numpy as np
from llm_search.models import *
from llm_search.successor_generator import *
from llm_search.state import State
from llm_search.state_evaluator import *
from llm_search.register import *
import heapq

# params = {
#     "model_name": "Qwen2.5-3B-Instruct",
#     "model_config": { "load_in_8bit": True},
#     "tokenizer_config": {}
# }
# m = get_model(**params)
# print(params)
# generation_args = {
#     "max_new_tokens": 1000,
#     "num_return_sequences": 5,
#     "do_sample": True,
#     "temperature": 0.7,
# }

# params = {
#     "model_name": "gemini-2.0-flash",
# }
# m = get_model(**params)
# generation_args = {
#     "max_output_tokens": 1000,
#     "candidate_count": 1,
#     "temperature": 0.7,
# }

# successor_generator = ProposeModelBasedSuccessorGenerator(m, generation_args)
# state_evaluator = VoteModelBasedStateEvaluator(m, generation_args)

class Solver(Register):
    registry = SOLVER_REGISTRY
    def __init__(self, sucessor_generator:SuccessorGenerator, state_evaluator:StateEvaluator, **kwargs):
        self._sucessor_generator = sucessor_generator
        self._state_evaluator = state_evaluator
        super().__init__(**kwargs)

    @abc.abstractmethod
    def solve(self, initial_state:State) -> State:
        raise NotImplementedError


class BeamSearchSolver(Solver):    
    def solve(self, initial_state:State):
        steps = self.__dict__.get("steps")
        states = [initial_state]
        heapq.heapify(states)
        for i in range(steps):
            s = heapq.heappop(states)
            successors = self._sucessor_generator.generate_successors(s)
            self._state_evaluator.evaluate_state_batch(successors)
            for succ in successors:
                heapq.heappush(states, succ)
        return heapq.heappop(states)
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["beam-search"]