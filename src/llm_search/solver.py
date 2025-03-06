import numpy as np
from models import *
from llm_search.successor_generator import ProposeModelBasedSuccessorGenerator
from llm_search.state import State
from llm_search.state_evaluator import VoteModelBasedStateEvaluator
import heapq

params = {
    "model_name": "Qwen2.5-3B-Instruct",
    "model_config": { "load_in_8bit": True},
    "tokenizer_config": {}
}
m = get_model(**params)
print(params)
generation_args = {
    "max_new_tokens": 1000,
    "num_return_sequences": 5,
    "do_sample": True,
    "temperature": 0.7,
}

# params = {
#     "model_name": "gemini-2.0-flash",
# }
# m = get_model(**params)
# generation_args = {
#     "max_output_tokens": 1000,
#     "candidate_count": 1,
#     "temperature": 0.7,
# }



successor_generator = ProposeModelBasedSuccessorGenerator(m, generation_args)
state_evaluator = VoteModelBasedStateEvaluator(m, generation_args)

states = [State( "1 2 4 6")]
heapq.heapify(states)
for i in range(3):
    s = heapq.heappop(states)
    print("-------------\nState:")
    s.print()
    successors = successor_generator.generate_successors(s)
    print("Successors:")
    for s_ in successors:
        s_.print()
    state_evaluator.evaluate_state_batch(successors)
    for succ in successors:
        heapq.heappush(states, succ)
print("Final state:")
heapq.heappop(states).print()