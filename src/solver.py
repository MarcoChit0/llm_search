import numpy as np
from models import *
from successor_generator import *
from state import *
from state_evaluator import *
import heapq

# problem_definition = """Problem: Given a set of four numbers, determine whether it is possible to reach 24 using those numbers and the basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).

# problem_definition = """For each task, provide a consice solution with minimal explanation."""
# params = {
#     "model_name": "llama-3.2-1B-Instruct",
#     "model_config": {"torch_dtype": torch.bfloat16},
#     "tokenizer_config": {}
# }
# m = get_model(**params)
# print(params)
# generation_args = {
#     "max_new_tokens": 500,
#     "num_return_sequences": 5,
#     "do_sample": True,
#     "temperature": 0.7,
# }

params = {
    "model_name": "gemini-1.5-flash-8b",
}
m = get_model(**params)
generation_args = {
    "max_output_tokens": 500,
    "candidate_count": 5,
    "temperature": 0.7,
}



successor_generator = ProposeModelBasedSuccessorGenerator(m, generation_args)
state_evaluator = VoteModelBasedStateEvaluator(m, generation_args)

states = [State( "1 2 4 6")]
heapq.heapify(states)
for i in range(3):
    s = heapq.heappop(states)
    s.print()
    successors = successor_generator.generate_successors(s)
    state_evaluator.evaluate_state_batch(successors)
    for succ in successors:
        heapq.heappush(states, succ)
print("Final state:")
heapq.heappop(states).print()