import os
import sys
import tap

from llm_search.solver import *
from llm_search.state import *
from llm_search.state_evaluator import *
from llm_search.successor_generator import *
from llm_search.models import *


class Parser(tap.Tap):
    # registries
    class_model: str
    class_successor_generator: str
    class_state_evaluator:str
    class_solver:str
    # generation parameters
    max_output_tokens: int
    candidate_count: int
    do_sample: bool
    temperature: float
    load_in_8bit: bool
    # solver parameters
    steps: int
    symmetry_level: str

    def configure(self) -> None:
        self.add_argument("-cm", "--class_model", choices=get_available_entries("model"), default="Qwen2.5-3B-Instruct")
        self.add_argument("-csg", "--class_successor_generator", choices=get_available_entries("successor_generator"), default="propose")
        self.add_argument("-cse", "--class_state_evaluator", choices=get_available_entries("state_evaluator"), default="vote")
        self.add_argument("-cs", "--class_solver", choices=get_available_entries("solver"), default="beam-search")
        self.add_argument("-t", "--max_output_tokens", default=1000)
        self.add_argument("-c", "--candidate_count", default=1)
        self.add_argument("--do_sample", default=True, action="store_false")
        self.add_argument("-temp", "--temperature", default=0.7)
        self.add_argument("-s", "--steps", default=3)
        self.add_argument("-l", "--load_in_8bit", default=True, action="store_false")
        self.add_argument("-sl", "--symmetry_level", choices=["none", "medium", "weak", "strong"] , default="weak")

    
if __name__ == "__main__":
    parser = Parser()

    import argcomplete 
    import re

    argcomplete.autocomplete(parser)
    parser.parse_args()

    model = get_registered_class(parser.class_model, "model").from_config({
        "model_name": parser.class_model,
        "model_config": {"load_in_8bit": parser.load_in_8bit},
        "tokenizer_config": {},
    })
    text_generation_args = {
        "max_output_tokens": parser.max_output_tokens,
        "candidate_count": parser.candidate_count,
        "do_sample": parser.do_sample,
        "temperature": parser.temperature,
    }
    text_generation_args = text_generation_args_mapping(type(model), text_generation_args)

    successor_generator = get_registered_class(parser.class_successor_generator, "successor_generator").from_config({
        "model": model,
        "text_generation_args": text_generation_args
    })
    state_evaluator = get_registered_class(parser.class_state_evaluator, "state_evaluator").from_config({
        "model": model,
        "text_generation_args": text_generation_args
    })
    solver = get_registered_class(parser.class_solver, "solver").from_config({
        "successor_generator": successor_generator,
        "state_evaluator": state_evaluator,
        "steps": parser.steps,
        "model": model,
        "symmetry_level": parser.symmetry_level
    })
    
    initial_state = State("1 2 4 6")

    final_state = solver.solve(initial_state)
    # verify whether the final state is correct
    if final_state is not None:
        path = []
        curr = final_state
        while curr is not None:
            path.append(curr)
            curr = getattr(curr, "_parent", None)  # Assumes each state has a 'parent' attribute.
        path.reverse()

        # Verify each transition in the solution path.
        for i in range(len(path) - 1):
            parent_state = path[i]
            child_state = path[i + 1]

            parent_nums = sorted(list(map(int, parent_state._data.split())))
            child_nums = sorted(list(map(int, child_state._data.split())))

            # In each move, one operation combines two numbers into one.
            if len(child_nums) != len(parent_nums) - 1:
                raise AssertionError("Invalid state transition: incorrect number of remaining numbers.")

            action = parent_state.get_action_to_child(child_state)
            numbers = re.findall(r'\d+', action)
            if len(numbers) < 3:
                raise AssertionError("Invalid action format; expected at least three numbers in the action: " + action)
            operand1, operand2, result = map(int, numbers[:3])

            # Compute the expected result.
            if '+' in action:
                computed = operand1 + operand2
            elif '-' in action:
                computed = operand1 - operand2
            elif '*' in action:
                computed = operand1 * operand2
            elif '/' in action:
                computed = operand1 / operand2
            else:
                raise ValueError("Invalid operator in action: " + action)

            if computed != result:
                raise AssertionError(f"Invalid computation in action \"{action}\": {operand1} ? {operand2} = {computed} (expected {result}).")

            # Simulate the operation on the parent's numbers.
            if operand1 in parent_nums and operand2 in parent_nums:
                temp_nums = parent_nums.copy()
                temp_nums.remove(operand1)
                temp_nums.remove(operand2)
                temp_nums.append(result)
                if sorted(temp_nums) != child_nums:
                    raise AssertionError("State transition did not result in the expected set of numbers.")
            else:
                raise AssertionError("Operands specified in the action are not present in the parent's state.")

        # Verify that the final state meets the 24 game win condition.
        final_numbers = list(map(int, path[-1]._data.split()))
        if len(final_numbers) != 1 or final_numbers[0] != 24:
            raise AssertionError("Final state does not equal 24. The solution is invalid.")

