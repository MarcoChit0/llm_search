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

    def configure(self) -> None:
        print(get_available_entries("model"))
        print(get_available_entries("state_evaluator"))
        print(get_available_entries("successor_generator"))
        print(get_available_entries("solver"))
        self.add_argument("-cm", "--class_model", choices=get_available_entries("model"), default="Qwen2.5-3B-Instruct")
        self.add_argument("-csg", "--class_successor_generator", choices=get_available_entries("successor_generator"), default="propose")
        self.add_argument("-cse", "--class_state_evaluator", choices=get_available_entries("state_evaluator"), default="vote")
        self.add_argument("-cs", "--class_solver", choices=get_available_entries("solver"), default="beam-search")
        self.add_argument("-t", "--max_output_tokens", default=1000)
        self.add_argument("-c", "--candidate_count", default=1)
        self.add_argument("--do_sample", default=True, action="store_true")
        self.add_argument("-temp", "--temperature", default=0.7)
        self.add_argument("-s", "--steps", default=3)
        self.add_argument("-l", "--load_in_8bit", default=False, action="store_true")

    
if __name__ == "__main__":
    parser = Parser()

    import argcomplete 

    argcomplete.autocomplete(parser)
    parser.parse_args()

    model = get_registered_class(parser.class_model, "model").from_config({
        "model_name": parser.class_model,
        "model_config": {"load_in_8bit": parser.load_in_8bit},
        "tokenizer_config": {}
    })
    generation_args = {
        "max_output_tokens": parser.max_output_tokens,
        "candidate_count": parser.candidate_count,
        "do_sample": parser.do_sample,
        "temperature": parser.temperature,
    }
    successor_generator = get_registered_class(parser.class_successor_generator, "successor_generator").from_config({
        "model": model,
        "text_generation_args": generation_args
    })
    state_evaluator = get_registered_class(parser.class_state_evaluator, "state_evaluator").from_config({
        "model": model,
        "text_generation_args": generation_args
    })
    solver = get_registered_class(parser.class_solver, "solver").from_config({
        "successor_generator": successor_generator,
        "state_evaluator": state_evaluator,
        "steps": parser.steps
    })
    
    initial_state = State("1 2 4 6")

    solver.solve(initial_state)


