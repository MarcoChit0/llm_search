import os
import sys
import tap

from llm_search.solver import *
from llm_search.state import *
from llm_search.state_evaluator import *
from llm_search.successor_generator import *
from llm_search.models import *


class Parser(tap.Tap):
    model_name: str
    model_config: dict
    tokenizer_config: dict
    max_output_tokens: int
    candidate_count: int
    do_sample: bool
    temperature: float
    steps: int
    load_in_8bit: bool

    def configure(self) -> None:
        self.add_argument("-m","--model_name", choices=get_available_models(), type=str, required=True)
        self.add_argument("-mc","--model_config", type=dict, default={})
        self.add_argument("-tc","--tokenizer_config", type=dict, default={})
        self.add_argument("-t","--max_output_tokens", type=int, default=1000)
        self.add_argument("-c","--candidate_count", type=int, default=1)
        self.add_argument("--do_sample", type=bool, default=True, action="store_true")
        self.add_argument("-temp","--temperature", type=float, default=0.7)
        self.add_argument("-s","--steps", type=int, default=3)
        self.add_argument("-l","--load_in_8bit", type=bool, default=False, action="store_true")

if __name__ == "__main__":
    pass

def run():
    parser = Parser()

    import argcomplete 

    argcomplete.autocomplete(parser)
    parser.parse_args()

