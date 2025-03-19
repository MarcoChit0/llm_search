from parser import Parser
from llm_search.solver import *
from llm_search.state import *
from llm_search.models import *
from llm_search.environments.environment import *
from llm_search.environments.game24.environment import *
from llm_search.environments.math.environment import *
import pandas as pd
import argcomplete     
import os
import datetime
from contextlib import redirect_stdout
from transformers import BitsAndBytesConfig

if __name__ == "__main__":
    parser = Parser()
    argcomplete.autocomplete(parser)
    parser.parse_args()
    parser.check_args()

    text_generation_args = {
        "max_output_tokens": parser.max_output_tokens,
        "candidate_count": parser.candidate_count,
        "do_sample": parser.do_sample,
        "temperature": parser.temperature,
    }
    model = get_registered_class(parser.model, "model").from_config({
        "model_name": parser.model,
        "model_config": {
            "quantization_config":BitsAndBytesConfig(load_in_8bit=parser.load_in_8bit),
            "device_map": parser.device},
        "tokenizer_config": {},
        "text_generation_args": text_generation_args
    })
    env = get_registered_class(parser.environment, "environment").from_config({
        "model": model,
        "successor_generator": parser.successor_generator,
        "state_evaluator": parser.state_evaluator,
    })
    solver = get_registered_class(parser.solver, "solver").from_config({
        "environment": env,
        "steps": parser.steps,
        "budget": parser.budget,
        "symmetry_level": parser.symmetry_level
    })

    experiment_dir = os.path.join("experiments", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(experiment_dir, exist_ok=True)

    def run_experiment(file_name):
        log_file = os.path.join(experiment_dir, file_name)
        results_file = os.path.join(experiment_dir, "results.csv")
        print(f"Running experiment for instance {file_name}")
        with open(log_file, "w") as log, redirect_stdout(log), open(results_file, "w+") as results:
            final_state = solver.solve()
            env.save_results(final_state, results)

    if parser.instance is not None:
        env.initialize(instance=parser.instance)
        initial_state = env.get_initial_state()
        run_experiment(f"instance_[{parser.instance}].log")
    else:
        for idx in range(parser.index_start, parser.index_end + 1):
            env.initialize(index=idx)
            initial_state = env.get_initial_state()
            run_experiment(f"index_[{idx}].log")


