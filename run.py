from parser import Parser
from llm_search.solver import *
from llm_search.state import *
from llm_search.models import *
from llm_search.environments.game24.environment import *
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

    model = get_registered_class(parser.model, "model").from_config({
        "model_name": parser.model,
        "model_config": {
            "quantization_config":BitsAndBytesConfig(load_in_8bit=parser.load_in_8bit),
            "attn_implementation":"flash_attention_2",
            "low_cpu_mem_usage":True, 
            "device_map": parser.device},
        "tokenizer_config": {},
    })
    text_generation_args = {
        "max_output_tokens": parser.max_output_tokens,
        "candidate_count": parser.candidate_count,
        "do_sample": parser.do_sample,
        "temperature": parser.temperature,
    }
    model = get_registered_class(parser.model, "model").from_config({
        "model_name": parser.model,
        "model_config": {"load_in_8bit": parser.load_in_8bit},
        "tokenizer_config": {},
        "text_generation_args": text_generation_args,
    })
    env = get_registered_class(parser.environment, "environment").from_config({
        "model": model,
        "successor_generator": parser.successor_generator,
        "state_evaluator": parser.state_evaluator
        
    })
    solver = get_registered_class(parser.solver, "solver").from_config({
        "environment": env,
        "steps": parser.steps,
        "model": model,
        "symmetry_level": parser.symmetry_level,
        "budget": parser.budget
    })

    experiment_dir = os.path.join("experiments", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(experiment_dir, exist_ok=True)
    results = []  

    def run_experiment(instance_id, initial_state, results, parser:Parser=parser):
        log_file = os.path.join(experiment_dir, f"instance_{instance_id}.log")
        with open(log_file, "w") as f, redirect_stdout(f):
            # print input information from parser
            print(f"Parser arguments: {parser}")
            print(f"Running experiment for instance {instance_id}")
            final_state = solver.solve(initial_state)
            correct = env.is_model_response_correct(initial_state, final_state)
        results.append({
            "Puzzle": initial_state._data,
            "Result": "Correct" if correct else "Incorrect" 
        })

    if parser.instance is not None:
        initial_state = State(parser.instance)
        run_experiment(f"new[{initial_state._data}]", initial_state, results)
    else:
        for i in range(parser.batch_index_start, parser.batch_index_end):
            initial_state = env.get_task(i)
            run_experiment(f"batch_[{i}]", initial_state, results)

    # Save all experiment results to a CSV file in the experiment directory
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(experiment_dir, "results.csv")
    results_df.to_csv(csv_file, index=False)
