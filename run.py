import logging
import os
import datetime
import pandas as pd
import argcomplete
from contextlib import redirect_stdout
from transformers import BitsAndBytesConfig

# Your imports:
from parser import Parser
from llm_search.register import *
from llm_search.solver import *
from llm_search.state import *
from llm_search.models import *
from llm_search.environments.environment import *
from llm_search.environments.game24.environment import *
from llm_search.environments.math.environment import *

# Global variables
state_file = ""
parameters_file = ""

if __name__ == "__main__":

    def check_parameters(parser):
        global parameters_file

        new_parameters_df = parser.to_dataframe().reset_index(drop=True)

        if os.path.exists(parameters_file):
            previous_parameters_df = pd.read_csv(parameters_file).reset_index(drop=True)
            if new_parameters_df.empty or previous_parameters_df.empty:
                logging.warning("One of the parameters tables is empty. Treating parameters as changed.")
            else:
                new_params = new_parameters_df.iloc[0].to_dict()
                prev_params = previous_parameters_df.iloc[0].to_dict()
                # Check that both have the same keys
                if set(new_params.keys()) != set(prev_params.keys()):
                    logging.warning("Parameters have changed (different keys).")
                else:
                    identical = True
                    for key in new_params:
                        if new_params[key] != prev_params[key]:
                            logging.warning(f"Parameter '{key}' changed from {prev_params[key]} to {new_params[key]}.")
                            identical = False
                            break
                        if identical:
                            return True
        else:
            logging.info("Parameters file does not exist. Creating a new one.")

        # Save the new parameters (either file is missing or parameters have changed)
        new_parameters_df.to_csv(parameters_file, index=False)
        return False

    
    def go_to_exit(**kwargs):
        error_message = kwargs.get("error_message", None)
        if error_message:
            logging.error(error_message, exc_info=True)
        info_message = kwargs.get("info_message", None)
        if info_message:
            logging.info(info_message, exc_info=True)
        logging.shutdown()
        exit(1)


    def check_exception(e):
        current_exception = e
        while current_exception:
            if not isinstance(current_exception, ExpectedError):
                go_to_exit(error_message=f"Critical error detected: {current_exception}")
            current_exception = current_exception.__cause__

    def run(df_state, task_name, **kwargs):
        log_file = open(os.path.join(experiment_dir, f"{task_name}__log.log"), "w")
        if not log_file:
            raise CriticalError(f"Could not open log file for task {task_name}.")
        
        msg = f"Task {task_name} started executing at time {datetime.datetime.now()}"
        logging.info(msg)
        print(msg)

        try:
            env.initialize(**kwargs, log_file=log_file)
            final_state: State = solver.solve(log_file=log_file)
            if final_state:
                final_state.print(log_file)
            log = ""
            status = "complete"
        except Exception as e:
            check_exception(e)
            logging.info(f"Error when solving task {task_name}: {e}", exc_info=True)
            final_state = None
            log = str(e)
            status = "incomplete"
        
        msg = f"Task {task_name} finished executing at time {datetime.datetime.now()}."
        logging.info(msg); print(msg)
        
        stats = env.get_statistics(final_state)
        if kwargs.get("save_state", True):
            idx = kwargs.get("index", None)
            df_state = df_state[df_state["index"] != idx]
            new_result = pd.DataFrame([{
                **stats,
                "status": status,
                "log": log,
            }])
            df_state = pd.concat([df_state, new_result], ignore_index=True)
            df_state.to_csv(state_file, index=False)
        
        log_file.close()
        task_name = None


    def should_skip_instance(state_df, instance_index, load_state):
        instance_state = state_df[state_df["index"] == instance_index]
        return not instance_state.empty and load_state


    parser = Parser()
    argcomplete.autocomplete(parser)
    parser.parse_args()
    parser.check_args()


    experiment_dir = os.path.join("experiments", parser.path)
    os.makedirs(experiment_dir, exist_ok=True)
    state_file = os.path.join(experiment_dir, "state.csv")
    parameters_file = os.path.join(experiment_dir, "parameters.csv")

    all_experiments_log = os.path.join(experiment_dir, "all_experiments.log")
    logging.basicConfig(
        filename=all_experiments_log,
        filemode='a',
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO
    )


    text_generation_args = {
        "max_output_tokens": parser.max_output_tokens,
        "candidate_count": parser.candidate_count,
        "do_sample": parser.do_sample,
        "temperature": parser.temperature,
    }
    model = get_registered_class(parser.model, "model").from_config({
        "model_name": parser.model,
        "model_config": {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=parser.load_in_8bit),
            "device_map": parser.device
        },
        "tokenizer_config": {},
        "text_generation_args": text_generation_args
    })
    env = get_registered_class(parser.environment, "environment").from_config({
        "model": model,
        "successor_generator": parser.successor_generator,
        "state_evaluator": parser.state_evaluator,
        "candidate_count": parser.candidate_count,
        "budget": parser.budget
    })
    solver = get_registered_class(parser.solver, "solver").from_config({
        "environment": env,
        "steps": parser.steps,
        "symmetry_level": parser.symmetry_level
    })


    if os.path.exists(state_file):
        df_state = pd.read_csv(state_file)
    else:
        df_state = pd.DataFrame(columns=env.get_columns()) 


    if parser.instance:
        run(df_state, f"instance_[{parser.instance}]", instance=parser.instance, save_state=parser.save_state)
    else:
        same_parameters = check_parameters(parser)
        for idx in range(parser.index_start, parser.index_end + 1):
            if same_parameters and should_skip_instance(df_state, idx, parser.load_state):
                msg = f"Skipping instance {idx}: already completed."
                logging.info(msg)
                print(msg)
                continue
            run(df_state, f"index_[{idx}]", index=idx, save_state=parser.save_state)
    go_to_exit(info_message=f"Experiment {parser.path} completed successfully.")
                