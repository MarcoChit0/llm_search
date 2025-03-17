import pandas as pd
from llm_search.environments.environment import *
from llm_search.models import *


class MathEnvironment(Environment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)

    def get_task(self, index: int) -> State:
        return State(self._df.iloc[index]["problem"])
    
    def is_model_response_correct(self, initial_state: State, final_state: State | None) -> bool:
        if final_state is None:
            return False
        return final_state._data == self._df.iloc[initial_state._data]["solution"]

    def wrap_successor_generator_prompt(self, model, state_generator) -> str:
        if type(model) == LlamaModel:
            """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."""
        else:
            """Please reason step by step, and put your final answer within \\boxed{}."""
