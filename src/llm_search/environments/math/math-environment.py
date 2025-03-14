import pandas as pd
from llm_search.environments.environment import *


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
