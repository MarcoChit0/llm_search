from __future__ import annotations
import pandas as pd
from llm_search.environments.environment import *
from llm_search.models import *
import csv

class MathState(State):
    def __init__(self, data: list[dict[str, str]], parent: State | None = None, action: str | None = None) -> None:
        super().__init__(data, parent, action)
    
    def is_symmetric(self, other: MathState, symmetry_level: str, model:Model) -> bool:
        if symmetry_level == "weak":
            p_tokens = 0.80
        elif symmetry_level in "medium":
            p_tokens = 0.60
        elif symmetry_level == "strong":
            p_tokens = 0.40
        else:
            raise CriticalError(f"Symmetry : Invalid symmetry level [{symmetry_level}].")
        return tokens_similarity(model.tokenize(self._data), model.tokenize(other._data), p_tokens)


class MathTask(Task):
    def __init__(self, problem:str, answer:str, solution: str, index: int | None = None, **kwargs):
        super().__init__(problem, answer, index, **kwargs)
        self._solution = solution

class MathEnvironment(Environment):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._df = pd.read_json("hf://datasets/HuggingFaceH4/MATH-500/test.jsonl", lines=True)

    def _initialize(self, **kwargs) -> None:
        index = kwargs.get("index")
        self._task = MathTask(self._df.iloc[index]["problem"], self._df.iloc[index]["answer"], self._df.iloc[index]["solution"], index)
        prompt = self.generate_math_solution_prompt() + self._task._problem
        try:
            response = self.generate(prompt, candidate_count=1, is_chat=True, chat_history=[])
        except Exception as e:
            raise CriticalError(f"_Initialize : {e}") from e
        self._initial_state =  MathState(response[0])

    def is_model_response_correct(self, **kwargs) -> bool:
        final_state = kwargs.get("final_state", None)
        if final_state is None:
            return False
        return self._df.iloc[self._task._index]["solution"] in final_state._data[-1]["content"]

    def get_statistics(self, final_state:MathState|None, **kwargs) -> dict[str, object]:
        row = self._model.get_statistics()
        error_message = kwargs.get("error_message", None)
        row.update({
            "problem": self._task._problem,
            "solution": self._task._solution,
            "answer": self._task._answer,
            "index": self._task._index,
            "correct": self.is_model_response_correct(final_state=final_state),
            "final_state": final_state._data if final_state else None,
            "message": error_message,
        })
        return row
        
    def get_columns(self): 
        return ["problem", "solution", "answer", "index", "correct", "final_state", "message"]

    @classmethod
    def get_entries(cls) -> list[str]:
        return ["math", "math-500", "math500"]
    
    def generate_math_solution_prompt(self) -> str:
        if type(self._model) == LlamaModel:
            prompt = """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\n"""
        else:
            prompt = """Solve the following math problem efficiently and clearly. Reason step by step, and put your final answer within \\boxed{}.\n\n"""    
        return prompt


    def expand(self, state: MathState, **kwargs) -> list[str]:
        log_file = kwargs.get("log_file", None)
        if log_file: log_file.write(f"expand({state})")
        chat_history = state._data
        prompt = (
            "Review the previous iterations and enhance the solution by integrating all available insights. "
            "Provide a clear, efficient, and logically structured step-by-step explanation, detailing all reasoning and intermediate steps. "
            "Conclude your explanation with the final answer enclosed in \\boxed{...}."
        )
        try:
            response = self.generate(prompt, is_chat=True, chat_history=chat_history)
        except Exception as e:
            raise ExpectedError(f"Expand : {e}") from e
        if log_file: log_file.write(f" -> produced {len(response)} successors\n")
        successors = []
        for i in range(len(response)):
            successors.append(MathState(response[i], parent=state, action=f"Action #{i}"))
        return successors
        
    def wrap_state_evaluation_prompt(self, state: State) -> str:
        state_evaluator = self.__dict__.get("state_evaluator")
        if state_evaluator != "vote":
            raise CriticalError("Wrap : Invalid state evaluator for this environment.")
        path = []
        cur = state
        while cur is not None:
            path.append(cur)
            cur = cur._parent
        path.reverse()
        prompt = "For the following problem, vote for the best solution, informing only its number.\n\n"
        prompt += "Problem Specification:\n" + path[0]._data + "\n\n"
        for i in range(len(state._parent._children)):
            prompt += f"Attempt {i}:\n" + path[i+1]._data + "\n\n"
        return prompt

    def evaluate(self, states: list[State]) -> None:
        state_evaluator = self.__dict__.get("state_evaluator")
        if state_evaluator == "vote":
            if not isinstance(states, list) or len(states) < 1:
                raise CriticalError(f"Evaluate : Invalid input for vote evaluation [{type(states)} | {len(states)}].")
            parent_state:State = states[0]._parent
            if parent_state is None:
                raise CriticalError("Evaluate : Missing parent_state.")
            try:
                prompt = self.wrap_state_evaluation_prompt(parent_state)
                voted_states = self.generate(prompt)
            except Exception as e:
                raise ExpectedError(f"Evaluate : {e}") from e
            states_batch_votes = {action:0 for action in parent_state._children.keys()}
            for voted_state in voted_states:
                if voted_state in states_batch_votes:
                    states_batch_votes[voted_state] += 1
            max_votes = max(states_batch_votes.values())
            best_actions = [action for action, votes in states_batch_votes.items() if votes == max_votes]
            best_action = np.random.choice(best_actions)
            parent_state._children[best_action]._value = 0
        else:
            raise CriticalError("Evaluate : State evaluator not implemented.")


    def is_goal_state(self, state: State) -> bool:
        chat_history = state._data
        last_response = chat_history[-1]['content']
        if self._df.iloc[self._task._index]["solution"] in last_response:
            return True
        return False