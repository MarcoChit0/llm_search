import pandas as pd
from llm_search.environments.environment import *
from llm_search.models import *
import csv


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
        response = self._model.generate_text(prompt, candidate_count=1)
        self._initial_state =  State(response[0])

    def is_model_response_correct(self, initial_state: State, final_state: State | None) -> bool:
        if final_state is None:
            return False
        return final_state._data == self._df.iloc[initial_state._data]["solution"]

    def save_results(self, final_state, file_pointer) -> None:
        path = []
        cur = final_state
        while cur is not None:
            path.append(cur)
            cur = cur._parent
        path.reverse()
        for i in range(len(path)):
            row = self._model.get_statistics()
            row.update({
                "problem": self._task._problem,
                "solution": self._task._solution,
                "answer": self._task._answer,
                "index": self._task._index,
                "response": path[i]._data,
                "reasoning_step": i,
                "action": path[i].get_action_to_child(path[i + 1]) if i < len(path) - 1 else ""
            })
            dict_to_csv(row, file_pointer)
        


    @classmethod
    def get_entries(cls) -> list[str]:
        return ["math", "math-500", "math500"]
    
    def generate_math_solution_prompt(self) -> str:
        if type(self._model) == LlamaModel:
            prompt = """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\n"""
        else:
            prompt = """Solve the following math problem efficiently and clearly. Reason step by step, and put your final answer within \\boxed{}.\n\n"""    
        return prompt


    def expand(self, state: State) -> list[str]:
        path = []
        cur = state
        while cur is not None:
            path.append(cur)
            cur = cur._parent
        path.reverse()
        
        prompt += "Problem Specification:\n" + self._task._problem + "\n\n"
        for i in range(len(path)):
            prompt += f"Reasoning attempt {i}:\n" + path[i]._data + "\n\n"
        prompt += "Refine the reasoning based on the provided attempts and present your final answer clearly."

        print("Prompt:")
        print(prompt)
        response = self._model.generate_text(prompt)
        print("Response:")
        print(response)
        successors = []
        for i in range(len(response)):
            successors.append(State(response[i], parent=state, action=f"Action #{i}"))
        return successors
        
    def wrap_state_evaluation_prompt(self, state: State) -> str:
        state_evaluator = self.__dict__.get("state_evaluator")
        if state_evaluator != "vote":
            raise ValueError("Invalid state evaluator for this environment.")
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
            assert isinstance(states, list) and len(states) > 1, "Invalid input for vote evaluation."
            parent_state:State = states[0]._parent
            if parent_state is None:
                raise ValueError("Missing the argument parent_state for vote evaluation.")
            voted_states = self._model.generate_text(self.wrap_state_evaluation_prompt(parent_state))
            states_batch_votes = {action:0 for action in parent_state._children.keys()}
            for voted_state in voted_states:
                if voted_state in states_batch_votes:
                    states_batch_votes[voted_state] += 1
            max_votes = max(states_batch_votes.values())
            best_actions = [action for action, votes in states_batch_votes.items() if votes == max_votes]
            best_action = np.random.choice(best_actions)
            parent_state._children[best_action]._value = 0
        else:
            raise ValueError("Invalid state evaluator for this environment.")


    def is_goal_state(self, state: State) -> bool:
        return state._data == self._df.iloc[state._data]["solution"]