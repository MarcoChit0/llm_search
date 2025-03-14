from llm_search.state import State
import re
from collections import Counter
import pandas as pd

class Environment:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self._path = "src/llm_search/environments/data.csv"
    
    def get_task(self, index:int) -> State:
        df = pd.read_csv(self._path)
        return State(df.iloc[index]["Puzzles"])
  
    def ground_truth(self,initial_state:State):
        def verify_recursively(l: list[int]):
            if len(l) == 1:
                if l[0] == 24:
                    return True
            for i in range(len(l)):
                for j in range(len(l)):
                    if i == j:
                        continue
                    for op in ['+', '-', '*', '/']:
                        if op == '/' and l[j] == 0:
                            continue
                        if op == '+':
                            new_value = l[i] + l[j]
                        elif op == '-':
                            new_value = l[i] - l[j]
                        elif op == '*':
                            new_value = l[i] * l[j]
                        elif op == '/':
                            new_value = l[i] / l[j]
                        new_l = l.copy()
                        new_l.remove(l[i])
                        new_l.remove(l[j])
                        new_l.append(new_value)
                        if verify_recursively(new_l):
                            return True
            return False
        
        return verify_recursively(list(map(int, initial_state._data.split())))


    def is_model_response_correct(self, initial_state: State, final_state: State | None):
        if final_state is None:
            if self.ground_truth(initial_state):
                print("There is a solution and the model did not found it.")
                return False
            else:
                return True

        final_state.print()

        # Verify that the final state is correct.
        path = []
        curr = final_state
        while curr is not None:
            path.append(curr)
            curr = getattr(curr, "_parent", None)
        path.reverse()

        # Verify each transition in the solution path.
        for i in range(len(path) - 1):
            parent_state = path[i]
            child_state = path[i + 1]

            parent_nums = sorted(list(map(int, parent_state._data.split())))
            child_nums = sorted(list(map(int, child_state._data.split())))

            if len(child_nums) != len(parent_nums) - 1:
                print(f"Invalid state transition: incorrect number of remaining numbers between states '{parent_state._data}' and '{child_state._data}'.")
                return False

            action = parent_state.get_action_to_child(child_state)
            numbers = re.findall(r'\d+', action)
            if len(numbers) < 3:
                print("Invalid action format; expected at least three numbers in the action: " + action)
                return False

            operand1, operand2, result = map(int, numbers[:3])

            if '+' in action:
                computed = operand1 + operand2
            elif '-' in action:
                computed = operand1 - operand2
            elif '*' in action:
                computed = operand1 * operand2
            elif '/' in action:
                computed = operand1 / operand2
            else:
                print("Invalid operator in action: " + action)
                return False

            if computed != result:
                print(f"Invalid computation in action \"{action}\": {operand1} ? {operand2} = {computed} (expected {result}).")
                return False

            parent_counter = Counter(parent_nums)
            if parent_counter[operand1] < 1 or parent_counter[operand2] < 1:
                print("Operands specified in the action are not present in the parent's state for action: " + action)
                return False

            parent_counter.subtract([operand1, operand2])
            parent_counter[result] += 1

            if parent_counter != Counter(child_nums):
                print("State transition did not result in the expected set of numbers for action: " + action)
                return False

        # Verify that the final state meets the 24-game win condition.
        final_numbers = list(map(int, path[-1]._data.split()))
        if len(final_numbers) != 1 or final_numbers[0] != 24:
            print("Final state does not equal 24. The solution is invalid.")
            return False

        return True