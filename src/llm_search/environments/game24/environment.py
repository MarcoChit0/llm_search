from llm_search.environments.environment import *
import os

class Game24Task(Task):
    def __init__(self, puzzle:str, answer: str, index:int|None = None, **kwargs):
        self._puzzle = puzzle
        super().__init__(
            problem= f"""For the numbers in the input sequence [{puzzle}], is it possible to reach 24 using only basic arithmetic operations?""",
            answer= answer,
            index=index,
            **kwargs)
    
    def get_initial_state(self) -> State:
        return State(self._puzzle, is_initial_state=True)

class Game24Environment(Environment):
    '''
    state._data :   str
                    "n1 ... ni", i <= 4
                    the numbers to be used in the 24 game
    action:
                    str
                    "nj operation nk = result (left: updated list)", 1 <= j, k <= i, operation in ['+', '-', '*', '/'] 
                    the operation to be applied to the numbers, and the updated list
    '''
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self._available_actions = {}
        self._data_file_path = os.path.join(os.path.dirname(__file__), "data.csv")

    def _initialize(self, **kwargs) -> None:
        idx = kwargs.get("index")
        puz = kwargs.get("instance")
        assert idx is not None or puz is not None, "Missing the argument index or puzzle for initialization."
        if idx:
            df = pd.read_csv(self._data_file_path)
            puzzle = df.iloc[idx]["Puzzles"]
        else:
            assert isinstance(puz, str), "Invalid puzzle."
            numbers = puz.split()
            assert len(numbers) <= 4, "Invalid puzzle length."
            assert all(num.isdigit() for num in numbers), "Invalid puzzle format."
            puzzle = puz
        self._task = Game24Task(puzzle, "Yes" if self.ground_truth(list(map(int, puzzle.split()))) else "No", idx)
        self._initial_state = State(puzzle)
  
    def ground_truth(self, puzzle: list[int]):
        if len(puzzle) == 1:
            if puzzle[0] == 24:
                return True
        for i in range(len(puzzle)):
            for j in range(len(puzzle)):
                if i == j:
                    continue
                for op in ['+', '-', '*', '/']:
                    if op == '/' and puzzle[j] == 0:
                        continue
                    if op == '+':
                        new_value = puzzle[i] + puzzle[j]
                    elif op == '-':
                        new_value = puzzle[i] - puzzle[j]
                    elif op == '*':
                        new_value = puzzle[i] * puzzle[j]
                    elif op == '/':
                        new_value = puzzle[i] / puzzle[j]
                    reduced_puzzle = puzzle.copy()
                    reduced_puzzle.remove(puzzle[i])
                    reduced_puzzle.remove(puzzle[j])
                    reduced_puzzle.append(new_value)
                    if self.ground_truth(reduced_puzzle):
                        return True
        return False

    def is_model_response_correct(self, task: Task, final_state: State | None):
        if final_state is None:
            if task._answer == "Yes":
                error_message = "There is a solution and the model did not found it."
                return False, error_message
            else:
                return True, "There are no solution and the model did not found it."

        final_state.print()

        path = []
        curr = final_state
        while curr is not None:
            path.append(curr)
            curr = getattr(curr, "_parent", None)
        path.reverse()
        
        for i in range(len(path) - 1):
            parent_state = path[i]
            child_state = path[i + 1]

            parent_nums = sorted(list(map(int, parent_state._data.split())))
            child_nums = sorted(list(map(int, child_state._data.split())))

            if len(child_nums) != len(parent_nums) - 1:
                error_message = (f"Invalid state transition: incorrect number of remaining numbers "
                                 f"between states '{parent_state._data}' and '{child_state._data}'.")
                return False, error_message

            action = parent_state.get_action_to_child(child_state)
            numbers = re.findall(r'\d+', action)
            if len(numbers) < 3:
                error_message = "Invalid action format; expected at least three numbers in the action: " + action
                return False, error_message

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
                error_message = "Invalid operator in action: " + action
                return False, error_message

            if computed != result:
                error_message = (f"Invalid computation in action \"{action}\": {operand1} ? {operand2} = {computed} "
                                 f"(expected {result}).")
                return False, error_message

            parent_counter = Counter(parent_nums)
            if parent_counter[operand1] < 1 or parent_counter[operand2] < 1:
                error_message = "Operands specified in the action are not present in the parent's state for action: " + action
                return False, error_message

            parent_counter.subtract([operand1, operand2])
            parent_counter[result] += 1

            if parent_counter != Counter(child_nums):
                error_message = "State transition did not result in the expected set of numbers for action: " + action
                return False, error_message
    
        final_numbers = list(map(int, path[-1]._data.split()))
        if len(final_numbers) != 1 or final_numbers[0] != 24:
            error_message = "Final state does not equal 24. The solution is invalid."
            return False, error_message

        return True, "There are a valid solution and the model found it."
    
    def save_results(self, final_state, file_pointer) -> None:
        is_correct, error_message = self.is_model_response_correct(self._task, final_state)
        row = {
            "puzzle": self._task._puzzle,
            "is_correct": is_correct,
            "message": error_message,
            "index": self._task._index,
        }
        row.update(self._model.get_statistics())
        dict_to_csv(row, file_pointer)

    def wrap_successor_generator_prompt(self, state) -> str:
        successor_generator = self.__dict__.get("successor_generator")
        if  successor_generator == "propose":
            return """Given a list of numbers, propose possible next steps using basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/). Each step must involve exactly two numbers from the list, and the result should replace those two numbers in a new list.

Rules:
- Only use basic arithmetic operations.
- Each operation should be displayed in the format:
  number [operation] number = result (left: updated list)
- List each possible next step on a separate line.

Example:

Input:  2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)

Now, generate the possible next steps for the following input:

Input: {input}
Possible next steps:""".format(input=state._data)
        elif successor_generator == "propose-all":
            return """Given a list of numbers, propose all possible next steps using basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/). Each step must involve exactly two numbers from the list, and the result should replace those two numbers in a new list.

Rules:
- Only use basic arithmetic operations.
- Each operation should be displayed in the format: 
number operation number = result (left: updated list)
- List each possible next step on a separate line.

Example:

Input:  1 2 3
All possible next steps:
1 + 2 = 3 (left: 3 3)
1 - 2 = -1 (left: -1 3)
1 * 2 = 2 (left: 2 3)
1 / 2 = 0.5 (left: 0.5 3)
1 + 3 = 4 (left: 2 4)
1 - 3 = -2 (left: -2 2)
1 * 3 = 3 (left: 2 3)
1 / 3 = 0.33 (left: 0.33 2)
2 + 1 = 3 (left: 3 3)
2 - 1 = 1 (left: 1 3)
2 * 1 = 2 (left: 2 3)
2 / 1 = 2.0 (left: 2.0 3)
2 + 3 = 5 (left: 1 5)
2 - 3 = -1 (left: -1 1)
2 * 3 = 6 (left: 1 6)
2 / 3 = 0.67 (left: 0.67 1)
3 + 1 = 4 (left: 2 4)
3 - 1 = 2 (left: 2 2)
3 * 1 = 3 (left: 2 3)
3 / 1 = 3.0 (left: 2 3.0)
3 + 2 = 5 (left: 1 5)
3 - 2 = 1 (left: 1 1)
3 * 2 = 6 (left: 1 6)
3 / 2 = 1.5 (left: 1 1.5)

Now, generate all the possible next steps for the following input:

Input: {input}
All possible next steps:""".format(input=state._data, candidate_steps='\n'.join(list(state._children.keys())))
        else:
            raise ValueError(f"Invalid successor generator: {successor_generator}")
        
    def apply_action(self, state, action):
        successor_generator = self.__dict__.get("successor_generator")
        if successor_generator in ['propose', 'propose-all']:
            successor_data = action.split('left: ')[1].replace(')', '').strip()
            return State(successor_data, state, action)
        else:
            raise ValueError(f"Invalid successor generator: {successor_generator}")
    
    def expand(self, state):
        successors = []
        for action in self.get_available_actions(state):
            successor = self.apply_action(state, action)
            successors.append(successor)
        return successors
        
    
    def get_available_actions(self, state):
        if state not in self._available_actions:
            successor_generator = self.__dict__.get("successor_generator")
            ACTION_PATTERN = r'^\s*\d+(?:\.\d+)?\s*[\+\-\/\*]\s*\d+(?:\.\d+)?\s*=\s*\d+(?:\.\d+)?\s*\(left:\s*(?:\d+(?:\.\d+)?(?:\s+\d+(?:\.\d+)?)*?)\)\s*$'
            if successor_generator in ["propose", "propose-all"]:
                prompt = self.wrap_successor_generator_prompt(state)
            else:
                raise ValueError(f"Invalid successor generator: {successor_generator}")
            
            response = self._model.generate_text(prompt)
            actions = []
            for r in response:
                for a in r.split('\n'):
                    if re.match(ACTION_PATTERN, a):
                        actions.append(a)
            self._available_actions[state] = actions
        return self._available_actions[state]
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["24-game", "game24"]

    def wrap_state_evaluation_prompt(self, state):
        state_evaluator = self.__dict__.get("state_evaluator")
        if state_evaluator == "vote":
            return """Given a list of candidate steps, select the best one to move toward the target number 24 using basic arithmetic operations: addition (+), subtraction (-), multiplication (*), and division (/).  

Rules:  
- Choose only one candidate step.  
- The response must contain **only** the selected step.  

Example:  

Input:  2 8 8 14  
Candidate steps:  
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 / 2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)

Vote: 14 - 8 = 6 (left: 2 6 8)  

Now, select the best step for the following input:  

Input: {input}  
Candidate steps:  
{candidate_steps}  

Vote:""".format(input=state._data, candidate_steps='\n'.join(list(state._children.keys())))
        else:
            raise ValueError(f"Invalid state evaluator: {state_evaluator}")
    
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
            raise ValueError(f"Invalid state evaluator: {state_evaluator}")
        
    def is_goal_state(self, state: State) -> bool:
        return state._data == "24"