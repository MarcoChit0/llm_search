from llm_search.environments.environment import *
import os

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
    
    def get_task(self, index:int) -> State:
        df = pd.read_csv(self._data_file_path)
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
        
        return verify_recursively(list(map(int, initial_state._data.split(' '))))


    def is_model_response_correct(self, initial_state: State, final_state: State | None):
        if final_state is None:
            if self.ground_truth(initial_state):
                print("There is a solution and the model did not found it.")
                return False
            else:
                return True

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
    
        final_numbers = list(map(int, path[-1]._data.split()))
        if len(final_numbers) != 1 or final_numbers[0] != 24:
            print("Final state does not equal 24. The solution is invalid.")
            return False

        return True
    
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
                print(prompt)
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