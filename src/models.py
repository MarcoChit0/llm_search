from __future__ import annotations
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os

class State:
    def __init__(self, message : dict, state : State | None = None, tokens : int = 0) -> None:
        if state is not None:
            state.add_children(self)
        self._message = message
        self._tokens = tokens
        self._next = []
        self._parent = state
    
    def add_children(self, child : State) -> None:
        self._next.append(child)
    
    def get_messages(self) -> list[dict]:
        messages = []
        current = self
        while current is not None:
            messages.insert(0, current._message)
            current = current._parent
        return messages


class Model:
    def __init__(self, **kwargs) -> None:
        
        self.__dict__.update(kwargs)
        model_path = self.__dict__.get('model_path')
        model_name = self.__dict__.get('model_name')
        token = os.getenv('HUGGINGFACE_TOKEN')
        
        if token is None:
            raise ValueError("HUGGINGFACE_TOKEN is required.")
        tokenizer_config:dict = self.__dict__.get('tokenizer_config', {})
        model_config:dict = self.__dict__.get('model_config', {})
        self._tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/{model_name}", **tokenizer_config, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(f"{model_path}/{model_name}", **model_config, token=token)
        self._generated_tokens = 0 

    
    def generate_text(self, prompt: str, state:State|None = None, **kwargs) -> list[State]:
        new_message = self.wrap_prompt(prompt)
        s = State(new_message, state)
        messages = s.get_messages()

        generator = pipeline(
            "text-generation", 
            model=self._model, 
            tokenizer=self._tokenizer)
        # response = [{'generated_text': messages:list[dict]}, {'generated_text': messages:list[dict]}, ...], where the last message is the newly produced text
        response = generator(messages, **kwargs) 

        states = []
        for candidate in response:
            candidate_message = candidate['generated_text'][-1]
            tokens = len(self._tokenizer.encode(candidate_message['content']))
            self._generated_tokens += tokens
            new_state = State(candidate_message, state, tokens)
            states.append(new_state)
        return states

    def wrap_prompt(self, prompt: str) -> dict:
        return {'role': 'system', 'content': prompt}
        
    @classmethod
    def get_available_models(cls) -> list[str]:
        raise NotImplementedError

class QwenModel(Model):
    def __init__(self, **kwargs) -> None:
        kwargs['model_path'] = 'Qwen'
        super().__init__(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        return ["Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
                "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-3B", "Qwen2.5-3B-Instruct"]

class LlamaModel(Model):
    def __init__(self, **kwargs) -> None:
        kwargs['model_path'] = 'meta-llama'
        if 'model_config' not in kwargs:
            kwargs['model_config'] = {"torch_dtype": torch.bfloat16}
        else:
            if 'torch_dtype' not in kwargs['model_config']:
                kwargs['model_config']['torch_dtype'] = torch.bfloat16
        super().__init__(**kwargs)

    @classmethod
    def get_available_models(cls) -> list[str]:
        return ["llama-3.2-1B", "llama-3.2-1B-Instruct",
                "llama-3.2-3B", "llama-3.2-3B-Instruct",
                "llama-3.1-8B", "llama-3.1-8B-Instruct",
                "llama-3.1-70B", "llama-3.1-70B-Instruct"]

def get_model(**kwargs) -> Model:
    model_name = kwargs.get('model_name')
    if not model_name:
        raise ValueError("model_name is required.")
    
    model_classes = [QwenModel, LlamaModel]
    for model_class in model_classes:
        if model_name in model_class.get_available_models():
            return model_class(**kwargs)
    
    raise ValueError(f"Model {model_name} is not available.")

params = {
    "model_name": "llama-3.2-1B-Instruct",
    "model_config": {"torch_dtype": torch.bfloat16},
    "tokenizer_config": {}
}
m = get_model(**params)

generation_args = {
    "max_new_tokens": 500,
    "num_return_sequences": 3,
    "do_sample": True,
    "temperature": 0.7,
}
message = [{"role": "system", "content": "Hello, how can I help you today?"}]
s = State(message[0])
p = "I want to book a flight to Paris."

r = m.generate_text(p, s, **generation_args)
for state in r:
    print(state.get_messages())
        