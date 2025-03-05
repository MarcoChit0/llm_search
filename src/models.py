from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os


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

    def generate_text(self, prompt: str, **kwargs) -> list[str]:
        print(kwargs)
        new_message = self.wrap_prompt(prompt)

        generator = pipeline(
            "text-generation", 
            model=self._model, 
            tokenizer=self._tokenizer)
        # response = [{'generated_text': messages:list[dict]}, {'generated_text': messages:list[dict]}, ...], where the last message is the newly produced text
        print()
        print(new_message)
        response = generator(new_message, **kwargs) 
        print(response)
        print()
        candidates = []
        for candidate in response:
            candidate_message = candidate['generated_text'][-1]['content']
            candidates.append(candidate_message)
            self._generated_tokens += len(self._tokenizer.encode(candidate_message))
        return candidates

    def wrap_prompt(self, prompt: str) -> list[dict]:
        return [{'role': 'user', 'content': prompt}]
        
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
    
    def wrap_prompt(self, prompt):
        return [{'role': 'system', 'content': 'Resolve the problem efficiently and clearly. Provide a concise solution with no explanation.'}] + super().wrap_prompt(prompt)

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