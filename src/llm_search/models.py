from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import abc
from llm_search.register import *

class Model(Register):
    registry = MODEL_REGISTRY

    @abc.abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> list[str]:
        raise NotImplementedError

class HuggingFaceModel(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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

    def wrap_prompt(self, prompt: str) -> list[dict]:
        return [{'role': 'user', 'content': prompt}]

    def generate_text(self, prompt: str, **kwargs) -> list[str]:
        new_message = self.wrap_prompt(prompt)

        generator = pipeline(
            "text-generation", 
            model=self._model, 
            tokenizer=self._tokenizer)
        # response = [{'generated_text': messages:list[dict]}, {'generated_text': messages:list[dict]}, ...], where the last message is the newly produced text
        response = generator(new_message, **kwargs) 
        candidates = []
        for candidate in response:
            candidate_message = candidate['generated_text'][-1]['content']
            candidates.append(candidate_message)
            self._generated_tokens += len(self._tokenizer.encode(candidate_message))
        return candidates

class QwenModel(HuggingFaceModel):
    def __init__(self, **kwargs) -> None:
        kwargs['model_path'] = 'Qwen'
        super().__init__(**kwargs)
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
                "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-3B", "Qwen2.5-3B-Instruct"]

class LlamaModel(HuggingFaceModel):
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
    def get_entries(cls) -> list[str]:
        return ["llama-3.2-1B", "llama-3.2-1B-Instruct",
                "llama-3.2-3B", "llama-3.2-3B-Instruct",
                "llama-3.1-8B", "llama-3.1-8B-Instruct",
                "llama-3.1-70B", "llama-3.1-70B-Instruct"]


from google import genai
from google.genai import types
class GeminiModel(Model):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=self.api_key)
        self.response_tokens = 0 
        self.prompt_tokens = 0
        self.total_tokens = 0
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

    def generate_text(self, prompt: str, **kwargs) -> list[str]:
        model_name = self.__dict__.get('model_name')
        generation_args = kwargs.get('generation_args', {})
        response = self.client.models.generate_content(model=model_name, contents=prompt, config=types.GenerateContentConfig(**generation_args))
        candidates = []
        for candidate in response.candidates:
            candidate_content = ""
            for part in candidate.content.parts:
                candidate_content += part.text
            candidates.append(candidate_content)

        self.response_tokens += response.usage_metadata.candidates_token_count
        self.prompt_tokens += response.usage_metadata.prompt_token_count
        self.total_tokens += response.usage_metadata.total_token_count

        return candidates