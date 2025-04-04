from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import abc
from llm_search.register import *
from llm_search.state import State

class Model(Register):
    registry = MODEL_REGISTRY

    def __init__(self, model_name:str, text_generation_args: dict[str, object] = {}, **kwargs) -> None:
        if issubclass(self.__class__, HuggingFaceModel):
            mapped_args = {
                "max_new_tokens": text_generation_args.get("max_output_tokens"),
                "num_return_sequences": text_generation_args.get("candidate_count"),
                "do_sample": text_generation_args.get("do_sample"),
                "temperature": text_generation_args.get("temperature"),
                "top_p": text_generation_args.get("top_p"),
                "top_k": text_generation_args.get("top_k"),
                "stop_strings": text_generation_args.get("stop_sequences"),
            }
        elif issubclass(self.__class__, GeminiModel):
            mapped_args = {
                "max_output_tokens": text_generation_args.get("max_output_tokens"),
                "candidate_count": text_generation_args.get("candidate_count"),
                "temperature": text_generation_args.get("temperature"),
                "top_p": text_generation_args.get("top_p"),
                "top_k": text_generation_args.get("top_k"),
                "stop_sequences": text_generation_args.get("stop_sequences"),
            }
        else:
            raise ValueError(f"Unsupported model class: {self.__class__.__name__}")

        # Store the remapped arguments
        self._text_generation_args = mapped_args
        self._model_name = model_name
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> list[str]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def tokenize(self, text:str) -> list[int]:
        raise NotImplementedError

class HuggingFaceModel(Model):
    prefix = None
    def __init__(self, model_name:str, text_generation_args: dict[str, object] = {}, model_config:dict={}, tokenizer_config:dict={}, **kwargs) -> None:
        super().__init__(model_name, text_generation_args, **kwargs)
        self.model_path = os.path.join(self.prefix, self._model_name)
        self.model_config = model_config
        self.tokenizer_config = tokenizer_config
        self.token = os.getenv('HUGGINGFACE_TOKEN')
        
        if self.token is None:
            raise ValueError("HUGGINGFACE_TOKEN is required.")
        
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, **self.tokenizer_config, token=self.token)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_path, **self.model_config, token=self.token)
        self._generated_tokens = 0 

    def wrap_prompt(self, prompt: str) -> list[dict]:
        return [{'role': 'user', 'content': prompt}]

    def generate_text(self, prompt: str, **kwargs) -> list[str]:
        new_message = self.wrap_prompt(prompt)
        text_generation_args = self._text_generation_args
        text_generation_args.update(kwargs)
        generator = pipeline(
            "text-generation", 
            model=self._model, 
            tokenizer=self._tokenizer)
        # response = [{'generated_text': messages:list[dict]}, {'generated_text': messages:list[dict]}, ...], where the last message is the newly produced text
        response = generator(new_message, **text_generation_args) 
        candidates = []
        for candidate in response:
            candidate_message = candidate['generated_text'][-1]['content']
            candidates.append(candidate_message)
            self._generated_tokens += len(self._tokenizer.encode(candidate_message))
        return candidates

    def tokenize(self, text:str) -> list[int]:
        return self._tokenizer.encode(text)

class QwenModel(HuggingFaceModel):
    prefix = "Qwen"
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["Qwen2.5-0.5B", "Qwen2.5-0.5B-Instruct",
                "Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct",
                "Qwen2.5-3B", "Qwen2.5-3B-Instruct"]

class LlamaModel(HuggingFaceModel):
    prefix = "meta-llama"

    def __init__(self, model_name:str, text_generation_args: dict[str, object] = {}, model_config:dict={}, tokenizer_config:dict={}, **kwargs) -> None:
        if 'torch_dtype' not in model_config:
            model_config['torch_dtype'] = torch.bfloat16 
        super().__init__(model_name, text_generation_args, model_config, tokenizer_config, **kwargs)
    
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
    def __init__(self, model_name:str, text_generation_args: dict[str, object] = {},  **kwargs) -> None:
        super().__init__(model_name, text_generation_args, **kwargs)
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
        text_generation_args = self._text_generation_args
        text_generation_args.update(kwargs)
        response = self.client.models.generate_content(model=self._model_name, contents=prompt, config=types.GenerateContentConfig(**text_generation_args))
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

    def tokenize(self, text:str) -> list[int]:
        # TODO: change this approach when google API provides a way to tokenize text
        # Currently, get the ASCII code for each character in the text
        return [ord(c) for c in text]
    
class ModelBasedClass:
    def __init__(self, model:Model, text_generation_args:dict, **kwargs):
        self._model = model
        self._text_generation_args = text_generation_args
        super().__init__(**kwargs)
    
    @abc.abstractmethod
    def get_prompt(self, state:State) -> str:
        raise NotImplementedError
