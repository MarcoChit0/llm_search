from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import os
import abc
from llm_search.register import *
from typing import Union


class Model(Register):
    registry = MODEL_REGISTRY

    def __init__(self, model_name:str, text_generation_args: dict[str, object] = {}, **kwargs) -> None:
        # Store the remapped arguments
        self._text_generation_args = map_text_generation_args(self.__class__, text_generation_args)
        self._model_name = model_name
        self._number_of_generate_calls = 0
        super().__init__(**kwargs)

    @abc.abstractmethod
    def generate(self, prompt:str, **kwargs) -> Union[list[str], list[list[dict[str, str]]]]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def tokenize(self, text:str) -> list[int]:
        raise NotImplementedError
    
    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_statistics(self) -> dict[str, object]:
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

    def reset(self):
        self._number_of_generate_calls = 0
        self._generated_tokens = 0
    
    def get_statistics(self) -> dict[str, object]:
        return {
            "generated_tokens": self._generated_tokens
        }

    def wrap_prompt(self, prompt: str) -> list[dict]:
        return [{'role': 'user', 'content': prompt}]


    def generate(self, prompt:str, **kwargs) -> Union[list[str], list[list[dict[str, str]]]]:
        is_chat = kwargs.get("is_chat", False)
        chat = kwargs.get("chat_history", []) + self.wrap_prompt(prompt)

        text_generation_args = {**self._text_generation_args, **map_text_generation_args(self.__class__, {**kwargs})}

        generator = pipeline("text-generation", model=self._model, tokenizer=self._tokenizer)
        response = generator(chat, **text_generation_args)

        outputs = []
        for candidate in response:
            generated_text = ""
            for part in candidate['generated_text']:
                generated_text += part['content']

            self._generated_tokens += len(self._tokenizer.encode(generated_text))
            
            if is_chat:
                outputs.append(chat + [{"role": "assistant", "content": generated_text}])
            else:
                outputs.append(generated_text)
        
        return outputs

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
        super().__init__(model_name, text_generation_args, **(kwargs or {}))
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        self.client = genai.Client(api_key=self.api_key)
        self._response_tokens = 0 
        self._prompt_tokens = 0
        self._total_tokens = 0
    
    def reset(self):
        self._number_of_generate_calls = 0
        self._response_tokens = 0
        self._prompt_tokens = 0
        self._total_tokens = 0
    
    def get_statistics(self) -> dict[str, object]:
        return {
            "response_tokens": self._response_tokens,
            "prompt_tokens": self._prompt_tokens,
            "total_tokens": self._total_tokens
        }
    
    @classmethod
    def get_entries(cls) -> list[str]:
        return ["gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-1.5-pro"]

    def _format_chat_prompt(self, chat_history: list[dict[str, str]], new_prompt: str) -> str:
        """
        Helper method to flatten the conversation history and new prompt into a single string.
        Each message is prefixed by its role (e.g. "User:" or "Assistant:").
        """
        formatted_history = "\n".join(f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history)
        # Add the new prompt with a "User:" prefix.
        if formatted_history:
            return f"{formatted_history}\nUser: {new_prompt}"
        else:
            return new_prompt

    def generate(self, prompt: str, **kwargs) -> Union[list[str], list[list[dict[str, str]]]]:
        # TODO: check whether this method is correct
        is_chat = kwargs.get("is_chat", False)
        chat_history = kwargs.get("chat_history", [])
        
        # If chat mode is active, format the prompt accordingly.
        conversation_prompt = self._format_chat_prompt(chat_history, prompt) if is_chat else prompt

        text_generation_args = map_text_generation_args(
            self.__class__, {**self._text_generation_args, **kwargs}
        )
        
        # Call Gemini's API with the flattened prompt.
        response = self.client.models.generate_content(
            model=self._model_name, 
            contents=conversation_prompt, 
            config=types.GenerateContentConfig(**text_generation_args)
        )

        outputs = []
        for candidate in response.candidates:
            candidate_content = ""
            for part in candidate.content.parts:
                candidate_content += part.text
            outputs.append(candidate_content)
        
        # Update token usage statistics.
        self._response_tokens += response.usage_metadata.candidates_token_count
        self._prompt_tokens += response.usage_metadata.prompt_token_count
        self._total_tokens += response.usage_metadata.total_token_count

        # If using chat mode, wrap the generated text in an assistant message.
        if is_chat:
            outputs = [chat_history + [{"role": "assistant", "content": candidate}] for candidate in outputs]
        
        return outputs

    def tokenize(self, text:str) -> list[int]:
        # TODO: change this approach when google API provides a way to tokenize text
        # Currently, get the ASCII code for each character in the text
        return [ord(c) for c in text]
    
def map_text_generation_args(model: Model, text_generation_args: dict[str, object]) -> dict[str, object]:
    new_args = {}
    if issubclass(model, HuggingFaceModel):
        mapping = {
            "max_output_tokens": "max_new_tokens",
            "candidate_count": "num_return_sequences",
            "do_sample": "do_sample",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop_sequences": "stop_strings",
        }
    elif issubclass(model, GeminiModel):
        mapping = {
            "max_output_tokens": "max_output_tokens",
            "candidate_count": "candidate_count",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop_sequences": "stop_sequences",
        }
    else:
        raise ValueError(f"Unsupported model class: {model.__name__}")
    
    for input_key, output_key in mapping.items():
        value = text_generation_args.get(input_key)
        if value is not None:
            new_args[output_key] = value
    return new_args
