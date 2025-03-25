from __future__ import annotations
import inspect
import abc
import logging

MODEL_REGISTRY = {}
ENVIRONMENT_REGISTRY = {}
SOLVER_REGISTRY = {}

REGISTRIES:dict[str, dict[str, Register]] = {
    "model": MODEL_REGISTRY,
    "environment": ENVIRONMENT_REGISTRY,
    "solver": SOLVER_REGISTRY
}

def register(registry:dict[str, Register], cls:Register) -> None:
    for entry in cls.get_entries():
        if entry in registry:
            raise ValueError(f"Duplicate entry {entry} in registry.")
        registry[entry.lower()] = cls

def get_registered_class(name:str, registry_name:str|None=None) -> Register:
    reg_class = None
    if registry_name is None:
        for registry in REGISTRIES.values():
            if name in registry:
                reg_class = registry[name]
                break
    else:
        registry = REGISTRIES.get(registry_name)
        if registry is None:
            raise ValueError(f"Invalid registry name {registry_name}.")
        reg_class = registry.get(name)
    if reg_class is None:
        raise ValueError(f"Invalid class name {name}.")
    return reg_class

class Register(abc.ABC):
    registry:dict[str, Register] | None = None
    
    def __init__(self, **kwargs:dict[str, object]):
        self.__dict__.update(kwargs)

    @classmethod
    def from_config(cls:Register, config: dict) -> Register:
        signature = inspect.signature(cls.__init__)
        valid_params = [param for param in signature.parameters if param != "self" and param != "kwargs"]

        # Validate required arguments
        for param in valid_params:
            if signature.parameters[param].default is inspect.Parameter.empty and param not in config:
                raise ValueError(f"Missing required argument {param}")
    
        return cls(**config)

    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.registry is None:
            raise ValueError("Missing registry attribute.")
        if not inspect.isabstract(cls):
            for entry in cls.get_entries():
                if entry in cls.registry:
                    raise ValueError(f"Duplicate entry {entry} in registry.")
                cls.registry[entry] = cls

    @classmethod
    @abc.abstractmethod
    def get_entries(cls) -> list[str]:
        raise NotImplementedError

def get_available_entries(registry:str|dict):
    if isinstance(registry, str):
        registry = REGISTRIES.get(registry)
        if registry is None:
            raise ValueError(f"Invalid registry name {registry}.")
    return list(registry.keys())

class ExpectedError(Exception):
    pass

class CriticalError(Exception):
    pass