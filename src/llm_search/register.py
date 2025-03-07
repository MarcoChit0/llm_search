from __future__ import annotations
import inspect
import abc

MODEL_REGISTRY = {}
STATE_EVALUATOR_REGISTRY = {}
SUCCESSOR_GENERATOR_REGISTRY = {}
REGISTRIES = {
    "model": MODEL_REGISTRY,
    "state_evaluator": STATE_EVALUATOR_REGISTRY,
    "successor_generator": SUCCESSOR_GENERATOR_REGISTRY
}

def register(registry:dict, cls:Register) -> None:
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
    def __init__(self, registry:dict, **kwargs) -> None:
        self.__dict__.update(kwargs)
        register(registry, self)

    @classmethod
    def from_config(cls, config:dict) -> Register:
        signature = inspect.signature(cls.__init__)
        valid_params = list(signature.parameters.keys())[1:]
        for param in valid_params:
            if signature.parameters[param].default is inspect.Parameter.empty and param not in config:
                raise ValueError(f"Missing required argument {param}")
        kwargs = {param:config.get(param) for param in valid_params if param in config}
        return cls(**kwargs)

    @classmethod
    @abc.abstractmethod
    def get_entries(cls) -> list[str]:
        raise NotImplementedError