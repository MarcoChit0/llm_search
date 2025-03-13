from __future__ import annotations
import inspect
import abc

MODEL_REGISTRY = {}
STATE_EVALUATOR_REGISTRY = {}
SUCCESSOR_GENERATOR_REGISTRY = {}
SOLVER_REGISTRY = {}

REGISTRIES:dict[str, dict[str, Register]] = {
    "model": MODEL_REGISTRY,
    "state_evaluator": STATE_EVALUATOR_REGISTRY,
    "successor_generator": SUCCESSOR_GENERATOR_REGISTRY,
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

        # Check if **kwargs is present
        has_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())

        # Validate required arguments
        for param in valid_params:
            if signature.parameters[param].default is inspect.Parameter.empty and param not in config:
                raise ValueError(f"Missing required argument {param}")

        kwargs = {param: config.get(param) for param in valid_params if param in config}

        # If **kwargs exists in the constructor, pass all remaining keys in config
        if has_kwargs:
            return cls(**config)

        return cls(**kwargs)

    
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