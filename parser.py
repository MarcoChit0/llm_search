import tap
from llm_search.register import get_available_entries
import datetime
import pandas as pd

class Parser(tap.Tap):
    # experiment parameters
    path: str
    save_state: bool
    load_state: bool

    # registries
    model: str
    solver:str
    environment:str

    # model parameters
    device: str
    max_output_tokens: int
    candidate_count: int
    do_sample: bool
    temperature: float
    load_in_8bit: bool
    
    # environment parameters
    successor_generator:str
    state_evaluator:str
    
    # solver parameters
    steps: int
    symmetry_level: str | None
    budget: int
    instance: str | None
    index_start: int | None
    index_end: int | None

    def configure(self) -> None:
        self.add_argument("-m", "--model", choices=get_available_entries("model"), default="Qwen2.5-3B-Instruct")
        self.add_argument("-s", "--solver", choices=get_available_entries("solver"), default="beam-search")
        self.add_argument("-e", "--environment", choices=get_available_entries("environment"), default="game24")

        self.add_argument("-d", "--device", default="auto")
        self.add_argument("-t", "--max_output_tokens", default=1000)
        self.add_argument("-c", "--candidate_count", default=1)
        self.add_argument("--do_sample", default=True, action="store_false")
        self.add_argument("-temp", "--temperature", default=0.7)
        self.add_argument("-l", "--load_in_8bit", default=True, action="store_false")
        
        self.add_argument("-b", "--budget", default=float("inf"))
        self.add_argument("--steps", default=3)
        self.add_argument("-sl", "--symmetry_level", choices=["weak", "medium", "strong"] , default=None)
        self.add_argument("-i", "--instance", default=None)
        self.add_argument("-is", "--index_start", default=None)
        self.add_argument("-ie", "--index_end", default=None)
        
        self.add_argument("-sg", "--successor_generator")
        self.add_argument("-se", "--state_evaluator")

        self.add_argument("-p", "--path", default=datetime.datetime.now().isoformat())
        self.add_argument("--save_state", default=True, action="store_false")
        self.add_argument("--load_state", default=True, action="store_false")

    def check_args(self) -> None:
        assert self.instance is not None or (self.index_start is not None and self.index_end is not None), "Either instance or index_start and index_end must be provided."
    
    def to_dataframe(self) -> pd.DataFrame:
        # save registries, model parameters, environment parameters, and solver parameters
        return pd.DataFrame({
            "model": [self.model],
            "solver": [self.solver],
            "environment": [self.environment],
            "device": [self.device],
            "max_output_tokens": [self.max_output_tokens],
            "candidate_count": [self.candidate_count],
            "do_sample": [self.do_sample],
            "temperature": [self.temperature],
            "load_in_8bit": [self.load_in_8bit],
            "successor_generator": [self.successor_generator],
            "state_evaluator": [self.state_evaluator],
            "steps": [self.steps],
            "symmetry_level": [self.symmetry_level],
            "budget": [self.budget],
        })