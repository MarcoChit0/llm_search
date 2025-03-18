import tap
from llm_search.register import get_available_entries

class Parser(tap.Tap):
    # registries
    model: str
    solver:str
    environment:str

    # text generation parameters
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
    symmetry_level: str
    budget: int
    instance: str | None
    batch_index_start: int | None
    batch_index_end: int | None

    def configure(self) -> None:
        self.add_argument("-m", "--model", choices=get_available_entries("model"), default="Qwen2.5-3B-Instruct")
        self.add_argument("-s", "--solver", choices=get_available_entries("solver"), default="beam-search")
        self.add_argument("-e", "--environment", choices=get_available_entries("environment"), default="game24")

        self.add_argument("-t", "--max_output_tokens", default=1000)
        self.add_argument("-c", "--candidate_count", default=1)
        self.add_argument("--do_sample", default=True, action="store_false")
        self.add_argument("-temp", "--temperature", default=0.7)
        self.add_argument("-l", "--load_in_8bit", default=True, action="store_false")
        
        self.add_argument("-b", "--budget", default=float("inf"))
        self.add_argument("--steps", default=3)
        self.add_argument("-sl", "--symmetry_level", choices=["none", "medium", "weak", "strong"] , default="weak")
        self.add_argument("-i", "--instance", default=None)
        self.add_argument("-bis", "--batch_index_start", default=None)
        self.add_argument("-bie", "--batch_index_end", default=None)
        
        self.add_argument("-sg", "--successor_generator")
        self.add_argument("-se", "--state_evaluator")

    def check_args(self) -> None:
        assert self.instance is not None or (self.batch_index_start is not None and self.batch_index_end is not None), "Either instance or batch_index_start and batch_index_end must be provided."