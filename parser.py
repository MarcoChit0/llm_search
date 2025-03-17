import tap
from llm_search.register import get_available_entries

class Parser(tap.Tap):
    # registries
    class_model: str
    class_state_evaluator:str
    class_solver:str
    # generation parameters
    max_output_tokens: int
    candidate_count: int
    do_sample: bool
    temperature: float
    load_in_8bit: bool
    # solver parameters
    steps: int
    symmetry_level: str
    instance: str | None
    batch_index_start: int | None
    batch_index_end: int | None

    def configure(self) -> None:
        self.add_argument("-cm", "--class_model", choices=get_available_entries("model"), default="Qwen2.5-3B-Instruct")
        self.add_argument("-cse", "--class_state_evaluator", choices=get_available_entries("state_evaluator"), default="vote")
        self.add_argument("-cs", "--class_solver", choices=get_available_entries("solver"), default="beam-search")
        self.add_argument("-t", "--max_output_tokens", default=1000)
        self.add_argument("-c", "--candidate_count", default=1)
        self.add_argument("--do_sample", default=True, action="store_false")
        self.add_argument("-temp", "--temperature", default=0.7)
        self.add_argument("-s", "--steps", default=3)
        self.add_argument("-l", "--load_in_8bit", default=True, action="store_false")
        self.add_argument("-sl", "--symmetry_level", choices=["none", "medium", "weak", "strong"] , default="weak")
        self.add_argument("-i", "--instance", default=None)
        self.add_argument("-bis", "--batch_index_start", default=None)
        self.add_argument("-bie", "--batch_index_end", default=None)
        
    def check_args(self) -> None:
        assert self.instance is not None or (self.batch_index_start is not None and self.batch_index_end is not None), "Either instance or batch_index_start and batch_index_end must be provided."