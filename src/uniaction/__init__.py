from .config import ExperimentConfig, load_config

__version__ = "1.0.0"
__all__ = ["UniActionPipeline", "ExperimentConfig", "load_config"]


def __getattr__(name):
    if name == "UniActionPipeline":
        from .pipeline import UniActionPipeline

        return UniActionPipeline
    raise AttributeError(name)
