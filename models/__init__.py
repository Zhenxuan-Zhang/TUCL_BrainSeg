
from .tucl import get_tucl

def get_model(model_name: str):
    """Factory method to get a model by name."""
    if  model_name == "tucl":
        return get_tucl()

    

    else:
        raise ValueError(f"Unknown model: {model_name}")
