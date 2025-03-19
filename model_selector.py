import torch
from models.get_tucl import get_tucl  # Import TUCL model loader

def get_model(model_name: str):
    
    if model_name == "tucl":
        return get_tucl()  # Load TUCL model
    else:
        raise ValueError(f"Unknown model: {model_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=[
        "tucl"
    ], help="Choose model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    args = parser.parse_args()

    # Load model
    model = get_model(args.model)
    
    # Load checkpoint if provided
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint['model'])
    model.eval()

    print(f"Loaded {args.model} from {args.checkpoint}")