import sys
import torch
from model_v1 import PositionAutoEncoder
from helpers import pos_to_tensor

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device("cpu")
    print(f"using {device}")

    # parse cl args
    fen = sys.argv[1] if len(sys.argv) > 1 else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    version = sys.argv[2] if len(sys.argv) > 2 else "4"
    # ------------------------

    # load model
    checkpoint = torch.load(f"../models/v{version}.pt")
    model = PositionAutoEncoder(checkpoint["hyperparameters"]).to(device)
    model.eval()
    model.load_state_dict(checkpoint["model"])
    # ------------------------

    # call model on inputs
    inputs = pos_to_tensor(fen, device).unsqueeze(0)
    embed = model.embed(inputs)
    # ------------------------

    print(embed)

if __name__ == "__main__":
    main()
