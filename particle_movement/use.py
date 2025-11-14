import torch
from model import WorldModel
from utils import visualize_movement, generate_samples


if __name__ == "__main__":
    model = WorldModel()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    sample = generate_samples(n=1, frame_size=20)
    with torch.no_grad():
        output = model(sample)
        pred_o_t = output['pred_o_ts'][0]

    visualize_movement(sample[0])
    visualize_movement(pred_o_t)