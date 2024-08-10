import json
import torch
from dnn import *
import os
import numpy as np
from regazzoni2022_mono_pred import *

# Normalize new data
def normalize_data(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min)

def denormalize_data(data, data_min, data_max):
    return data * (data_max - data_min) + data_min



def get_prediction(example_input):

    # Load model configuration
    with open('model_config.json', 'r') as f:
        model_config = json.load(f)

    # Load normalization parameters
    with open('normalization_params.json', 'r') as f:
        normalization_params = json.load(f)

    input_min = np.array(normalization_params['input_min'])
    input_max = np.array(normalization_params['input_max'])

    output_min = np.array(normalization_params['output_min'])
    output_max = np.array(normalization_params['output_max'])

    X_new_normalized = normalize_data(example_input, input_min, input_max)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert to tensor and move to device
    X_new_tensor = torch.tensor(X_new_normalized, dtype=torch.float32).to(device)

    # Load the model
    model = LargerNN(model_config['input_size'], model_config['hidden_sizes'], model_config['output_size']).to(device)

    # Load the state dictionary
    model.load_state_dict(torch.load('model_state.pth'))
    model.eval()

    # Predict
    with torch.no_grad():
        predictions = model(X_new_tensor)

    # Denormalize predictions if needed
    predictions_np = predictions.cpu().numpy()
    y_pred_denormalized = denormalize_data(predictions_np, output_min, output_max)

    return y_pred_denormalized

if __name__ == 'main':
    # Prepare input data (example input)
    example_input = torch.load('filtered_output.pt')[0, :]
    y_pred_denormalized = get_prediction(example_input)

    # Prepare input data (example input)
    example_output = torch.load('filtered_input.pt').numpy()[0, :]

    # error_pred = np.abs(y_pred_denormalized - example_output) / example_output
    #
    # # Post-process predictions if needed
    print(y_pred_denormalized)
    print(example_output)
    # print(error_pred)

