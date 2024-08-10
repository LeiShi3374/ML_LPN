'''
Evaluates error between simulation results and patient-specific metrics that we 
aim to match.
'''

import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
from prediction import *

from regazzoni2022_mono_pred import Simulation

# ----------------------------- MAIN CODE ------------------------------------ #
def gen_results(y_test, case_type, sim_parameters):
    # Create an instance of the Simulation class
    simulation = Simulation(sim_parameters)

    # Set parameters
    # y_pred = [2.47014160e+01, 1.78618673e-02, 2.09048271e+00, 1.24385327e-01,
    #          1.03915676e-01, 1.19570255e-01, 3.10267568e-01, 5.02145253e-02,
    #          1.72481899e+01, 7.64159317e+01, 4.09196510e+01, 4.60950012e+01]
    #
    # y_orig = [2.49059842e+01, 1.41062380e-02, 1.85244397e+00, 1.20549375e-01,
    #          1.24153034e-01, 1.09650453e-01, 3.21833094e-01, 5.11867428e-02,
    #          1.78679284e+01, 7.91088126e+01, 4.17553168e+01, 4.88132920e+01]


    # Get the new parameters from the baseline
    simulation.get_parameters(y_test)

    # Run simulation
    results_dict = simulation.integrate()

    # Save input parameters to results_dict
    results_dict['parameters'] = simulation.parameters

    # Extract metrics from simulation
    simulation.compute_clinical_metrics(results_dict)
    sim_metrics = results_dict['clinical_metrics']

    combined_input = simulation.record_input()
    combined_output = simulation.record_output(sim_metrics)

    combined_result = np.concatenate((combined_input, combined_output))

    # Create output directory if it doesn't exist
    output_dir = os.path.join(current_dir, f'evaluate_{case_type}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, 'sim_metrics.txt'), 'w') as file:
        for metric, metric_value in sim_metrics.items():
            file.write(f"{metric}: {metric_value['Value']} {metric_value['Units']} \n")

    simulation.plot_results(results_dict, output_dir, font_size=12)

    # Save simulation results
    # simulation.save_results(results_dict, output_dir)

    # Plot patient-specific metrics and simulation metrics on bar plot
    # simulation.plot_metrics(sim_metrics, os.path.join(output_dir, 'metric_comparison.png'))

    return combined_result

# ---------------------------------------------------------------------------- #
# For parallel execution in differential_evolution, need to use the following   
if __name__ == "__main__":

    # Load parameters of model (as dictionary) (all in mmHg/mL units)
    # Specify the name of the file (without .py extension)
    # parameters_file = 'regazzoni2022_parameters'
    # parameters_module = importlib.import_module(parameters_file) # Dynamically import the module
    # Get the path of the parameters file
    # parameters_filepath = parameters_module.__file__
    # Get the parameters dictionary from the module
    # parameters = parameters_module.parameters

    # Get directory of this script
    current_dir = os.path.dirname(os.path.realpath(__file__))

    # General sim parameters
    n_cardiac_cyc = 10
    dt = 0.001 # Here this is a psudo number to determine how many
    save_last_n_cardiac_cycles = 2


    num = 150
    example_input = torch.load('filtered_output.pt')[num, :]
    y_pred = get_prediction(example_input)
    y_orig = torch.load('filtered_input.pt').numpy()[num, :]

    y_pred_results = gen_results(y_pred, 'pred', [n_cardiac_cyc, dt, save_last_n_cardiac_cycles])
    print(y_pred_results)

    y_orig_results = gen_results(y_orig, 'orig', [n_cardiac_cyc, dt, save_last_n_cardiac_cycles])
    print(y_orig_results)