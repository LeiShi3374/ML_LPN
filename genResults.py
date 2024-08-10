import os
import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from tqdm import tqdm

from regazzoni2022_mono import Simulation

# ----------------------------- MAIN CODE ------------------------------------ #
def gen_results(sim_parameters):
    # Create an instance of the Simulation class
    simulation = Simulation(sim_parameters)

    # Get the new parameters from the baseline
    simulation.get_parameters()

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

    return combined_result

def process_case(case_index, progress_queue, lock, output_file):
    combined_result = gen_results([n_cardiac_cyc, dt, save_last_n_cardiac_cycles])
    
    # Save results every 10000 cases
    if case_index % 10000 == 0:
        with lock:
            df = pd.DataFrame([combined_result])
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', header=False, index=False)
            else:
                df.to_csv(output_file, mode='w', header=False, index=False)
    
    # Notify progress
    progress_queue.put(1)
    return combined_result

def update_progress(progress_queue, num_cases):
    pbar = tqdm(total=num_cases)
    while True:
        progress = progress_queue.get()
        if progress is None:  # Check for termination signal
            break
        pbar.update(progress)
    pbar.close()

# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # General sim parameters
    n_cardiac_cyc = 10
    dt = 0.001 # Here this is a pseudo number to determine how many
    save_last_n_cardiac_cycles = 2

    num_cases = 200000
    num_workers = multiprocessing.cpu_count()  # Use all available CPUs

    # Get directory of this script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    output_file = os.path.join(current_dir, 'combined_results.csv')

    # Create a manager for the progress queue and a lock
    manager = Manager()
    progress_queue = manager.Queue()
    lock = manager.Lock()

    # Create a pool of workers
    with Pool(processes=num_workers) as pool:
        # Start a background thread to update the progress bar
        progress_thread = multiprocessing.Process(target=update_progress, args=(progress_queue, num_cases))
        progress_thread.start()

        # Map process_case function to all case indices
        results = [pool.apply_async(process_case, (i, progress_queue, lock, output_file)) for i in range(num_cases)]
        results = [result.get() for result in results]

        # Signal the progress thread to terminate
        progress_queue.put(None)
        progress_thread.join()

        # Convert the results to a DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, mode='a', header=False, index=False)  # Final save to ensure all results are included

    print('All cases processed and saved to combined_results.csv')
