# scGRN-Entropy

This project is designed to calculate Cell differentiation trajectory and pseudotime.

## Dependencies

Ensure you have the following Python libraries installed:

- numpy
- pandas
- matplotlib
- scanpy
- anndata
- networkx
- rpy2

Also, make sure your R environment is configured correctly.

## File Structure

- `scGRN-Entropy-main.py`: Main program file.
- `utils.py`: Contains utility functions.
- `getMST.py`: Functions related to building the minimum spanning tree.
- `distance_matrix.py`: Calculates distance matrix.
- `caculate_TranProb_PTime_Entropy.py`: Calculates transition probabilities and pseudotime.

## Usage

1. Ensure the R environment is installed and the `R_HOME` path is set.
2. Place your data files in the `data` directory.
3. Run the main program:

   ```bash
   python scGRN-Entropy-main.py
4. Results will be saved in the result directory.
   
## Features

Reads RDS files and converts them to AnnData objects.
Calculates Gene Regulatory Networks (GRN).
Computes transition probabilities.
Builds the minimum spanning tree, calculates pseudotime, and visualizes trajectories.
Sample Data
The default data file is germline-human-female_li.rds, which can be modified in scGRN-Entropy-main.py.

Visualization
After running the program, the results will include visualizations of pseudotime trajectories.
