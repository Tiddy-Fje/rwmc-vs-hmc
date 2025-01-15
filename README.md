## README for RWMC vs HMC Analysis

### Project Overview
This project was developed as part of the Stochastic Simulation course at EPFL. It provides insight on how to choose between RWMC and HMC based on problem characteristics. 

See `report.pdf` for a detailed analysis and discussion of the project. It covers an overview of the theoretical foundations of both methods, proofs of important properties of HMC, exploration and interpretation of how parameters, such as step size for RWMC and mass for HMC influence performance. To systematically assess and compare their efficiency, among others a similarity metric and effective sample size are used. Specifically, two case studies are considered : a 2D toy example with tunable complexity and a higher-dimensional 10D birthweight dataset. 
	
### How to Run the Analysis
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Set up the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate regression
   ```
3. Open and run the files.

### Authors
- Aude Maier
- Tara Fjellman

For questions or collaborations, please reach out via email.

### Acknowledgments
- Professor Fabio Nobile : Provided the project statement and lecture material.

### Notes
- This project adheres to EPFL guidelines for reproducible research. Ensure all code modifications are documented, and credit is given where due.