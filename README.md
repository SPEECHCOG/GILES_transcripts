### Code for Räsänen & Kocharov (2024). "Age-Dependent Analysis and Stochastic Generation of Child-Directed Speech", Proc. CogSci-2024, Rotterdam, Netherlands.

Program code for training a GPT-2 architecture language model (LM) from CHILDES transcripts and generating new child-directed speech transcripts using the model. 

### Contents:

- `GILES_main.py`: The main script for training the model and generating text using it.
- `measure_unique_utterances.py`: Script for analyzing the proportion of unique utterances in the generated texts that never occur in CHILDES (or counting unique utterances in CHILDES).
- `analyze_results.m`: Plotting the results based on the corpus analyses (MATLAB).

### Main dependencies

- tensorflow==2.12.1
- numpy==1.23.5
- tensorflow-text==2.12.0

### Instructions:

- Download and prepare CHILDES transcript data into age-specific bins using `<missing>`
- Run `GILES_main.py` to train the model and generate transcripts with it (after setting data paths inside the file).

To analyze the data linguistically, you will need `<explanation missing>`, and then run `compare_datasets.py` of that test suite. 
