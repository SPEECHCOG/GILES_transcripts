### Code for Räsänen & Kocharov (2024). "Age-Dependent Analysis and Stochastic Generation of Child-Directed Speech", Proc. CogSci-2024, Rotterdam, Netherlands.

Program code for training a GPT-2 architecture language model (LM) from CHILDES transcripts and generating new child-directed speech transcripts using the model. 

### Contents:
- `GILES_main.py`: The main script for training the model and generating text using it.
- `measure_unique_utterances.py`: Script for analyzing the proportion of unique utterances in the generated texts that never occur in CHILDES (or counting unique utterances in CHILDES).
- `analyze_results.m`: Plotting the results based on the corpus analyses (MATLAB).
- `get_childes_naenglish.R`: The R script to download NA English corpora.
- `childes_to_ao_dataset.py`: Extraction of CHILDES transcript data from downloaded CSV-files into age-specific bins.
### Main dependencies
**Python**

For LM training:

- tensorflow==2.12.1
- tensorflow-text==2.12.0

For evaluation of the generated data:
- evaluate (https://huggingface.co/docs/evaluate/en/installation)
- matplotlib
- pandas
- scipy
- stanza==1.8.2 (https://stanfordnlp.github.io/stanza/)

**R**
- childesr

### Instructions:

- Download CHILDES transcripts from NA English corpora using `get_childes_naenglish.R` as CSV-files (see the list of corpora in the script). The script uses `childesr` library (https://github.com/langcog/childesr). Setup a path to directory, where you want to store CSV-files.
- Extract CHILDES transcript data from downloaded CSV-files into age-specific bins using `childes_to_ao_dataset.py`
```
childes_to_ao_dataset.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR
```
- Run `GILES_main.py` to train the model and generate transcripts with it (after setting data paths inside the file).
- Run `compare_datasets.py` to extract linguistic features for the given texts (or collections of texts):
```
compare_datasets.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR --sample-data --analyze-data
```
The same code give a possibility to check statistical significance of feature changes along with (1) an age of a target child and (2) a number of words in a collection of texts corresponding to an age bin.
```
compare_datasets.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR --sample-data --analyze-data --statistical_tests
```

- Run `analyze_results.m` to plot the feature comparison of the generated transcripts against original transcripts.
