# extracting info from data through pre-trained LLM


## file structure
- `01_train_suffix.py` is the main function to run and deals with processing all the cmd-line args
  - *suffix* is much simpler (and doesn't require any model gradients)
  - when we implement *classification* and *clustering*, these should also go into new, different files
- `scripts` is a folder for running sweeps over experiments
  - for example, `scripts/submit_sweep.py` loops over cmd-line args and calls `01_train.py`
  - each file saves a pkl of results into a folder and after the sweep is run the `analyze` notebooks load and aggregate these into a dataframe
- `data.py` holds the code for generating datasets
  - it uses files in the `data_utils` folder

## testing
- to check if the pipeline seems to work, install pytest then run `pytest` from the repo's root directory

## recent changes
- 09/04
  - much broader support for datasets including nli
  - added tests
- 09/03
  - note: args that start with `use_` are boolean
- 09/02
  - changed defaults in `01_train_suffix.py`: defaults to suffix, max_digit = 10, beam_width_suffix=4
  - `data.py` returns data + `check_answer_func`
  - moved `train` function into `__main__` and refactored into `train_prefix.py` and `train_suffix.py`
  - gpu / parallelization may not work properly for `train_prefix.py`