# extracting info from data through pre-trained LLM

- `01_train.py` has all the main code for running/algorithms
  - this code is split up based on whther we are searching for a *prefix* or *suffix* (algorithms are very different)
  - *suffix* is much simpler (and doesn't require any model gradients)
- `scripts` is a folder for running sweeps over experiments
- `data.py` holds the code for generating datasets (currently synthetic but soon to extend beyond)

# recent changes
- 09/03
  - note: args that start with `use_` are boolean
- 09/02
  - changed defaults in `01_train.py`: defaults to suffix, max_digit = 10, beam_width_suffix=4
  - `data.py` returns data + `check_answer_func`
  - moved `train` function into `__main__` and refactored into `train_prefix.py` and `train_suffix.py`
  - gpu / parallelization may not work properly for `train_prefix.py`