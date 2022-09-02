# extracting info from data through pre-trained LLM

- `01_train.py` has all the main code for running/algorithms (might move some of it to `search.py`)
  - this code is split up based on whther we are searching for a *prefix* or *suffix* (algorithms are very different)
  - *suffix* is much simpler (and doesn't require any model gradients)
- `scripts` is a folder for running sweeps over experiments
- `data.py` holds the code for generating datasets (currently synthetic but soon to extend beyond)