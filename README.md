# README

This repository provides the source code of the ReLU neural network for calculating Hamming distance and Levenshtein distance.

## repository description

- relu_hamming.py
    - The ReLU neural network for calculating Hamming distance.
- relu_levenshtein.py
    - The ReLU neural network for calculating Levenshtein distance.
- sampledata
    - This folder contains the data to be referenced during program execution.

## Guidelines for running the program

If you wish to perform distance calculation or learning, you need to make changes to the following two points in the source code.

You will modify the weights within the `set_weight_for_debug` function. 
If you want to perform distance calculation, use `weights[0][0][0]=1`
If you want to perform learning, use `random.uniform(0,1)`.

```
def set_weight_for_debug(model, seq_length):

...

        # for calculating correct distance
        #weights[0][0][0] = 1

        # for learning 
        weights[0][0][0] = random.uniform(0,1)
```

The three functions within the `main()` provide the following experiments:

- `measure()` measures the time for distance calculation.
- `measure_cnst_model()` measures the time for model construction.
- `training()` performs distance learning.

Parameters specifying the length of strings and an input file names when running are defined within each function.

```
def main():
    #measure()
    #measure_cnst_model()
    training()
```

