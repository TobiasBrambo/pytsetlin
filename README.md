# PyTsetlin


README.md under construction...

A low-code, feature-POOR, Pythonic implementation of a Coalesced Tsetlin Machine. This is not intended to be a feature-rich or speed-optimized implementation; see relevant repositories like [
TMU](https://github.com/cair/tmu) and [green-tsetlin](https://github.com/ooki/green_tsetlin) for that. However, it's intended to be an easy-to-use TM programmed in Python, with the intent of making it accessible to plug-and-play new ideas and be able to get some results, either on an input level or TM memory level. Also, since the implementation is written entirely in Python, the code can be compared with the theoretical concepts presented in the papers, potentially making it easier to grasp.

Even though this repo is not focused on speed, I have made some functions compatible for Numba compilation. Without this, the code would be so slow that it deems the implementation unusable.

## Installation

1. Install package to environment to use in other projects:
```bash
pip install git+https://github.com/Sebastianostby/pytsetlin.git # install pytsetlin in env.
```

2. Clone or template this repository and install the required dependencies:

```bash
cd pytsetlin
pip install -r requirements.txt
```

## Example

Here's a basic example of how to use the Tsetlin Machine:

```python
from pytsetlin import TsetlinMachine
from pytsetlin.data.mnist import get_mnist

X_train, X_test, y_train, y_test = get_mnist()

tm = TsetlinMachine(n_clauses=500,
                    threshold=625,
                    s=10.0,
                    n_threads=20)

tm.set_train_data(X_train, y_train)

tm.set_eval_data(X_test, y_test)

r = tm.train(training_epochs=10)

# progress bar for visualization
>>> Eval Acc: 96.31, Best Eval Acc: 96.31 (10): 100%|██████████| 10/10 [01:03<00:00,  6.30s/it]
```

```python
print(r)
>>> {'train_time': [12.25, 5.77, 5.42, 4.96, 6.83, 4.71, 4.58, 4.88, 4.11, 5.9], 'eval_acc': [91.56, 92.97, 93.45, 94.42, 94.24, 94.71, 94.82, 95.1, 95.11, 96.31], 'best_eval_acc': 96.31, 'best_eval_epoch': 10}
```
Note performance may vary depending on system! 

## Literature References

* Core Papers 

     * [The Tsetlin Machine](https://arxiv.org/abs/1804.01508) introduces the Tsetlin Machine.

     * [Coalesced Multi-Output Tsetlin Machines](https://arxiv.org/abs/2108.07594) the variation this repo is largly based on.

* Other Cool Developments 

    * [Sparse Tsetlin Machine](https://arxiv.org/abs/2405.02375)


## Notes

1. Input data must be binary (dtype=np.uint8 for features, np.uint32 for labels)
2. The implementation uses Numba for efficient computation
3. Memory is allocated automatically when training begins


## License

MIT Licence

