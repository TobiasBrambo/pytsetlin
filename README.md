# PyTsetlin


README.md under construction...

A low-code, feature-POOR, Pythonic implementation of a Coalesced Tsetlin Machine. This is not intended to be a feature-rich or speed-optimized implementation; see relevant repositories like _, _, for that. However, it's intended to be an easy-to-use TM programmed in Python, making it easy to plug-and-play new ideas to quickly test get some results, either on a input level or TM structure level. Also, by being completely written in Python, it's easily readable to compare the implementation to the theory presented in the papers, maybe making it easier to understand.

Even though this repo is not focused on speed, I have made some heavy functions compatible for Numba compilation. Without this, the code is so slow that it makes the implementation unusable.

## Installation

Clone or fork this repository and install the required dependencies:

```bash
cd pytsetlin
pip install -r requirements.txt
```

## Example

Here's a basic example of how to use the Tsetlin Machine:

```python
from data.mnist import get_mnist

from tsetlin_machine import TsetlinMachine

X_train, X_test, y_train, y_test = get_mnist()

tm = TsetlinMachine(n_clauses=500,
                    threshold=625,
                    s=10.0)

tm.set_train_data(X_train, y_train)

tm.set_eval_data(X_test, y_test)

r = tm.train()

print(r)

>>> {'train_time': [40.81219186200178, 27.42472163400089, 25.56161936299759, 24.745437929996115, 23.14931125700241, 22.180752667001798, 21.975613653994515, 21.790488855003787, 21.90930388400011, 20.580369818999316], 'eval_acc': [40.81219186200178, 27.42472163400089, 25.56161936299759, 24.745437929996115, 23.14931125700241, 22.180752667001798, 21.975613653994515, 21.790488855003787, 21.90930388400011, 20.580369818999316], 'best_eval_acc': 96.46}

```

## Notes

1. Input data must be binary (dtype=np.uint8 for features, np.uint32 for labels)
2. The implementation uses Numba for efficient computation
3. Memory is allocated automatically when training begins


## License

MIT Licence

## References

Add paper refs
