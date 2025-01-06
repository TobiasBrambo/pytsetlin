import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.xor import get_xor


from tsetlin_machine import TsetlinMachine






if __name__ == "__main__":

    x, y = get_xor(n_rows = 4100, noise_fraction=0.3)
    xt, yt = get_xor(n_rows = 820, noise_fraction=0.0)


    tm = TsetlinMachine(n_clauses=1000,
                        threshold=2000,
                        s=15.0)

    tm.set_train_data(x, y)

    tm.set_eval_data(xt, yt)

    tm.train(training_epochs=1000)
