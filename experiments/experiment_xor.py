
# experiments_xor.py 

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.xor import get_xor


from tsetlin_macine import TsetlinMachine






if __name__ == "__main__":

    x, y = get_xor(n_rows = 4100, flip_fraction=0.2, add_negated=True)
    xt, yt = get_xor(n_rows = 820, flip_fraction=0.0, add_negated=True)


    tm = TsetlinMachine(n_clauses=10,
                        threshold=20,
                        s=5.0,
                        n_literals=6)

    tm.set_train_data(x, y)

    tm.set_eval_data(xt, yt)

    tm.train(training_epochs=1000)


    # import numpy as np
    # from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    # from tmu.models.classification.vanilla_classifier import TMClassifier

    # import tqdm

    # tm = TMCoalescedClassifier(
    #         number_of_clauses=1000,
    #         T=2000,
    #         s=15.0,
    #         platform='CPU',
    #         boost_true_positive_feedback=0
    #     )

    # for epoch in tqdm.tqdm(range(1000)):

    #     tm.fit(x.astype(np.uint32), y.astype(np.uint32))

    #     res = 100 * (tm.predict(xt) == yt).mean()
    #     print(res)
    #     if res == 100:
    #         break