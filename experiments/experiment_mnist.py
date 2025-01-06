import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mnist import get_mnist

from tsetlin_macine import TsetlinMachine






if __name__ == "__main__":

    X_train, X_test, y_train, y_test = get_mnist()


    # import numpy as np
    # from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    # from tmu.models.classification.vanilla_classifier import TMClassifier

    # import tqdm

    # tm = TMCoalescedClassifier(
    #         number_of_clauses=100,
    #         T=62,
    #         s=10.0,
    #         platform='CPU',
    #         boost_true_positive_feedback=1,
    #     )

    # for epoch in tqdm.tqdm(range(20)):

    #     tm.fit(X_train.astype(np.uint32), y_train.astype(np.uint32))

    #     res = (tm.predict(X_test) == y_test).mean()

    #     print(res)

    tm = TsetlinMachine(n_clauses=500,
                        threshold=625,
                        s=10.0)

    tm.set_train_data(X_train, y_train)

    tm.set_eval_data(X_test, y_test)

    tm.train(training_epochs=20)
    



    # WHY DOES IT NOT CRASH WHEN I DONT ADD NEGATED LITS< AND HARD DEFINE N LITS, SHOULD IT NOT FAIL, lit_k + n_lits here??????????