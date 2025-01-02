

from numba import njit
import numpy as np


from core.feedback import evaluate_clauses_training, get_update_p, update_clauses, evaluate_clause



@njit
def train_epoch(cb, wb, x, y, threshold, s, n_outputs, n_literals):

    indices = np.arange(x.shape[0], dtype=np.int32) # this does not need to be calulated at every epoch...
    np.random.shuffle(indices)

    for indice in indices:
        
        clause_outputs = evaluate_clauses_training(x[indice], cb)

        pos_update_p = get_update_p(wb, clause_outputs, threshold, y[indice], True)



        update_ps = np.zeros(n_outputs, dtype=np.float32)

        for i in range(n_outputs):

            if i == y[indice]:
                update_ps[i] = 0.0
            else:
                update_ps[i] = get_update_p(wb, clause_outputs, threshold, i, False)    

        if np.sum(update_ps) == 0.0:
            return
        
        not_target = np.random.randint(n_outputs)

        while not_target == y[indice]:
            not_target = np.random.randint(n_outputs)

        neg_update_p = update_ps[not_target]


        update_clauses(cb, wb, clause_outputs, pos_update_p, neg_update_p, y[indice], not_target, x[indice], n_literals, s)

@njit
def classify(x, clause_block, weight_block, threshold, n_outputs, n_literals):

    max_class_sum = -threshold
    max_class = 0

    all_class_sums = np.zeros(n_outputs, dtype=np.float32)

    clause_outputs = np.zeros(clause_block.shape[0], dtype=np.int32)


    clause_outputs = evaluate_clause(x, clause_block, n_literals)


    for i in range(n_outputs):
        class_sum = np.dot(weight_block[i].astype(np.float32), clause_outputs.astype(np.float32))
        all_class_sums[i] = class_sum

        if class_sum > max_class_sum:
            max_class_sum = class_sum
            max_class = i
    
    return max_class
  
@njit
def eval_predict(x_eval, cb, wb, threshold, n_outputs, n_literals):
    

    y_pred = np.zeros(x_eval.shape[0], dtype=np.int32)

    for i in range(x_eval.shape[0]):

        y_pred[i] = classify(x_eval[i], cb, wb, threshold, n_outputs, n_literals)


    return y_pred

