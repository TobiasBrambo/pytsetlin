

from numba import njit, prange
import numpy as np


from core.feedback import evaluate_clauses_training, get_update_p, update_clauses, evaluate_clause



@njit
def train_epoch(cb, wb, x, y, threshold, s, n_outputs, n_literals):

    cb = np.ascontiguousarray(cb)
    wb = np.ascontiguousarray(wb)

    indices = np.arange(x.shape[0], dtype=np.int32) # this does not need to be calulated at every epoch?
    np.random.shuffle(indices)

    clause_outputs = np.ones(cb.shape[0], dtype=np.uint8)

    for indice in indices:
        literals = x[indice]
        target = y[indice]
        
        evaluate_clauses_training(literals, cb, n_literals, clause_outputs)
        
        update_ps = np.zeros(n_outputs, dtype=np.float32)

        for i in range(n_outputs):
            update_ps[i] = get_update_p(wb, clause_outputs, threshold, i, i == target)
        
        if np.sum(update_ps) == 0.0:
            continue
            
        not_target_candidates = np.arange(n_outputs)
        not_target_candidates = not_target_candidates[not_target_candidates != target]
        not_target = np.random.choice(not_target_candidates)
        
        pos_update_p = update_ps[target]
        neg_update_p = update_ps[not_target]
        
        update_clauses(cb, wb, clause_outputs, pos_update_p, neg_update_p, 
                      target, not_target, literals, n_literals, s)


    return


@njit
def classify(x, clause_block, weight_block, n_literals):

    clause_outputs = evaluate_clause(x, clause_block, n_literals)

    class_sums = np.dot(weight_block.astype(np.float32), clause_outputs.astype(np.float32))

    return np.argmax(class_sums)
  
@njit(parallel=True)
def eval_predict(x, cb, wb, n_literals):
    y_pred = np.zeros(x.shape[0], dtype=np.int32)
    
    for i in prange(x.shape[0]):
        y_pred[i] = classify(x[i], cb, wb, n_literals)
    
    return y_pred
