

from core import executor
import tqdm
import numpy as np



class TsetlinMachine:
    def __init__(self,
                 n_clauses:int = 50,
                 s:float = 5.0,
                 threshold:int = 100,
                 n_literals = None):

        self.n_clauses = n_clauses        
        self.s = s    
        self.threshold = threshold    

        self.n_literals = n_literals
        self.n_outputs = None

        self.C = None        
        self.W = None

        self.x_train = None
        self.y_train = None

        self.x_eval = None
        self.y_eval = None



    def set_train_data(self, instances:np.array, targets:np.array, feature_negation:bool=True):
        

        if not isinstance(instances, np.ndarray):
            raise ValueError("x_train must be of type np.ndarray, x_train type: {}".format(type(instances)))

        if instances.shape[0] != targets.shape[0]:
            raise ValueError("Data x_train and y_train must have the same number of examples: {} != {}".format(instances.shape[0], targets.shape[0]))

        if instances.dtype != np.uint8:
            raise ValueError("Data x_train must be of type np.uint8, was: {}".format(instances.dtype))

        if targets.dtype != np.uint32:
            raise ValueError("Data y_train must be of type np.uint32, was: {}".format(targets.dtype))
        

        if feature_negation:
            self.n_literals = instances.shape[1]
            self.x_train = np.c_[instances, 1 - instances]

        else:
            self.n_literals = instances.shape[1] // 2
            self.x_train = instances

        self.y_train = targets

        self.n_outputs = len(np.unique(self.y_train)) 


    def set_eval_data(self, instances:np.array, targets:np.array, feature_negation:bool=True):
        
        if not isinstance(instances, np.ndarray):
            raise ValueError("x_eval must be of type np.ndarray, x_eval type: {}".format(type(instances)))

        if instances.shape[0] != targets.shape[0]:
            raise ValueError("Data x_eval and y_eval must have the same number of examples: {} != {}".format(instances.shape[0], targets.shape[0]))

        if instances.dtype != np.uint8:
            raise ValueError("Data x_eval must be of type np.uint8, was: {}".format(instances.dtype))

        if targets.dtype != np.uint32:
            raise ValueError("Data y_eval must be of type np.uint32, was: {}".format(targets.dtype))
        
        if feature_negation:
            self.x_eval = np.c_[instances, 1 - instances]
        else:
            self.x_eval = instances

        self.y_eval = targets

    
    def allocate_memory(self):


        if (self.n_literals is None) or (self.n_outputs is None):
            raise ValueError("failed to allocate memory, make sure data is set using set_train_data() and set_eval_data()")


        self.C = np.zeros((self.n_clauses, 2*self.n_literals), dtype=np.int8)

        self.W = np.random.choice(np.array([-1, 1]), size=(self.n_outputs, self.n_clauses), replace=True).astype(np.int32)


    def reset(self):
        pass


    def train(self, 
              training_epochs:int=10,
              eval_freq:int=1,
              hide_progress_bar:bool=False,
              early_stop_at:float=100.0):


        self.allocate_memory()

        r = {}

        best_eval_acc = "N/A"
        eval_score = "N/A"

        with tqdm.tqdm(total=training_epochs, disable=hide_progress_bar) as progress_bar:
            progress_bar.set_description(f"[0/{training_epochs}], Eval Acc: {eval_score}, Best Eval Acc: {best_eval_acc}")


            for epoch in range(training_epochs):

                executor.train_epoch(
                    self.C, self.W, self.x_train, self.y_train, 
                    self.threshold, self.s, self.n_outputs, self.n_literals
                    )
                
                if (epoch+1) % eval_freq == 0:

                    y_hat = executor.eval_predict(self.x_eval, self.C, self.W, self.n_literals)

                    eval_score = round(100 * np.mean(y_hat == self.y_eval), 2)

                    if best_eval_acc == 'N/A' or eval_score > best_eval_acc:
                        best_eval_acc = round(eval_score, 2)


                progress_bar.set_description(f"[{epoch+1}/{training_epochs}]: Eval Acc: {eval_score}, Best Eval Acc: {best_eval_acc}") 
                progress_bar.update(1)

                if not eval_score == 'N/A':
                    if eval_score >= early_stop_at:
                        break
    

    def predict(self, x):
        
        return executor.classify(x, self.C, self.W, self.n_literals)




if __name__ == "__main__":

    tm = TsetlinMachine()

    tm.train()