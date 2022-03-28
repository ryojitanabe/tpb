"""
Run an optuna optimizer on a BBOB benchmark suite.
"""
import numpy as np
import cocoex  # only experimentation module
import optuna

def run(bbob_suite='bbob-biobj', budget_multiplier=20, optimizer='motpe'):        
    ### input
    output_folder = optimizer

    ### prepare
    suite = cocoex.Suite(bbob_suite, "", "")
    observer = cocoex.Observer(bbob_suite, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()
    optuna_seed = 1
    
    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing

        if problem.dimension > 20:
            break
        
        def my_objectives(trial):
            x = []
            for i in range(problem.dimension):
                x.append(trial.suggest_uniform('x{}'.format(i), -5, 5))
            y1, y2 = problem(x)
            return y1,y2                    

        sampler = optuna.samplers.MOTPESampler(n_startup_trials=11*fun.dimension-1, n_ehvi_candidates=24, seed=optuna_seed)
        study = optuna.create_study(directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(my_objectives, n_trials=budget_multiplier*problem.dimension)
        
        optuna_seed += 1        
        minimal_print(problem, final=problem.index == len(suite) - 1)        
                
if __name__ == '__main__':
    np.random.seed(seed=1)
    run_speed_test(bbob_suite='bbob-biobj', budget_multiplier=20)    
