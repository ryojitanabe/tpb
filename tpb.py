"""
A Two-phase Framework with a Bezier Simplex-based Interpolation Method

This is based on "example_experiment_for_beginners.py" provided by the COCO software.
"""
import numpy as np
import cocoex  # only experimentation module
import logging
import os
import torch
import torch_bsf
import click
import pygmo as pg
# from scipy.optimize import fmin_slsqp
# from smac.facade.func_facade import fmin_smac
import pybobyqa
#from pyDOE import lhs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ScalarProblem():
    """
    Define a single-objective scalar optimization problem.

    Attributes
    ----------
    fun : callable object
        A function to be minimized.
    scalarizing_fun: string
        The name of a scalarizing function.
    weight: float 
        A weight factor
    normalization: string
        A normalization method.
    approx_ideal: 1-d float ndarray
        An approximate ideal point for normalization.
    approx_nadir: 1-d float ndarray
        An approximate nadir point for normalization.
    max_fevals: int
        The maximum number of function evaluations.
    fx0: 1-d float ndarray
        An objective vector of the initial solution x0, which is generated on the center of the solution space (0, ..., 0). This is used only for the normalization method in HMO-CMA-ES.
    x_archive: float N-d list 
        A list of all N-dimensional solutions found in this optimization.
    f_archive: float M-d list 
        A list of all M-dimensional objective vectors found in this optimization.
    bsf_scalar_val: float
        A scalar value of the best-so-far solution in terms of the scalarizing function.
    bsf_x: float 1-d ndarray
        The best-so-far solution in terms of the scalarizing function.
    bsf_f: float 1-d ndarray
        The objective vector of the best-so-far solution in terms of the scalarizing function.
    fevals: float
        The number of function evaluations used so far in this scalar optimization.
    """    
    def __init__(self, fun, scalarizing_fun, weight, normalization, approx_ideal, approx_nadir, max_fevals, fx0=None):
        self.fun = fun
        self.scalarizing_fun = scalarizing_fun
        self.weight = weight
        self.normalization = normalization
        self.approx_ideal = approx_ideal
        self.approx_nadir = approx_nadir
        self.fx0 = fx0
        self.max_fevals = max_fevals
        
        # The ideal point and the nadir point are the same for some reason. This case occurs on a few multimodal functions, e.g., a run of an optimizer succeeded on f_1, but failed on f_2. To avoid the division by zero, the normalization is not performed.
        if self.approx_ideal is not None and np.allclose(self.approx_ideal, self.approx_nadir):
            self.normalization = 'no'

        # If w = 0 or 1, the normalization is unnecessary.
        if np.allclose(self.weight, 0) or np.allclose(self.weight, 1):
            self.normalization = 'no'
            
        self.x_archive = []
        self.f_archive = []        
        self.bsf_scalar_val = 1e+20
        self.bsf_x = None
        self.bsf_f = None
        self.fevals = 0

    def weighted_sum_2d(self, obj_vec):
        """   
        Evaluate the objective vector f(x) by a weighted sum function. 

        **NOTE**: This is only for bi-objective optimization.

        Parameters
        ----------
        obj_vec: 1-d float list
            The objective vector of the solution.

        Returns
        ----------
        scalar_val: float
            The scalarizing function value of f(x).
        ----------
        """
        if self.normalization == 'ideal_nadir':
            obj_vec = np.array(obj_vec)
            normalized_obj_vec = (obj_vec - self.approx_ideal) / (self.approx_nadir - self.approx_ideal)
            scalar_val = self.weight * normalized_obj_vec[0] + (1 - self.weight) * normalized_obj_vec[1]
        elif self.normalization == 'hmo':
            normalized_obj_vec = (obj_vec / abs(self.fx0))
            scalar_val = self.weight * normalized_obj_vec[0] + (1 - self.weight) * normalized_obj_vec[1]
        elif self.normalization == 'no':
            scalar_val = self.weight * obj_vec[0] + (1 - self.weight) * obj_vec[1]
        else:
            logger.error("%s is not defined.", self.normalization)
            exit(1)
            
        return scalar_val            

    def tchebycheff_mul_2d(self, obj_vec):
        """   
        Evaluate the objective vector f(x) by a weighted tchebycheff function. 

        **NOTE**: This is only for bi-objective optimization.

        Parameters
        ----------
        obj_vec: 1-d float list
            The objective vector of the solution.

        Returns
        ----------
        scalar_val: float
            The scalarizing function value of f(x).
        ----------
        """
        tch_value = -1e+20
        
        if self.normalization == 'no':
            # TODO: How should we set the approximated ideal point in the first run of a DFO? Unlike EMO algorithms (e.g., MOEA/D), no solution is available.
            logger.warning("A given solution was valuated by either of objective functions.")
            tch_value = self.weight * obj_vec[0] + (1 - self.weight) * obj_vec[1]                
        elif self.normalization == 'ideal_nadir':
            obj_vec = np.array(obj_vec)
            nor_obj_vec = (obj_vec - self.approx_ideal) / (self.approx_nadir - self.approx_ideal)
            weight_vec = np.array([self.weight, 1-self.weight])
            for w, f in zip(weight_vec, nor_obj_vec):
                tch_value = max(tch_value, w * abs(f - 0))
        else:
            logger.error("%s is not defined.", self.normalization)
            exit(1)
                        
        return tch_value

    def eval_scalar_fun(self, obj_vec):
        """   
        Evaluate the objective vector f(x) by a scalarizing function. 

        Parameters
        ----------
        obj_vec: 1-d float list
            The objective vector of the solution.

        Returns
        ----------
        scalar_val: float
            The scalarizing function value of f(x).
        ----------
        """               
        scalar_val = 1e+20
        if self.scalarizing_fun == 'ws':
            scalar_val = self.weighted_sum_2d(obj_vec)
        elif self.scalarizing_fun == 'tch':
            scalar_val = self.tchebycheff_mul_2d(obj_vec)
        else:
            logger.error("%s is not defined.", self.scalarizing_fun)
            exit(1)
            
        return scalar_val
    
    def eval(self, x):
        """   
        Evaluate the solution x by the objective functions. Then, evaluate its objective vector f(x) by a scalarizing function. 

        Parameters
        ----------
        x: 1-d float ndarray
            The solution.

        Returns
        ----------
        scalar_val: float
            The scalarizing function value of f(x).
        ----------
        """               
        # The SciPy implementations of some DFO optimizers (e.g., the SLSQP) do not terminate at a given maximum number of solutions. The following forces an optimizer to terminate the search by giving 1e+20 for any x. The following prevents the optimizer from using an additional fevals in terms of the objective functions.
        if self.fevals >= self.max_fevals:
            return 1e+20            

        # Evaluate the solution x by the objective functions
        obj_vec = self.fun(x)
        self.fevals += 1
        self.x_archive.append(x)
        self.f_archive.append(obj_vec)

        # Evaluate the objective vector f(x) by the scalar function
        scalar_val = self.eval_scalar_fun(obj_vec)

        # Update best-so-far information
        if scalar_val < self.bsf_scalar_val:
            self.bsf_scalar_val = scalar_val
            self.bsf_x = np.copy(x)
            self.bsf_f = np.copy(obj_vec)
            
        return scalar_val

    def best_x_f(self, fsf_x_archive, fsf_f_archive):
        """   
        Find the best so-far solution and its objective vector in terms of a given scalar function.

        Parameters
        ----------
        fsf_f_archive: float M-d list 
            A list of all M-dimensional objective vectors found so far.
        fsf_x_archive: float N-d list 
            A list of all N-dimensional solutions found so far.

        Returns
        ----------
        best_x: 1-d float ndarray
            The best solution found so far.
        best_f: 1-d float ndarray
            The objective vector of the best solution found so far.
        ----------
        """
        best_scalar_val = 1e+20
        best_x = None
        best_f = None
        
        for x, obj_vec in zip(fsf_x_archive, fsf_f_archive):
            scalar_val = self.eval_scalar_fun(obj_vec) 
            if scalar_val < best_scalar_val:
                best_scalar_val = scalar_val
                best_x = np.copy(x)
                best_f = np.copy(obj_vec)
                
        return best_x, best_f
        
def bsf_nondom_x(scalarizing_fun, weight_factors, fsf_f_archive, fsf_x_archive):
    """   
    Find the best so-far non-dominated solutions for a given weight vector set. The size of the non-dominated solution set must be the size of "weight_factors" or lower.

    TODO: Do we need to take into account "dominated" solutions?
    
    Parameters
    ----------
    scalarizing_fun: string
        The name of a scalarizing function.
    weight_factors: float list 
        A list of weight factors
    fsf_f_archive: float M-d list 
        A list of all M-dimensional objective vectors found so far.
    fsf_x_archive: float N-d list 
        A list of all N-dimensional solutions found so far.

    Returns
    ----------
    bsf_x_archive_data: float N-d list 
        A list of all N-dimensional solutions found best so far.
    bsf_f_archive_data: float M-d list 
        A list of all M-dimensional objective vectors found best so far.
    ndr: 1-d int ndarray
        The non-domination ranks of the best solutions for the scalar problems.
    ----------
    """            
    bsf_f_archive = []
    bsf_x_archive = []                
    bsf_points = np.array(fsf_f_archive)[np.array(fsf_f_archive).argmin(axis=0)]
    approx_ideal = bsf_points.min(axis=0)
    approx_nadir = bsf_points.max(axis=0)
    normalization = 'ideal_nadir'

    for weight in weight_factors:                
        # Find the best solution for the scalar problem with weight. Then, add it to the archive.
        my_fun = ScalarProblem(fun=None, scalarizing_fun=scalarizing_fun, weight=weight, normalization=normalization, approx_ideal=approx_ideal, approx_nadir=approx_nadir, max_fevals=0)
        bsf_x, bsf_f = my_fun.best_x_f(fsf_x_archive, fsf_f_archive)
        bsf_f_archive.append(bsf_f)
        bsf_x_archive.append(bsf_x)

        # Delete solutions duplicated in bsf_x. 
        del_mask = np.full(len(fsf_x_archive), True)
        for i, arc_x in enumerate(fsf_x_archive):
            if np.allclose(arc_x, bsf_x, rtol=0, atol=1e-12):
                del_mask[i] = False
        fsf_x_archive = np.array(fsf_x_archive)[del_mask]
        fsf_f_archive = np.array(fsf_f_archive)[del_mask]

    # https://esa.github.io/pygmo2/mo_utils.html
    # ndf (list of 1D NumPy int array): the non dominated fronts, dl (list of 1D NumPy int array): the domination list, dc (1D NumPy int array): the domination count, ndr (1D NumPy int array): the non domination ranks
    #ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=bsf_f_archive)
    _, _, _, ndr = pg.fast_non_dominated_sorting(points=bsf_f_archive)
        
    return bsf_x_archive, bsf_f_archive, ndr

def tpb(bbob_suite='bbob-biobj', budget_multiplier=20, n_weights=3, opt_budget_rate=0.9, optimizer='bobyqa', scalarizing_fun='ws', bez_degree=2, interpolator='bez'):
    """   
    Run the TPB instance on the bi-objective BBOB function set.

    Parameters
    ----------
    bbob_suite: string
        The name of a bbob suite (bbob, bbob-noisy, bbob-biobj, bbob-largescale, bbob-mixint, bbob-biobj-mixint)
    budget_multiplier: int
        A budget_multiplier. For example, the maximum number of function evaluations is 20 * dim when budget_multiplier = 20.
    n_weights: int
        The number of weights.
    opt_budget_rate: float
        A budget rate used in the first optimization phase. For example, the maximum number of function evaluations in the first phase is 36 when the total budget = 40 and opt_budget_rate=0.9.
   optimizer: string
        The name of a single-objective optimizer.
   bez_degree: int
        A degree parameter for the Bezier simplex.
   scalarizing_fun: string
        A scalarizing function (ws or tch)
   interpolator: string
        An interpolator (bez or no). "no" means that the interpolation is not performed.
    """
    
    ### input
    output_folder = f"{optimizer}_{interpolator}_w{n_weights}_ob{opt_budget_rate}_bezdeg{bez_degree}_{scalarizing_fun}_b{budget_multiplier}"
    
    ### prepare
    suite = cocoex.Suite(bbob_suite, "", "")
    observer = cocoex.Observer(bbob_suite, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        lbounds = np.full(problem.dimension, -5)
        ubounds = np.full(problem.dimension, 5)
        bbob_bounds = [(-5, 5)] * problem.dimension

        # TPB is performed only for {2, 3, 5, 10, 20}.
        if problem.dimension > 20:
            break
        
        # 1. The first optimization phase        
        # fsf_x_archive and fsf_f_archive maintain all solutions X and f(X) found so far, respectively.
        fsf_x_archive = []
        fsf_f_archive = []

        # A budget for optimizing a single scalar problem. 
        single_opt_budget = int(np.floor(opt_budget_rate * (budget_multiplier * problem.dimension) / n_weights))

        # Evenly generate the weight factors. For example, when n_weights = 3, weight_factors = [0. 0.5, 1]
        # Note that this works only for two-objective problems. For details, see Section 4.2 in the GECCO paper.
        weight_factors = np.linspace(0, 1, n_weights)

        # Rearrange the order of weight factors. The extreme weight factors should be the first and second. The remaining weight factors are sorted in an ascending order.
        # e.g., [0.   0.25 0.5  0.75 1.  ] => [0.   1.   0.25 0.5  0.75]        
        tmp = weight_factors[1]
        weight_factors[1] = weight_factors[-1]
        weight_factors[-1] = tmp
        
        tmp_arr = weight_factors[2:]
        tmp_arr =  np.sort(tmp_arr)
        weight_factors = np.concatenate([weight_factors[:2], tmp_arr])                                      

        # Step 1. First, a DFO optimizes each objective with w=0, 1 to obtain approximated ideal and nadir points. Then, the DFO is applied to the scalar problems with the remaining weight factors. Note that all objective vectors are normalized only in this phase.
        for i, weight in enumerate(weight_factors):            
            # It is assumed that the first and second weight factors are to optimize each objective(i.e., w=0.0 and w=1.0).
            # This should be "if i < NUM_OBJECTIVES"
            if i in [0, 1]:
                normalization = 'no'
                approx_ideal = None
                approx_nadir = None
            else:
                # Find M extreme points from all points found so far
                bsf_points = np.array(fsf_f_archive)[np.array(fsf_f_archive).argmin(axis=0)]
                approx_ideal = bsf_points.min(axis=0)
                approx_nadir = bsf_points.max(axis=0)                
                normalization = 'ideal_nadir'
            
            my_fun = ScalarProblem(fun=problem, scalarizing_fun=scalarizing_fun, weight=weight, normalization=normalization, approx_ideal=approx_ideal, approx_nadir=approx_nadir, max_fevals=single_opt_budget)

            # Set the initial solution x0. For the first time, x0 is set to the center of the solution domain (0, ..., 0) in the case of the bi-objective BBOB functions. After that, x0 is set to the best solution found so far in terms of a scalar function g with each w.
            if i == 0:
                x0 = problem.initial_solution
            else:
                x0, _ = my_fun.best_x_f(fsf_x_archive, fsf_f_archive)

            # Run a single-objective optimizer on a given scalar problem
            if optimizer == 'bobyqa':
                res = pybobyqa.solve(my_fun.eval, x0, bounds=(lbounds, ubounds), maxfun=single_opt_budget, do_logging=False, print_progress=False, seek_global_minimum=False)
            # elif optimizer == 'slsqp':
            #     res = fmin_slsqp(my_fun.eval, x0, bounds=bbob_bounds, acc=1e-11, iter=3000, disp=False)
            else:
                logger.error("%s is not defined.", optimizer)                
                exit(1)

            # Add all solutions generated in this trial to fsf_x_archive and fsf_f_archive
            fsf_x_archive.extend(my_fun.x_archive)
            fsf_f_archive.extend(my_fun.f_archive)

        # The following is for TPB2, which randomly generate (11*problem.dimension-1) solutions by Latin hypercube sampling, instead of BOBYQA.
        # 1. The first phase in TPB2
        # sample = lhs(problem.dimension, 11*problem.dimension-1, criterion='center')
        # # Linearly map each solution from [0,1]^dim to [-5,5]^dim
        # sample = (ubounds - lbounds) * sample + lbounds
        # for x in sample:
        #     obj_x = problem(x)
        #     fsf_x_archive.append(x)
        #     fsf_f_archive.append(obj_x)            
            
        # 2. The interpolation phase
        if interpolator == 'bez':        
            bsf_x_archive, bsf_f_archive, ndr = bsf_nondom_x(scalarizing_fun, weight_factors, fsf_f_archive, fsf_x_archive)
            
            # len(fsf_x_archive) is the number of function evaluations used in the first phase.
            intp_budget = budget_multiplier * problem.dimension - len(fsf_x_archive)
               
            # It is impossible to interporate a single solution.
            # TODO: What should we do ELSE? Should we perform a random search? Should we perform the restart phase with a DFO again?
            if len(bsf_x_archive) > 1 and intp_budget > 0:
                intp_w_vecs = []
                for w in weight_factors:
                    intp_w_vecs.append([w, 1-w])

                # The following is copied from https://gitlab.com/hmkz/pytorch-bsf#how-to-use
                ts = torch.tensor(intp_w_vecs)
                xs = torch.tensor(bsf_x_archive)

                # Train a Bezier Simplex model
                # NOTE: "gpus=0" causes an error in pytorch-bsf. The argument "auto_select_gpus=True" in line 215 in bezier_simplex.py should be "auto_select_gpus=False" when the option "gpus" is set to 0.
                # https://gitlab.com/hmkz/pytorch-bsf/-/blob/master/torch_bsf/bezier_simplex.py
                bs = torch_bsf.fit(params=ts, values=xs, degree=bez_degree, max_epochs=100, gpus=0)

                t = []
                # Generate uniformly-distributed parameters in the range [0, 1]. Note that "+2" is for (0, 1) and (1, 0), which are removed later.
                # TODO: "+2" works only for two-objective problems. It should be "+NUM_OBJECTIVES".
                for t0 in np.linspace(0, 1, intp_budget+2):                
                    t.append([t0, 1-t0])

                # Delete the first (0, 1) and the last (1, 0) since xs include them
                t.pop(0)
                t.pop(-1)
                
                intp_X = bs(t)
                intp_X = intp_X.to('cpu').detach().numpy().copy()

                # This shuffling method slightly improves the anytime performance of TPB without any additional computation.
                intp_X = np.random.permutation(intp_X)

                # Evalute the interpolated solutions by the objective function. obj_x may be unnecessary.
                for x in intp_X:
                    obj_x = problem(x)
                    
        # End the optimization on the single problem instance.
        minimal_print(problem, final=problem.index == len(suite) - 1)
               
# python tpb.py -bm 20 -nw 3 -obr 0.9 -o bobyqa -bd 2 -sf ws -i bez    
@click.command()
@click.option('--budget_multiplier', '-bm', required=True, default=20, type=int, help='A budget_multiplier (e.g., "20" * dim).')
@click.option('--n_weights', '-nw', required=True, default=3, type=int, help='The number of weights.')
@click.option('--opt_budget_rate', '-obr', required=True, default=0.9, type=float, help='A budget rate used in the first optimization phase. For example, the maximum number of function evaluations in the first phase is 36 when the total budget = 40 and opt_budget_rate=0.9.')
@click.option('--optimizer', '-o', required=True, default='bobyqa', type=str, help='A single-objective optimizer.')
@click.option('--bez_degree', '-bd', required=False, default=2, type=int, help='A degree parameter for the Bezier simplex.')
@click.option('--scalarizing_fun', '-sf', required=True, default='ws', type=str, help='A scalarizing function.')        
@click.option('--interpolator', '-i', required=True, default='bez', type=str, help='An interpolator.')
def run(budget_multiplier, n_weights, opt_budget_rate, optimizer, bez_degree, scalarizing_fun, interpolator):    
    tpb(bbob_suite='bbob-biobj', budget_multiplier=budget_multiplier, n_weights=n_weights, opt_budget_rate=opt_budget_rate, optimizer=optimizer, scalarizing_fun=scalarizing_fun, bez_degree=bez_degree, interpolator=interpolator)
    
if __name__ == '__main__':
    np.random.seed(seed=1)
    run()    
