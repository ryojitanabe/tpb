%
% This is based on exampleexperiment.m provided by the COCO software.
% https://github.com/numbbo/coco/blob/master/code-experiments/build/matlab/exampleexperiment.m
%
% This script runs random search for BUDGET_MULTIPLIER*DIM function
% evaluations on a COCO suite and can serve also as a timing experiment.
%
% This example experiment allows also for easy implementation of independent
% restarts by simply increasing NUM_OF_INDEPENDENT_RESTARTS. To make this
% effective, the algorithm should have at least one more stopping criterion
% than just a maximal budget.
%
more off; % to get immediate output in Octave

%%%%%%%%%%%%%%%%%%%%%%%%%
% Experiment Parameters %
%%%%%%%%%%%%%%%%%%%%%%%%%
BUDGET_MULTIPLIER = 20; % algorithm runs for BUDGET_MULTIPLIER*dimension funevals
NUM_OF_INDEPENDENT_RESTARTS = 1e9; % max. number of independent algorithm
% restarts; if >0, make sure that the
% algorithm is not always doing the same thing
% in each run (which is typically trivial for
% randomized algorithms)

%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare Experiment    %
%%%%%%%%%%%%%%%%%%%%%%%%%

% choose a test suite and a matching logger, for
% example one of the following:
%
% bbob               24 unconstrained noiseless single-objective functions
% bbob-biobj         55 unconstrained noiseless bi-objective functions
% [bbob-biobj-ext     92 unconstrained noiseless bi-objective functions]
% bbob-largescale    24 unconstrained noiseless single-objective functions in large dimensions
% [bbob-constrained* 48 constrained noiseless single-objective functions]
% bbob-mixint        24 unconstrained noiseless single-objective functions with mixed-integer variables
% bbob-biobj-mixint  92 unconstrained noiseless bi-objective functions with mixed-integer variables
%
% Suites with a star are partly implemented but not yet fully supported.
%
suite_name = 'bbob-biobj';
observer_name = suite_name;
observer_options = strcat('result_folder: KRVEA_bm20', ...
    [' algorithm_name: KRVEA '...
    ' algorithm_info: Implementation of PlatEMO ']);

% initialize suite and observer with default options,
% to change the default, see 
% http://numbbo.github.io/coco-doc/C/#suite-parameters and
% http://numbbo.github.io/coco-doc/C/#observer-parameters
% for details.
suite = cocoSuite(suite_name, '', '');
observer = cocoObserver(observer_name, observer_options);

% set log level depending on how much output you want to see, e.g. 'warning'
% for fewer output than 'info'.
cocoSetLogLevel('info');

% keep track of problem dimension and #funevals to print timing information:
printeddim = 1;
doneEvalsAfter = 0; % summed function evaluations for a single problem
doneEvalsTotal = 0; % summed function evaluations per dimension
printstring = '\n'; % store strings to be printed until experiment is finished

%%%%%%%%%%%%%%%%%%%%%%%%%
% Run Experiment        %
%%%%%%%%%%%%%%%%%%%%%%%%%
while true
    % get next problem and dimension from the chosen suite:
    problem = cocoSuiteGetNextProblem(suite, observer);
    if ~cocoProblemIsValid(problem)
        break;
    end
    dimension = cocoProblemGetDimension(problem);

    if dimension > 20
      break;
    end    
    
    % printing
    if printeddim < dimension
      if printeddim > 1
        elapsedtime = toc;
        printstring = strcat(printstring, ...
            sprintf('   COCO TIMING: dimension %d finished in %e seconds/evaluation\n', ...
            printeddim, elapsedtime/double(doneEvalsTotal)));
        tic;
      end
      doneEvalsTotal = 0;
      printeddim = dimension;
      tic;
    end
    
    % restart functionality: do at most NUM_OF_INDEPENDENT_RESTARTS+1
    % independent runs until budget is used:
    i = -1; % count number of independent restarts
    while (BUDGET_MULTIPLIER*dimension > (cocoProblemGetEvaluations(problem) + ...
                                          cocoProblemGetEvaluationsConstraints(problem)))
        i = i+1;
        if (i > 0)
            fprintf('INFO: algorithm restarted\n');
        end
        doneEvalsBefore = cocoProblemGetEvaluations(problem) + ...
                          cocoProblemGetEvaluationsConstraints(problem);
        
	lbounds = -5 * ones(1, dimension);
	ubounds = 5 * ones(1, dimension);	
		
	my_f = @(x)dummyFun(@cocoEvaluateFunction, problem, x);
	my_f1 = @(x)dummyFun1(my_f, x);
	my_f2 = @(x)dummyFun2(x);

	try
	  platemo('objFcn',{my_f1, my_f2},'lower',lbounds,'upper',ubounds,'algorithm',@KRVEA, 'maxFE',BUDGET_MULTIPLIER* dimension);
	catch ME
	  warning('The MOEA unexpectedly terminated');
	  display(cocoProblemGetName(problem))	  
	end
	  	
        % check whether things went wrong or whether experiment is over:
        doneEvalsAfter = cocoProblemGetEvaluations(problem) + ...
                         cocoProblemGetEvaluationsConstraints(problem);
        if cocoProblemFinalTargetHit(problem) == 1 ||...
                doneEvalsAfter >= BUDGET_MULTIPLIER * dimension
            break;
        end
        if (doneEvalsAfter == doneEvalsBefore)
            fprintf('WARNING: Budget has not been exhausted (%d/%d evaluations done)!\n', ....
                doneEvalsBefore, BUDGET_MULTIPLIER * dimension);
            break;
        end
        if (doneEvalsAfter < doneEvalsBefore)
            fprintf('ERROR: Something weird happened here which should not happen: f-evaluations decreased');
        end
        if (i >= NUM_OF_INDEPENDENT_RESTARTS)
            break;
        end
    end
    
    doneEvalsTotal = doneEvalsTotal + doneEvalsAfter;
end

elapsedtime = toc;
printstring = strcat(printstring, ...
    sprintf('   COCO TIMING: dimension %d finished in %e seconds/evaluation\n', ...
    printeddim, elapsedtime/double(doneEvalsTotal)));
fprintf(printstring);

cocoObserverFree(observer);
cocoSuiteFree(suite);
