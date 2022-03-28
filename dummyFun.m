function [f1_value, f2_value] = dummyFun(f, problem, x)
  f_values = f(problem, x);
  f1_value = f_values(1);
  f2_value = f_values(2); 
end
