function f1_value = dummyFun1(f, x)
  [f1_value, f2_value] = f(x);
  save('tmp_f2_value.mat','f2_value') 
end
