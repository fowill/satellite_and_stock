function [result,p_value]=ROOS(y_real,benchmark,y_model)
result=((y_real-y_model)'*(y_real-y_model))/((y_real-benchmark)'*(y_real-benchmark));
result=(1-result)*100;
p_value=Perform_CW_test(y_real,benchmark,y_model);
end