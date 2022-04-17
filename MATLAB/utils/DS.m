function [result,p_value]=DS(y_real,y_model)
a=sign(y_real).*sign(y_model);
result=1-sum(a==-1)/length(y_real);
a(a>-1)=1;
a(a==-1)=0;

a=a-0.5;
meana=mean(a);
stda=std(a);

t=meana./stda*sqrt(length(y_real)-1);
p_value=1-normcdf(t,0,1);
end