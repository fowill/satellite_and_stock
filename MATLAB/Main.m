%% Load data
clear 
tic
addpath('data')
addpath('utils')
load xy_all
%% Forecast process 
for k = 1:5
    h = k;  %horizons 
for j = 1:length(markets_name)
    xy_table = xy_close{j}; 
    [x,y,date] = generate_xy_from_xytable(xy_table, h);   
    [L,N] = size(x);
    initial = 500;   % initial window length    
    parfor i = 1:N   
       for t = initial:L
           xx = x(1:t-h,i);
           xx = [ones(t-h,1), xx];
           x_test = [1, x(t,i)];
           yy = y(1:t-h,:);

           beta = pinv(xx'*xx)*(xx'*yy);

           y_pred(t,i) = x_test*beta;
           y_bench(t,i) = mean(yy);
           y_real(t,i) = y(t,:);
           beta_all_temp(t,i) = beta(2); 
        end
    end
    beta_all{k}(j,:) = beta_all_temp(end,:);
    clear beta_all_temp
    y_pred = mean(y_pred,2);
    y_bench = mean(y_bench,2);
    y_real = mean(y_real,2);
    
    y_pred = y_pred(initial:end,:);
    y_bench = y_bench(initial:end,:);
    y_real = y_real(initial:end,:);  
    
    [table(j,2*k-1),table(j,2*k)] = ROOS(y_real,y_bench,y_pred);
    [table_DS(j,2*k-1),table_DS(j,2*k)] = DS(y_real,y_pred);
    cssed = cumsum((y_real-y_bench).^2-(y_real-y_pred).^2);
    CSSED{k}{j}=cssed;               
end
k
end
toc 
save('data/figure_table.mat','table','table_DS','CSSED','markets_name');
