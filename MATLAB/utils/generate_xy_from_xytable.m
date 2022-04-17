function [x,y,date]=generate_xy_from_xytable(xy_table,h)
%% generate x
    x_table = xy_table(:,1:end-1);
    [~,N] = size(x_table);
    delta_x = [zeros(1,N);log(x_table{2:end,:})-log(x_table{1:end-1,:})]; 
    delta_count = delta_x;  
    [aa,bb] = find(delta_x~=0);
    for i=1:length(aa)
        delta_count(aa(i),bb(i))=aa(i);
    end
    temp_index = x_table;
    delta_count(1,:) = 1;
    delta_count(delta_count==0) = nan;
    temp_index{:,:} = delta_count;
    temp_index = retime(temp_index,'regular','previous','TimeStep',hours(24));
    delta_count = [zeros(1,N);temp_index{3:end,:}-temp_index{2:end-1,:}]+0.00001;

    xy_table{:,1:end-1} = delta_x./delta_count;
    xy_table = xy_table(~isnan(xy_table{:,end}),:);
    x = xy_table{1:end,1:end-1};
  %% generate y    
    y = xy_table{:,end};
    y = (log(y(1+h:end,:))-log(y(1:end-h,end)))/h;%sqrt(h);
  %% Match time
    y = y(2:end,:);
    x = x(1:end-h-1,:);
  %% genetae date
    date = xy_table.Time(1:end-h-1,:);
    date.Format='yyyy-MM-dd';
end