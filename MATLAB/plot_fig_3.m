clear
addpath('data')
addpath('utils')
load xy_all
c1=[107 155  195]./255;
c1=[218 230  240]./255;
c2=[244 170  17]./255;

c1=[204 204  204]./255;
c3=[224 113 46]./255;
%c3=[244 170  17]./255;
b=readtable('RWI.xlsx');
RWI=[b(:,1),b(:,3)];

RWI=timetable(RWI{:,1},RWI{:,2},'VariableNames',{'RWI'});
RWI.Time.TimeZone='UTC';


a=readtable('BDI.csv');
BDI=a(:,1:2);
for i=1:length(BDI{:,1})
    value(i,:)=str2double(cell2mat(BDI{i,2}));
end
BDI=timetable(BDI{:,1},value,'VariableNames',{'BDI'});
BDI.Time.TimeZone='UTC';
load xy_all.mat
xy=xy_close{1};
counts=xy(:,1);
counts{:,1}=sum(xy{:,1:48},2);
xx=synchronize(BDI,counts);
xx=synchronize(xx,RWI);
xx=retime(xx,"daily","linear");
xx=xx(~isnan(xx{:,1}),:);
xx=xx(~isnan(xx{:,2}),:);
xx=xx(xx.Time<'2021-10-10',:);
figure('Color',[1 1 1]);
x3=xx.tokyo;
x2=xx.BDI;
x1=xx.RWI;
xx.Time.Format='yyyy-MM-dd';
xx1=log(x1(2:end,:))-log(x1(1:end-1,:));
xx2=log(x2(2:end,:))-log(x2(1:end-1,:));
xx3=log(x3(2:end,:))-log(x3(1:end-1,:));

xx1=zscore(xx1);
xx2=zscore(xx2);
xx3=zscore(xx3);



x1=mapminmax(x1')';
x2=mapminmax(x2')';
x3=mapminmax(x3')';
x1=(x1+1)./2;
x2=(x2+1)./2;
x3=(x3+1)./2;

h1=area(x1,'FaceColor',c1,'EdgeColor','none');
hold on 
h2=area(x2,'FaceColor',c2,'EdgeColor','none');
h3=plot(x3,'LineWidth',2,'Color',c3);
set(gca,'XLim',[0 365*5-80])
set(gca,'XTick',[0 365 365*2 365*3 365*4],'XTickLabel', ...
    ['2017-01-01' ;'2018-01-01' ;'2019-01-01'; '2020-01-01' ;'2021-01-01'], ...
    'FontSize',10,'FontWeight','bold');

set(gca,'LineWidth', 1);

legend([h3,h2,h1],{'GNC','BDI','RWI/ISL'}, ...
    "Box","off","FontSize",13,"Orientation","horizontal",'Location','south','FontWeight','bold')







