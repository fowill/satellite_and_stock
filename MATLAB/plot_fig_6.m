clear 
addpath('data')
addpath('utils')
load figure_table.mat
for i=1:33
    CSSED_high(i)=CSSED{1}{i}(end);
end
[~,I]=sort(CSSED_high,"descend");
for i=1:5
CSSED{i}=CSSED{i}(I);
%data_all{1}{i}=data_all{1}{i}(I);
markets_name_sort{i}=markets_name(I);
end





c_title=[217,217,217]./255;
c_title=[231,156,65]./255;
c_title=[249 225 209]./255;
%c_title=[168 187 218]./255;
%c1=[249 225 209]./255;

color{1}=[234,69,65]./255;
color{2}=[220,111,49]./255;
color{3}=[240,169,28]./255;
color{4}=[91,188,214]./255;
color{4}=[70,129,214]./255;
color{5}=[150,150,150]./255;


figure('Color',[1 1 1])
for j=1:length(markets_name)
    subplot(5,7,j)
    hold on 
    for k=1:5
        k=6-k;
    plot(CSSED{k}{j}*10000,'LineWidth',2,'Color',color{k});
    end
    axis([0,length(CSSED{k}{j}),-0.1,1.5]) 
    if j>7
    axis([0,length(CSSED{k}{j}),-0.1,1])
    end
    if j>14
    axis([0,length(CSSED{k}{j}),-0.1,0.8])
    end
    if j>21
    axis([0,length(CSSED{k}{j}),-0.1,0.6])
    end
        if j>26
    axis([0,length(CSSED{k}{j}),-0.1,0.4])
    end
%     axis([-0.05,0.15,-34,0])
%     set(gca,'YTick',-33:1:-1);
     set(gca,'XTick',[0 250 500]);
     set(gca,'XTicklabel',['2019' ;'2020' ;'2021'],'FontSize',10,'FontWeight','bold');
%     set(gca,'YTicklabel',[{''};flipud(markets_name_sort{i});{""}],'FontSize',10)
%     set(gca,'tickdir','out')
%     set(gca,'ticklength',[0.01 0])
%         



    ylim=get(gca,'ylim'); 
    xlim=get(gca,'xlim'); 
    
    a1=310;
   % plot([a1,a1],[-0.1,1],'LineWidth',1.5,'LineStyle','--','Color',[0.3,0.3,0.3]);
    plot(xlim,[0,0],'LineWidth',1.5,'LineStyle','--','Color',[0.3,0.3,0.3]);
    
    box on 
   % title(markets_name{j})
    ax=get(gca);
    ax_position=ax.Position;
    annotation('rectangle',[ax_position(1),ax_position(2)+ax_position(4),ax_position(3),0.02],'FaceColor',c_title);
    annotation('textbox',[ax_position(1),ax_position(2)+ax_position(4),ax_position(3),0.02],'String',markets_name_sort{1}{j}, ...
        HorizontalAlignment='center',VerticalAlignment='middle',FontSize=11,LineWidth=1,FontWeight='bold');


        annotation('textbox',[ax_position(1),ax_position(2),ax_position(3),ax_position(4)],'String',' ', ...
    HorizontalAlignment='center',VerticalAlignment='middle',FontSize=8,LineWidth=1);
end
