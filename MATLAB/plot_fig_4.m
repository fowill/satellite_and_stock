clear
addpath('data')
addpath('utils')
load figure_table
for i=1:5
[~,I]=sort(table(:,2*i-1),1,"descend");
table(:,2*i-1:2*i)=table(I,2*i-1:2*i);
markets_name_sort{i}=markets_name(I);
end


x_arix=1:1:34;
c1=[245,10,10]./255;
c2=[91,188,214]./255;
c3=[217,217,217]./255;
c3=[168 187 218]./255;
sz=250;
Linewidth=1;
I=size(table,2)/2;
figure('Color',[1 1 1])
for i=1:I
    
    subplot(1,I,i)
    hold on
    for j=1:33
        if table(j,2*i)>0.1
        h1=scatter(table(j,2*i-1),-j,sz,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',c2,'LineWidth',Linewidth);
        else
        h2=scatter(table(j,2*i-1),-j,sz,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',c1,'LineWidth',Linewidth);
        end
    end
  
    ylim=get(gca,'ylim'); % 
    a1=mean(table(:,2*i-1));
    plot([a1,a1],ylim,'LineWidth',2,'LineStyle','--','Color',c1);
    plot([0,0],ylim,'LineWidth',2,'LineStyle','--','Color',[0 0 0]);
     
        set(gca,'YTick',-33:1:-1);
        set(gca,'XTick',[0 0.05,0.1]);
        set(gca,'YTicklabel',[flipud(markets_name_sort{i});{""}],'FontSize',10,'FontWeight','bold')
        set(gca,'tickdir','out')
        set(gca,'ticklength',[0 0])
        axis([-0.05,0.15,-34,0])
        %set(gca,'xminortick','on')
         ax=get(gca);
    ax_position=ax.Position;
    annotation('rectangle',[ax_position(1),ax_position(2)+ax_position(4),ax_position(3),0.05],'FaceColor',c3);
    annotation('textbox',[ax_position(1),ax_position(2)+ax_position(4),ax_position(3),0.05],'String',['h=',num2str(i)], ...
    HorizontalAlignment='center',VerticalAlignment='middle',FontSize=13,LineWidth=1,FontWeight='bold');

         


    box off
    ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
   set(ax2,'YTick', []);
   set(ax2,'XTick', []);
   set(ax2,'LineWidth', 1.2);
   box on
% 
%      


end
legend([h2,h1],{'Significantly positive','Positive but not significantly'}, ...
    "Box","off","FontSize",13,"Orientation","horizontal",'Location','south','FontWeight','bold')



