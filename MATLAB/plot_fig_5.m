clear
addpath('data')
addpath('utils')
load figure_table

x_arix=1:1:29;
c1=[245,10,10]./255;
c2=[91,188,214]./255;
c3=[217,217,217]./255;
%c3=[168 187 218]./255;
sz=250;
Linewidth=1;
N=size(table_DS,2)/2;
figure('Color',[1 1 1])

for i=1:N
[~,I]=sort(table_DS(:,2*i-1),1,"descend");
table_DS(:,2*i-1:2*i)=table_DS(I,2*i-1:2*i);
markets_name_sort{i}=markets_name(I);
end



for i=1:N
    
    subplot(1,N,i)
    hold on
    for j=1:33

        if  table_DS(j,2*i-1)<0.5
        h1=scatter(table_DS(j,2*i-1),-j,sz,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',[1 1 1],'LineWidth',Linewidth);
        end
        if table_DS(j,2*i)>0.1 && table_DS(j,2*i-1)>=0.5
        h2=scatter(table_DS(j,2*i-1),-j,sz,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',c2,'LineWidth',Linewidth);
        end
        if table_DS(j,2*i)<0.1 && table_DS(j,2*i-1)>=0.5
        h3 =scatter(table_DS(j,2*i-1),-j,sz,'MarkerEdgeColor',[0,0,0],'MarkerFaceColor',c1,'LineWidth',Linewidth);
        end
      end
  % grid on 
   set(subplot(1,N,i),'GridColor','k','GridAlpha',0.15);
   axis([0.4,0.7,-34,0])
    box on 
    ylim=get(gca,'ylim'); 
    a1=mean(table_DS(:,2*i-1));
    plot([a1,a1],ylim,'LineWidth',1.5,'LineStyle','--','Color',c1);
    hold on
    plot([0.5,0.5],ylim,'LineWidth',1.5,'LineStyle','--','Color',[0 0 0]);
    ax=get(gca);
    ax_position=ax.Position;
    annotation('rectangle',[ax_position(1),ax_position(2)+ax_position(4),ax_position(3),0.05],'FaceColor',c3);
    annotation('textbox',[ax_position(1),ax_position(2)+ax_position(4),ax_position(3),0.05],'String',['h=',num2str(i)], ...
    HorizontalAlignment='center',VerticalAlignment='middle',FontSize=13,LineWidth=1,FontWeight='bold');

      
        set(gca,'YTick',-33:1:-1);
        set(gca,'XTick',[0.45,0.5,0.55,0.6,0.65]);
        set(gca,'YTicklabel',[flipud(markets_name_sort{i});{""}],'FontSize',10,'FontWeight','bold')
        set(gca,'tickdir','out')
        set(gca,'ticklength',[0 0])
   
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


end
legend([h3,h2,h1],{'Significantly higher than 50%','Higher than 50% but not significant','Lower than 50%'}, ...
    "Box","off","FontSize",13,"Orientation","horizontal",'Location','south','FontWeight','bold')



