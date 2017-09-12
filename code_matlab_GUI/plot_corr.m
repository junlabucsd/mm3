function [] = plot_corr(pool_corr, pool_data)

%% calculate correlation efficient
ft1 = fittype('a*x+b');
fit_a = zeros(11,11);
fit_b = zeros(11,11);
% fit_pval = zeros(11,11);
corr_coef = zeros(11,11);
corr_pval = zeros(11,11);

for i = 1:11
    for j = 1:11
       fit_temp = fit(pool_data(:,i),pool_data(:,j),ft1);
       [corr_coef(i,j),corr_pval(i,j)] = corr(pool_data(:,i),pool_data(:,j),'type','Spearman');
       fit_a(i,j) = fit_temp.a;
       fit_b(i,j) = fit_temp.b;
    end
end

x_fit = 0.5:0.1:1.5;

%% plot: correlations - scattereed
colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator

positions = [400, 400, 1255, 1350];

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'C/<C>',...
          'D/<D>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>'...
          'B/<B>',};

fig1 = figure;
set(fig1,'Position',positions(1,:));
hold on;

k = 1;
for i = 1:11
    for j = 1:11
        h = subplot('Position',[j/12.85,(12-i)/12.85,1/13,1/13]);
%         h = subplot(8,8,k);
        if i~=j
            hold on           
            h1 = plot(pool_data(:,j),pool_data(:,i));
            h1.Color = colors(1,:); set(h1,'LineWidth',0.5,'Markersize',2.5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');
        end
                
        if i==11
            xlabel(labels(j),'fontsize',20)
        end
        xlim([0.4 1.6])
        set(gca,'XScale','linear','XTick',[],'XTickLabel',{})
        
        if j==1
            ylabel(labels(i),'fontsize',20) 
        end
        ylim([0.4 1.6])
        set(gca,'YScale','linear','YTick',[],'YTickLabel',{});

%         if j==1
%             set(gca,'XScale','linear','XTick',[0.4 1 1.6],'XTickLabel',{'0.4','1','1.6'})
%             set(gca,'YScale','linear','YTick',[0.4 1 1.6],'YTickLabel',{'0.4','1','1.6'})
%         end
        set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])  
        
        k = k+1;
    end
end
% text(-4.5, 12, '\delta(\tau_{cyc})\neq0','FontSize',30);
% text(-4.5, 12, '\delta(S_0)\neq0','FontSize',30);
% text(-4.5, 12, '\delta(\lambda)\neq0','FontSize',30);

%% plot: correlations - 2d histogram
colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator

positions = [400, 400, 1255, 1350];

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'C/<C>',...
          'D/<D>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>'...
          'B/<B>',};

fig1 = figure;
set(fig1,'Position',positions(1,:));
hold on;

k = 1;
for i = 1:11
    for j = 1:11
        h = subplot('Position',[j/12.85,(12-i)/12.85,1/13,1/13]);
%         h = subplot(8,8,k);
        if i~=j
            hold on     
            
            Xedges = 0.4:0.07:1.6;
            Yedges = 0.4:0.07:1.6;
            h2 = histogram2(pool_data(:,j),pool_data(:,i),Xedges,Yedges,'DisplayStyle','tile','ShowEmptyBins','on','EdgeColor','none');
        end
                
        if i==11
            xlabel(labels(j),'fontsize',20)
        end
        xlim([0.4 1.6])
        set(gca,'XScale','linear','XTick',[],'XTickLabel',{})
        
        if j==1
            ylabel(labels(i),'fontsize',20) 
        end
        ylim([0.4 1.6])
        set(gca,'YScale','linear','YTick',[],'YTickLabel',{});

%         if j==1
%             set(gca,'XScale','linear','XTick',[0.4 1 1.6],'XTickLabel',{'0.4','1','1.6'})
%             set(gca,'YScale','linear','YTick',[0.4 1 1.6],'YTickLabel',{'0.4','1','1.6'})
%         end
        set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])  
        
        k = k+1;
    end
end
% text(-4.5, 12, '\delta(\tau_{cyc})\neq0','FontSize',30);
% text(-4.5, 12, '\delta(S_0)\neq0','FontSize',30);
% text(-4.5, 12, '\delta(\lambda)\neq0','FontSize',30);

%% plot: correlations - binned
colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator

positions = [400, 400, 1255, 1350];

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'C/<C>',...
          'D/<D>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>'...
          'B/<B>',};

fig2 = figure;
set(fig2,'Position',positions(1,:));
hold on;

k = 1;
for i = 1:11
    for j = 1:11
        h = subplot('Position',[j/12.85,(12-i)/12.85,1/13,1/13]);
%         h = subplot(8,8,k);
        if i~=j
            hold on
            h1 = errorbar(pool_corr{j,i}(1,:),pool_corr{j,i}(3,:),pool_corr{j,i}(4,:));
            h1.Color = colors(1,:); set(h1,'LineWidth',0.5,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');
        end
        
                
        if i~=j
            hold on
            y_fit = fit_a(j,i)*x_fit + fit_b(j,i);
            
            f1 = plot(x_fit,y_fit);
            f1.Color = colors(1,:); set(f1,'LineWidth',0.25,'Markersize',5,'Marker','None','MarkerFaceColor',[1 1 1],'LineStyle','-');
        end
        
                
        if i==11
            xlabel(labels(j),'fontsize',20)
        end
        xlim([0.4 1.6])
        set(gca,'XScale','linear','XTick',[],'XTickLabel',{})
        
        if j==1
            ylabel(labels(i),'fontsize',20) 
        end
        ylim([0.4 1.6])
        set(gca,'YScale','linear','YTick',[],'YTickLabel',{});
        
        if j==1 && i==11
%             set(gca,'XScale','linear','XTick',[0.4 1 1.6],'XTickLabel',{'0.4','1','1.6'})
%             set(gca,'YScale','linear','YTick',[0.4 1 1.6],'YTickLabel',{'0.4','1','1.6'})
        end

        set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])
        
        text(0.4, 0.45, ['Corr = ' num2str(corr_coef(j,i),2)],'FontSize',12);
        
        k = k+1;
    end
end
% text(-4.5, 12, '\delta(\tau_{cyc})\neq0','FontSize',30);
% text(-4.5, 12, '\delta(S_0)\neq0','FontSize',30);
% text(-4.5, 12, '\delta(\lambda)\neq0','FontSize',30);

end