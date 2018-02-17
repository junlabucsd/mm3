function [] = plot_corr_auto(pool_corr, pool_data_m, pool_data_d)

%% calculate correlation efficient
ft1 = fittype('a*x+b');
fit_a = zeros(11,1);
fit_b = zeros(11,1);
% fit_pval = zeros(11,1);
corr_coef = zeros(11,1);
corr_pval = zeros(11,1);

for i = 1:11
    
    pool_data_tmp = [pool_data_m(:,i),pool_data_d(:,i)]';

    pool_data_isnan = any(isnan(pool_data_tmp));
    pool_data_tmp = pool_data_tmp(:,find(pool_data_isnan==0))';
    
    fit_temp = fit(pool_data_tmp(:,1),pool_data_tmp(:,2),ft1);
    [corr_coef(i,1),corr_pval(i,1)] = corr(pool_data_tmp(:,1),pool_data_tmp(:,2),'type','Pearson');
    fit_a(i,1) = fit_temp.a;
    fit_b(i,1) = fit_temp.b;
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

positions = [400, 400, 1255, 200];

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'B/<B>',...
          'C/<C>',...
          'D/<D>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>'};

fig1 = figure;
set(fig1,'Position',positions(1,:));
hold on;

k = 1;
for i = 1:11
    h = subplot('Position',[i/13.85,1/13.85,1/14,1]);
    hold on           
    h1 = plot(pool_data_d(:,i),pool_data_m(:,i));
    h1.Color = colors(1,:); set(h1,'LineWidth',0.5,'Markersize',2.5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');
    
    xlabel(labels(i),'fontsize',20)
    xlim([0.4 1.6])
    set(gca,'XScale','linear','XTick',[],'XTickLabel',{})

    if i==1
        ylabel('daughter cell','fontsize',20) 
    end
    ylim([0.4 1.6])
    set(gca,'YScale','linear','YTick',[],'YTickLabel',{});

    set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])  

    k = k+1;
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

positions = [400, 400, 1255, 200];

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'B/<B>',...
          'C/<C>',...
          'D/<D>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>'};

fig1 = figure;
set(fig1,'Position',positions(1,:));
hold on;

k = 1;
for i = 1:11
    h = subplot('Position',[i/13.85,1/13.85,1/14,1]);
    hold on           

    Xedges = 0.4:0.07:1.6;
    Yedges = 0.4:0.07:1.6;
    h2 = histogram2(pool_data_d(:,i),pool_data_m(:,i),Xedges,Yedges,'DisplayStyle','tile','ShowEmptyBins','on','EdgeColor','none');
    
    xlabel(labels(i),'fontsize',20)
    xlim([0.4 1.6])
    set(gca,'XScale','linear','XTick',[],'XTickLabel',{})

    if i==1
        ylabel('daughter cell','fontsize',20) 
    end
    ylim([0.4 1.6])
    set(gca,'YScale','linear','YTick',[],'YTickLabel',{});

    set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])  

    k = k+1;
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

positions = [400, 400, 1255, 200];

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'B/<B>',...
          'C/<C>',...
          'D/<D>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>',};

fig2 = figure;
set(fig2,'Position',positions(1,:));
hold on;

k = 1;
for i = 1:11

    h = subplot('Position',[i/13.85,1/13.85,1/14,1]);
    hold on
    h1 = errorbar(pool_corr{i,i}(1,:),pool_corr{i,i}(3,:),pool_corr{i,i}(4,:));
    h1.Color = colors(1,:); set(h1,'LineWidth',0.5,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');


    hold on
    y_fit = fit_a(i,1)*x_fit + fit_b(i,1);

    f1 = plot(x_fit,y_fit);
    f1.Color = colors(1,:); set(f1,'LineWidth',0.25,'Markersize',5,'Marker','None','MarkerFaceColor',[1 1 1],'LineStyle','-');


    xlabel(labels(i),'fontsize',20)
    xlim([0.4 1.6])
    set(gca,'XScale','linear','XTick',[],'XTickLabel',{})

    if i==1
        ylabel(labels(i),'fontsize',20) 
    end
    ylim([0.4 1.6])
    set(gca,'YScale','linear','YTick',[],'YTickLabel',{});

    set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])

    text(0.4, 0.45, ['Corr = ' num2str(corr_coef(i,1),2)],'FontSize',11);

    k = k+1;
        
end
% text(-4.5, 12, '\delta(\tau_{cyc})\neq0','FontSize',30);
% text(-4.5, 12, '\delta(S_0)\neq0','FontSize',30);
% text(-4.5, 12, '\delta(\lambda)\neq0','FontSize',30);

end