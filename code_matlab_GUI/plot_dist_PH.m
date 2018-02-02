function [] = plot_dist_PH(pool_data,pool_data_norm,pool_CV)

%% plot: distributions
colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator

positions = [400, 400, 1255, 200];

% labels = {'S_{1/2}}',...
%           '\lambda (1/hour)',...
%           'S_0 ({\mu}m)',...
%           '\tau_{cyc} (min)',...
%           'S_d ({\mu}m)',...
%           'S_b ({\mu}m)',...
%           '\tau (min)',...
%           '\Delta_d ({\mu}m)'...
%           '\Delta_i ({\mu}m)',};

% labels = {'S_{1/2}',...
%           '\lambda (1/hour)',...
%           'S_0 ({\mu}m)',...
%           '\tau_{cyc} (min)',...
%           'S_d ({\mu}m)',...
%           'S_b ({\mu}m)',...
%           '\tau (min)',...
%           '\Delta_d ({\mu}m)'...
%           't_i (min)',};

labels = {'S_{1/2}',...
          '\lambda (1/hour)',...
          'S_0 ({\mu}m)',...
          '\tau_{cyc} (min)',...
          'S_d ({\mu}m)',...
          'S_b ({\mu}m)',...
          '\tau (min)',...
          '\Delta_d ({\mu}m)'...
          'S_i ({\mu}m)',};

bin_wid = 0.1;
      
fig = figure;
set(fig,'Position',positions(1,:));
hold on;

k = 1;
for j = [1 2 5 6 7 8]
    h = subplot('Position',[j/10.85,1/10.85,1/11,1]);
%         h = subplot(8,8,k);
    if pool_CV(j) > 0.04
        hold on           
        [f, x] = hist(pool_data(:,j), 16);
        h1 = plot(x,f/length(pool_data(:,j)));
        h1.Color = colors(1,:);
        set(h1,'LineWidth',1,'LineStyle','-');
    end
%         text(0.4, 0.45, ['CV=' num2str(pool_CV(1,j),2)],'FontSize',14);

    ylim([0 0.5])
    set(gca,'YScale','linear','YTick',[],'YTickLabel',{});
    
    if j==1
        ylabel('PDF','fontsize',20)
    end
    
    xlabel(labels(j),'fontsize',20) 
    tick1 = mean(pool_data(:,j))-3*std(pool_data(:,j));
    tick2 = mean(pool_data(:,j));
    tick3 = mean(pool_data(:,j))+3*std(pool_data(:,j));
    bd1 = mean(pool_data(:,j))-4*std(pool_data(:,j));
    bd2 = mean(pool_data(:,j))+4*std(pool_data(:,j));
    
    set(gca,'TickLength',[0.05 0.1],'fontsize',10,'TickDir','out','PlotBoxAspectRatio',[1 1 1])  
    
    set(gca,'XScale','linear','XTick',[tick1 tick2 tick3],'XTickLabel',{num2str(tick1,'%.2f'),num2str(tick2,'%.2f'),num2str(tick3,'%.2f')},'fontsize',11)
    xlim([bd1 bd2])
   
    k = k+1;
end

%% plot: distributions - normalized
colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator

positions = [400, 400, 1255, 200];

% labels = {'S_{1/2}/<S_{1/2}>',...
%           '\lambda/<\lambda>',...
%           'S_0/<S_0>',...
%           '\tau_{cyc}/<\tau_{cyc}>',...
%           'S_d/<S_d>',...
%           'S_b/<S_b>',...
%           '\tau/<\tau>',...
%           '\Delta_d/<\Delta_d>'...
%           '\Delta_i/<\Delta_i>',};

labels = {'S_{1/2}/<S_{1/2}>',...
          '\lambda/<\lambda>',...
          'S_0/<S_0>',...
          '\tau_{cyc}/<\tau_{cyc}>',...
          'S_d/<S_d>',...
          'S_b/<S_b>',...
          '\tau/<\tau>',...
          '\Delta_d/<\Delta_d>'...
          't_i/<t_i>',};

bin_wid = 0.08;
      
fig = figure;
set(fig,'Position',positions(1,:));
hold on;

k = 1;
for j = [1 2 5 6 7 8]
    h = subplot('Position',[j/10.85,1/10.85,1/11,1]);
%         h = subplot(8,8,k);
    if pool_CV(j) > 0.04
        hold on           
        [f, x] = hist(pool_data_norm(:,j), (max(pool_data_norm(:,j))-min(pool_data_norm(:,j)))/bin_wid);
        h1 = plot(x,f/length(pool_data_norm(:,j)));
        h1.Color = colors(1,:);
        set(h1,'LineWidth',1,'LineStyle','-');
    end
        text(0.4, 0.45, ['CV=' num2str(pool_CV(1,j),2)],'FontSize',14);
    
    if pool_CV(j) > 0.04 && j>9 %j==8
        
        x_raw = pool_data_norm(pool_data_norm(:,j)>0,j);
        pd = fitdist(x_raw,'Lognormal');

%         [h,p] = chi2gof(log(x_raw));
        [h,p] = chi2gof(x_raw,'cdf',{@logncdf,pd.mu,pd.sigma},'nbins',10); %chi^2 test with lognormal dist with estimated parameters (define bins)
%         x_make = normrnd(1e4,1,[1e5,1]); %chi^2 test with norm dist
%         [h,p] = chi2gof(x_make,'cdf',{@normcdf,1e4,1},'nbins',1e2);

%           R = lognrnd(pd.mu,pd.sigma,[1e4,1]);
%           [h,p] = chi2gof(R,'cdf',{@logncdf,pd.mu,pd.sigma},'nbins',100); %chi^2 test with ARTIFICIAL lognormal dist with estimated parameters (define bins)
        
%         [h,p] = ztest((log(x_raw)-0.95*pd.mu)/pd.sigma,0,1); %ztest: does not apply here
%         pd = fitdist(x_raw,'Normal'); %ztest: does not apply here
%         [h,p] = ztest(x_raw-mean(x_raw)/std(x_raw),0,1);
        
        x_value = x;
        y = pdf(pd,x_value);

        h2 = plot(x_value,y*bin_wid);
        h2.Color = colors(8,:);
        set(h2,'LineWidth',1,'LineStyle','--');
        
        text(0.4, 0.38, ['\mu=' num2str(pd.mu,2)],'FontSize',14,'Color',colors(8,:));
        text(0.4, 0.33, ['\sigma=' num2str(pd.sigma,2)],'FontSize',14,'Color',colors(8,:));
        text(0.4, 0.28, ['p-value=' num2str(p,2)],'FontSize',14,'Color',colors(8,:));
        text(0.4, 0.23, ['reject? ' num2str(h,1)],'FontSize',14,'Color',colors(8,:));
    end

    ylim([0 0.5])
    set(gca,'YScale','linear','YTick',[],'YTickLabel',{});
    
    if j==1
        ylabel('PDF','fontsize',20)
    end
    
    xlabel(labels(j),'fontsize',20) 
    set(gca,'XScale','linear','XTick',[],'XTickLabel',{})
    xlim([0.4 1.6])
    
    if j==1
%         set(gca,'XScale','linear','XTick',[0.4 1 1.6],'XTickLabel',{'0.4','1','1.6'})
    end
    
    set(gca,'TickLength',[0.05 0.1],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1])  

       
    k = k+1;
end
