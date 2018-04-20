clear all; clc; 
% close all;

%% load data
cell_data = load('/path/to/cells_foci.mat');

px_to_mu = 0.11;
t_int = 2.0;
start_cut = 0;
end_cut = 1500;

% extract data
L = length(fieldnames(cell_data));
fnames = fieldnames(cell_data);

channel = cell(1,1);
Foci_all = [];


for i = 1:L
    
    N = length( cell_data.(fnames{i}).times );
          
    channel = cell_data.(fnames{i}); 
    
    if isempty(channel.foci_h) == 0 && channel.division_time < end_cut %&& channel.peak > 0 && channel.peak < 2000 
    
        for j = 1:N
            
            if isempty(channel.foci_h) == 0

                iscell_foci = double(iscell(channel.foci_h));

                if iscell_foci == 1
                    Foci_all = [Foci_all; [channel.foci_h{1,j}']];
                end

                if iscell_foci == 0
                    Foci_all = [Foci_all; [channel.foci_h(j,1)']];
                end 
                
            end

        end
        
    end
    
    if mod(i,100)==0
        i
    end
    
end

% save('../../analysis/IW_foci_20171026.mat');


%% plot
% close all;
% load('../../analysis/IW_foci_20171026.mat');

foci_counter = length(Foci_all(:,1));
bin_wid_scale = 500;

colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator

positions = [400, 400, 420, 400];


% %-------------Fitted Peak intensity of focus------------
%-------------Total intensity of focus------------
bin_wid4 = 2*mean(Foci_all(:,1))/bin_wid_scale;
      
fig = figure;
set(fig,'Position',positions(1,:));
hold on;
      
[f, x] = hist(Foci_all(:,1), (max(Foci_all(:,1))-min(Foci_all(:,1)))/bin_wid4);
h1 = plot(x,f/length(foci_counter));
h1.Color = colors(1,:);
set(h1,'LineWidth',1,'LineStyle','-');


set(gca,'YScale','linear','YTick',[],'YTickLabel',{});

% xlabel('Total intensity of focus (AU)','fontsize',20) 
% set(gca,'XScale','linear','XTick',[0 30 60 90 120 150 180 210 240]*1e2,'XTickLabel',{'0','','','','1.2\times10^4','','','','2.4\times10^4'})
set(gca,'XScale','linear','XTick',[0 30 60 90 120 150 180 210 240]*2e2,'XTickLabel',{'','','','','','','','',''})
xlim([0 mean(Foci_all(:,1))+5*std(Foci_all(:,1))])

set(gca,'TickLength',[0.025 0.05],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[1 1 1]) 

%% fit the intensity weighting distribution
% close all;
%%------multi-gaussian fitting
[fit1 gof] = fit(x.', f.','gauss2');
% [fit1 gof] = fit(x.', f.','gauss2','Lower',[80 5e3 0 20 1.0e4 0],'Upper',[160 8e3 1e4 80 2.5e4 1e4])

g1 = fit1.a1*exp(-((x-fit1.b1)./fit1.c1).^2);
g2 = fit1.a2*exp(-((x-fit1.b2)./fit1.c2).^2);
% g3 = fit1.a3*exp(-((x-fit1.b3)./fit1.c3).^2);
figure;
hold on;

plot(fit1,x,f);
plot(x,g1,'r--');
plot(x,g2,'b--');
% plot(x,g3,'m--');

pg1 = normcdf(x,fit1.b1,fit1.c1/(2^0.5));
pg2 = normcdf(x,fit1.b2,fit1.c2/(2^0.5));

peak_ind_x = zeros(1,length(x));
thr_x = 0;

for i=1:length(x)
    if min(pg1(i),1-pg1(i)) >= min(pg2(i),1-pg2(i))
        peak_ind_x(i) = 1;
    else
        peak_ind_x(i) = 2;
    end
end

for i=1:length(x)-1
    if peak_ind_x(i+1)-peak_ind_x(i)>0
        thr_x = (x(i+1)+x(i))/2;
    end
end

