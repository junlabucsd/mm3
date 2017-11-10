function [] = plot_time(pool_data)

%% positions and colors
colors = [46 49 146;
          28 117 188;
          0 167 157;
          57 181 74;
          141 198 63;
          251 176 64;
          241 90 41;
          239 65 54]/255; %illustrator
      
positions = [400, 400, 640, 240;
             400, 400*2, 640, 240;
             400, 400*3, 640, 240;
             400*2, 400, 480, 480;
             400*2, 400*2, 480, 480;
             400*2, 400*3, 480, 480;
             400*3, 400, 480, 480;
             400*3, 400*2, 480, 480;
             400*3, 4+00*3, 480, 480;
             400*4, 400, 480, 480;
             400*4, 400*2, 480, 480;
             400*4, 400*3, 480, 480;
             400*4, 400*4, 480, 480];

%% ----bin data for time course
N_edges_t = 20;
edges_t = zeros(1,N_edges_t);
edges_t(1,1) = min(pool_data.birth_time);
center_all_t = zeros(1,N_edges_t-1);

discrt_time = cell(1,N_edges_t-1);

mean_discrt_tau_t = zeros(1,N_edges_t-1);
std_discrt_tau_t = zeros(1,N_edges_t-1);

mean_discrt_S_b_t = zeros(1,N_edges_t-1);
std_discrt_S_b_t = zeros(1,N_edges_t-1);

mean_discrt_S_d_t = zeros(1,N_edges_t-1);
std_discrt_S_d_t = zeros(1,N_edges_t-1);

mean_discrt_delta_S_t = zeros(1,N_edges_t-1);
std_discrt_delta_S_t = zeros(1,N_edges_t-1);

mean_birth_t = zeros(1,N_edges_t-1);
std_birth_t = zeros(1,N_edges_t-1);

for p=2:N_edges_t
    edges_t(1,p) = edges_t(1,1) + (p-1)*(max(pool_data.birth_time)-min(pool_data.birth_time))/(N_edges_t-1);
    center_all_t(1,p-1) = (edges_t(1,p)+edges_t(1,p-1))/2;
end 

k = zeros(N_edges_t-1,1) + 1;
l = zeros(N_edges_t-1,1) + 1;
m = zeros(N_edges_t-1,1) + 1;
n = zeros(N_edges_t-1,1) + 1;
for j=1:length(pool_data.birth_time)
    for p=1:N_edges_t-1
        if (pool_data.birth_time(1,j)-edges_t(1,p))*(pool_data.birth_time(1,j)-edges_t(1,p+1))<0
            discrt_time{1,p}(l(p),1) = pool_data.newborn_length(1,j);
            discrt_time{1,p}(l(p),2) = pool_data.added_length(1,j);
            discrt_time{1,p}(l(p),3) = pool_data.division_length(1,j);
            discrt_time{1,p}(l(p),4) = pool_data.generation_time(1,j);
            l(p) = l(p)+1;
        end
    end
end

for p=1:N_edges_t-1
    if isempty(discrt_time{1,p}) == 0
        mean_discrt_tau_t(1,p) = mean(discrt_time{1,p}(:,4));
        std_discrt_tau_t(1,p) = std(discrt_time{1,p}(:,4));
        
        mean_discrt_S_b_t(1,p) = mean(discrt_time{1,p}(:,1));
        std_discrt_S_b_t(1,p) = std(discrt_time{1,p}(:,1));
        
        mean_discrt_S_d_t(1,p) = mean(discrt_time{1,p}(:,3));
        std_discrt_S_d_t(1,p) = std(discrt_time{1,p}(:,3));
        
        mean_discrt_delta_S_t(1,p) = mean(discrt_time{1,p}(:,2));
        std_discrt_delta_S_t(1,p) = std(discrt_time{1,p}(:,2));
    end
end 

%% time course plot
%---------generation time vs time
fig = figure;
set(fig,'Position',positions(1,:));
hold on;

h1 = plot(pool_data.birth_time,pool_data.generation_time);
h1.Color = colors(1,:); set(h1,'LineWidth',1,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');

h2 = plot(center_all_t,mean_discrt_tau_t);
h2.Color = colors(1,:); set(h2,'LineWidth',5,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','-');

xlabel('time (min)','fontsize',20)
xlim([0 1500])
set(gca,'XScale','linear','XTick',[0 100 200 300 400 500 600],'XTickLabel',{'0','','','300','','','600'})

ylabel('generation time (min)','fontsize',20) 
ylim([0 100])
set(gca,'YScale','linear','YTick',[0 15 30 45 60],'YTickLabel',{'0','','30','','60'});

set(gca,'TickLength',[0.0125 0.025],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[3.5 1 1]);

%---------size vs time
fig = figure;
set(fig,'Position',positions(2,:));
hold on;

h1 = plot(pool_data.birth_time,pool_data.newborn_length);
h1.Color = colors(1,:); set(h1,'LineWidth',1,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');

h3 = plot(pool_data.birth_time,pool_data.division_length);
h3.Color = colors(8,:); set(h3,'LineWidth',1,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');

h5 = plot(pool_data.birth_time,pool_data.added_length);
h5.Color = colors(5,:); set(h5,'LineWidth',1,'Markersize',5,'Marker','o','MarkerFaceColor',[1 1 1],'LineStyle','None');

h2 = plot(center_all_t,mean_discrt_S_b_t);
h2.Color = colors(1,:); set(h2,'LineWidth',5,'Markersize',5,'Marker','None','MarkerFaceColor',[1 1 1],'LineStyle','-');

h4 = plot(center_all_t,mean_discrt_S_d_t);
h4.Color = colors(8,:); set(h4,'LineWidth',5,'Markersize',5,'Marker','None','MarkerFaceColor',[1 1 1],'LineStyle','-');

h6 = plot(center_all_t,mean_discrt_delta_S_t);
h6.Color = colors(5,:); set(h6,'LineWidth',5,'Markersize',5,'Marker','None','MarkerFaceColor',[1 1 1],'LineStyle','-');


xlabel('time (min)','fontsize',20)
xlim([0 1500])
set(gca,'XScale','linear','XTick',[0 100 200 300 400 500 600],'XTickLabel',{'0','','','300','','','600'})

ylabel('length ({\mu}m)','fontsize',20) 
ylim([0 8])
set(gca,'YScale','linear','YTick',[0 1 2 3 4 5 6 7 8],'YTickLabel',{'0','','2','','4','','6','','8'});

set(gca,'TickLength',[0.0125 0.025],'fontsize',20,'TickDir','out','PlotBoxAspectRatio',[3.5 1 1]);

legend([h2 h4 h6], 'newborn length','division length','added length')