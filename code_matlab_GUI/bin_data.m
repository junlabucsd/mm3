function [binned_data, l] = bin_data(N_edges,data_X,data_Y)

%----bin data for correlations------
edges = zeros(1,N_edges);
% edges(1,1) = max(min(data_X/nanmean(data_X)),0.4);
% edges(1,1) = min(data_X/nanmean(data_X)); % fixed lower boundary = min(data)
edges(1,1) = 0.5; % fixed lower boundary
center_all = zeros(1,N_edges-1);

bin_X = cell(1,N_edges-1);

binned_data = zeros(4,N_edges-1);


for p=2:N_edges
    edges(1,p) = edges(1,1) + (p-1)*1/(N_edges-1); % fixed upper boundary
%     edges(1,p) = edges(1,1) + (p-1)*(max(data_X/nanmean(data_X))-edges(1,1))/(N_edges-1); % fixed upper boundary = max(data)
    center_all(1,p-1) = (edges(1,p)+edges(1,p-1))/2;
end   
    
l = zeros(N_edges-1,1) + 1;
L = length(data_X);
for j=1:L
    for p=1:N_edges-1
        if (data_X(j,1)-edges(1,p))*(data_X(j,1)-edges(1,p+1))<=0
            bin_X{1,p}(l(p),1) = data_X(j,1);
            bin_X{1,p}(l(p),2) = data_Y(j,1);
            l(p) = l(p)+1;
        end
    end
end

for p=1:N_edges-1
    if isempty(bin_X{1,p}) == 0 && l(p)>0.01*L % 1% (for Perturbation data)
%     if isempty(bin_X{1,p}) == 0 && l(p)>0.0455*L %2 sigma
%     if isempty(bin_X{1,p}) == 0 && l(p)>50 %fixed number (for Sattar's data)
        binned_data(1,p) = nanmean(bin_X{1,p}(:,1));
        binned_data(2,p) = nanstd(bin_X{1,p}(:,1));
        
        binned_data(3,p) = nanmean(bin_X{1,p}(:,2));
        binned_data(4,p) = nanstd(bin_X{1,p}(:,2));

    end
end 


end