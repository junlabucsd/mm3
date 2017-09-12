function output_data = filter_data(input_data)

output_data = input_data;

% mean_data = mean(input_data);
% std_data = std(input_data);
% 
% filtered_data(:,1) = (input_data(:,1) < mean_data(1)+2.0*std_data(1)) & (input_data(:,1) > mean_data(1)-2.0*std_data(1));
% logic_data = filtered_data(:,1);
% 
% for i=2:8
%     if i~=3 && i~=4
%         filtered_data(:,i) = (input_data(:,i) < mean_data(i)+2.0*std_data(i)) & (input_data(:,i) > mean_data(i)-2.0*std_data(i));
%         logic_data = logic_data.*logic_data;
%     end
% end
% 
% L = length(input_data(:,1));
% for i=1:L;
%     if logic_data(L+1-i)==0
%         output_data(L+1-i,:) = [];
%     end
% end
    
end