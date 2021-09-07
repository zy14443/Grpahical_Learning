%% Plot

% figure;
% plot(1:cost_sum, n_off_total_all([1,2,4],1:cost_sum)./sum(n_off_total_all(:,1:cost_sum),1),'LineWidth', 2)
% hold on;
% line(1:cost_sum,  n_LP_all([1,2,4],:)./n_LP_total.*(ones(1,cost_sum)), 'LineStyle','--','LineWidth', 2)
% xlabel('T')
% ylabel('Sample proportion')
% legend('Node 1', 'Node 3', 'Node 5')

figure;
heatmap(theta_matrix)
colormap parula

%% LP
figure;
heatmap(reshape(n_LP_all./sum(n_LP_all),10,10),'CellLabelColor','none')
colormap parula

figure;
bar_1 = bar3(reshape(n_LP_all./sum(n_LP_all),10,10));
for k = 1:length(bar_1)
    zdata = bar_1(k).ZData;
    bar_1(k).CData = zdata;
    bar_1(k).FaceColor = 'interp';
end

%% Lyapunov
figure;
heatmap(reshape(n_off_total./sum(n_off_total),10,10),'CellLabelColor','none')
colormap parula

figure;
for kk=1:1000:ceil(n_LP_total)
    bar_2 = bar3(reshape(n_off_total_all(:,kk)./sum(n_off_total),10,10));

%     bar_2 = bar3(reshape(n_off_total./sum(n_off_total),10,10));
    for k = 1:length(bar_2)
        zdata = bar_2(k).ZData;
        bar_2(k).CData = zdata;
        bar_2(k).FaceColor = 'interp';
    end
    zlim([0,max(n_off_total./sum(n_off_total))])
    pause(1/48);
end

%% Greedy
figure;
heatmap(reshape(n_greedy_total./sum(n_greedy_total),10,10),'CellLabelColor','none')
colormap parula

% witerObj = VideoWriter('myVideo.avi');
% writerObj.FrameRate = 48;
% open(writerObj);

figure;
for kk=1:1000:cost_sum_greedy
    bar_2 = bar3(reshape(n_greedy_total_all(:,kk),10,10));

%     bar_2 = bar3(reshape(n_off_total./sum(n_off_total),10,10));
    for k = 1:length(bar_2)
        zdata = bar_2(k).ZData;
        bar_2(k).CData = zdata;
        bar_2(k).FaceColor = 'interp';
    end
    zlim([0,max(n_greedy_total)])
    pause(1/48);
% frame = getframe(gcf);
% writeVideo(writerObj, frame);

end

% close(writerObj);
%% Random
figure;
heatmap(reshape(n_rand_total./sum(n_rand_total),10,10),'CellLabelColor','none')
colormap parula

figure;
for kk=1:1000:cost_sum_rand
    bar_2 = bar3(reshape(n_rand_total_all(:,kk)./sum(n_rand_total),10,10));

%     bar_2 = bar3(reshape(n_off_total./sum(n_off_total),10,10));
    for k = 1:length(bar_2)
        zdata = bar_2(k).ZData;
        bar_2(k).CData = zdata;
        bar_2(k).FaceColor = 'interp';
    end
    zlim([0,max(n_rand_total./sum(n_rand_total))])
    pause(1/48);

end
%% Sample Complexity

figure;
plot(1:5, n_mean_compare(:,1), '-*','LineWidth', 2)
hold on;

plot(1:5, n_mean_compare(:,2), '-o','LineWidth', 2)
xlabel('n-Hop')
ylabel('Sample Complexity')
legend('Random Sampling', 'Lyapunov Sampling')
xticks([1,2,3,4,5])