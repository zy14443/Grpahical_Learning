%% Plot reward rate
figure;
h = plot(BB, (reward_off_B'./BB), '-x', 'LineWidth', 4);
hold on;
h = plot(BB, (reward_B'./BB), '-o', 'LineWidth', 4);
hold on;

h = plot(BB, (reward_opt_B'./BB), '--', 'LineWidth', 2, 'Color', [0.25,0.25,0.25]);
hold on;
xlabel('B')
ylabel('Reward Rate')
legend('LyOff', 'LyOn', '\pi*', 'Location','northeast')
    
%% Constraint violation
figure;
h=plot(BB, penalty_off_B'./BB-c, '-x', 'LineWidth', 4);
hold on;
h=plot(BB, penalty_B'./BB-c, '-o', 'LineWidth', 4);
hold on;

xlabel('B')
ylabel('Constraint Violation')
legend('LyOff', 'LyOn', 'Location','northeast')

%% Fraction of time
figure;

plot(BB, N_opt_B(:,2),'--', 'LineWidth', 2, 'Color', [0.25,0.25,0.25]);
hold on;
plot(BB, N_opt_B(:,1),'--', 'LineWidth', 2,'Color',[0.6350,0.0780,0.1840]);
hold on;

plot(BB, N_off_B(:,2),'-x', 'LineWidth', 4, 'Color', [0.8500,0.3250,0.0980]);
hold on;
plot(BB, N_off_B(:,1),'-x', 'LineWidth', 4, 'Color', [0.4660,0.6740,0.1880]);
hold on;

plot(BB, N_B(:,2), '-o', 'LineWidth', 4, 'Color', [0,0.4470,0.7410]);
hold on;
plot(BB, N_B(:,1), '-o', 'LineWidth', 4, 'Color', [0.4940,0.1840,0.5560]);
hold on;

xlabel('B')
ylabel('Fraction of Time Allocation')
legend('\pi* Arm 2', '\pi* Arm 1','LyOff Arm 2','LyOff Arm 1','LyOn Atm 2','LyOn Atm 1',    'Location', 'East')

%% Plot changing K

h=plot(2:14:100, penalty_B_K(1,1:7:50)/BB(1)-c, '-x', 'LineWidth', 4);
hold on;
xlabel('K')
ylabel('Constraint Violation')
legend('B=2\times10^4','B=3\times10^4',...
    'B=4\times10^4',...
    'Location','northwest')

