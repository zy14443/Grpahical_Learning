%% Environment
% Bernoulli random variables

alpha = 2;
c = 0.8;

K_max = 100;

mu_x_all = [];
for i=1:K_max/2
    mu_x_all = [mu_x_all,[0.4,0.6]*1.005^(i-1)];
end
mu_y_all = [];
for i=1:K_max/2
    mu_y_all = [mu_y_all,[0.6,0.3]*1.01^(i-1)];
end
mu_r_all = [];
for i=1:K_max/2
    mu_r_all = [mu_r_all,[0.6,0.3]*1.01^(i-1)];
end

BB = [1000:1000:10000];%[1000:2000:8000,10000:10000:100000]; %total budget
penalty_off_B_K = zeros(length(BB),K_max/2);
penalty_B_K = zeros(length(BB),K_max/2);

reward_opt_B_K = zeros(length(BB),K_max/2);
reward_off_B_K = zeros(length(BB),K_max/2);
reward_B_K = zeros(length(BB),K_max/2);
%%
for K=2

%cost
mu_x = mu_x_all(1:K);
%penalty
mu_y = mu_y_all(1:K);
%reward
mu_r = mu_r_all(1:K);

r_off = mu_r./mu_x;
y_off = mu_y./mu_x;

y_max = max(mu_y./mu_x);
r_max = max(mu_r./mu_x);
mu_min = min(mu_x);



N_repeat = 5000; %repetitions

%% Lyapunov Offline
tic
N_off_B = zeros(length(BB),K);
reward_off_B = zeros(length(BB),1);
penalty_off_B = zeros(length(BB),1);

v_off_0=1;
delta_off_0=0.9;

for kk = 1:length(BB)
    
    B = BB(kk);
    
    N_off_total = zeros(N_repeat,K);
    reward_off_total = zeros(N_repeat,K);
    penalty_off_total = zeros(N_repeat,K);

    parfor counter = 1:N_repeat
        %empirical cost
        sum_x_off = zeros(K,1);

        %empirical penalty
        sum_y_off = zeros(K,1);

        %empirical reward
        sum_r_off = zeros(K,1);

        %empirical number of pulls
        N_off = zeros(K,1);
        cost_sum_off = 0;

        Q_off = 0;
  
        V_off = v_off_0*sqrt(B);
        delta_off = delta_off_0/sqrt(B);
        while cost_sum_off<=B

            psi_Q_off = -V_off*r_off+Q_off*y_off;

            [min_temp, k_select] = min(psi_Q_off); 

            %observe results
            N_off(k_select) = N_off(k_select)+1;
            cost = (rand<mu_x(k_select));
            sum_x_off(k_select) = sum_x_off(k_select)+ cost;
            y_obs = (rand<mu_y(k_select));
            sum_y_off(k_select) = sum_y_off(k_select)+ y_obs;    
            sum_r_off(k_select) = sum_r_off(k_select)+ (rand<mu_r(k_select));
            cost_sum_off = cost_sum_off+cost;

            %update Q
            Q_off = max([0,Q_off+y_obs-(c-delta_off)*cost]);

        end
        N_off_total(counter,:) = N_off';
        reward_off_total(counter,:) = sum_r_off';
        penalty_off_total(counter,:) = sum_y_off';
    end

    N_off_mean = mean(N_off_total,1);
    reward_off_mean = mean(reward_off_total,1);
    penalty_off_mean = mean(penalty_off_total,1);
    
    N_off_B(kk,:) = N_off_mean/sum(N_off_mean);
    reward_off_B(kk,:) = sum(reward_off_mean);
    penalty_off_B(kk,:) = sum(penalty_off_mean);
end
penalty_off_B_K(:,K/2) = penalty_off_B;
reward_off_B_K(:,K/2) = reward_off_B;
toc

%% Lyapunov Online
tic
N_B = zeros(length(BB),K);
reward_B = zeros(length(BB),1);
penalty_B = zeros(length(BB),1);

v_0=1; 
delta_0 = 2;

for kk = 1:length(BB)
    
    B = BB(kk);
    
    N_total = zeros(N_repeat,K);
    reward_total = zeros(N_repeat,K);
    penalty_total = zeros(N_repeat,K);


    parfor counter = 1:N_repeat


        %empirical cost
        sum_x = zeros(K,1);

        %empirical penalty
        sum_y = zeros(K,1);

        %empirical reward
        sum_r = zeros(K,1);

        %empirical number of pulls
        N = zeros(K,1);
        cost_sum = 0;

        %initialization
        for i = 1:K
            for j = 1:ceil(log(2*B/mu_min))
                N(i) = N(i)+1;
                cost = (rand<mu_x(i));
                sum_x(i) = sum_x(i)+ cost;
                sum_y(i) = sum_y(i)+ (rand<mu_y(i));
                sum_r(i) = sum_r(i)+ (rand<mu_r(i));
                cost_sum = cost_sum+cost;
            end
        end

        %LyConf main loop
        Q = 0;
        V = v_0*sqrt(B);
        delta = delta_0*sqrt(1/B);
        while cost_sum <=B

            %empirical estimation
            rad_emp = sqrt(2*alpha*log(sum(N))./N);
            r_hat = sum_r./sum_x;
            y_hat = sum_y./sum_x;
            mu_x_hat = sum_x./N;

            %empirical Lyapunov drift-plus-penalty
            psi_Q = -V*r_hat+Q*y_hat;

            %confidence bound
            cb = rad_emp.*(V*(1+r_hat)./mu_x_hat+Q*(1+y_hat)./mu_x_hat);

            %arm selection
            [min_temp, k_select] = min(psi_Q-cb);

            %observe results
            N(k_select) = N(k_select)+1;
            cost = (rand<mu_x(k_select));
            sum_x(k_select) = sum_x(k_select)+ cost;
            y_obs = (rand<mu_y(k_select));
            sum_y(k_select) = sum_y(k_select)+ y_obs;    
            sum_r(k_select) = sum_r(k_select)+ (rand<mu_r(k_select));
            cost_sum = cost_sum+cost;

            %update Q
            Q = max([0,Q+y_obs-(c-delta)*cost]);
        end
        N_total(counter,:) = N';
        reward_total(counter,:) = sum_r';
        penalty_total(counter,:) = sum_y';

    end

    N_mean = mean(N_total,1);  
    reward_mean = mean(reward_total,1);
    penalty_mean = mean(penalty_total,1);
    
    N_B(kk,:) = N_mean/sum(N_mean);
    reward_B(kk,:) = sum(reward_mean);
    penalty_B(kk,:) = sum(penalty_mean);
end

penalty_B_K(:,K/2) = penalty_B;
reward_B_K(:,K/2) = reward_B;

toc

%% Optimal Stationary Randomized Policy

N_opt_B = zeros(length(BB),K);
reward_opt_B = zeros(length(BB),1);
penalty_opt_B = zeros(length(BB),1);

beta_prime = 0;

for kk = 1:length(BB)
    
    B = BB(kk);
    
    fun=@(z) -(mu_r*z')/(mu_x*z');

    z_0=0.5*ones(1,K);
    
    c_prime = (penalty_off_B(kk)/BB(kk)-c);
    
    A = mu_y-(c+beta_prime*c_prime)*mu_x;
    b = 0;

    Aeq = ones(1,K);
    beq = 1;

    lb = zeros(1,K);
    ub = ones(1,K);
    options = optimoptions(@fmincon,'Display','off');

    z_star = fmincon(fun,z_0,A,b,Aeq,beq,lb,ub,[],options);
    
%     fun(z_star)
    
    reward_opt_total = zeros(N_repeat,K);
    penalty_opt_total = zeros(N_repeat,K);


    N_opt_B(kk,:) = z_star;
    reward_opt_B(kk,:) = -fun(z_star)*B;%sum(reward_opt_mean);
    penalty_opt_B(kk,:) = (mu_y*z_star')/(mu_x*z_star')*B; %sum(penalty_opt_mean);
end

reward_opt_B_K(:,K/2) = reward_opt_B;

end


%% Plot

figure;

plot(BB, penalty_opt_B'./BB -c, '-*', 'LineWidth', 2);
hold on;
plot(BB, penalty_off_B'./BB -c, '-*', 'LineWidth', 2);
hold on;
plot(BB, penalty_B'./BB -c, '-x', 'LineWidth', 2);
xlabel('Budget B')
ylabel('Constraint Violation')
legend('Lyapunov Offline', 'Lyapunov Online')

figure;
plot(sqrt(BB), (reward_opt_B-reward_off_B)', '-*', 'LineWidth', 2);
hold on;

plot(sqrt(BB), (reward_opt_B-reward_B)', '-*', 'LineWidth', 2);
hold on;
xlabel('Budget B')
ylabel('Reward')
legend('Lyapunov Offline', 'Lyapunov Online')


figure;
plot(BB, N_off_B,'-*', 'LineWidth', 2);
hold on;
plot(BB, N_opt_B,'--', 'LineWidth', 2);
hold on;
plot(BB, N_B, '-x', 'LineWidth', 2);
xlabel('Budget B')
ylabel('Fraction of time allocation')
legend('Lyapunov Offline Arm 1', 'Lyapunov Offline Arm 2','Lyapunov Online Arm 1', 'Lyapunov Offline Arm 2', 'Location', 'East')

figure
plot(BB, reward_B-reward_off_B, '-x', 'LineWidth', 2);

