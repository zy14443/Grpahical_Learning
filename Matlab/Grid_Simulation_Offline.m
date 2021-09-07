%% Grid Graph

grid_m = 10;
grid_n = 10;
theta_matrix = 0.1*ones(grid_m, grid_n);
%%
n_hop_max = 1;

A_matrix_all = zeros(grid_m*grid_n,grid_m*grid_n);
counter = 1;
for j=1:grid_m
    for i=1:grid_n
        A_matrix_all(:,counter) = grid_vector(i,j, theta_matrix, n_hop_max,grid_m,grid_n);
        counter = counter+1;
    end
end

%% LP-Approximation
epsilon = 0.01;
delta = 1 - 0.95;
theta = reshape(theta_matrix,[],1);

K = grid_m*grid_n;
f = ones(1,K);
lb = zeros(1,K);
ub = [];
M_constraint=double(A_matrix_all);

b = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/epsilon^2;
% b = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/epsilon^2;

Aeq=[];
beq=[];
n_LP_all = linprog(f,-M_constraint,-b,Aeq,beq,lb,ub);
n_LP_total = sum(n_LP_all);

% y_LP_all = linprog(-b,M_constraint',f,Aeq,beq,lb,ub);
%% Greedy Method Offline

N_max = 6e5; % total budget
Q_greedy = b';
n_greedy_total = zeros(length(theta),1);
cost_sum_greedy = 0;

theta_hat_matrix = 1*ones(grid_m, grid_n);
% A_matrix_hat_all = zeros(grid_m*grid_n,grid_m*grid_n);
% counter = 1;
% for j=1:grid_m
%  for i=1:grid_n
%      A_matrix_hat_all(:,counter) = grid_vector(i,j, theta_hat_matrix, n_hop_max,grid_m,grid_n);
%      counter = counter+1;
%  end
% end
% M_constraint_hat=double(A_matrix_hat_all);

obs_matrix = zeros(grid_m, grid_n);
cnt_matrix = zeros(grid_m, grid_n);


n_greedy_total_all = zeros(length(theta),N_max);
while cost_sum_greedy<N_max
     n_next = eye(length(theta));
     psi_greedy = Q_greedy*(-M_constraint*n_next);
     [min_temp, k_select] = min(psi_greedy); 
     if (sum(psi_greedy==min_temp)>1)
        k_select_rand = find(psi_greedy==min_temp);
        rand_temp = randperm(length(k_select_rand));
        k_select = k_select_rand(rand_temp(1));
     end        
     
     n_greedy_total(k_select) =  n_greedy_total(k_select)+1;
     
     [result_matrix,flag_matrix] = grid_simulator(k_select, theta_matrix, n_hop_max,grid_m,grid_n); 
     obs_matrix = obs_matrix+result_matrix.*flag_matrix;
     cnt_matrix = cnt_matrix+flag_matrix;
     theta_hat_matrix(flag_matrix>0) = obs_matrix(flag_matrix>0)./cnt_matrix(flag_matrix>0);
     
%      theta_hat_matrix_ucb = min(theta_hat_matrix+sqrt(2*log(1/delta)./cnt_matrix),ones(grid_m,grid_n));     
%      A_matrix_hat_all = zeros(grid_m*grid_n,grid_m*grid_n);
%      counter = 1;
%      for j=1:grid_m
%          for i=1:grid_n
%              A_matrix_hat_all(:,counter) = grid_vector(i,j, theta_hat_matrix_ucb, n_hop_max,grid_m,grid_n);
%              counter = counter+1;
%          end
%      end
%      M_constraint_hat=double(A_matrix_hat_all);
     
     cost_sum_greedy = cost_sum_greedy+1;
     Q_greedy = max(zeros(1,length(theta)),Q_greedy-reshape(flag_matrix,1,[]));
     n_greedy_total_all(:,cost_sum_greedy) = n_greedy_total;

     if (sum(Q_greedy)==0)%sum(sum(cnt_matrix<reshape(b,grid_m,grid_n)))==0
         break;
     end
end
mse_greedy = sum(sum((theta_hat_matrix-theta_matrix).^2));


%% Lyapunov Method Offline
N_max = 5e5; % total budget

Q_off = zeros(1,length(theta)); 
n_off_total = zeros(length(theta),1);
cost_sum = 0;

theta_hat_matrix = 1*ones(grid_m, grid_n);
% A_matrix_hat_all = zeros(grid_m*grid_n,grid_m*grid_n);
% counter = 1;
% for j=1:grid_m
%  for i=1:grid_n
%      A_matrix_hat_all(:,counter) = grid_vector(i,j, theta_hat_matrix, n_hop_max,grid_m,grid_n);
%      counter = counter+1;
%  end
% end
% M_constraint_hat=double(A_matrix_hat_all);

obs_matrix = zeros(grid_m, grid_n);
cnt_matrix = zeros(grid_m, grid_n);


n_off_total_all = zeros(length(theta),N_max);
while cost_sum<N_max
     n_next = n_off_total+eye(length(theta));
     psi_off = Q_off*(-M_constraint*n_next);
     [min_temp, k_select] = min(psi_off); 
     if (sum(psi_off==min_temp)>1)
        k_select_rand = find(psi_off==min_temp);
        rand_temp = randperm(length(k_select_rand));
        k_select = k_select_rand(rand_temp(1));
     end

     n_off_total(k_select) =  n_off_total(k_select)+1;

     [result_matrix,flag_matrix] = grid_simulator(k_select, theta_matrix, n_hop_max,grid_m,grid_n); 
     obs_matrix = obs_matrix+result_matrix.*flag_matrix;
     cnt_matrix = cnt_matrix+flag_matrix;
     theta_hat_matrix(flag_matrix>0) = obs_matrix(flag_matrix>0)./cnt_matrix(flag_matrix>0);
     
%      theta_hat_matrix_ucb = min(theta_hat_matrix+sqrt(2*log(1/delta)./cnt_matrix),ones(grid_m,grid_n));
%      A_matrix_hat_all = zeros(grid_m*grid_n,grid_m*grid_n);
%      counter = 1;
%      for j=1:grid_m
%          for i=1:grid_n
%              A_matrix_hat_all(:,counter) = grid_vector(i,j, theta_hat_matrix_ucb, n_hop_max,grid_m,grid_n);
%              counter = counter+1;
%          end
%      end
%      M_constraint_hat=double(A_matrix_hat_all);
     
     cost_sum = cost_sum+1;
     Q_off = max(zeros(1,length(theta)),Q_off+(b-M_constraint*n_off_total)'); %reshape(cnt_matrix,[],1)
     n_off_total_all(:,cost_sum) = n_off_total;

     if (sum(Q_off)==0)%sum(sum(cnt_matrix<reshape(b,grid_m,grid_n)))==0
         break;
     end
end

mse_Lyapunov = sum(sum((theta_hat_matrix-theta_matrix).^2));


%% Random Sampling

N_max = 4e5; % total budget

n_rand_total = zeros(length(theta),1);
cost_sum_rand = 0;

theta_hat_matrix = 0.5*ones(grid_m, grid_n);

obs_matrix = zeros(grid_m, grid_n);
cnt_matrix = zeros(grid_m, grid_n);

n_rand_total_all = zeros(length(theta),N_max);
while cost_sum_rand<N_max
     
     k_select_rand = randperm(grid_m*grid_n); 
     k_select = k_select_rand(1);

     n_rand_total(k_select) =  n_rand_total(k_select)+1;

     [result_matrix,flag_matrix] = grid_simulator(k_select, theta_matrix, n_hop_max,grid_m,grid_n); 
     obs_matrix = obs_matrix+result_matrix.*flag_matrix;
     cnt_matrix = cnt_matrix+flag_matrix;
     theta_hat_matrix(flag_matrix>0) = obs_matrix(flag_matrix>0)./cnt_matrix(flag_matrix>0);

     cost_sum_rand = cost_sum_rand+1;
     n_rand_total_all(:,cost_sum_rand) = n_rand_total;

     if (sum(sum(cnt_matrix<reshape(b,grid_m,grid_n)))==0)
         break;
     end
end

mse_random = sum(sum((theta_hat_matrix-theta_matrix).^2));


