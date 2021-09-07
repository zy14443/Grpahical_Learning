%     parfor counter = 1:N_repeat
%         %empirical cost
%         sum_x_opt = zeros(K,1);
% 
%         %empirical penalty
%         sum_y_opt = zeros(K,1);
% 
%         %empirical reward
%         sum_r_opt = zeros(K,1);
%         cost_optimal = 0;
%         while cost_optimal<=B
% 
%             if rand<z_star(1)
%                k_select = 1;
%             else
%                k_select = 2;
%             end
% 
%             cost = (rand<mu_x(k_select));
%             sum_x_opt(k_select) = sum_x_opt(k_select)+ cost;
%             y_obs = (rand<mu_y(k_select));
%             sum_y_opt(k_select) = sum_y_opt(k_select)+ y_obs;    
%             sum_r_opt(k_select) = sum_r_opt(k_select)+ (rand<mu_r(k_select));
%             cost_optimal = cost_optimal+cost;
% 
%         end
%         reward_opt_total(counter,:) = sum_r_opt';
%         penalty_opt_total(counter,:) = sum_y_opt';
%     end

%     reward_opt_mean = mean(reward_opt_total,1);
%     penalty_opt_mean = mean(penalty_opt_total,1);
%     