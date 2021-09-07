% K = 5; % total number of nodes
M = 100; % number of experiments to take average
N_max = 1e6; % total budget

epsilon = 0.01;
delta = 1 - 0.95;


M = [1,0,0.42749794,0,0;
    0,1,0.42749794,0,0;
    0,0,1,0.42749794,0.42749794;
    0,0,0,1,0;
    0,0,0,0,1;];

%%
results=[];

K=3;
% theta_0=0.3;
k_range = 2:20;
theta_range = 0.1:0.05:0.9;

n_sim=length(theta_range);

n_LP_total = zeros((n_sim),1);
n_Bernstein = zeros((n_sim),1);
% n_LP_total_emp = zeros((n_sim),1);
% n_Bernstein_emp = zeros((n_sim),1);
for kk=1:length(theta_range)
%     K=k_range(kk);
    theta_0=theta_range(kk);
%     n_generation = K-1;
%     [A_chain] = adjacency_matrix_generator(ones(1,n_generation+1),n_generation); %Chain graph
%     A_chain_up = triu(A_chain);

%     ng=kk;
%     m0=0; m1=1; m2=2;        % The bracings for full and genreal binary trees
%     nd=2; 
%     nn=(nd^(ng+1)-1)/(nd-1);
%     B=Full_Binary_Branching(nn,m0,m2,0);   
    
    n_generation = 1;
    n_branch = K-1;
    [A_star] = adjacency_matrix_generator([n_branch,zeros(1,n_branch)],n_generation); %Star graph
    A_star_up = triu(A_star);

    %Power Law
%     seed =[0 1 0 0 1;1 0 0 1 0;0 0 0 1 0;0 1 1 0 0;1 0 0 0 0];
%     A_power_law = SFNG(K, 2, seed);
%     A_power_law_up = triu(A_power_law);
%     PL_Equation = PLplot(A_power_law);

%% Offline with Known Parameters
    

    tree_structure = A_star_up;
    K=size(tree_structure,1);
    
    theta = ones(size(tree_structure,1),1)*theta_0;
    tree_structure_theta = eye(size(tree_structure))+diag(theta)*(tree_structure);

    f = ones(1,K);
    lb = zeros(1,length(theta));
    ub = [];
    M_constraint=double(tree_structure_theta');

    b = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/epsilon^2;
    % b = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/epsilon^2;

    Aeq=[];
    beq=[];
    n_LP_all = linprog(f,-M_constraint,-b,Aeq,beq,lb,ub);
    n_LP_total(kk,1) = sum(n_LP_all);
    
%     y_LP_all = linprog(-b,M_constraint',f,Aeq,beq,lb,ub);
    n_Bernstein(kk,1) = K*log(2/delta)*(2*theta_0*(1-theta_0)+2*epsilon/3)/(epsilon*epsilon);

%% Offline with Unknown Parameters
%     theta = ones(K,1)*theta_0;
% 
% %     tree_structure = A_star;
%     tree_structure_theta = eye(size(tree_structure))+diag(theta)*(tree_structure);
% 
%     f = ones(1,K);
%     lb = zeros(1,length(theta));
%     ub = [];
%     M_constraint=double(tree_structure_theta');
% 
%     % b = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/epsilon^2;
%     b = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/epsilon^2;
% 
%     Aeq=[];
%     beq=[];
%     n_LP_all_emp = linprog(f,-M_constraint,-b,Aeq,beq,lb,ub);
%     n_LP_total_emp(kk) = sum(n_LP_all_emp);
% 
%     y_LP_all_emp = linprog(-b,M_constraint',f,Aeq,beq,lb,ub);
%   
%     n_Bernstein_emp(kk) = K*log(3/delta)*(2*theta_0*(1-theta_0)+3*epsilon)/(epsilon*epsilon);

end
%% Lyapunov Method

Q_off = b';%zeros(1,length(theta));
n_off_total = zeros(length(theta),1);
cost_sum = 0;

n_off_total_all = zeros(length(theta),N_max);
while cost_sum<N_max
     n_next = n_off_total+eye(length(theta));
     psi_off = Q_off*(-M_constraint*n_next);
     [min_temp, k_select] = min(psi_off); 
     
     n_off_total(k_select) =  n_off_total(k_select)+1;
     cost_sum = cost_sum+1;
     Q_off = max(zeros(1,length(theta)),Q_off-M_constraint(:,k_select)');%(b-M_constraint*n_off_total)');
     n_off_total_all(:,cost_sum) = n_off_total;

     if (sum(Q_off)==0)
         break;
     end
end

plot(1:cost_sum, n_off_total_all([1,3,5],1:cost_sum)./sum(n_off_total_all(:,1:cost_sum),1),'LineWidth', 2)
hold on;
line(1:cost_sum,  n_LP_all([1,3,5],:)./n_LP_total.*(ones(1,cost_sum)), 'LineStyle','--','LineWidth', 2)
xlabel('T')
ylabel('Sample proportion')
legend('Node 1', 'Node 3', 'Node 5')
%%
% figure;
% t = k_range;
t=theta_range;
% plot(t,n_LP_total);
hold on;
plot(t,mean(n_LP_total,2),'-', 'LineWidth',2);

hold on;
set(gca, 'YScale', 'log')
% results = [results;n_LP_total_emp'];
hold on;
xlabel('K')
ylabel('sample complexity')

legend('K=6','K=5', 'K=4','K=3','Location','Northeast')
grid on;
% plot(t,n_Bernstein,'-');
% hold on;
% plot(t,n_Bernstein_emp,'--');
% hold on;
%%
for kkk=[1]
plot(t,results(kkk,:),'-','LineWidth',2);
hold on;
end
xlabel('\theta')
ylabel('Sample complexity')
% legend('\theta=0.1','\theta=0.2','\theta=0.3','\theta=0.4','\theta=0.5','Location','southeast')
%% Known Parameters
% K=500;
theta_0=0.8;

k_range = 10:10:100;
% theta_range = 0.1:0.1:0.9;

n_sim=length(k_range);

n_empirical_Bernstein_New = cell((n_sim),1);
n_LP_total = zeros((n_sim),1);
n_Bernstein = zeros((n_sim),1);


for kk=1:length(k_range)
    K=k_range(kk);
%     theta_0=theta_range(kk);
%% Online with Known Parameters
% n_empirical_Bernstein_New = zeros(size(theta)); 
    n_current = zeros(M,K);
    theta = ones(K,1)*theta_0;
    
    n_generation = K-1;
    [A_chain] = adjacency_matrix_generator(ones(1,n_generation+1),n_generation); %Chain graph
    A_chain_up = triu(A_chain);
    
    tree_structure = A_chain;
    tree_structure_theta = eye(size(tree_structure))+diag(theta)*(tree_structure);

    for i=1:M %running M times to take averag

        counter=1;        
        n_history = cell(K,1); 

        n_current_temp=zeros(1,K);
        r = ones(size(theta));
        
        while counter < N_max
            if (sum(r)==0)
                break
            end

%             tree_structure_theta = eye(size(tree_structure))+diag(theta)*(tree_structure);

            p_hat = tree_structure_theta*r;        
            [tmax,flip] = max(p_hat);

            n_current_temp(flip) = n_current_temp(flip)+1;                      

            x = rand;
            if (x <= theta(flip))
                n_history{flip} = [n_history{flip};1];

                unlearned_child = tree_structure(flip,:).*r';                
                unlearned_child(flip)=0;
                unlearned_index=1:K;
                unlearned_index = unlearned_index(unlearned_child > 0 & unlearned_child <=1);      

                for cc=unlearned_index

                    x = rand;
                    if (x <= theta(cc))
                        n_history{cc} = [n_history{cc};1];
                    else
                       n_history{cc} = [n_history{cc};0];
                    end
                    j=cc;
                   
                    n_target = log(2/delta)*(2*theta(j)*(1-theta(j))+2/3*epsilon)/(epsilon*epsilon);% Bernstein
                    r(j) = length(n_history{j}) < n_target;

                end

            else
                n_history{flip} = [n_history{flip};0];
            end

            j=flip;
   
            n_target = log(2/delta)*(2*theta(j)*(1-theta(j))+2/3*epsilon)/(epsilon*epsilon);% Bernstein
            r(j) = length(n_history{j}) < n_target;

            counter = counter+1;
        end
        
        n_current(i,:)=n_current_temp;


    end

    n_empirical_Bernstein_New{kk} = mean(n_current, 1);

    disp(kk);
    
    f = ones(1,K);
    lb = zeros(1,length(theta));
    ub = [];
    M_constraint=double(tree_structure_theta');

    b = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/epsilon^2;
    % b = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/epsilon^2;

    Aeq=[];
    beq=[];
    n_LP_all = linprog(f,-M_constraint,-b,Aeq,beq,lb,ub);
    n_LP_total(kk) = sum(n_LP_all);

    n_Bernstein(kk) = K*log(2/delta)*(2*theta_0*(1-theta_0)+2*epsilon/3)/(epsilon*epsilon);

end

%%

t=k_range;
figure;
n_empirical_Bernstein_New_mean = cellfun(@sum,n_empirical_Bernstein_New);
plot(t,n_empirical_Bernstein_New_mean);
hold on;
plot(t,n_LP_total);
hold on;
plot(t,n_Bernstein);


figure;
plot(t,n_empirical_Bernstein_New_mean./n_LP_total);


% save('Known_chain_K_10_100_theta_0_8','theta_0','k_range','n_empirical_Bernstein_New','n_LP_total','n_Bernstein');

%% Unknown Parameters
% K=500;
theta_0=0.2;

k_range = 2:10:100;
% theta_range = 0.1:0.1:0.9;

n_sim=length(k_range);

n_empirical_Bernstein_Unknown = cell((n_sim),1);
n_LP_total_emp = zeros((n_sim),1);
n_Bernstein_emp = zeros((n_sim),1);

for kk=1:length(k_range)
    K=k_range(kk);
%% Online with Unknown Parameters
% n_empirical_Bernstein_New = zeros(size(theta)); 

    n_current = zeros(M,K);
    theta = ones(K,1)*theta_0;

    n_generation = K-1;
    [A_chain] = adjacency_matrix_generator(ones(1,n_generation+1),n_generation); %Chain graph
    A_chain_up = triu(A_chain);

    
    tree_structure = A_chain;
    
    parfor i=1:M %running M times to take averag

        counter=1;        
        n_history = cell(K,1); 

        n_current_temp=zeros(1,K);
        r = ones(size(theta));
        theta_hat = ones(size(theta))*0.5;

        while counter < N_max      
            if (sum(r)==0)
                break
            end


            tree_structure_theta_hat = eye(size(tree_structure))+diag(theta_hat)*(tree_structure);
            
            p_hat = tree_structure_theta_hat*r;        
            [tmax,flip] = max(p_hat);

            n_current_temp(flip) = n_current_temp(flip)+1;

            x = rand;
            if (x <= theta(flip))
                n_history{flip} = [n_history{flip};1];

                unlearned_child = tree_structure(flip,:).*r';               
                unlearned_child(flip)=0;
                unlearned_index=1:K;
                unlearned_index = unlearned_index(unlearned_child > 0 & unlearned_child <=1);     

                for cc=unlearned_index
              
                    x = rand;
                    if (x <= theta(cc))
                        n_history{cc} = [n_history{cc};1];
                    else
                       n_history{cc} = [n_history{cc};0];
                    end
                    j=cc;
                    variance_emp = var(n_history{j});
                    theta_hat(j)= mean(n_history{j});
                    n_target = log(3/delta)*(2*variance_emp+3*epsilon)/(epsilon*epsilon); %empirical Bernstein
            %         n_target = log(2/delta)*(2*theta(j)*(1-theta(j))+2/3*epsilon)/(epsilon*epsilon);% Bernstein
                    r(j) = length(n_history{j}) < n_target;
                   
                end
            else
                n_history{flip} = [n_history{flip};0];
            end

            j=flip;
            variance_emp = var(n_history{j});
            theta_hat(j)= mean(n_history{j});
            n_target = log(3/delta)*(2*variance_emp+3*epsilon)/(epsilon*epsilon); %empirical Bernstein
    %         n_target = log(2/delta)*(2*theta(j)*(1-theta(j))+2/3*epsilon)/(epsilon*epsilon);% Bernstein
            r(j) = length(n_history{j}) < n_target;

            counter = counter+1;
        end
        n_current(i,:)=n_current_temp;


    end
    
    n_empirical_Bernstein_Unknown{kk} = mean(n_current, 1);

    disp(kk);
    
    tree_structure_theta = eye(size(tree_structure))+diag(theta)*(tree_structure);
    f = ones(1,K);
    lb = zeros(1,length(theta));
    ub = [];
    M_constraint=double(tree_structure_theta');

    % b = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/epsilon^2;
    b = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/epsilon^2;

    Aeq=[];
    beq=[];
    n_LP_all_emp = linprog(f,-M_constraint,-b,Aeq,beq,lb,ub);

    n_LP_total_emp(kk) = sum(n_LP_all_emp);

    n_Bernstein_emp(kk) = K*log(3/delta)*(2*theta_0*(1-theta_0)+3*epsilon)/(epsilon*epsilon);

end

t=k_range;
figure;
n_empirical_Bernstein_Unknown = cellfun(@sum,n_empirical_Bernstein_Unknown);
plot(t,n_empirical_Bernstein_Unknown);
hold on;
plot(t,n_LP_total);
hold on;
plot(t,n_Bernstein_emp);


figure;
plot(t,n_empirical_Bernstein_Unknown./n_LP_total);


save('Unknown_chain_K_2_100_theta_0_2','theta_0','k_range','n_empirical_Bernstein_Unknown','n_LP_total_emp','n_Bernstein_emp');

%%

  % n_generation = 2;
    % n_branch = 3;
    % [A_full_tree,K] = adjacency_matrix_generator([ones(1,(1-n_branch^(n_generation+1))/(1-n_branch))*n_branch,zeros(1,n_branch^n_generation)],n_generation); %Full tree
%%
%             unlearned_index=1:K;
%             unlearned_index(flip)=0;
%             unlearned_index = unlearned_index(unlearned_child > 0 & unlearned_child <=1);            
%    
%             x = rand(length(unlearned_index),1);
% 
%             x_run = x - theta(unlearned_index);
% 
%             flip_index=1:K;
%             flip_index_1 = flip_index(x_run<=0);
%             flip_index_0 = flip_index(x_run>0);
% 
%             n_history(flip_index_1) = cellfun(@(x) [x;1],n_history(flip_index_1),'UniformOutput',false);
%             n_history(flip_index_0) = cellfun(@(x) [x;0],n_history(flip_index_0),'UniformOutput',false);
% 
%             %variance_emp = var_emp(n_history{j});
%             %n_target(j) = log(3/delta)*(2*variance_emp(j)+3*epsilon)/(epsilon*epsilon); %empirical Bernstein
%             n_target = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/(epsilon*epsilon);% Bernstein
%             L_history = cellfun(@length,n_history);
%             r =  L_history < n_target;
