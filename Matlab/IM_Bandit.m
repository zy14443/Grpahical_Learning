n_V = 5;
n_E = 4;

% theta = rand(n_V,1);
theta = ones(n_V,1);

M_adj = [0,0,1,0,0;
         0,0,1,0,0;
         0,0,0,1,1;
         0,0,0,0,0;
         0,0,0,0,0;];

M_p = M_adj.*rand(size(M_adj));
     
G = diag(theta)+M_p;

%Calculate expected influence (Exact Oracle)
influence_opt = zeros(n_V,1);
for i=flip(1:size(G,1))
    
    influence_opt(i) = G(i,i)*(1+sum(influence_opt(G(i,:)>0)'.*G(i,G(i,:)>0)));
    
end

%% Influence Maximization with Bandit

n=10000; %horizon

N = 1000; %number of simulations

G_reulst_all = zeros(size(G));
N_pull_all = zeros(size(G));
n_influence_all = zeros(N,1);

for n_sim = 1:N
    tic
    % V_index = ones(n_V,1)*inf;
    N_pull = zeros(size(G));
    G_result = zeros(size(G));

    G_hat = ones(size(G)).*(G>0);
    n_influence = 0;

    for t=1:n
        influence = zeros(n_V,1);
        for i=flip(1:size(G_hat,1))
            influence(i) = G_hat(i,i)*(1+sum(influence(G_hat(i,:)>0)'.*G_hat(i,G_hat(i,:)>0)));
        end

    %     influence = zeros(n_V,1);
    %     for i=flip(1:size(G_hat_ucb,1))
    %         influence(i) = G_hat_ucb(i,i)*(1+sum(influence(G_hat_ucb(i,:)>0)'.*G_hat_ucb(i,G_hat_ucb(i,:)>0)));
    %     end

        [max_inf,action] = max(influence);

        %update G_result and N_pull
        G_rand = (rand(size(G))<G);  
        [G_result,N_pull,n_influence] = update(G_result,N_pull,G_rand, action, n_influence);
        G_hat(N_pull>0) = G_result(N_pull>0)./N_pull(N_pull>0);

        UCB = ones(size(G)).*(G>0);
        UCB(N_pull>0) = sqrt(3*log(t)./N_pull(N_pull>0));

        G_hat_ucb = G_hat+UCB;
        G_hat_ucb(G_hat_ucb>1)=1;
    end
    toc
    G_reulst_all = G_reulst_all+G_result;
    N_pull_all = N_pull_all+N_pull;
    n_influence_all(n_sim,1) = n_influence;

end
