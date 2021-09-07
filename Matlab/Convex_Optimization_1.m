%Convex Optimization Test
delta = 1-0.95;
epsilon = 0.05;
K=100;
counter = 1;
% for theta_1 = 0.1:0.01:0.9

    theta = ones(1,K)*0.3;
    
    f = ones(1,K);
%     fun = @(x)norm(x);
%     x0 = zeros(1,length(theta));
    
    lb = zeros(1,length(theta));
    ub = [];
    
    M_constraint=double(tree_structure_theta');
%      M=[1,0,0,0,0;
%         0,1,0,0,0;
%         theta(1),theta(2),1,0,0;
%         0,0,theta(3),1,0;
%         0,0,theta(3),0,1;];
    
%     M=[1,0,0,0,0,0,0;
%    theta(1),1,0,0,0,0,0;
%    theta(1),0,1,0,0,0,0;
%    theta(1),0,0,1,0,0,0;
%    theta(1),0,0,0,1,0,0;
%    theta(1),0,0,0,0,1,0;
%    theta(1),0,0,0,0,0,1;];

%   M=[1,0,0,0,0,0,0;
%    theta(1),1,0,0,0,0,0;
%    theta(1)*theta(2),theta(2),1,0,0,0,0;
%    theta(3)*theta(2)*theta(1),theta(3)*theta(2),theta(3),1,0,0,0;
%    theta(4)*theta(3)*theta(2)*theta(1),theta(4)*theta(3)*theta(2),theta(4)*theta(3),theta(4),1,0,0;
%    theta(5)*theta(4)*theta(3)*theta(2)*theta(1),theta(5)*theta(4)*theta(3)*theta(2),theta(5)*theta(4)*theta(3),theta(5)*theta(4),theta(5),1,0;
%    theta(6)*theta(5)*theta(4)*theta(3)*theta(2)*theta(1),theta(6)*theta(5)*theta(4)*theta(3)*theta(2),theta(6)*theta(5)*theta(4)*theta(3),theta(6)*theta(5)*theta(4),theta(6)*theta(5),theta(6),1;];

%    b = log(2/delta)*(2*theta.*(1-theta)+2/3*epsilon)/epsilon^2;
   b = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/epsilon^2;
   Aeq=[];
   beq=[];
   x = linprog(f,-M_constraint,-b,Aeq,beq,lb,ub);
%    x = fmincon(fun,x0,-M,-b,Aeq,beq,lb,ub);

    n(counter,:) = sum(x);    
%     n4(counter) = x(4);
%     epsilon(counter) = x(length(theta)+1);
%     counter = counter+1;
%     theta_1
% end

%%
% epsilon_0 = epsilon;
figure;
t = 0.1:0.01:0.9;
plot(t,n)
hold on;
% plot(t,n(:,1))
% hold on;
% plot(t,n(:,2))
% hold on;
% plot(t,n(:,3))
legend('K=3','K=4','K=5','K=6','K=7')
% legend('0.1 ind','0.25 ind','0.5 ind','0.1 chain','0.25 chain','0.5 chain','0.75 chain')
xlabel('theta')
ylabel('total sample complexity')


%%

count = 1;
epsilon_min = epsilon;
for theta_1 = 0.1:0.01:0.9
    
%     alpha_1 = 1-2*theta_1;
%     alpha_2 = 1;
%     alpha_3 = 1;
    
    sigma_1 = (1-theta_1)*theta_1;
    sigma_2 = (1-theta_1)*theta_1;
    sigma_3 = (1-theta_1)*theta_1;
    sigma_4 = (1-theta_1)*theta_1;
    sigma_5 = (1-theta_1)*theta_1;
    sigma_6 = (1-theta_1)*theta_1;
    sigma_7 = (1-theta_1)*theta_1;
    
%     c_prime = N/((alpha_1+alpha_2+alpha_3)*log(2/delta));
%     
%     epsilon_min = (1+sqrt(1+18*c_prime*(alpha_1*sigma_1+alpha_2*sigma_2+alpha_3*sigma_3)/(alpha_1+alpha_2+alpha_3)))/(3*c_prime);
    
    
    n_1_Bernstein_graph(count) = log(2/delta)*(2*sigma_1+2/3*epsilon_min)/(epsilon_min*epsilon_min);
    n_2_Bernstein_graph(count) = log(2/delta)*(2*sigma_2+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;
    n_3_Bernstein_graph(count) = log(2/delta)*(2*sigma_3+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;
    n_4_Bernstein_graph(count) = log(2/delta)*(2*sigma_4+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;
    n_5_Bernstein_graph(count) = log(2/delta)*(2*sigma_5+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;
    n_6_Bernstein_graph(count) = log(2/delta)*(2*sigma_6+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;
    n_7_Bernstein_graph(count) = log(2/delta)*(2*sigma_7+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;


    count=count+1;
end
%%
t = 0.1:0.01:0.9;
% plot(t,epsilon)
% hold on;
plot(t,n_1_Bernstein_graph+n_2_Bernstein_graph+n_3_Bernstein_graph+n_4_Bernstein_graph+n_5_Bernstein_graph+n_6_Bernstein_graph,'--')
hold on;
legend('K=3','K=4','K=5','K=6','K=7')
% +n_4_Bernstein_graph+n_5_Bernstein_graph+n_6_Bernstein_graph+n_7_Bernstein_graph
% plot(t,n_2_Bernstein_graph)
% hold on;
% plot(t,n_3_Bernstein_graph)