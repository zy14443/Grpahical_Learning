%Convex Optimization Test
delta = 1-0.95;
N=100;

counter = 1;
for theta_1 = 0.1:0.01:0.9

    theta = [theta_1,0.5,0.5];%ones(1,3)*theta_1;
    % delta = 1-0.95;

    fun = @(x)x(length(theta)+1);

    lb = zeros(1,length(theta)+1);
    ub = [N*ones(1,length(theta)),1];

    A = [ones(1,length(theta)),0];
    b = N;
    Aeq = [];
    beq = [];

    n0 = N/length(theta);
    x0 = [n0*ones(1,length(theta)),1];

    nonlcon = @(x)circlecon(x,theta);
    x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,nonlcon);
    
    n(counter,:) = x(1:length(theta));    
%     n4(counter) = x(4);
    epsilon(counter) = x(length(theta)+1);
    counter = counter+1;
    theta_1
end

%%
% epsilon_0 = epsilon;
t = 0.1:0.01:0.9;
% plot(t,epsilon)
% hold on;
plot(t,n(:,1))
hold on;
plot(t,n(:,2))
hold on;
plot(t,n(:,3))
legend('independent','n=3','n=4','n=5','n=6','n=7')
% legend('0.1 ind','0.25 ind','0.5 ind','0.1 chain','0.25 chain','0.5 chain','0.75 chain')
xlabel('theta')
ylabel('epsilon')

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
    
%     c_prime = N/((alpha_1+alpha_2+alpha_3)*log(2/delta));
%     
%     epsilon_min = (1+sqrt(1+18*c_prime*(alpha_1*sigma_1+alpha_2*sigma_2+alpha_3*sigma_3)/(alpha_1+alpha_2+alpha_3)))/(3*c_prime);
    
    
    n_1_Bernstein_graph(count) = log(2/delta)*(2*sigma_1+2/3*epsilon_min)/(epsilon_min*epsilon_min);
    n_2_Bernstein_graph(count) = log(2/delta)*(2*sigma_2+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;
    n_3_Bernstein_graph(count) = log(2/delta)*(2*sigma_3+2/3*epsilon_min)/(epsilon_min*epsilon_min)-n_1_Bernstein_graph(count)*theta_1;

    count=count+1;
end
%%
t = 0.1:0.01:0.9;
% plot(t,epsilon)
% hold on;
plot(t,n_1_Bernstein_graph)
hold on;
plot(t,n_2_Bernstein_graph)
hold on;
plot(t,n_3_Bernstein_graph)