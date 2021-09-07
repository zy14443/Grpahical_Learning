K = 1000; % total number of nodes
M = 100; % number of experiments to take average
N_max = 2e7; % total budget

epsilon = 0.01;
delta = 1 - 0.95;

theta = zeros(K,1);

% X = 0:.01:1;
% for par = 0.5:0.3:2
%     y = betapdf(X,par,par);
%     plot(X,y)
%     hold on
% end


%% Fixed K, change theta distribution
K=100;
t = 0.5:0.01:2;
n_Bernstein_L_sum = zeros(size(t));
n_Bernstein_L_emp_sum = zeros(size(t));
n_Hoeffding_sum = zeros(size(t));
n_average=10000;

counter=1;
for alpha=t
    
    for tt=1:n_average
        theta = betarnd(alpha,alpha,1,K);

        n_Hoeffding = log(2/delta)/(2*epsilon*epsilon)*ones(size(theta));
        n_Bernstein_L = log(2/delta)*(2*theta.*(1-theta)+2*epsilon/3)/(epsilon*epsilon);
        n_Bernstein_L_emp = log(3/delta)*(2*theta.*(1-theta)+3*epsilon)/(epsilon*epsilon);

        n_Bernstein_L_sum(counter) = n_Bernstein_L_sum(counter)+sum(n_Bernstein_L);
        n_Bernstein_L_emp_sum(counter) = n_Bernstein_L_emp_sum(counter)+sum(n_Bernstein_L_emp);
        n_Hoeffding_sum(counter) = n_Hoeffding_sum(counter)+sum(n_Hoeffding);
    end

    counter=counter+1;
end



plot(t,n_Hoeffding_sum/n_average,'-','color',[0.8500, 0.3250, 0.0980]);
hold on;
plot(t,n_Bernstein_L_sum/n_average,'-','color',[0.9290, 0.6940, 0.1250]);
hold on;
plot(t,n_Bernstein_L_emp_sum/n_average,'-','color',[0.4940, 0.1840, 0.5560]);
hold on;

%% Sanity check on estimation
K=100;
M=1000;
t = 0.5:0.01:2;
alpha=2;

theta = betarnd(alpha,alpha,1,K);
n_current = zeros(M,K);
correct_estimate = zeros(M,K);

parfor i=1:M %running M times to take averag
    counter=1;
    
    n_history = cell(K,1);    
    n_current_temp=zeros(1,K);
    r = zeros(size(theta));
    tic
    while counter < N_max
        
        if (sum(r)==K)
            break;
        end
        
        [tmax,flip] = min(r);
        n_current_temp(flip) = n_current_temp(flip)+1;
        x = rand;
        if (x <= theta(flip))
            n_history{flip} = [n_history{flip};1];
        else
            n_history{flip} = [n_history{flip};0];     
        end 
        
        
        j=flip;         
        variance_emp = var_emp(n_history{j});
        n_target = log(3/delta)*(2*variance_emp+3*epsilon)/(epsilon*epsilon);
        r(j) = length(n_history{j}) >= n_target;
 
        counter=counter+1;
        
    end
    
    correct_estimate_temp=zeros(1,K);
    for kk=1:K
        m_emp = mean_emp(n_history{kk});
        correct_estimate_temp(kk)=abs(m_emp-theta(kk));
    end
    toc

    n_current(i,:)=n_current_temp;
    correct_estimate(i,:)=correct_estimate_temp;
end
save('sanity_check_H','correct_estimate','n_current','theta')
correct_rate = sum(correct_estimate<epsilon);
%%
n_Hoeffding = log(2/delta)/(2*epsilon*epsilon)*ones(size(theta));
for i =2:K
    n_Hoeffding(i) = n_Hoeffding(i)+n_Hoeffding(i-1);
end

%%
n_Bernstein_L = zeros(size(theta));
n_Bernstein_L_emp = zeros(size(theta));

% 
% for i = 1:K
%     theta(i) = betarnd(0.5,0.5); %low variance
% end

for i = 1:K
    n_Bernstein_L(i) = log(2/delta)*(2*theta(i)*(1-theta(i))+2*epsilon/3)/(epsilon*epsilon);
    n_Bernstein_L_emp(i) = log(3/delta)*(2*theta(i)*(1-theta(i))+3*epsilon)/(epsilon*epsilon);
end
    
for i = 2:K
    n_Bernstein_L(i) = n_Bernstein_L(i)+n_Bernstein_L(i-1);
    n_Bernstein_L_emp(i) = n_Bernstein_L_emp(i)+n_Bernstein_L_emp(i-1);
end

%%
% Online Bernstein
% n_empirical_Bernstein_L = zeros(size(theta)); 
n_current = zeros(M,K);
tic
parfor i=1:M %running M times to take averag
    counter=1;
    
    n_history = cell(K,1);    
    n_current_temp=zeros(1,K);
    r = zeros(size(theta));
    
    while counter < N_max
        
        if (sum(r)==K)
            break;
        end
        
        [tmax,flip] = min(r);
        n_current_temp(flip) = n_current_temp(flip)+1;
        x = rand;
        if (x <= theta(flip))
            n_history{flip} = [n_history{flip};1];
        else
            n_history{flip} = [n_history{flip};0];     
        end 
        
        
        j=flip;         
        variance_emp = var_emp(n_history{j});
        n_target = log(3/delta)*(2*variance_emp+3*epsilon)/(epsilon*epsilon);
        r(j) = length(n_history{j}) >= n_target;
 
        counter=counter+1;
        
    end
    

    n_current(i,:)=n_current_temp;
end
toc
n_empirical_Bernstein_L = mean(n_current, 1);

for i = 2:K
    n_empirical_Bernstein_L(i) = n_empirical_Bernstein_L(i)+n_empirical_Bernstein_L(i-1);
end

save('n_empirical_Bernstein_L','n_current','theta');

%%
n_Bernstein_U = zeros(size(theta));
n_Bernstein_U_emp = zeros(size(theta));

% for i = 1:K
%     theta(i) = betarnd(1,1); %uniform
% end

for i = 1:K
    n_Bernstein_U(i) = log(2/delta)*(2*theta(i)*(1-theta(i))+2*epsilon/3)/(epsilon*epsilon);
    n_Bernstein_U_emp(i) = log(3/delta)*(2*theta(i)*(1-theta(i))+3*epsilon)/(epsilon*epsilon);
end
    
for i = 2:K
    n_Bernstein_U(i) = n_Bernstein_U(i)+n_Bernstein_U(i-1);
    n_Bernstein_U_emp(i) = n_Bernstein_U_emp(i)+n_Bernstein_U_emp(i-1);
end
%%
% Online Bernstein
% n_empirical_Bernstein_U = zeros(size(theta)); 
n_current = zeros(M,K);
tic
parfor i=1:M %running M times to take averag
    counter=1;
    
    n_history = cell(K,1);    
    n_current_temp=zeros(1,K);
    r = zeros(size(theta));
    tic
    while counter < N_max
        
        if (sum(r)==K)
            break;
        end
        
        [tmax,flip] = min(r);
        n_current_temp(flip) = n_current_temp(flip)+1;
        x = rand;
        if (x <= theta(flip))
            n_history{flip} = [n_history{flip};1];
        else
            n_history{flip} = [n_history{flip};0];     
        end 
        
        
        j=flip;         
        variance_emp = var_emp(n_history{j});
        n_target = log(3/delta)*(2*variance_emp+3*epsilon)/(epsilon*epsilon);
        r(j) = length(n_history{j}) >= n_target;
 
        counter=counter+1;
        
    end
    toc

    n_current(i,:)=n_current_temp;
end
toc

n_empirical_Bernstein_U = mean(n_current, 1);

for i = 2:K
    n_empirical_Bernstein_U(i) = n_empirical_Bernstein_U(i)+n_empirical_Bernstein_U(i-1);
end
save('n_empirical_Bernstein_U','n_current','theta');
%%
n_Bernstein_H = zeros(size(theta));
n_Bernstein_H_emp =  zeros(size(theta));

% for i = 1:K
%     theta(i) = betarnd(2,2); %high variance
% end

for i = 1:K
    n_Bernstein_H(i) = log(2/delta)*(2*theta(i)*(1-theta(i))+2*epsilon/3)/(epsilon*epsilon);
    n_Bernstein_H_emp(i) = log(3/delta)*(2*theta(i)*(1-theta(i))+3*epsilon)/(epsilon*epsilon);
end
 
for i = 2:K
    n_Bernstein_H(i) = n_Bernstein_H(i)+n_Bernstein_H(i-1);
    n_Bernstein_H_emp(i) = n_Bernstein_H_emp(i) + n_Bernstein_H_emp(i-1);
end

%%
% Online Bernstein
% n_empirical_Bernstein_L = zeros(size(theta)); 
n_current = zeros(M,K);
tic
parfor i=1:M %running M times to take averag
    counter=1;
    
    n_history = cell(K,1);    
    n_current_temp=zeros(1,K);
    r = zeros(size(theta));
    tic
    while counter < N_max
        
        if (sum(r)==K)
            break;
        end
        
        [tmax,flip] = min(r);
        n_current_temp(flip) = n_current_temp(flip)+1;
        x = rand;
        if (x <= theta(flip))
            n_history{flip} = [n_history{flip};1];
        else
            n_history{flip} = [n_history{flip};0];     
        end 
        
        
        j=flip;         
        variance_emp = var_emp(n_history{j});
        n_target = log(3/delta)*(2*variance_emp+3*epsilon)/(epsilon*epsilon);
        r(j) = length(n_history{j}) >= n_target;
 
        counter=counter+1;
        
    end
    toc

    n_current(i,:)=n_current_temp;
end
toc
n_empirical_Bernstein_H = mean(n_current, 1);

for i = 2:K
    n_empirical_Bernstein_H(i) = n_empirical_Bernstein_H(i)+n_empirical_Bernstein_H(i-1);
end

save('n_empirical_Bernstein_H','n_current','theta');
%%

figure;
t = 100:K;
plot(t,n_Hoeffding(100:K));
hold on;
plot(t,n_Bernstein_L(100:K),'-','color',[0.8500, 0.3250, 0.0980]);
hold on;
plot(t,n_Bernstein_U(100:K),'-','color',[0.9290, 0.6940, 0.1250]);
hold on;
plot(t,n_Bernstein_H(100:K),'-','color',[0.4940, 0.1840, 0.5560]);
hold on;

plot(t,n_Bernstein_L_emp(100:K),'--','color',[0.8500, 0.3250, 0.0980]);
hold on;
plot(t,n_Bernstein_U_emp(100:K),'--','color',[0.9290, 0.6940, 0.1250]);
hold on;
plot(t,n_Bernstein_H_emp(100:K),'--','color',[0.4940, 0.1840, 0.5560]);
hold on;

plot(t,n_empirical_Bernstein_L,'--','color',[0.8500, 0.3250, 0.0980]);
hold on;
plot(t,n_empirical_Bernstein_U,'--','color',[0.9290, 0.6940, 0.1250]);
hold on;
plot(t,n_empirical_Bernstein_H,'--','color',[0.4940, 0.1840, 0.5560]);
hold on;

plot(t,n_Bernstein_L_emp(100:K)./n_Bernstein_L(100:K),'--','color',[0.8500, 0.3250, 0.0980]);
hold on;

plot(t,n_Bernstein_U_emp(100:K)./n_Bernstein_L(100:K),'--','color',[0.9290, 0.6940, 0.1250]);
hold on;

plot(t,n_Bernstein_H_emp(100:K)./n_Bernstein_L(100:K),'--','color',[0.4940, 0.1840, 0.5560]);


legend('Heoffding', 'Emp Bernstein Low', 'Bernstein Low', 'Emp Bernstein Uniform', 'Bernstein Uniform', 'Emp Bernstein High', 'Bernstein Uniform', 'Location', 'Northwest')
xlabel('\fontname{Times New Roman}\fontsize{16}Number of nodes')
ylabel('\fontname{Times New Roman}\fontsize{14}Sample complexity')
% plot(t,n_empirical_Bernstein_L);
% hold on;

%%
K=100;

n_distribution = sum(n_current,2);
figure;
histogram(n_distribution, 'Normalization','pdf')
hold on;
xline(n_Hoeffding(K),'LineWidth', 2, 'Color', 'r');
xlim([0,2e6]);
line([n_Hoeffding(K), n_Hoeffding(K)], ylim, 'LineWidth', 2, 'Color', 'r');
hold on;
