% single coin

p_1 = 0.71;

N=1000; %repeat

TT = [100,500,1000, 5000, 10000,1000000]; %horizon
B_repeat = zeros(N,length(TT));
counter=1;
for T = TT
    
    for kk = 1:N
        B_0 = 1;
        p_1_sum=0;
        p_hat = 0.5;
        p_ucb = 0.5;
        p_lcb = 0.5;

        B_history = zeros(1,T);


        for t=1:T
            x=rand;

            p_1_sum = p_1_sum+(x<=p_1);
            p_hat = p_1_sum/t;

            p_ucb = p_hat+sqrt(2*log(1+t*log(t)*log(t))/t);

            p_lcb = p_hat-sqrt(2*log(1+t*log(t)*log(t))/t);

            f_t_star = 2*p_1-1; %Kelly

            f_t = max(2*p_lcb-1,0); %empirical mean
            if f_t>=1
                f_t = 1-0.01;
            end

            B_history(t) = p_1*log((1+f_t_star)/(1+f_t))+(1-p_1)*log((1-f_t_star)/(1-f_t));
    %         f_t = max(2*p_ucb-1,0); %ucb

    %         f_t = max(2*p_lcb-1,0); %lcb

    %         if t==1
    %             B_history(t) = B_0+f_t*B_0*(x<=p_1)-f_t*B_0*(x>p_1);
    %         else
    %             B_history(t) = B_history(t-1)+f_t*B_history(t-1)*(x<=p_1)-f_t*B_history(t-1)*(x>p_1);
    %         end

        end

        B_repeat(kk,counter)=sum(B_history);
        
    end
    counter=counter+1;
end
%%
X = TT;

% figure
plot(X,mean(B_repeat,1),'LineWidth',2)
set(gca, 'XScale', 'log')
hold on
% plot(X,mean(n_beta,1),'LineStyle','-.','Color','b','LineWidth',2)
xlabel('T');
ylabel('Regret');
legend('empirical 0.51','LCB 0.51', 'empirical 0.71', 'LCB 0.71','Location','Northwest');

%% double coins

p_1 = 0.55;
p_2 = 0.65;

N=10000; %repeat

TT = [1000:1000:10000]; %horizon
B_repeat = zeros(N,length(TT));

counter = 1;
for T=TT

    for kk = 1:N
%         B_0 = 100;

        p_1_sum=0;
        n_1_sum=0;
        p_1_hat = 0.5;
        p_1_ucb = 0.5;
        p_l_lcb = 0.5;

        p_2_sum=0;
        n_2_sum=0;
        p_2_hat = 0.5;
        p_2_ucb = 0.5;
        p_2_lcb = 0.5;

    %     B_history = zeros(1,T);


        for t=1:T

            ft = 1+t*log(t)*log(t);
            if p_1_hat > p_2_hat
                n_1_sum = n_1_sum+1;
                x_1=rand;        
                p_1_sum = p_1_sum+(x_1<=p_1);
                p_1_hat = p_1_sum/n_1_sum;        
                p_1_ucb = p_1_hat+sqrt(2*log(T*T)/n_1_sum);     
                p_l_lcb = p_1_hat-sqrt(2*log(T*T)/n_1_sum);
            else
                n_2_sum = n_2_sum+1;
                x_2=rand;        
                p_2_sum = p_2_sum+(x_2<=p_2);
                p_2_hat = p_2_sum/n_2_sum;        
                p_2_ucb = p_2_hat+sqrt(2*log(T*T)/n_2_sum);       
                p_2_lcb = p_2_hat-sqrt(2*log(T*T)/n_2_sum);
            end

        end

        B_repeat(kk,counter) = max(p_1,p_2)*T-(p_1_sum+p_2_sum);
        
    end
    counter = counter+1;
end
%%
X = TT;

figure
plot(X,mean(B_repeat,1),'LineWidth',2)
set(gca, 'XScale', 'linear')
hold on

ylim([0,60])
% plot(X,mean(n_beta,1),'LineStyle','-.','Color','b','LineWidth',2)
xlabel('T');
ylabel('B');
legend('empirical mean','UCB','LCB','offline')

%%
i=1;
for p = 0.1:0.01:0.45
    H_p(i) = binary_entropy(p);
    i=i+1;
end

plot(0.1:0.01:0.45,(1-H_p));

