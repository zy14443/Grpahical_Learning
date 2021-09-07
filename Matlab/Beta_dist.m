delta = 0.05;
epsilon = 0.01;

for kk = 1:100
    counter=1;
    for theta = 0.01:0.01:0.99

        N=100000;
        Y = zeros(N,1);

        a = 1;
        b = 1;

        for n=1:N
            x=rand;

            if x<theta
                Y(n)=1;
                a = a+1;
            else
                Y(n)=0;
                b = b+1;
            end

            if a>1 && b>1
                theta_mode = (a-1)/(a+b-2);
                beta_prob = betacdf(theta_mode+epsilon,a,b)-betacdf(theta_mode-epsilon,a,b);
                if beta_prob>1-delta
                    n_beta(kk,counter) = n;
                    counter = counter+1;
                    break;
                end

            end


    %         Bernstein_Interval = sqrt(2*theta*(1-theta)*log(2/delta)/n)+2/3*log(2/delta)/n;
    % 
    %         if Bernstein_Interval < epsilon
    %             n_Bernstein(counter) = n;
    %             counter = counter+1;
    %             break;
    %         end
        end
    end
end

%%
X = 0.01:0.01:0.99;
% y1 = betapdf(X,a,b);
figure
% plot(X,y1,'Color','r','LineWidth',2)

plot(X,n_Bernstein,'Color','r','LineWidth',2)

hold on
plot(X,mean(n_beta,1),'LineStyle','-.','Color','b','LineWidth',2)
xlabel('\theta');
ylabel('n');
legend('Bernstein','Bayesian');