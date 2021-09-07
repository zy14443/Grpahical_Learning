function [c,ceq] = circlecon(x,theta)

delta = 1-0.95;

%independent
% M=[1,0,0;
%    0,1,0;
%    0,0,1;];
%    0,0,0,1];

%Chain graph
% M=[1,0,0,0,0,0;
%    theta(1),1,0,0,0,0;
%    theta(1)*theta(2),theta(2),1,0,0,0;
%    theta(1)*theta(2)*theta(3), theta(2)*theta(3),theta(3),1,0,0;
%    theta(1)*theta(2)*theta(3)*theta(4),theta(2)*theta(3)*theta(4),theta(3)*theta(4),theta(4),1,0;
%    theta(1)*theta(2)*theta(3)*theta(4)*theta(5),theta(2)*theta(3)*theta(4)*theta(5),theta(3)*theta(4)*theta(5),theta(4)*theta(5),theta(5),1;];

%Tree
M=[1,0,0;
   theta(1),1,0;
   theta(1),0,1;];
%    theta(1),0,0,1,0,0,0;
%    theta(1),0,0,0,1,0,0;
%    theta(1),0,0,0,0,1,0;
%    theta(1),0,0,0,0,0,1;];

%V_graph
% M=[1,0,0;
%    0,1,0;
%    theta(1),theta(2),1;];

n_p = M*x(1:length(theta))';

sigma_sq = zeros(size(theta));

for i = 1:length(theta)
    sigma_sq(i) = theta(i)*(1-theta(i));
    c(i) = log(2/delta)*(1+sqrt(1+18*n_p(i)*sigma_sq(i)/log(2/delta)+1))/(3*n_p(i))-x(length(theta)+1);
end

ceq = [];