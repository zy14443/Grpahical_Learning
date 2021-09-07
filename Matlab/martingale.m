kk=1;
zz=15:21;
p = zeros(length(zz),1);
for z=zz

TT = 1e5;



for T=TT

q=10;
% z=12;
epsilon = 0.5;
Y_0 = 0;

N_exp = 10000;
result = 0;
for counter=1:N_exp
   Y = zeros(1,T);
   for i=1:T

        if i==1
            Y(i)=q;
        else
            if Y(i-1)<q
                Y(i)=Y(i-1)+1;
            else
                x=rand;
                if x<=(1-epsilon)/2
                    Y(i) = Y(i-1)+1;
                else
                    Y(i)= Y(i-1)-1;
                end
            end
        end
    end
    result = result+(sum(Y>z)>0);
end

end
p(kk) = result/N_exp;
kk = kk+1;
disp(kk);

end

plot(zz,p);
hold on;
%%
% plot(TT,p);
% hold on;
% 
% xlabel('N')
% ylabel('P')
% legend('y-q*=3','y-q*=4','y-q*=5','y-q*=6','y-q*=7','y-q*=8');