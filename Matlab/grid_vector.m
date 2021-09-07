function [A_vector] = grid_vector(i,j, theta_matrix, n_hop_max,grid_m,grid_n)
    A_matrix = zeros(grid_m,grid_n);

    for r=1:grid_m
        for c=1:grid_n
            distance = abs(r-i)+abs(c-j);
            if (distance>1)
                continue;
            else
                if (distance ==0)
                    A_matrix(r,c) = 1;
                elseif (distance == 1)
                    A_matrix(r,c)=theta_matrix(i,j);
                end
            end
        end
    end

    for n_hop = 2:n_hop_max
        for r=1:grid_m
            for c=1:grid_n
               distance = abs(r-i)+abs(c-j);
               if (distance == n_hop)
                    A_matrix(r,c) = A_matrix(r-sign(r-i),c)*theta_matrix(r-sign(r-i),c)+A_matrix(r,c-sign(c-j))*theta_matrix(r,c-sign(c-j));            
               end
            end
        end
    end

    A_vector = reshape(A_matrix, [], 1);
end