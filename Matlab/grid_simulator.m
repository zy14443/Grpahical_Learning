function [result_matrix,flag_matrix] = grid_simulator(k_select, theta_matrix, n_hop_max,grid_m,grid_n)

    i = mod(k_select, grid_m);
    if (i==0)
        i=10;
    end
    j = ceil(k_select / grid_m);

    result_matrix = zeros(grid_m, grid_n);
    flag_matrix = zeros(grid_m, grid_n);
    
    result_matrix(i,j) = (rand<theta_matrix(i,j));
    flag_matrix(i,j) = 1;
    
    if (result_matrix(i,j)==0)
        return
    else
        for r=1:grid_m
            for c=1:grid_n
                distance = abs(r-i)+abs(c-j);                
                if (distance == 1)
                    result_matrix(r,c)=(rand<theta_matrix(r,c));
                    flag_matrix(r,c) = 1;
                end
            end
        end
        for n_hop = 2:n_hop_max
            for r=1:grid_m
                for c=1:grid_n
                   distance = abs(r-i)+abs(c-j);
                   if (distance == n_hop &&  (result_matrix(r-sign(r-i),c)==1 || result_matrix(r,c-sign(c-j))==1))
                       result_matrix(r,c)=(rand<theta_matrix(r,c));
                       flag_matrix(r,c) = 1;            
                   end
                end
            end
        end        
    end
    
end