p_1 = 0.5;

H_p = binary_entropy(p_1);

epsilon=0.05;
delta = 0.05;

n_1 = ((1-delta)*log(1/(2*epsilon))-1)/H_p;