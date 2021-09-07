function variance = var_emp(n_history)

if isempty(n_history)
    variance = inf;
else
    variance = var(n_history);
end