function average = mean_emp(n_history)

if isempty(n_history)
    average = 0.5;
else
    average = mean(n_history);
end