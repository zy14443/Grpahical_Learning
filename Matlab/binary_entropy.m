function H = binary_entropy(p)
    H = (-p*log2(p)-(1-p)*log2(1-p));
end