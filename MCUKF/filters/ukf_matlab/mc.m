function [Cx, Cy] = mc(sigma, X_hat, L, m, D, W) 
% kernel function
G=@(x)exp(-(x^2/2*sigma^2));
entropy_x = zeros(1, L);
entropy_y = zeros(1, m);
E = D - W*X_hat;
for i = 1:L
    entropy_x(i) = G(E(i));
end
for i = 1:m
    entropy_y(i) = G(E(i+L));
end
Cx = diag(entropy_x);
Cy = diag(entropy_y);
