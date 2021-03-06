function [y,Y,P,Y1] = ut(f,X,Wm,Wc,n,R)  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Unscented Transformation  
% UT转换函数  
% Input:  
% f: nonlinear map  
% X: sigma points  
% Wm: weights for mean  
% Wc: weights for covariance  
% n: number of outputs of f  
% R: additive covariance  
% Output:  
% y: transformed mean  
% Y: transformed smapling points  
% P: transformed covariance  
% Y1: transformed deviations  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
L = size(X,2);  
y = zeros(n,1);  
Y = zeros(n,L);  
for k=1:L  
    Y(:,k) = f(X(:,k));  
    y = y+Wm(k)*Y(:,k); 
    disp(y)
end  
Y1 = Y-y(:,ones(1,L));  
P = Y1*diag(Wc)*Y1'+R;   % 为什么是R? 为什么作者对Q设置成对角矩阵?