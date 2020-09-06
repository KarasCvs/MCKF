function [x,P] = mcukf(fstate, x, P, hmeas, z, Q, R, k)  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% UKF Unscented Kalman Filter for nonlinear dynamic systems  
% ���𿨶����˲���Unscented Kalman Filter�������������ڶ�̬������ϵͳ  
% for nonlinear dynamic system (noises are assumed as additive):  
%   x_k+1 = f(x_k) + w_k  
%   z_k = h(x_k) + v_k  
% w ~ N(0,Q) meaning w is gaussian noise with covariance Q  
% v ~ N(0,R) meaning v is gaussian noise with covariance R  
% =============================����˵��=================================  
% Inputs:   
% fstate  -[function]: ״̬����f(x)  
%     x   -     [vec]: ״̬������� "a priori" state estimate  
%     P   -     [mat]: ����������� "a priori" estimated state covariance  
% hmeas   -[function]: ���ⷽ��h(x)  
%     z   -     [vec]: ��������     current measurement  
%     Q   -     [mat]: ״̬��������w(t) process noise covariance  
%     R   -     [mat]: ���ⷽ������v(t) measurement noise covariance  
% Output:  
%     x   -     [mat]: ״̬������� "a posteriori" state estimate  
%     P   -     [mat]: ���������� "a posteriori" state covariance  
% MC:
%     Sp  -          : P��cholesky�ֽ�
%     Sr  -          : R��cholesky�ֽ�
%     S  -           : P��R�Խ�ƴ�Ӻ��cholesky�ֽ�
%     P_hat  -       : Sp*C_x*Sp'
%     R_hat  -       : Sr*C_y*Sr'
%     H      -       : measurement slope matrix, (P1^-1*P12)'
%     W      -       : S^-1[I; H]
%     xi     -       : ��������, ���x-x_hat��R
% =====================================================================  
% By Yi Cao at Cranfield University, 04/01/2008  
% Modified by JD Liu 2010-4-20  
% Modified by zhangwenyu, 12/23/2013  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
if nargin<7  
    error('Not enough inputarguments!');  
end  
% ��ʼ����Ϊ�˼򻯺�������lamda�Ĺ��̱�Ĭ��  
L = numel(x);                                 %numer of states  
m = numel(z);                                 %numer of measurements  
alpha = 1e-3;                                 %default, tunable  
ki = abs(0);                                       %default, tunable  
beta = 2;                                     %default, tunable  
%mc
eps = 1e-7;                                    %������������۳���
sigma = 6;                                  %kernel bandwidth
% UTת������  
lambda = alpha^2*(L+ki)-L;                    %scaling factor  
c = L+lambda;                                 %scaling factor  
Wm = [lambda/c 0.5/c+zeros(1,2*L)];           %weights for means  
Wc = Wm;  
Wc(1) = Wc(1)+(1-alpha^2+beta);               %weights for covariance  
c = sqrt(c);  
X = sigmas(x,P,c);                            %sigma points around x x_hat(k-1|k-1)
[x1,X1,P1,X2] = ut(fstate,X,Wm,Wc,L,Q,k);       %unscented transformation of process  x1=x_hat(k|k-1)��UT�任��sigma��ļ�Ȩ��ֵ. X2 = X1-x1�ľ�һ��ֵ
[z1,Z1,P2,Z2] = ut(hmeas,X1,Wm,Wc,m,R,k);       %unscented transformation of measurments % P1����P(k|k-1), z1����y_hat 
r = sqrt(R)*randn(m, 1);                      % �������
% �˲�����  UTת����MCû��ϵ
% Sp = chol(P1);
% Sr = chol(R);
[l, d] = ldl(P1); Sp = l*chol(d);
[l, d] = ldl(R); Sr = l*chol(d);
S = [Sp zeros(size(Sp, 1), size(Sr, 2)); zeros(size(Sr, 1), size(Sp, 2)), Sr]; %����ƴ�Ӿ���S
P12 = X2*diag(Wc)*Z2';                        %transformed cross-covariance  % X2=X1-x1, x1�Ǽ�Ȩ��ֵ.
% MC����H, W, D
H = (P1\P12)';
W = S\[eye(size(S, 1)-size(H, 1)); H];
D = S\[x1;z-z1+H*x1];
X_hat = (W'*W)\W'*D;
EstimateRate = 1;count = 0;
while EstimateRate > eps
    [Cx, Cy] = mc(sigma, X_hat, L, m, D, W);
    Cx = Cx + eye(L)*1.0e-5;
    Cy = Cy + eye(m)*1.0e-5;
    P_hat = Sp/Cx*Sp';
    R_hat = Sr/Cy*Sr';
    
    K = P_hat*H'/(H*P_hat*H' + R_hat);  
    X_hat1 = x1 + K*(z - z1); % y�ܿ���������, ��y����z?
    EstimateRate = norm(X_hat1-X_hat) / norm(X_hat); 
    X_hat = X_hat1;
    count = count+1;
end
% disp(count);
x = X_hat1;                              %state update
K(isnan(K)==1) = 0;
P = (eye(L)-K*H)*P1*(eye(L)-K*H)' + K*R*K';                                %covariance update
disp(eig(P));
disp("P=");disp(P);
end