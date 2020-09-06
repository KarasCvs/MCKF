%n=3; %number of state  
clc;  
clear;  
n=4;  
t=0.2;  
q=1; %std of process  
r=0.8; %std of measurement 
rng(2);
Q=q^2*eye(n); % covariance of process  
R=r^2*eye(n); % covariance of measurement  
f=@(x, k)real([x(1)+t*x(3);sin(x(2)+t*x(4));x(3)+t*x(2);x(4)+t*x(1)]); % 4d nonlinear state equations  
h=@(x, k)real([sqrt(x(1)+1);0.8*x(2)+0.3*x(1);x(3);x(4)]);  
% f=@(x, k)0.5*x+2*cos(1.2*k)+25*(x/(1+x^2)); % 1d nonlinear state equations  
% h=@(x, k)x^2/20;  
% measurement equation  
s=[0.3;0.2;1;2];  % initial state
% s = 0.1;
x=s+q*randn(n,1); %initial state % initial state with noise  
x_u = x;
x_mc = x;
P_mc = eye(n); P_u = eye(n); % initial state covraiance  
N=20; % total dynamic steps  
mcxV = zeros(n,N); %mcukf estmate % allocate memory  
uxV = zeros(n, N); %ukf
sV = zeros(n,N); %actual  
zV = zeros(n,N);  

for k=1:N  
    disp(k);
    z = h(s, k) + r*randn(n, 1)+20*randn(n, 1); % 1d measurments
%     z = h(s) + r*randn(n, 1); orignal
    sV(:,k)= s; % save actual state  
    zV(:,k) = z; % save measurment  
    [x_mc, P_mc] = mcukf(f,x_mc,P_mc,h,z,Q,R,k); % mcukf  
    mcxV(:,k) = x_mc; % save estimate
    [x_u, P_u] = ukf(f,x_u,P_u,h,z,Q,R);
    uxV(:,k) = x_u; % save estimate
    s = f(s, k) + q*randn(n,1); % 1d update process  
%     s = f(s) + q*randn(n,1); % orignal update process 
end
% MSE
% uMSE = 0; mcMSE = 0;
% for k=1:N
%     uMSE = uMSE + (sV(:, k) - uxV(:, k)).^2;
%     mcMSE = mcMSE + (sV(:, k) - mcxV(:, k)).^2;
% end
uMSE = mean((zV - uxV).^2, 2);
mcMSE = mean((zV - mcxV).^2, 2);

fprintf("uMSE=%.2f,\n", uMSE);
fprintf("mcMSE=%.2f\n",mcMSE);
for k=1:n % plot results  
    subplot(n,1,k)  
    plot(1:N, sV(k,:), '-', 1:N, mcxV(k,:), '--', 1:N, uxV(k,:), '-.', 1:N,zV(k,:),'*');
%     plot(1:N, sV(k,:), '-', 1:N, uxV(k,:), '--',1:N,zV(k,:),'*')
%     plot(1:N, sV(k,:), '-', 1:N, mcxV(k,:), '--',1:N,zV(k,:),'*')
    legend("Real", "MCUKF", "UKF", "Measurement ");
end  