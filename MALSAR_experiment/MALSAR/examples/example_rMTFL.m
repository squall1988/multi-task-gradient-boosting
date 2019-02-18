%% file example_rMTFL.m
% this file shows the usage of Least_rMTFL.m function 
% and study how to detect outlier tasks. 
%
%% OBJECTIVE
%  argmin_W ||X(P+Q) - Y||_F^2 + lambda1*||P||_{1,2} + lambda2*||Q^T||_{1,2}
%   s.t. W = P + Q
%
%% Copyright (C) 2012 Jiayu Zhou, and Jieping Ye
%
% You are suggested to first read the Manual.
% For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
% Last modified on April 16, 2012.
%
%% Related papers
%
% [1] Gong, P. and Ye, J. and Zhang, C. Robust Multi-Task Feature Learning,
% Submitted, 2012
%

clear;
clc;
close all;

addpath('../MALSAR/functions/rMTFL/'); % load function 
addpath('../MALSAR/utils/'); % load utilities

%rng('default');     % reset random generator. Available from Matlab 2011.

scores = []
for i = 1:10
    % school data
    load(['../data/school_mat/school_re', num2str(i), '.mat'])

    opts.init = 0;      % guess start point from data. 
    opts.tFlag = 1;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-6;   % tolerance. 
    opts.maxIter = 500; % maximum iteration number of optimization.

    rho_1 = 90;%   rho1: P
    rho_2 = 280; %   rho2: Q

    [W funcVal P Q] = Least_rMTFL(train_input, train_output, rho_1, rho_2, opts);
    
    for task=1:139
        test_output_hat{task} = test_input{task} * W(:,task);
        resi{task} = test_output{task} - test_output_hat{task};
        RMSE(task) = sqrt(mean(resi{task}.^2));
        %NRMSE(task) = sqrt(mean(resi{task}.^2))/(max(resi{task})-min(resi{task}));
    end
    fprintf(sprintf('RMSE: %f\n',mean(RMSE)));
    %fprintf(sprintf('NRMSE: %f\n',mean(NRMSE)));
    scores = [scores, mean(RMSE)]
    %scores = [scores, mean(NRMSE)]
end
fprintf(sprintf('Mean RMSE: %f +/- %f\n',mean(scores),var(scores)));
