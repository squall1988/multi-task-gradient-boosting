%% file example_Trace.m
% this file shows the usage of Least_Trace.m function 
% and study the low-rank patterns. 
%
%% OBJECTIVE
% argmin_W { sum_i^t (0.5 * norm (Y{i} - X{i}' * W(:, i))^2) 
%            + rho1 \|W\|_*}
%  where \|W\|_* = sum(svd(W, 0)) is the trace norm
%
%% LICENSE
%   This program is free software: you can redistribute it and/or modify
%   it under the terms of the GNU General Public License as published by
%   the Free Software Foundation, either version 3 of the License, or
%   (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%   Copyright (C) 2011 - 2012 Jiayu Zhou and Jieping Ye 
%
%   You are suggested to first read the Manual.
%   For any problem, please contact with Jiayu Zhou via jiayu.zhou@asu.edu
%
%   Last modified on June 3, 2012.
%
%% Related papers
%
% [1] Ji, S. and Ye, J. An Accelerated Gradient Method for Trace Norm Minimization, ICML 2009
%


clear;
clc;
close;

addpath('../MALSAR/functions/low_rank/'); % load function
addpath('../MALSAR/utils/'); % load utilities

lambda = [1 10 100 200 500 1000 2000];

%rng('default');     % reset random generator. Available from Matlab 2011.
opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-5;   % tolerance. 
opts.maxIter = 1500; % maximum iteration number of optimization.

tn_val = zeros(length(lambda), 1);
rk_val = zeros(length(lambda), 1);
log_lam  = log(lambda);


scores = []
for i = 1:10
    % school data
    load(['./school_mat/school_re', num2str(i), '.mat'])
    [W funcVal] = Least_Trace(train_input, train_output, 100, opts);
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
fprintf(sprintf('Mean RMSE: %f\n',mean(scores)));