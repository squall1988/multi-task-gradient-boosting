clc
clear
opts.max_iter =500;
opts.max_iter_fista =300;
opts.max_iter_ADMM=50;
opts.rel_tol  =10^-3;
opts.rel_tol_fista=10^-3;
opts.tol_ADMM=10^-2;
opts.rho=2;

% VSTG_MTL requires MALSAR by Jiayu, et al., 2011
% http://jiayuzhou.github.io/MALSAR/
addpath(genpath('MALSAR'))
%% regression: school dataset

K=9;
hyp = [2,8,2,3];

scores = []
for i = 1:10
    % data
    load(['./school_mat/school_re', num2str(i), '.mat'])
    [U,V,fun] = VSTG_MTL_regress(train_input,train_output,K,hyp,opts);
    W = U*V;
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
