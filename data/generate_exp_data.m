clc
clear
load('./sarcos_inv.mat')

result = cell(7, 1);
for i =22:28
    result{i-21} = [sarcos_inv(:, 1:21), sarcos_inv(:, i), ones(size(sarcos_inv, 1), 1)*(i-21)];
end

result = cell2mat(result);
