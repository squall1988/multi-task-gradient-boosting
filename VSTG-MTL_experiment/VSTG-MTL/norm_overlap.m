% copyright 2012 Andreas Argyriou
% GPL License http://www.gnu.org/copyleft/gpl.html

function [ normw , alpha ] = norm_overlap( w, k )

% Compute k overlap norm
% alpha is a subgradient

d = length(w);
% d
% size(w)
[beta, ind] = sort(abs(w), 'descend');

s = sum(beta(k:d));
temp = s;
found = false;
% size(beta)
% k=2;
for r=0:k-2
    r
    k-2
    beta(k-r-1)
    beta
    beta(k-r)
  if ( (temp >= (r+1)*beta(k-r)) && (temp < (r+1)*beta(k-r-1)) )
    found = true;
    break;
  else
    temp = temp + beta(k-r-1);
  end
end
if (~found)
  r=k-1;
end

alpha(1:k-r-1) = beta(1:k-r-1);
alpha(k-r:d) = temp / (r+1);
alpha = alpha';
[dummy,rev]=sort(ind,'ascend');
alpha = sign(w) .* alpha(rev);

if k-r-1==0
    normw = sqrt(temp^2/(r+1));
else
    normw = sqrt( beta(1:k-r-1)'*beta(1:k-r-1) + temp^2/(r+1) );
end

end
