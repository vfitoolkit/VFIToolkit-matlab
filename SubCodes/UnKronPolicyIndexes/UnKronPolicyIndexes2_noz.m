function Policy=UnKronPolicyIndexes2_noz(PolicyKron, n_daprime1, n_daprime2, n_a, vfoptions)
% For models with no z (and no e, no semiz), two daprime dimensions.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(2,N_a);
%        PolicyKron(1,:) indexes the optimal choice for daprime1
%        PolicyKron(2,:) indexes the optimal choice for daprime2
%        If vfoptions.gridinterplayer==1, PolicyKron is (4,N_a): rows 1-2 are the
%        Kron indices, rows 3 and 4 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1+l_daprime2,n_a);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+l_daprime2+2,n_a).
% Handy trick: You can pass N_a in place of n_a to skip unpacking that dimension.

l_daprime1=length(n_daprime1);
l_daprime2=length(n_daprime2);

divisors1=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]
divisors2=cumprod([1,n_daprime2(1:end-1)])';   % [l_daprime2,1]

if l_daprime1==1 && l_daprime2==1
    % Fast path: every decision dimension is single, so the mixed-radix un-flatten
    % is the identity and PolicyKron already holds the un-Kron'd policy indices (the
    % grid-interp-layer rows pass through). A bare reshape (shares data, no vertcat,
    % no mod/floor temporaries) reproduces the output below exactly.
    if vfoptions.gridinterplayer==1
        Policy=reshape(PolicyKron,[l_daprime1+l_daprime2+2,n_a]);
    else
        Policy=reshape(PolicyKron,[l_daprime1+l_daprime2,n_a]);
    end
elseif vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            PolicyKron(3,:);
            PolicyKron(4,:)];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+2,n_a]);
else
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1];
    Policy=reshape(Policy,[l_daprime1+l_daprime2,n_a]);
end

end
