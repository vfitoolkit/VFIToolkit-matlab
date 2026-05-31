function Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_daprime1, n_daprime2, n_daprime3, n_daprime4, n_daprime5, n_a, n_z, N_j, vfoptions)
% For models with z, five daprime dimensions, finite horizon.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(5,N_a,N_z,N_j);
%        PolicyKron(k,:,:,:) indexes the optimal choice for daprime k (k=1,2,3,4,5)
%        If vfoptions.gridinterplayer==1, PolicyKron is (7,N_a,N_z,N_j): rows 1-5 are the
%        Kron indices, rows 6 and 7 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1+l_daprime2+l_daprime3+l_daprime4+l_daprime5,n_a,n_z,N_j);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+l_daprime2+l_daprime3+l_daprime4+l_daprime5+2,n_a,n_z,N_j).
% Handy trick: You can pass N_a in place of n_a, or N_z in place of n_z, to skip unpacking that dimension.

l_daprime1=length(n_daprime1);
l_daprime2=length(n_daprime2);
l_daprime3=length(n_daprime3);
l_daprime4=length(n_daprime4);
l_daprime5=length(n_daprime5);

divisors1=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]
divisors2=cumprod([1,n_daprime2(1:end-1)])';   % [l_daprime2,1]
divisors3=cumprod([1,n_daprime3(1:end-1)])';   % [l_daprime3,1]
divisors4=cumprod([1,n_daprime4(1:end-1)])';   % [l_daprime4,1]
divisors5=cumprod([1,n_daprime5(1:end-1)])';   % [l_daprime5,1]

if vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            mod(floor((PolicyKron(3,:)-1)./divisors3),n_daprime3(:))+1;
            mod(floor((PolicyKron(4,:)-1)./divisors4),n_daprime4(:))+1;
            mod(floor((PolicyKron(5,:)-1)./divisors5),n_daprime5(:))+1;
            PolicyKron(6,:);
            PolicyKron(7,:)];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+l_daprime3+l_daprime4+l_daprime5+2,n_a,n_z,N_j]);
else
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            mod(floor((PolicyKron(3,:)-1)./divisors3),n_daprime3(:))+1;
            mod(floor((PolicyKron(4,:)-1)./divisors4),n_daprime4(:))+1;
            mod(floor((PolicyKron(5,:)-1)./divisors5),n_daprime5(:))+1];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+l_daprime3+l_daprime4+l_daprime5,n_a,n_z,N_j]);
end

end
