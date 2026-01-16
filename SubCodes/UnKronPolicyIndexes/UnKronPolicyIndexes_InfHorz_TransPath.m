function PolicyPath=UnKronPolicyIndexes_InfHorz_TransPath(PolicyPathKron, n_d, n_a, n_z,T,vfoptions,justfirstdim)
% Can use vfoptions OR simoptions
% Input: PolicyKron is (2,N_a,N_z,T) first dim indexes the optimal choice for d and aprime
%                      (1,N_a,N_z,T) if there is no d
%    vfoptions.gridinterplayer=1 will mean the first dimension has one extra value (so 3 if d, 2 without)
% Output: Policy is (l_d+l_a,n_a,n_z,T);

% Really this is just the same things as when T is N_j, so just redirect rather than duplicate
PolicyPath=UnKronPolicyIndexes_Case1_FHorz(PolicyPathKron,n_d,n_a,n_z,T,vfoptions);
if justfirstdim==1
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);
end

end