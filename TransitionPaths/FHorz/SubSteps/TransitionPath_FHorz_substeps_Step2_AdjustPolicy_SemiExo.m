function [Policy_dsemiexoPath,Policy_aprimePath,PolicyProbsPath,PolicyValuesPath]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy_SemiExo(PolicyIndexesPath,T,n_d,n_a,n_z,n_e,N_j,N_a,N_z,N_e,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions)
% Semi-exogenous state: extract the policy needed to iterate the agent distribution, plus the PolicyValues used for AggVars.
% Unlike the generic Step2, the semiz dist iteration needs BOTH the aprime index and the d2 (semi-exo) index,
% because the semiz transition depends on d2.
% PolicyIndexesPath is [nrows,N_a,N_bothz,N_j,T-1] (N_e==0) or [nrows,N_a,N_bothz,N_e,N_j,T-1] (N_e>0), bothz=(semiz,z).
% Outputs (per the SemiExo dist single-step raws, [N_a*N_bothze,(N_probs,)N_j,T-1]):
%   Policy_dsemiexoPath : the d2 index
%   Policy_aprimePath   : the aprime index (lower & upper points when gridinterplayer==1)
%   PolicyProbsPath     : interpolation probabilities (only when gridinterplayer==1)
%   PolicyValuesPath    : [N_a,N_j,N_bothze,l_d+l_aprime,T-1] grid values for AggVars (semiz disguised as z)

N_semiz=prod(simoptions.n_semiz);
N_bothz=N_semiz*max(N_z,1);
N_bothze=N_semiz*max(N_z,1)*max(N_e,1); % N_semiz*N_z*N_e (composite carried in the dist)

l_d=length(n_d);
l_d1=l_d-simoptions.l_dsemiz;
nrows=size(PolicyIndexesPath,1);

% Normalize PolicyIndexesPath to canonical [nrows,N_a,N_bothze,N_j,T-1] (a,bothze,j ordering).
% Storage ordering matches the generic: slowOLG is (a,bothz,[e],j); fastOLG is (a,j,bothz,[e]).
if transpathoptions.fastOLG==0
    PolicyIndexesPath=reshape(PolicyIndexesPath,[nrows,N_a,N_bothze,N_j,T-1]); % folds bothz,e -> bothze
else % fastOLG: (a,j,bothz,[e]) -> (a,bothz,[e],j) -> fold to (a,bothze,j)
    if N_e==0
        PolicyIndexesPath=permute(reshape(PolicyIndexesPath,[nrows,N_a,N_j,N_bothze,T-1]),[1,2,4,3,5]);
    else
        PolicyIndexesPath=permute(reshape(PolicyIndexesPath,[nrows,N_a,N_j,N_bothz,N_e,T-1]),[1,2,4,5,3,6]);
        PolicyIndexesPath=reshape(PolicyIndexesPath,[nrows,N_a,N_bothze,N_j,T-1]);
    end
end

% PolicyValues for AggVars: convert the normalized policy indexes to grid values, treating bothze as a single exo state
% (semiz disguised as z). Mirrors the generic Step2 slowOLG construction. The normalized layout is always (a,bothze,j),
% so we always call with fastOLG=0 (regardless of transpathoptions.fastOLG).
PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,N_bothze,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); % [N_a,N_j,N_bothze,l_d+l_aprime,T-1] (fastOLG ordering needed for AggVars)

% aprime index (single endogenous state)
Policy_aprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,:),[N_a*N_bothze,N_j,T-1]);
% d2 index (the l_dsemiz decision variables, immediately after the l_d1 standard ones)
if simoptions.l_dsemiz==1
    Policy_dsemiexoPath=reshape(PolicyIndexesPath(l_d1+1,:,:,:,:),[N_a*N_bothze,N_j,T-1]);
elseif simoptions.l_dsemiz==2
    Policy_dsemiexoPath=reshape(PolicyIndexesPath(l_d1+1,:,:,:,:)+n_d(l_d1+1)*(PolicyIndexesPath(l_d1+2,:,:,:,:)-1),[N_a*N_bothze,N_j,T-1]);
else
    error('simoptions.l_dsemiz>2 not yet implemented for transition paths with semiz')
end

PolicyProbsPath=[];
if vfoptions.gridinterplayer==1
    % Build lower/upper aprime points and the interpolation probabilities (from the L2 index; last row is L2flag, second-last is L2 index)
    aprimeProbs_upper=reshape((PolicyIndexesPath(end-1,:,:,:,:)-1)/(vfoptions.ngridinterp+1),[N_a*N_bothze,1,N_j,T-1]); % prob of upper grid point
    Policy_aprimePath=reshape(Policy_aprimePath,[N_a*N_bothze,1,N_j,T-1]);
    Policy_aprimePath=repmat(Policy_aprimePath,1,2,1,1);
    Policy_aprimePath(:,2,:,:)=Policy_aprimePath(:,2,:,:)+1; % upper grid point
    PolicyProbsPath=ones([N_a*N_bothze,2,N_j,T-1],'gpuArray');
    PolicyProbsPath(:,1,:,:)=1-aprimeProbs_upper; % lower
    PolicyProbsPath(:,2,:,:)=aprimeProbs_upper; % upper
end

end
