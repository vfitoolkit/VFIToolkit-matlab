function [Policy_dsemizPath,Policy_aprimePath,PolicyProbsPath,PolicyValuesPath]=TransitionPath_InfHorz_substeps_Step2_AdjustPolicy_SemiExo(PolicyIndexesPath,T,n_d,n_a,N_a,N_z,N_e,d_gridvals,aprime_gridvals,vfoptions,simoptions)
% Semi-exogenous state: extract the policy needed to iterate the agent distribution, plus the PolicyValues used for AggVars.
% Unlike the generic Step2, the semiz dist iteration needs BOTH the aprime index and the d2 (semi-exo) index, because
% the semiz transition depends on d2.
% PolicyIndexesPath is [l_d+l_a(+2),N_a,N_bothz,T-1] (N_e==0) or [l_d+l_a(+2),N_a,N_bothz,N_e,T-1] (N_e>0), bothz=(semiz,z),
% with d1 rows, then d2 rows, then aprime (then L2 index and L2flag with the grid interpolation layer).
% Outputs, over the composite state (a,semiz,z,e) [semiz fastest; N_bothze=N_semiz*max(N_z,1)*max(N_e,1)]:
%   Policy_dsemizPath : [N_a*N_bothze,T-1], the joint d2 index (cpu)
%   Policy_aprimePath : [N_a*N_bothze,T-1] the aprime index; with the grid interpolation layer [N_a*N_bothze,2,T-1],
%                       the lower and upper grid points (cpu)
%   PolicyProbsPath   : [N_a*N_bothze,2,T-1] the probabilities of the lower and upper grid points with the grid
%                       interpolation layer (cpu), [] otherwise
%   PolicyValuesPath  : [N_a,N_bothze,l_d+l_a,T-1] grid values for AggVars (semiz disguised as z; gpu)
% Uses vfoptions.gridinterplayer and vfoptions.ngridinterp, as those are what produced PolicyIndexesPath.

if ~isfield(simoptions,'n_semiz')
    simoptions.n_semiz=0;
end
if ~isfield(simoptions,'l_dsemiz')
    simoptions.l_dsemiz=1; % by default, only one decision variable influences the semi-exogenous state
end

N_semiz=prod(simoptions.n_semiz);
N_bothze=N_semiz*max(N_z,1)*max(N_e,1);

l_d=length(n_d);
l_d2=simoptions.l_dsemiz;
l_d1=l_d-l_d2;
n_d2=n_d(end-l_d2+1:end);
nrows=size(PolicyIndexesPath,1);

% Fold (bothz,e) into a single composite exogenous dimension
PolicyIndexesPath=reshape(PolicyIndexesPath,[nrows,N_a,N_bothze,T-1]);

% PolicyValues for AggVars: semiz (and z, e) disguised as a single exogenous state
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,N_bothze,T-1,d_gridvals,aprime_gridvals,vfoptions,1);
PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_bothze,T-1]),[2,3,1,4]); % [N_a,N_bothze,l_d+l_a,T-1]

% aprime index (single endogenous state)
Policy_aprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:),[N_a*N_bothze,T-1]);
% d2 index, joint over the l_dsemiz decision variables (immediately after the l_d1 standard ones)
Policy_dsemizPath=reshape(PolicyIndexesPath(l_d1+1,:,:,:),[N_a*N_bothze,T-1]);
for ii=2:l_d2
    Policy_dsemizPath=Policy_dsemizPath+prod(n_d2(1:ii-1))*(reshape(PolicyIndexesPath(l_d1+ii,:,:,:),[N_a*N_bothze,T-1])-1);
end

PolicyProbsPath=[];
if vfoptions.gridinterplayer==1
    % Lower/upper aprime points and their probabilities, from the L2 index (second-last row) and L2flag (last row)
    L2index=reshape(PolicyIndexesPath(end-1,:,:,:),[N_a*N_bothze,1,T-1]);
    L2flag=reshape(PolicyIndexesPath(end,:,:,:),[N_a*N_bothze,1,T-1]); % 1=force lower, 2=usual, 3=force upper
    L2index(L2flag==1)=1;                       % force all weight to lower grid point
    L2index(L2flag==3)=vfoptions.ngridinterp+2; % force all weight to upper grid point
    PolicyProbsPath=zeros(N_a*N_bothze,2,T-1,'gpuArray');
    PolicyProbsPath(:,2,:)=(L2index-1)/(vfoptions.ngridinterp+1); % probability of upper grid point
    PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
    Policy_aprimePath=reshape(Policy_aprimePath,[N_a*N_bothze,1,T-1]); % reinterpret as lower grid index
    Policy_aprimePath=repelem(Policy_aprimePath,1,2,1);
    Policy_aprimePath(:,2,:)=Policy_aprimePath(:,2,:)+1; % upper grid index
    PolicyProbsPath=gather(PolicyProbsPath);
end

% Agent distribution iteration is performed on cpu
Policy_aprimePath=gather(Policy_aprimePath);
Policy_dsemizPath=gather(Policy_dsemizPath);

end
