function AgentDistPathTree=AgentDistOnUncertainTransPath_FHorz(AgentDist_initial, jequaloneDist, PricePath, ParamPath, PolicyPathTree, AgeWeightsParamNames, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,N_j,pi_z, Parameters, transpathoptions, simoptions)
% The agent distribution on every node of the tree of an uncertain transition path of a finite horizon (OLG) model,
% given the policy on the tree (from ValueFnOnUncertainTransPath_FHorz). The FHorz counterpart of
% AgentDistOnUncertainTransPath_InfHorz. See UncertainTransPathTree for the aggregate shock S and the tree, and
% ValueFnOnUncertainTransPath_FHorz for the forms of PricePath and ParamPath.
%
% AgentDist_initial is the distribution in period 1 (where S is always S_initial), [n_a,n_z,N_j] as for
% AgentDistOnTransPath_Case1_FHorz.
% jequaloneDist, the distribution of the newborns, can be [n_a,n_z] (the same in every period), a path [n_a,n_z,T], or
% given per S as jequaloneDist.S1, .S2, ..., each of those forms (newborns that depend on S). The age weights
% (AgeWeightsParamNames) are a parameter as usual, and so can be on the ParamPath, per S when they depend on S.
%
% Each history carries the whole distribution of agents: a node's distribution is that of the economy on that history,
% not a probability-weighted share of it. The policy chosen at a node is chosen before next period's S is known, so
% every age j>=2 of every child of a node is the node's distribution moved on one period by its policy. The newborns
% and the age weights are those of the CHILD's period and S, so when they depend on S the children of one node start
% from different distributions: each child's start is computed from its parent with a 2-period
% AgentDistOnTransPath_Case1_FHorz (the parent's node then the child's). The tree is then done one segment (a chain of
% single-child nodes) at a time with AgentDistOnTransPath_Case1_FHorz.
%
% AgentDistPathTree is the same shape as the output of AgentDistOnTransPath_Case1_FHorz, with the last dimension being
% the nodes of the tree instead of the periods: [n_a,n_z,N_j,nNodes].

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~exist('simoptions','var')
    simoptions=struct();
end
if isfield(simoptions,'n_semiz') && prod(simoptions.n_semiz)>0
    error('AgentDistOnUncertainTransPath_FHorz: semi-exogenous states (simoptions.n_semiz) are not yet implemented for uncertain transition paths')
end
if (isfield(simoptions,'experienceasset') && simoptions.experienceasset>=1) || (isfield(simoptions,'experienceassetz') && simoptions.experienceassetz>=1)
    error('AgentDistOnUncertainTransPath_FHorz: experience assets are not yet implemented for uncertain transition paths')
end

%% The tree, and the prices and parameters on it (parameters can be vectors, e.g. age-dependent: k-by-nNodes)
transpathoptions.uncertaintreevectorparams=1;
[Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions);
PricePathNames=fieldnames(PricePathNodes);
ParamPathNames=fieldnames(ParamPathNodes);
nNodes=Tree.nNodes;

Policyshape=size(PolicyPathTree);
if Policyshape(end)~=nNodes
    error('AgentDistOnUncertainTransPath_FHorz: the last dimension of PolicyPathTree must be the %i nodes of the tree',nNodes)
end
NPolicy=prod(Policyshape(1:end-1));
PolicyPathTree=reshape(PolicyPathTree,[NPolicy,nNodes]);
Distshape=size(AgentDist_initial);
NDist=numel(AgentDist_initial);

%% The newborns on the tree, jeqNodes [NJ1,nNodes], always handed on as an explicit path
% AgentDistOnTransPath_Case1_FHorz reads a jequaloneDist whose last dimension equals T as a path, so a plain [n_a,n_z]
% would be misread whenever a segment happens to be n_z periods long; an explicit [NJ1,Tseg] path never is.
N_e=0;
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
end
NJ1=prod(n_a)*max(prod(n_z),1)*max(N_e,1);
jeqNodes=zeros(NJ1,nNodes);
if isstruct(jequaloneDist)
    for SS=unique(Tree.nodeS)'
        if ~isfield(jequaloneDist,['S',num2str(SS)])
            error('AgentDistOnUncertainTransPath_FHorz: jequaloneDist is given per S, but has no field S%i (some node is in S=%i)',SS,SS)
        end
        temp=gather(jequaloneDist.(['S',num2str(SS)]));
        nodes_S=find(Tree.nodeS==SS);
        if numel(temp)==NJ1
            jeqNodes(:,nodes_S)=repmat(reshape(temp,[NJ1,1]),1,length(nodes_S));
        elseif numel(temp)==NJ1*T
            temp=reshape(temp,[NJ1,T]);
            jeqNodes(:,nodes_S)=temp(:,Tree.nodedate(nodes_S));
        else
            error('AgentDistOnUncertainTransPath_FHorz: jequaloneDist.S%i must be [n_a,n_z] (the same every period) or [n_a,n_z,T]',SS)
        end
    end
else
    temp=gather(jequaloneDist);
    if numel(temp)==NJ1
        jeqNodes=repmat(reshape(temp,[NJ1,1]),1,nNodes);
    elseif numel(temp)==NJ1*T
        temp=reshape(temp,[NJ1,T]);
        jeqNodes=temp(:,Tree.nodedate);
    else
        error('AgentDistOnUncertainTransPath_FHorz: jequaloneDist must be [n_a,n_z] (the same every period), [n_a,n_z,T], or given per S as jequaloneDist.S1, .S2, ...')
    end
end

AgentDistPathTree=zeros(NDist,nNodes);

%% A segment N_j periods long makes an age-dependent parameter's segment path square, which
% PricePathParamPath_FHorz_StructToMatrix reads as N_j-by-T with a warning; here it is built that way on purpose, so that
% warning is turned off for the calls on the segments, and back on (as it was) when this command returns
warningstate_square=warning('off','VFIToolkit:SquarePathEntry');
cleanup_square=onCleanup(@() warning(warningstate_square));

%% Forward over the segments: a parent segment always has a smaller number than its children
for ss=1:Tree.nSeg
    segnodes=Tree.segnodes{ss};
    L=length(segnodes);
    parentnode=Tree.segparent(ss);
    if parentnode==0
        AgentDist_seg=reshape(gather(AgentDist_initial),[NDist,1]);
    else
        % The start of a child of a branching node: the parent's distribution moved on one period by the parent's policy,
        % with the newborns and age weights of the child (a 2-period path, the parent's node then the child's; the policy
        % of the second period is never used, so it repeats the parent's)
        twonodes=[parentnode,segnodes(1)];
        PricePath_2=struct();
        for pp=1:length(PricePathNames)
            PricePath_2.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})(twonodes);
        end
        ParamPath_2=struct();
        for pp=1:length(ParamPathNames)
            ParamPath_2.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,twonodes);
        end
        PolicyPath_2=gpuArray(reshape(PolicyPathTree(:,[parentnode,parentnode]),[Policyshape(1:end-1),2]));
        AgentDistPath_2=AgentDistOnTransPath_Case1_FHorz(reshape(AgentDistPathTree(:,parentnode),Distshape), reshape(jeqNodes(:,twonodes),[NJ1,2]), PricePath_2, ParamPath_2, PolicyPath_2, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, 2,Parameters, transpathoptions, simoptions);
        AgentDistPath_2=gather(reshape(AgentDistPath_2,[NDist,2]));
        AgentDist_seg=AgentDistPath_2(:,2);
    end
    if L==1
        % A single node (it is a branching node): its distribution is the start
        AgentDistPathTree(:,segnodes)=AgentDist_seg;
    else
        PricePath_seg=struct();
        for pp=1:length(PricePathNames)
            PricePath_seg.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})(segnodes);
        end
        ParamPath_seg=struct();
        for pp=1:length(ParamPathNames)
            ParamPath_seg.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,segnodes); % k-by-L, k=1 or N_j
        end
        PolicyPath_seg=gpuArray(reshape(PolicyPathTree(:,segnodes),[Policyshape(1:end-1),L]));
        AgentDistPath_seg=AgentDistOnTransPath_Case1_FHorz(reshape(AgentDist_seg,Distshape), reshape(jeqNodes(:,segnodes),[NJ1,L]), PricePath_seg, ParamPath_seg, PolicyPath_seg, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, L,Parameters, transpathoptions, simoptions);
        AgentDistPathTree(:,segnodes)=gather(reshape(AgentDistPath_seg,[NDist,L]));
    end
end

%% The same shape as AgentDistOnTransPath_Case1_FHorz, with the nodes in place of the periods
AgentDistPathTree=reshape(AgentDistPathTree,[Distshape,nNodes]);

end
