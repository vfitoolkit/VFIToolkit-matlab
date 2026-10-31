function AgentDistPathTree=AgentDistOnUncertainTransPath_InfHorz(AgentDist_initial, PricePath, ParamPath, PolicyPathTree, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,pi_z, Parameters, transpathoptions, simoptions)
% The agent distribution on every node of the tree of an uncertain transition path, given the policy on the tree
% (from ValueFnOnUncertainTransPath_InfHorz). See UncertainTransPathTree for the aggregate shock S and the tree.
%
% AgentDist_initial is the distribution in period 1, before any S is drawn after T1 (or before S_1 is drawn, if T1=1).
% Each history carries the whole distribution of agents: a node's distribution is that of the economy on that history,
% not a probability-weighted share of it. The policy chosen at a node is chosen before next period's S is known, so
% every child of a node starts from the same distribution, the node's distribution moved on one period by its policy.
% The tree is done one segment (a chain of single-child nodes) at a time with AgentDistOnTransPath_InfHorz.
%
% AgentDistPathTree is the same shape as the output of AgentDistOnTransPath_InfHorz, with the last dimension being
% the nodes of the tree instead of the periods.

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~exist('simoptions','var')
    simoptions=struct();
end
if ndims(pi_z)==3
    error('AgentDistOnUncertainTransPath_InfHorz: a pi_z that varies over the path is not implemented for uncertain transition paths')
end
if isfield(simoptions,'pi_e') && size(simoptions.pi_e,2)>1
    error('AgentDistOnUncertainTransPath_InfHorz: an e (simoptions.pi_e) that varies over the path is not implemented for uncertain transition paths')
end

%% The tree, and the prices and parameters on it
[Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions);
PricePathNames=fieldnames(PricePathNodes);
ParamPathNames=fieldnames(ParamPathNodes);
nNodes=Tree.nNodes;

Policyshape=size(PolicyPathTree);
if Policyshape(end)~=nNodes
    error('AgentDistOnUncertainTransPath_InfHorz: the last dimension of PolicyPathTree must be the %i nodes of the tree',nNodes)
end
NPolicy=prod(Policyshape(1:end-1));
PolicyPathTree=reshape(PolicyPathTree,[NPolicy,nNodes]);
NDist=numel(AgentDist_initial);

AgentDistPathTree=zeros(NDist,nNodes);
AgentDistNext=zeros(NDist,Tree.nSeg); % for each segment ending at a branching node, the distribution its children start from
Distpershape=[];

%% Forward over the segments: a parent segment always has a smaller number than its children
for ss=1:Tree.nSeg
    segnodes=Tree.segnodes{ss};
    L=length(segnodes);
    endnode=segnodes(end);
    if Tree.segparent(ss)==0
        AgentDist_seg=AgentDist_initial;
    else
        AgentDist_seg=reshape(AgentDistNext(:,Tree.nodeseg(Tree.segparent(ss))),size(AgentDist_initial));
    end
    if Tree.nodenchild(endnode)==0
        Tseg=L;
        segnodes_ext=segnodes;
    else
        % One extra period, to get the distribution the children start from. The policy in that extra period is never
        % used (the last period of AgentDistOnTransPath_InfHorz is not moved on), so it repeats the last node.
        Tseg=L+1;
        segnodes_ext=[segnodes,endnode];
    end
    PricePath_seg=struct();
    for pp=1:length(PricePathNames)
        PricePath_seg.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})(segnodes_ext);
    end
    ParamPath_seg=struct();
    for pp=1:length(ParamPathNames)
        ParamPath_seg.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(segnodes_ext);
    end
    PolicyPath_seg=reshape(PolicyPathTree(:,segnodes_ext),[Policyshape(1:end-1),Tseg]);

    AgentDistPath_seg=AgentDistOnTransPath_InfHorz(AgentDist_seg, PricePath_seg, ParamPath_seg, PolicyPath_seg, n_d,n_a,n_z,pi_z,Tseg,Parameters,transpathoptions,simoptions);

    if isempty(Distpershape)
        temp=size(AgentDistPath_seg);
        Distpershape=temp(1:end-1);
    end
    AgentDistPath_seg=gather(reshape(AgentDistPath_seg,[NDist,Tseg]));
    AgentDistPathTree(:,segnodes)=AgentDistPath_seg(:,1:L);
    if Tree.nodenchild(endnode)>0
        AgentDistNext(:,ss)=AgentDistPath_seg(:,L+1);
    end
end

AgentDistPathTree=reshape(AgentDistPathTree,[Distpershape,nNodes]);

end
