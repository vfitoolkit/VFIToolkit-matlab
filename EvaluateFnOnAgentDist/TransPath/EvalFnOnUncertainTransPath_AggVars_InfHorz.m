function AggVarsTree=EvalFnOnUncertainTransPath_AggVars_InfHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePath, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z, d_grid,a_grid,z_grid, transpathoptions, simoptions)
% The aggregate variables on every node of the tree of an uncertain transition path, given the agent distribution and
% policy on the tree. See UncertainTransPathTree for the aggregate shock S and the tree.
% Each node is evaluated as its own economy (a node's distribution is that of the economy on its history), one segment
% (a chain of single-child nodes) at a time with EvalFnOnTransPath_AggVars_InfHorz.
%
% AggVarsTree.(name).Mean is 1-by-nNodes. The aggregate of history h in period t is
% AggVarsTree.(name).Mean(Tree.histories(h,t)), where Tree=UncertainTransPathTree(T,T1,T2,pi_S,S_initial).

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~exist('simoptions','var')
    simoptions=struct();
end
% A _tminus1 or _tplus1 input would need the value from the parent node, or the expectation over the children; not
% implemented for uncertain transition paths
FnNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnNames)
    tempnames=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{ff}));
    for ii=1:length(tempnames)
        if endsWith(tempnames{ii},'_tminus1') || endsWith(tempnames{ii},'_tplus1')
            error('EvalFnOnUncertainTransPath_AggVars_InfHorz: FnsToEvaluate.%s uses %s; _tminus1 and _tplus1 inputs are not implemented for uncertain transition paths',FnNames{ff},tempnames{ii})
        end
    end
end

%% The tree, and the prices and parameters on it
[Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions);
PricePathNames=fieldnames(PricePathNodes);
ParamPathNames=fieldnames(ParamPathNodes);
nNodes=Tree.nNodes;

Policyshape=size(PolicyPathTree);
Distshape=size(AgentDistPathTree);
if Policyshape(end)~=nNodes || Distshape(end)~=nNodes
    error('EvalFnOnUncertainTransPath_AggVars_InfHorz: the last dimension of AgentDistPathTree and PolicyPathTree must be the %i nodes of the tree',nNodes)
end
PolicyPathTree=reshape(PolicyPathTree,[prod(Policyshape(1:end-1)),nNodes]);
AgentDistPathTree=reshape(AgentDistPathTree,[prod(Distshape(1:end-1)),nNodes]);

for ff=1:length(FnNames)
    AggVarsTree.(FnNames{ff}).Mean=zeros(1,nNodes);
end

for ss=1:Tree.nSeg
    segnodes=Tree.segnodes{ss};
    L=length(segnodes);
    endnode=segnodes(end);
    if Tree.nodenchild(endnode)==0
        Tseg=L;
        segnodes_ext=segnodes;
    else
        % A segment ending at a branching node can be a single node, and a path of one period is not a transition path
        % to EvalFnOnTransPath_AggVars_InfHorz, so it is given one extra period (repeating the last node) and that is dropped
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
    AgentDistPath_seg=reshape(AgentDistPathTree(:,segnodes_ext),[Distshape(1:end-1),Tseg]);

    AggVarsPath_seg=EvalFnOnTransPath_AggVars_InfHorz(FnsToEvaluate, gpuArray(AgentDistPath_seg), gpuArray(PolicyPath_seg), PricePath_seg, ParamPath_seg, Parameters, Tseg, n_d, n_a, n_z, d_grid, a_grid, z_grid, transpathoptions, simoptions);

    for ff=1:length(FnNames)
        temp=gather(AggVarsPath_seg.(FnNames{ff}).Mean);
        AggVarsTree.(FnNames{ff}).Mean(segnodes)=temp(1:L);
    end
end

end
