function [AggVarsTree,AggVarsGivenChildTree]=EvalFnOnUncertainTransPath_AggVars_FHorz(FnsToEvaluate, AgentDistPathTree, PolicyPathTree, PricePath, ParamPath, Parameters, T, T1, T2, pi_S, S_initial, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, transpathoptions, simoptions)
% The aggregate variables on every node of the tree of an uncertain transition path of a finite horizon (OLG) model,
% given the agent distribution and policy on the tree (from AgentDistOnUncertainTransPath_FHorz and
% ValueFnOnUncertainTransPath_FHorz). The FHorz counterpart of EvalFnOnUncertainTransPath_AggVars_InfHorz. See
% UncertainTransPathTree for the aggregate shock S and the tree, and ValueFnOnUncertainTransPath_FHorz for the forms of
% PricePath and ParamPath (parameters can depend on age, k-by-nNodes on the tree).
% Each node is evaluated as its own economy (a node's distribution is that of the economy on its history), one segment
% (a chain of single-child nodes) at a time with EvalFnOnTransPath_AggVars_Case1_FHorz.
%
% AggVarsTree.(name).Mean is 1-by-nNodes. The aggregate of history h in period t is
% AggVarsTree.(name).Mean(Tree.histories(h,t)), where Tree=UncertainTransPathTree(T,T1,T2,pi_S,S_initial).
%
% _tminus1 and _tplus1 inputs (as EvalFnOnTransPath_AggVars_Case1_FHorz: _tplus1 of prices, _tminus1 of prices, parameters
% and aggregates, with transpathoptions.initialvalues for period 1):
%   _tminus1 is the value at the parent node.
%   _tplus1 at a node with one child is the child's price, and at a period T node the node's own (the final eqm). At a
%     branching node it is evaluated once for each child c, with c's price; AggVarsGivenChildTree.(name)(c) is the
%     aggregate at the parent of c given that the next node is c, and AggVarsTree.(name).Mean at the branching node is
%     the expectation, sum_c Tree.nodecondprob(c)*AggVarsGivenChildTree.(name)(c).
%   AggVarsGivenChildTree.(name) is 1-by-nNodes, for each FnsToEvaluate that uses _tplus1 (NaN at the period 1 node,
%   which has no parent); below a node with one child it is that node's Mean. It is struct() if none uses _tplus1.
%   Not implemented: _tminus1 of an aggregate whose FnsToEvaluate uses _tplus1 (at a branching node its value depends
%   on the child).

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~exist('simoptions','var')
    simoptions=struct();
end
FnNames=fieldnames(FnsToEvaluate);
tplus1FnNames={}; % the FnsToEvaluate that use a _tplus1 input
usetminus1=0;
for ff=1:length(FnNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnNames{ff}));
    if any(endsWith(temp,'_tplus1'))
        tplus1FnNames=[tplus1FnNames,FnNames(ff)];
    end
    if any(endsWith(temp,'_tminus1'))
        usetminus1=1;
    end
end
if usetminus1==1 && ~isfield(transpathoptions,'initialvalues')
    error('EvalFnOnUncertainTransPath_AggVars_FHorz: a _tminus1 input is used, so transpathoptions.initialvalues must be given (it supplies the period 1 values)')
end

%% The tree, and the prices and parameters on it (parameters can be vectors, e.g. age-dependent: k-by-nNodes)
transpathoptions.uncertaintreevectorparams=1;
[Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions);
PricePathNames=fieldnames(PricePathNodes);
ParamPathNames=fieldnames(ParamPathNodes);
nNodes=Tree.nNodes;

[~,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,ParamPathNames,{},transpathoptions);
temp=intersect(tminus1AggVarsNames,tplus1FnNames);
if ~isempty(temp)
    error('EvalFnOnUncertainTransPath_AggVars_FHorz: %s_tminus1 is used, but FnsToEvaluate.%s uses _tplus1, so at a branching node its value depends on which child comes next and its _tminus1 is not defined; not implemented for uncertain transition paths',temp{1},temp{1})
end

Policyshape=size(PolicyPathTree);
Distshape=size(AgentDistPathTree);
if Policyshape(end)~=nNodes || Distshape(end)~=nNodes
    error('EvalFnOnUncertainTransPath_AggVars_FHorz: the last dimension of AgentDistPathTree and PolicyPathTree must be the %i nodes of the tree',nNodes)
end
PolicyPathTree=reshape(PolicyPathTree,[prod(Policyshape(1:end-1)),nNodes]);
AgentDistPathTree=reshape(AgentDistPathTree,[prod(Distshape(1:end-1)),nNodes]);

for ff=1:length(FnNames)
    AggVarsTree.(FnNames{ff}).Mean=zeros(1,nNodes);
end
AggVarsGivenChildTree=struct();
for ff=1:length(tplus1FnNames)
    AggVarsGivenChildTree.(tplus1FnNames{ff})=nan(1,nNodes);
end

% A segment N_j periods long makes an age-dependent parameter's segment path square, which
% PricePathParamPath_FHorz_StructToMatrix reads as N_j-by-T with a warning; here it is built that way on purpose, so that
% warning is turned off for the calls on the segments, and back on (as it was) when this command returns
warningstate_square=warning('off','VFIToolkit:SquarePathEntry');
cleanup_square=onCleanup(@() warning(warningstate_square));
for ss=1:Tree.nSeg
    segnodes=Tree.segnodes{ss};
    L=length(segnodes);
    endnode=segnodes(end);
    if Tree.nodenchild(endnode)==0
        Tseg=L;
        segnodes_ext=segnodes;
    else
        % A segment ending at a branching node can be a single node, and a path of one period is not a transition path
        % to EvalFnOnTransPath_AggVars_Case1_FHorz, so it is given one extra period (repeating the last node) and that is dropped
        Tseg=L+1;
        segnodes_ext=[segnodes,endnode];
    end
    PricePath_seg=struct();
    for pp=1:length(PricePathNames)
        PricePath_seg.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})(segnodes_ext);
    end
    ParamPath_seg=struct();
    for pp=1:length(ParamPathNames)
        ParamPath_seg.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,segnodes_ext); % k-by-Tseg, k=1 or N_j
    end
    PolicyPath_seg=reshape(PolicyPathTree(:,segnodes_ext),[Policyshape(1:end-1),Tseg]);
    AgentDistPath_seg=reshape(AgentDistPathTree(:,segnodes_ext),[Distshape(1:end-1),Tseg]);

    % The _tminus1 values of the first node of the segment are those of its parent node (a segment that starts in
    % period 1 has no parent, and keeps transpathoptions.initialvalues). Segments are numbered by the date of their
    % first node, so the parent node is already done.
    transpathoptions_seg=transpathoptions;
    parentnode=Tree.segparent(ss);
    if parentnode>0
        for pp=1:length(tminus1priceNames)
            transpathoptions_seg.initialvalues.(tminus1priceNames{pp})=PricePathNodes.(tminus1priceNames{pp})(parentnode);
        end
        for pp=1:length(tminus1paramNames)
            transpathoptions_seg.initialvalues.(tminus1paramNames{pp})=ParamPathNodes.(tminus1paramNames{pp})(:,parentnode);
        end
        for ff=1:length(tminus1AggVarsNames)
            transpathoptions_seg.initialvalues.(tminus1AggVarsNames{ff})=AggVarsTree.(tminus1AggVarsNames{ff}).Mean(parentnode);
        end
    end

    AggVarsPath_seg=EvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate, gpuArray(AgentDistPath_seg), gpuArray(PolicyPath_seg), PricePath_seg, ParamPath_seg, Parameters, Tseg, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, transpathoptions_seg, simoptions);

    for ff=1:length(FnNames)
        temp=gather(AggVarsPath_seg.(FnNames{ff}).Mean);
        AggVarsTree.(FnNames{ff}).Mean(segnodes)=temp(1:L);
    end

    if ~isempty(tplus1FnNames)
        % Within the segment each node has one child, the next node, and its _tplus1 is that node's price
        for ff=1:length(tplus1FnNames)
            AggVarsGivenChildTree.(tplus1FnNames{ff})(segnodes(2:end))=AggVarsTree.(tplus1FnNames{ff}).Mean(segnodes(1:end-1));
        end
        if Tree.nodenchild(endnode)>0
            % A branching node: above, its _tplus1 was its own price (the repeated last node), so the FnsToEvaluate that
            % use _tplus1 are evaluated again for each child c, as a path of two periods, the branching node then c,
            % of which only the first period is kept (the second period's dist and policy are those of the branching
            % node, and are not used). Every FnsToEvaluate is passed, so that any aggregate used as _tminus1 is still
            % one; only those that use _tplus1 are kept. Its _tminus1 values are those of its parent node.
            transpathoptions_end=transpathoptions;
            parentnode=Tree.nodeparent(endnode);
            if parentnode>0
                for pp=1:length(tminus1priceNames)
                    transpathoptions_end.initialvalues.(tminus1priceNames{pp})=PricePathNodes.(tminus1priceNames{pp})(parentnode);
                end
                for pp=1:length(tminus1paramNames)
                    transpathoptions_end.initialvalues.(tminus1paramNames{pp})=ParamPathNodes.(tminus1paramNames{pp})(:,parentnode);
                end
                for ff=1:length(tminus1AggVarsNames)
                    transpathoptions_end.initialvalues.(tminus1AggVarsNames{ff})=AggVarsTree.(tminus1AggVarsNames{ff}).Mean(parentnode);
                end
            end
            children=Tree.nodefirstchild(endnode)+(0:Tree.nodenchild(endnode)-1);
            PolicyPath_end=reshape(PolicyPathTree(:,[endnode,endnode]),[Policyshape(1:end-1),2]);
            AgentDistPath_end=reshape(AgentDistPathTree(:,[endnode,endnode]),[Distshape(1:end-1),2]);
            for cc=1:length(children)
                PricePath_c=struct();
                for pp=1:length(PricePathNames)
                    PricePath_c.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})([endnode,children(cc)]);
                end
                ParamPath_c=struct();
                for pp=1:length(ParamPathNames)
                    ParamPath_c.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,[endnode,children(cc)]);
                end
                AggVarsPath_c=EvalFnOnTransPath_AggVars_Case1_FHorz(FnsToEvaluate, gpuArray(AgentDistPath_end), gpuArray(PolicyPath_end), PricePath_c, ParamPath_c, Parameters, 2, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, transpathoptions_end, simoptions);
                for ff=1:length(tplus1FnNames)
                    temp=gather(AggVarsPath_c.(tplus1FnNames{ff}).Mean);
                    AggVarsGivenChildTree.(tplus1FnNames{ff})(children(cc))=temp(1);
                end
            end
            w=reshape(Tree.nodecondprob(children),1,[]);
            for ff=1:length(tplus1FnNames)
                AggVarsTree.(tplus1FnNames{ff}).Mean(endnode)=sum(w.*AggVarsGivenChildTree.(tplus1FnNames{ff})(children));
            end
        end
    end
end

end
