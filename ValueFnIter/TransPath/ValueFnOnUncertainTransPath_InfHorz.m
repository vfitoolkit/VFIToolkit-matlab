function [VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_InfHorz(PricePath, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% The value fn and policy on every node of the tree of an uncertain transition path, given the prices.
% See UncertainTransPathTree for the aggregate shock S (pi_S, T1, T2, S_initial), the tree, and the forms PricePath and
% ParamPath can take (a parameter that depends on S is N_S-by-T).
%
% V_final and Policy_final are structures with one field per S, V_final.S1, V_final.S2, ...: the value fn and policy
% of the final stationary eqm of each S. Every node in period T is at the final stationary eqm of its S. Only the S
% that some history can be in at period T are needed.
%
% Households know pi_S: at a node whose children differ (a branching node, which is a period t with T1<=t<=T2), the
% next period value fn is the expectation over the children, sum_{S'} pi_S(S_t,S') V_{t+1,S'}. Everywhere else the
% next period is a single node, and the tree is solved one segment (a chain of single-child nodes) at a time, each an
% ordinary deterministic transition solved by ValueFnOnTransPath_InfHorz.
%
% Outputs are the same shape as those of ValueFnOnTransPath_InfHorz, with the last dimension being the nodes of the
% tree instead of the periods: e.g. VPathTree is [n_a,n_z,nNodes]. The value fn of history h in period t is
% VPathTree(:,:,Tree.histories(h,t)), where Tree=UncertainTransPathTree(T,T1,T2,pi_S,S_initial).

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~exist('vfoptions','var')
    vfoptions=struct();
end
% Local search is not implemented on the tree (ValueFnOnTransPath_InfHorz would ignore localsearch=1 with a warning
% in every segment, and localsearch=2 would restrict every segment without any verification sweep)
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
if vfoptions.localsearch~=0
    error('ValueFnOnUncertainTransPath_InfHorz: vfoptions.localsearch is not implemented for uncertain transition paths')
end
% Shocks that vary over the path, given as grids, would have to be cut into the dates of each segment; not implemented.
% (Shocks given by vfoptions.ExogShockFn/EiidShockFn whose parameters are in ParamPath are fine, as ParamPath is.)
if ndims(pi_z)==3 || ndims(z_grid)==3 || (ndims(z_grid)==2 && size(z_grid,2)>1 && size(z_grid,1)==sum(n_z) && ~all(size(z_grid)==[prod(n_z),length(n_z)]))
    error('ValueFnOnUncertainTransPath_InfHorz: z_grid and pi_z that vary over the path are not implemented for uncertain transition paths')
end
if isfield(vfoptions,'pi_e') && size(vfoptions.pi_e,2)>1
    error('ValueFnOnUncertainTransPath_InfHorz: an e (vfoptions.pi_e) that varies over the path is not implemented for uncertain transition paths')
end

%% The tree, and the prices and parameters on it
[Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions);
PricePathNames=fieldnames(PricePathNodes);
ParamPathNames=fieldnames(ParamPathNodes);
nNodes=Tree.nNodes;

finalS=unique(Tree.nodeS(Tree.nodedate==T));
for ii=1:length(finalS)
    if ~isfield(V_final,['S',num2str(finalS(ii))]) || ~isfield(Policy_final,['S',num2str(finalS(ii))])
        error('ValueFnOnUncertainTransPath_InfHorz: V_final and Policy_final need a field S%i (a history can be in S=%i in period T)',finalS(ii),finalS(ii))
    end
end
Vshape=size(V_final.(['S',num2str(finalS(1))]));
Policyshape=size(Policy_final.(['S',num2str(finalS(1))]));
NV=prod(Vshape);
NPolicy=prod(Policyshape);

VPathTree=zeros(NV,nNodes);
PolicyPathTree=zeros(NPolicy,nNodes);
Vpershape=[];
Policypershape=[];

%% Backward over the segments: a child segment always has a larger number than its parent
for ss=Tree.nSeg:-1:1
    segnodes=Tree.segnodes{ss};
    L=length(segnodes);
    endnode=segnodes(end);
    if Tree.nodenchild(endnode)==0
        % The segment runs to period T, where it is at the final stationary eqm of its S
        Tseg=L;
        segnodes_ext=segnodes;
        V_final_seg=V_final.(['S',num2str(Tree.nodeS(endnode))]);
        Policy_final_seg=Policy_final.(['S',num2str(Tree.nodeS(endnode))]);
    else
        % The segment ends at a branching node: the next period value fn is the expectation over its children. It goes
        % in as the period Tseg=L+1 value fn of a transition of L+1 periods, of which the first L are this segment. The
        % prices and parameters of that extra period are never used (V_final is given), so it repeats the last node.
        Tseg=L+1;
        segnodes_ext=[segnodes,endnode];
        children=Tree.nodefirstchild(endnode):(Tree.nodefirstchild(endnode)+Tree.nodenchild(endnode)-1);
        % nodecondprob is the pi_S of each child given this node; pruning means every one is positive, so no 0*(-Inf)
        V_final_seg=reshape(sum(VPathTree(:,children).*Tree.nodecondprob(children)',2),Vshape);
        Policy_final_seg=reshape(PolicyPathTree(:,children(1)),Policyshape); % only stored in period Tseg, which is dropped
    end
    PricePath_seg=struct();
    for pp=1:length(PricePathNames)
        PricePath_seg.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})(segnodes_ext);
    end
    ParamPath_seg=struct();
    for pp=1:length(ParamPathNames)
        ParamPath_seg.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(segnodes_ext);
    end

    [VPath_seg,PolicyPath_seg]=ValueFnOnTransPath_InfHorz(PricePath_seg, ParamPath_seg, Tseg, gpuArray(V_final_seg), Policy_final_seg, Parameters, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);

    if isempty(Vpershape)
        temp=size(VPath_seg);
        Vpershape=temp(1:end-1);
        temp=size(PolicyPath_seg);
        Policypershape=temp(1:end-1);
    end
    VPath_seg=gather(reshape(VPath_seg,[NV,Tseg]));
    PolicyPath_seg=gather(reshape(PolicyPath_seg,[NPolicy,Tseg]));
    VPathTree(:,segnodes)=VPath_seg(:,1:L);
    PolicyPathTree(:,segnodes)=PolicyPath_seg(:,1:L);
end

%% Same shape as ValueFnOnTransPath_InfHorz, with the nodes in place of the periods
VPathTree=reshape(VPathTree,[Vpershape,nNodes]);
PolicyPathTree=reshape(PolicyPathTree,[Policypershape,nNodes]);

end
