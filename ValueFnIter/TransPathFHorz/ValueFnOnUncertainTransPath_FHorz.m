function [VPathTree,PolicyPathTree]=ValueFnOnUncertainTransPath_FHorz(PricePath, ParamPath, T, T1, T2, pi_S, S_initial, V_final, Policy_final, Parameters, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% The value fn and policy on every node of the tree of an uncertain transition path of a finite horizon (OLG) model,
% given the prices. The FHorz counterpart of ValueFnOnUncertainTransPath_InfHorz. See UncertainTransPathTree for the
% aggregate shock S (pi_S, T1, T2, S_initial), the tree, and the forms PricePath and ParamPath can take: a parameter
% that depends on S is N_S-by-T, or per S as ParamPath.(name).S1, .S2, ..., whose fields can be N_j-by-T (one that
% depends on age and on S); an age-dependent parameter that does not depend on S is N_j-by-T, as on a deterministic path
% (give it per S if N_j=N_S, where an N_j-by-T entry would be read as depending on S). Prices are scalars.
%
% V_final and Policy_final are structures with one field per S, V_final.S1, V_final.S2, ...: the value fn and policy
% of the final stationary eqm of each S, as ValueFnIter_Case1_FHorz gives them ([n_a,n_z,N_j]). Every node in period T
% is at the final stationary eqm of its S. Only the S that some history can be in at period T are needed.
%
% Households know pi_S: at a node whose children differ (a branching node), the next period value fn of every age is
% the expectation over the children, sum_{S'} pi_S(S_t,S') V_{t+1,S'}. Everywhere else the next period is a single
% node, and the tree is solved one segment (a chain of single-child nodes) at a time, each an ordinary deterministic
% transition solved by ValueFnOnTransPath_Case1_FHorz (with transpathoptions.fastOLG as given).
%
% DISCOUNT FACTORS READ AT THE CHILD, as in ValueFnOnUncertainTransPath_InfHorz: 'name' in DiscountFactorParamNames is
% read at the node (it can depend on age, e.g. survival probabilities), 'name_tplus1' at the next node (a stochastic
% discount factor, a scalar price or parameter): at a branching node it goes inside the expectation.
%
% LOCAL SEARCH, as in ValueFnOnTransPath_Case1_FHorz: vfoptions.localsearch=1 is ignored here with a warning (a single
% backward pass has no verification sweep to certify a restricted answer against), and localsearch=2 forces it, each
% segment's window centred on staying put.
%
% Not implemented (refused): semi-exogenous states, experience assets, vfoptions.exoticpreferences, and z or e grids
% that vary over the path (age-dependent ones are fine).
%
% Outputs are the same shape as those of ValueFnOnTransPath_Case1_FHorz, with the last dimension being the nodes of the
% tree instead of the periods: e.g. VPathTree is [n_a,n_z,N_j,nNodes]. The value fn of history h in period t is
% VPathTree(:,:,:,Tree.histories(h,t)), where Tree=UncertainTransPathTree(T,T1,T2,pi_S,S_initial).

if ~exist('transpathoptions','var')
    transpathoptions=struct();
end
if ~exist('vfoptions','var')
    vfoptions=struct();
end
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
if vfoptions.localsearch==1
    warning(['ValueFnOnUncertainTransPath_FHorz: vfoptions.localsearch=1 is IGNORED here and the value and policy ' ...
        'are computed with a full unrestricted search, as in ValueFnOnTransPath_Case1_FHorz. Set vfoptions.localsearch=2 ' ...
        'to force the restricted search anyway.'])
    vfoptions.localsearch=0;
elseif vfoptions.localsearch>2
    error('vfoptions.localsearch must be 0, 1 (ignored here, with a warning) or 2 (forced)')
end
% Not implemented on the tree, refused by name rather than ignored
if isfield(vfoptions,'n_semiz') && prod(vfoptions.n_semiz)>0
    error('ValueFnOnUncertainTransPath_FHorz: semi-exogenous states (vfoptions.n_semiz) are not yet implemented for uncertain transition paths')
end
if (isfield(vfoptions,'experienceasset') && vfoptions.experienceasset>=1) || (isfield(vfoptions,'experienceassetz') && vfoptions.experienceassetz>=1)
    error('ValueFnOnUncertainTransPath_FHorz: experience assets are not yet implemented for uncertain transition paths')
end
if isfield(vfoptions,'exoticpreferences') && ~strcmp(vfoptions.exoticpreferences,'None')
    error('ValueFnOnUncertainTransPath_FHorz: vfoptions.exoticpreferences is not yet implemented for uncertain transition paths')
end
% Shocks that vary over the path, given as grids, would have to be cut into the dates of each segment; not implemented.
% Age-dependent ones are fine (the detection is that of ExogShockSetup_FHorz_TPath).
% (Shocks given by vfoptions.ExogShockFn/EiidShockFn whose parameters are in ParamPath are fine, as ParamPath is.)
N_z=prod(n_z);
zpathvarying=0;
if ndims(z_grid)==4
    zpathvarying=1;
elseif ndims(z_grid)==3 && ~all(size(z_grid)==[prod(n_z),length(n_z),N_j])
    zpathvarying=1;
elseif ndims(z_grid)==2 && size(z_grid,2)>1 && ~all(size(z_grid)==[prod(n_z),length(n_z)]) && ~all(size(z_grid)==[sum(n_z),N_j])
    zpathvarying=1;
end
if ndims(pi_z)==4
    zpathvarying=1;
elseif ndims(pi_z)==3 && ~all(size(pi_z)==[N_z,N_z,N_j]) && ~all(size(pi_z)==[N_z,N_z,N_j-1])
    zpathvarying=1;
end
if zpathvarying==1
    error('ValueFnOnUncertainTransPath_FHorz: z_grid and pi_z that vary over the path are not implemented for uncertain transition paths (age-dependent ones are fine)')
end
if isfield(vfoptions,'pi_e') && ndims(vfoptions.pi_e)==3
    error('ValueFnOnUncertainTransPath_FHorz: an e (vfoptions.pi_e) that varies over the path is not implemented for uncertain transition paths (an age-dependent one is fine)')
end

%% The tree, and the prices and parameters on it (parameters can be vectors, e.g. age-dependent: k-by-nNodes)
transpathoptions.uncertaintreevectorparams=1;
[Tree,PricePathNodes,ParamPathNodes]=UncertainTransPathTree(T,T1,T2,pi_S,S_initial,PricePath,ParamPath,transpathoptions);
PricePathNames=fieldnames(PricePathNodes);
ParamPathNames=fieldnames(ParamPathNodes);
nNodes=Tree.nNodes;

%% Discount factors read at the child (name_tplus1), and their product at each node
if ischar(DiscountFactorParamNames)
    DiscountFactorParamNames={DiscountFactorParamNames};
end
DFchildNames={}; % the name of each discount factor read at the child, without the _tplus1
DFnodeNames={}; % the discount factors read at the node
for DFparam_c=1:length(DiscountFactorParamNames)
    if endsWith(DiscountFactorParamNames{DFparam_c},'_tplus1')
        DFchildNames=[DFchildNames,{DiscountFactorParamNames{DFparam_c}(1:end-7)}];
    else
        DFnodeNames=[DFnodeNames,DiscountFactorParamNames(DFparam_c)];
    end
end
DFchildvals=ones(1,nNodes); % at each node, the product of the discount factors read at the child, at their values at that node
for DFparam_c=1:length(DFchildNames)
    if any(strcmp(DFchildNames{DFparam_c},DFnodeNames))
        error('ValueFnOnUncertainTransPath_FHorz: DiscountFactorParamNames contains both %s and %s_tplus1; %s would be applied twice, at the node and at the child',DFchildNames{DFparam_c},DFchildNames{DFparam_c},DFchildNames{DFparam_c})
    end
    if any(strcmp(DFchildNames{DFparam_c},PricePathNames))
        DFchildvals=DFchildvals.*PricePathNodes.(DFchildNames{DFparam_c});
    elseif any(strcmp(DFchildNames{DFparam_c},ParamPathNames))
        if size(ParamPathNodes.(DFchildNames{DFparam_c}),1)>1
            error('ValueFnOnUncertainTransPath_FHorz: DiscountFactorParamNames contains %s_tplus1, but %s is not a scalar; a discount factor read at the child must be a scalar',DFchildNames{DFparam_c},DFchildNames{DFparam_c})
        end
        DFchildvals=DFchildvals.*ParamPathNodes.(DFchildNames{DFparam_c});
    else
        error('ValueFnOnUncertainTransPath_FHorz: DiscountFactorParamNames contains %s_tplus1, but %s is not in PricePath or ParamPath (a discount factor read at the child must take a value at each node)',DFchildNames{DFparam_c},DFchildNames{DFparam_c})
    end
end
if ~isempty(DFchildNames)
    % Within a segment, the discount factors read at the child go in as one extra parameter, in each period the
    % product at the next node of the segment
    if any(strcmp('DFtplus1_UncertainTPath',[PricePathNames;ParamPathNames]))
        error('ValueFnOnUncertainTransPath_FHorz: DFtplus1_UncertainTPath is a name used internally, please rename your parameter')
    end
    DiscountFactorParamNames=[DFnodeNames,{'DFtplus1_UncertainTPath'}];
end

finalS=unique(Tree.nodeS(Tree.nodedate==T));
for ii=1:length(finalS)
    if ~isfield(V_final,['S',num2str(finalS(ii))]) || ~isfield(Policy_final,['S',num2str(finalS(ii))])
        error('ValueFnOnUncertainTransPath_FHorz: V_final and Policy_final need a field S%i (a history can be in S=%i in period T)',finalS(ii),finalS(ii))
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

%% A segment N_j periods long makes an age-dependent parameter's segment path square, which
% PricePathParamPath_FHorz_StructToMatrix reads as N_j-by-T with a warning; here it is built that way on purpose, so that
% warning is turned off for the calls on the segments, and back on (as it was) when this command returns
warningstate_square=warning('off','VFIToolkit:SquarePathEntry');
cleanup_square=onCleanup(@() warning(warningstate_square));

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
        % The segment ends at a branching node: the next period value fn (of every age) is the expectation over its
        % children. It goes in as the period Tseg=L+1 value fn of a transition of L+1 periods, of which the first L are
        % this segment. The prices and parameters of that extra period are never used (V_final is given), so it
        % repeats the last node.
        Tseg=L+1;
        segnodes_ext=[segnodes,endnode];
        children=Tree.nodefirstchild(endnode):(Tree.nodefirstchild(endnode)+Tree.nodenchild(endnode)-1);
        % nodecondprob is the pi_S of each child given this node; pruning means every one is positive, so no 0*(-Inf).
        % The discount factors read at the child go inside the expectation (all ones when there are none).
        V_final_seg=reshape(sum(VPathTree(:,children).*(reshape(Tree.nodecondprob(children),1,[]).*DFchildvals(children)),2),Vshape);
        Policy_final_seg=reshape(PolicyPathTree(:,children(1)),Policyshape); % only stored in period Tseg, which is dropped
    end
    PricePath_seg=struct();
    for pp=1:length(PricePathNames)
        PricePath_seg.(PricePathNames{pp})=PricePathNodes.(PricePathNames{pp})(segnodes_ext);
    end
    ParamPath_seg=struct();
    for pp=1:length(ParamPathNames)
        ParamPath_seg.(ParamPathNames{pp})=ParamPathNodes.(ParamPathNames{pp})(:,segnodes_ext); % k-by-Tseg, k=1 or N_j
    end
    if ~isempty(DFchildNames)
        % Period k of the segment discounts with the value at its next node, segnodes(k+1). At the last node there is
        % none: a branching node has them inside V_final_seg, and a period T node is V_final (never computed).
        ParamPath_seg.DFtplus1_UncertainTPath=ones(1,Tseg);
        ParamPath_seg.DFtplus1_UncertainTPath(1:L-1)=DFchildvals(segnodes(2:L));
    end

    [VPath_seg,PolicyPath_seg]=ValueFnOnTransPath_Case1_FHorz(PricePath_seg, ParamPath_seg, Tseg, gpuArray(V_final_seg), Policy_final_seg, Parameters, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);

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

%% Same shape as ValueFnOnTransPath_Case1_FHorz, with the nodes in place of the periods
VPathTree=reshape(VPathTree,[Vpershape,nNodes]);
PolicyPathTree=reshape(PolicyPathTree,[Policypershape,nNodes]);

end
