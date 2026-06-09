function varargout=HeteroAgentStationaryEqm_FHorz_PType2L(n_d, n_a, n_z, N_j, Names_i, N_i, n_p, pi_z, d_grid, a_grid, z_grid, jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightParamNames, TopPTypeDistParamNames, PTypeDistParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions)
% Two-level permanent-type general equilibrium dispatcher (finite-horizon).
%
% Top level is named (Names_i, cell of length N_topi). Within each top type
% the bottom level is numeric (N_i is either a scalar applied to every top,
% or a struct keyed by Names_i giving the per-top bottom count).
% Parameters/options that vary at the top level must be supplied as structs
% keyed by Names_i (or, where supported, as vectors of length N_topi for
% TopPTypeDistParamNames). Parameters/options that vary at the bottom level
% within a top are handled by the existing single-level PType machinery.
%
% Internally we build a flat PTypeStructure indexed by every (top,bottom)
% pair; the _subfn loops over that flat list and calls the non-PType
% ValueFnIter_Case1_FHorz / StationaryDist_FHorz_Case1 /
% EvalFnOnAgentDist_AggVars_FHorz_Case1 once per pair, then accumulates
% AggVars using combined weights = topweight*bottomweight.
%
% v1 scope:
%   - plain global GE prices (no per-ptype struct-keyed prices)
%   - plain GeneralEqmEqns (heteroagentoptions.GEptype must be empty)
%   - plain intermediateEqns (heteroagentoptions.intermediateEqnsptype not allowed)
%   - no CustomModelStats
%   - jequaloneDist must be numeric or a struct keyed by top names
% These restrictions are enforced with explicit errors below.

%% Check 'double fminalgo'
if exist('heteroagentoptions','var')
    if isfield(heteroagentoptions,'fminalgo')
        if length(heteroagentoptions.fminalgo)>1
            if isfield(heteroagentoptions,'toleranceGEcondns')
                heteroagentoptions.toleranceGEcondns=heteroagentoptions.toleranceGEcondns.*ones(1,length(heteroagentoptions.fminalgo));
            else
                heteroagentoptions.toleranceGEcondns=10^(-4).*ones(1,length(heteroagentoptions.fminalgo));
            end
            if isfield(heteroagentoptions,'toleranceGEprices')
                heteroagentoptions.toleranceGEprices=heteroagentoptions.toleranceGEprices.*ones(1,length(heteroagentoptions.fminalgo));
            else
                heteroagentoptions.toleranceGEprices=10^(-4).*ones(1,length(heteroagentoptions.fminalgo));
            end
            temp=heteroagentoptions.fminalgo;
            temp2=heteroagentoptions.toleranceGEcondns;
            temp3=heteroagentoptions.toleranceGEprices;
            heteroagentoptions.fminalgo=heteroagentoptions.fminalgo(1:end-1);
            heteroagentoptions.toleranceGEcondns=heteroagentoptions.toleranceGEcondns(1:end-1);
            heteroagentoptions.toleranceGEprices=heteroagentoptions.toleranceGEprices(1:end-1);
            p_eqm_previous=HeteroAgentStationaryEqm_FHorz_PType2L(n_d, n_a, n_z, N_j, Names_i, N_i, n_p, pi_z, d_grid, a_grid, z_grid, jequaloneDist, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightParamNames, TopPTypeDistParamNames, PTypeDistParamNames, GEPriceParamNames, heteroagentoptions, simoptions, vfoptions);
            for pp=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{pp})=p_eqm_previous.(GEPriceParamNames{pp});
            end
            heteroagentoptions.fminalgo=temp(end);
            heteroagentoptions.toleranceGEcondns=temp2(end);
            heteroagentoptions.toleranceGEprices=temp3(end);
        end
    end
end


%% Defaults
N_p=prod(n_p);
if isempty(n_p)
    N_p=0;
end

if exist('heteroagentoptions','var')==0
    heteroagentoptions=struct();
end
if ~isfield(heteroagentoptions,'fminalgo'),         heteroagentoptions.fminalgo=1; end
if ~isfield(heteroagentoptions,'multiGEcriterion'), heteroagentoptions.multiGEcriterion=1; end
if ~isfield(heteroagentoptions,'multiGEweights'),   heteroagentoptions.multiGEweights=ones(1,length(fieldnames(GeneralEqmEqns))); end
if ~isfield(heteroagentoptions,'toleranceGEprices'), heteroagentoptions.toleranceGEprices=10^(-4); end
if ~isfield(heteroagentoptions,'toleranceGEcondns'), heteroagentoptions.toleranceGEcondns=10^(-4); end
if ~isfield(heteroagentoptions,'maxiter'),           heteroagentoptions.maxiter=200*length(GEPriceParamNames); end
if N_p~=0 && ~isfield(heteroagentoptions,'p_grid')
    error('You have set n_p to a non-zero value, but not declared heteroagentoptions.p_grid')
end
if ~isfield(heteroagentoptions,'GEptype'),          heteroagentoptions.GEptype={}; end
if ~isfield(heteroagentoptions,'constrainpositive'), heteroagentoptions.constrainpositive={}; end
if ~isfield(heteroagentoptions,'constrain0to1'),    heteroagentoptions.constrain0to1={}; end
if ~isfield(heteroagentoptions,'constrainAtoB'),    heteroagentoptions.constrainAtoB={}; end
if isfield(heteroagentoptions,'constrainAtoB') && ~isempty(heteroagentoptions.constrainAtoB)
    if prod(heteroagentoptions.constrainAtoB)>0
        if ~isfield(heteroagentoptions,'constrainAtoBlimits')
            error('You have used heteroagentoptions.constrainAtoB, but are missing heteroagentoptions.constrainAtoBlimits')
        end
    end
end
if ~isfield(heteroagentoptions,'verbose'),          heteroagentoptions.verbose=0; end
if ~isfield(heteroagentoptions,'verboseaccuracy1'), heteroagentoptions.verboseaccuracy1=4; end
if ~isfield(heteroagentoptions,'verboseaccuracy2'), heteroagentoptions.verboseaccuracy2=6; end
if ~isfield(heteroagentoptions,'pricehistory'),     heteroagentoptions.pricehistory=0; end
if ~isfield(heteroagentoptions,'outputGEstruct'),   heteroagentoptions.outputGEstruct=1; end
if ~isfield(heteroagentoptions,'outputgather'),     heteroagentoptions.outputgather=1; end

%% v1 restrictions
if ~iscell(Names_i)
    error('HeteroAgentStationaryEqm_FHorz_PType2L requires Names_i as a cell array of top-level PType names.')
end
if ~isempty(heteroagentoptions.GEptype)
    error('heteroagentoptions.GEptype is not supported by HeteroAgentStationaryEqm_FHorz_PType2L v1; use plain GE conditions.')
end
if isfield(heteroagentoptions,'intermediateEqnsptype') && any(heteroagentoptions.intermediateEqnsptype)
    error('heteroagentoptions.intermediateEqnsptype is not supported by HeteroAgentStationaryEqm_FHorz_PType2L v1.')
end
if isfield(heteroagentoptions,'CustomModelStats')
    error('heteroagentoptions.CustomModelStats is not supported by HeteroAgentStationaryEqm_FHorz_PType2L v1.')
end
for pp=1:length(GEPriceParamNames)
    if isstruct(Parameters.(GEPriceParamNames{pp}))
        error('Per-ptype struct-keyed GE prices are not supported by HeteroAgentStationaryEqm_FHorz_PType2L v1 (GE price %s).', GEPriceParamNames{pp})
    end
end
if isa(jequaloneDist,'function_handle')
    error('jequaloneDist as function_handle is not supported by HeteroAgentStationaryEqm_FHorz_PType2L v1; pass it numerically or as a struct keyed by Names_i.')
end

heteroagentoptions.useCustomModelStats=0;

if heteroagentoptions.fminalgo==0
    heteroagentoptions.outputGEform=1;
elseif heteroagentoptions.fminalgo==5
    heteroagentoptions.outputGEform=1;
    heteroagentoptions.outputgather=0;
elseif heteroagentoptions.fminalgo==7
    heteroagentoptions.outputGEform=1;
else
    heteroagentoptions.outputGEform=0;
end

temp=size(heteroagentoptions.multiGEweights);
if temp(2)==1
    heteroagentoptions.multiGEweights=heteroagentoptions.multiGEweights';
end
if length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns))
    error('length(heteroagentoptions.multiGEweights)~=length(fieldnames(GeneralEqmEqns))')
end

heteroagentoptions.verboseaccuracy1=['	%s: %8.',num2str(heteroagentoptions.verboseaccuracy1),'f \n'];
heteroagentoptions.verboseaccuracy2=['	%s: %8.',num2str(heteroagentoptions.verboseaccuracy2),'f \n'];

AggVarNames=fieldnames(FnsToEvaluate);
nGEprices=length(GEPriceParamNames);

PTypeStructure.numFnsToEvaluate=length(fieldnames(FnsToEvaluate));

%% Top-level setup
N_topi=length(Names_i);

% Resolve per-top N_i_top (length-N_topi vector of bottom counts)
N_i_top=zeros(1,N_topi);
for ii_top=1:N_topi
    iistr_top=Names_i{ii_top};
    if isstruct(N_i)
        N_i_top(ii_top)=N_i.(iistr_top);
    else
        N_i_top(ii_top)=N_i;
    end
end
N_pairs=sum(N_i_top);

% Build bottom names per top following the single-level PType convention
% ('ptype001', 'ptype002', ...). Pair keys use 'topname__ptypeNNN'.
BottomNames=cell(1,N_topi);
for ii_top=1:N_topi
    BottomNames{ii_top}=cell(1,N_i_top(ii_top));
    for ii_bot=1:N_i_top(ii_top)
        if ii_bot<10
            BottomNames{ii_top}{ii_bot}=['ptype00',num2str(ii_bot)];
        elseif ii_bot<100
            BottomNames{ii_top}{ii_bot}=['ptype0',num2str(ii_bot)];
        else
            BottomNames{ii_top}{ii_bot}=['ptype',num2str(ii_bot)];
        end
    end
end

% Flat pair list
PTypeStructure.N_pairs=N_pairs;
PTypeStructure.Names_i=Names_i;
PTypeStructure.N_i_top=N_i_top;
PTypeStructure.iistr=cell(1,N_pairs);
PTypeStructure.toptag=cell(1,N_pairs);
PTypeStructure.bottomtag=cell(1,N_pairs);
PTypeStructure.itop=zeros(1,N_pairs);
PTypeStructure.ibot=zeros(1,N_pairs);
k=0;
for ii_top=1:N_topi
    for ii_bot=1:N_i_top(ii_top)
        k=k+1;
        PTypeStructure.iistr{k}=[Names_i{ii_top},'__',BottomNames{ii_top}{ii_bot}];
        PTypeStructure.toptag{k}=Names_i{ii_top};
        PTypeStructure.bottomtag{k}=BottomNames{ii_top}{ii_bot};
        PTypeStructure.itop(k)=ii_top;
        PTypeStructure.ibot(k)=ii_bot;
    end
end

%% Per-pair peel (top then bottom)
% jequaloneDist: top-only peel here, bottom-level peeled in inner block.
for k=1:N_pairs
    iistr=PTypeStructure.iistr{k};
    iistr_top=PTypeStructure.toptag{k};
    iistr_bot=PTypeStructure.bottomtag{k};
    ii_top=PTypeStructure.itop(k);
    ii_bot=PTypeStructure.ibot(k);

    % --- Options: top peel with PType_Options_2L (passes bottom-keyed structs through), then bottom peel with PType_Options ---
    vfoptions_top=PType_Options_2L(vfoptions,iistr_top);
    simoptions_top=PType_Options_2L(simoptions,iistr_top);
    PTypeStructure.(iistr).vfoptions=PType_Options(vfoptions_top,iistr_bot);
    PTypeStructure.(iistr).simoptions=PType_Options(simoptions_top,iistr_bot);
    PTypeStructure.(iistr).simoptions.outputasstructure=0;

    if heteroagentoptions.verbose==1
        fprintf('Setting up, pair %i of %i: %s\n',k,N_pairs,iistr)
    end

    % --- n_d, n_a, d_grid, a_grid: top then bottom ---
    [n_d_top,n_a_top,d_grid_top,a_grid_top]=PType_setup_da(iistr_top,n_d,n_a,d_grid,a_grid);
    [PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).a_grid]=PType_setup_da(iistr_bot,n_d_top,n_a_top,d_grid_top,a_grid_top);
    if PTypeStructure.(iistr).n_d(1)==0
        PTypeStructure.(iistr).l_d=0;
    else
        PTypeStructure.(iistr).l_d=length(PTypeStructure.(iistr).n_d);
    end
    PTypeStructure.(iistr).l_a=length(PTypeStructure.(iistr).n_a);

    % --- N_j: top then bottom ---
    if isstruct(N_j) && isfield(N_j,iistr_top)
        N_j_top=N_j.(iistr_top);
    else
        N_j_top=N_j;
    end
    if isstruct(N_j_top) && isfield(N_j_top,iistr_bot)
        PTypeStructure.(iistr).N_j=N_j_top.(iistr_bot);
    else
        PTypeStructure.(iistr).N_j=N_j_top;
    end

    % --- Exogenous shocks: top peel (mode 1), then bottom peel (mode 3) ---
    [n_z_top,z_grid_top,pi_z_top,PTypeStructure.(iistr).vfoptions]=PType_setup_ExogShocks(ii_top,iistr_top,N_topi,n_z,z_grid,pi_z,PTypeStructure.(iistr).vfoptions,1);
    [PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).vfoptions]=PType_setup_ExogShocks(ii_bot,iistr_bot,N_i_top(ii_top),n_z_top,z_grid_top,pi_z_top,PTypeStructure.(iistr).vfoptions,3);
    PTypeStructure.(iistr).N_z=prod(PTypeStructure.(iistr).n_z);
    if PTypeStructure.(iistr).N_z==0
        PTypeStructure.(iistr).l_z=0;
    else
        PTypeStructure.(iistr).l_z=length(PTypeStructure.(iistr).n_z);
    end
    if prod(PTypeStructure.(iistr).simoptions.n_e)==0
        PTypeStructure.(iistr).l_e=0;
    else
        PTypeStructure.(iistr).l_e=length(PTypeStructure.(iistr).simoptions.n_e);
    end

    % --- ReturnFn / DiscountFactor: top peel, then bottom peel ---
    [ReturnFn_top,DiscountFactorParamNames_top]=PType_setup_ReturnFnDiscountFactor(iistr_top,ReturnFn,DiscountFactorParamNames);
    [PTypeStructure.(iistr).ReturnFn,PTypeStructure.(iistr).DiscountFactorParamNames]=PType_setup_ReturnFnDiscountFactor(iistr_bot,ReturnFn_top,DiscountFactorParamNames_top);

    % --- Parameters: top peel (struct only, mode 1), then bottom peel (struct+vector, mode 3) ---
    Parameters_top=PType_setup_Parameters(ii_top,iistr_top,N_topi,Parameters,1);
    PTypeStructure.(iistr).Parameters=PType_setup_Parameters(ii_bot,iistr_bot,N_i_top(ii_top),Parameters_top,3);

    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNamesFn(PTypeStructure.(iistr).ReturnFn,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).Parameters);

    % --- gridsinGE checks for this pair ---
    heteroagentoptions.gridsinGE(k)=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'ExogShockFn')
        tempExogShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.ExogShockFn);
        if ~isempty(intersect(tempExogShockFnParamNames,GEPriceParamNames))
            heteroagentoptions.gridsinGE(k)=1;
        end
    end
    if isfield(PTypeStructure.(iistr).vfoptions,'EiidShockFn')
        tempEiidShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.EiidShockFn);
        if ~isempty(intersect(tempEiidShockFnParamNames,GEPriceParamNames))
            heteroagentoptions.gridsinGE(k)=1;
        end
    end

    if heteroagentoptions.gridsinGE(k)==0
        [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J;
        PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J;
    end
    PTypeStructure.(iistr)=rmfield(PTypeStructure.(iistr),'z_grid');
    PTypeStructure.(iistr)=rmfield(PTypeStructure.(iistr),'pi_z');
    if isfield(PTypeStructure.(iistr).simoptions,'ExogShockFn')
        PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'ExogShockFn');
    end
    PTypeStructure.(iistr).vfoptions.alreadygridvals=1;
    PTypeStructure.(iistr).simoptions.alreadygridvals=1;

    % --- Semi-exogenous state ---
    heteroagentoptions.gridsinGE_semiexo(k)=0;
    if isfield(PTypeStructure.(iistr).vfoptions,'n_semiz') && prod(PTypeStructure.(iistr).vfoptions.n_semiz)>0
        if isfield(PTypeStructure.(iistr).vfoptions,'SemiExoShockFn')
            tempExogShockFnParamNames=getAnonymousFnInputNames(PTypeStructure.(iistr).vfoptions.SemiExoShockFn);
            if ~isempty(intersect(tempExogShockFnParamNames,GEPriceParamNames))
                heteroagentoptions.gridsinGE_semiexo(k)=1;
            end
        end

        % Trailing-dim-of-length-N_topi or N_i_top(ii_top) handled by PType_setup_ExogShocks above is for z; semiz lives in vfoptions and uses the existing flat slice convention.
        if size(PTypeStructure.(iistr).vfoptions.semiz_grid,ndims(PTypeStructure.(iistr).vfoptions.semiz_grid))==N_i_top(ii_top)
            otherdims=repmat({':'},1,ndims(PTypeStructure.(iistr).vfoptions.semiz_grid)-1);
            PTypeStructure.(iistr).vfoptions.semiz_grid=PTypeStructure.(iistr).vfoptions.semiz_grid(otherdims{:},ii_bot);
        end
        if isfield(PTypeStructure.(iistr).vfoptions,'pi_semiz')
            if size(PTypeStructure.(iistr).vfoptions.pi_semiz,ndims(PTypeStructure.(iistr).vfoptions.pi_semiz))==N_i_top(ii_top)
                otherdims=repmat({':'},1,ndims(PTypeStructure.(iistr).vfoptions.pi_semiz)-1);
                PTypeStructure.(iistr).vfoptions.pi_semiz=PTypeStructure.(iistr).vfoptions.pi_semiz(otherdims{:},ii_bot);
            end
        end

        PTypeStructure.(iistr).vfoptions=SemiExogShockSetup_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        PTypeStructure.(iistr).simoptions.semiz_gridvals_J=PTypeStructure.(iistr).vfoptions.semiz_gridvals_J;
        PTypeStructure.(iistr).simoptions.pi_semiz_J=PTypeStructure.(iistr).vfoptions.pi_semiz_J;
        PTypeStructure.(iistr).vfoptions.alreadygridvals_semiexo=1;
        PTypeStructure.(iistr).simoptions.alreadygridvals_semiexo=1;
    end

    % --- jequaloneDist (top peel only here; numeric or top-struct) ---
    if isstruct(jequaloneDist)
        if isfield(jequaloneDist,iistr_top)
            PTypeStructure.(iistr).jequaloneDist=jequaloneDist.(iistr_top);
        else
            error(['You must input jequaloneDist for top permanent type ',iistr_top])
        end
    else
        PTypeStructure.(iistr).jequaloneDist=jequaloneDist;
    end

    % --- AgeWeightParamNames: top peel ---
    PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightParamNames;
    if isstruct(AgeWeightParamNames)
        if isfield(AgeWeightParamNames,iistr_top)
            PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightParamNames.(iistr_top);
        else
            error(['You must input AgeWeightParamNames for top permanent type ',iistr_top])
        end
    end

    % --- FnsToEvaluate (uses pair index in the flat list to mask which fns apply) ---
    l_d_temp=PTypeStructure.(iistr).l_d;
    l_a_temp=PTypeStructure.(iistr).l_a;
    l_z_temp=PTypeStructure.(iistr).l_z+PTypeStructure.(iistr).l_e;
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp,WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,PTypeStructure.iistr,k,l_d_temp,l_a_temp,l_z_temp,0);
    PTypeStructure.(iistr).FnsToEvaluate=FnsToEvaluate_temp;
    PTypeStructure.(iistr).FnsToEvaluateParamNames=FnsToEvaluateParamNames_temp;
    PTypeStructure.(iistr).WhichFnsForCurrentPType=WhichFnsForCurrentPType;
    PTypeStructure.(iistr).FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii;
end

%% Top-level weights from TopPTypeDistParamNames (elementwise product over names)
topptweights=ones(N_topi,1);
for kk=1:length(TopPTypeDistParamNames)
    val=Parameters.(TopPTypeDistParamNames{kk});
    if isstruct(val)
        v=zeros(N_topi,1);
        for ii_top=1:N_topi
            v(ii_top)=val.(Names_i{ii_top});
        end
        val=v;
    end
    topptweights=topptweights.*val(:);
end

%% Bottom weights per top and combined flat ptweights
PTypeStructure.ptweights=zeros(N_pairs,1);
for k=1:N_pairs
    iistr=PTypeStructure.iistr{k};
    ii_top=PTypeStructure.itop(k);
    bottomweight=PTypeStructure.(iistr).Parameters.(PTypeDistParamNames{1});
    PTypeStructure.ptweights(k,1)=topptweights(ii_top)*bottomweight;
end
PTypeStructure.topptweights=topptweights;

%% intermediateEqns (plain only)
heteroagentoptions.useintermediateEqns=0;
if isfield(heteroagentoptions,'intermediateEqns')
    heteroagentoptions.useintermediateEqns=1;
    intEqnNames=fieldnames(heteroagentoptions.intermediateEqns);
    nIntEqns=length(intEqnNames);
    heteroagentoptions.intermediateEqnsCell=cell(1,nIntEqns);
    for gg=1:nIntEqns
        temp=getAnonymousFnInputNames(heteroagentoptions.intermediateEqns.(intEqnNames{gg}));
        heteroagentoptions.intermediateEqnParamNames(gg).Names=temp;
        heteroagentoptions.intermediateEqnsCell{gg}=heteroagentoptions.intermediateEqns.(intEqnNames{gg});
    end
end

%% GE eqns (plain only)
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);
GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    GeneralEqmEqnParamNames(gg).Names=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end

%% GE price vector (no struct-keyed prices in v1 — scalars only)
GEparamsvec0=[];
GEpriceindexes=zeros(nGEprices,1);
for pp=1:nGEprices
    GEparamsvec0=[GEparamsvec0;reshape(gather(Parameters.(GEPriceParamNames{pp})),[],1)];
    GEpriceindexes(pp)=length(Parameters.(GEPriceParamNames{pp}));
end
GEpriceindexesB=[0; cumsum(GEpriceindexes)];
GEpriceindexes=[[1; 1+cumsum(GEpriceindexes(1:end-1))],cumsum(GEpriceindexes)];

[GEparamsvec0,heteroagentoptions]=ParameterConstraints_TransformParamsToUnconstrained(GEparamsvec0,GEpriceindexesB,GEPriceParamNames,heteroagentoptions,1);

%% pricehistory preallocation
if heteroagentoptions.pricehistory==1
    GEpricepath=zeros(length(GEparamsvec0),heteroagentoptions.maxiter);
    GEcondnpath=zeros(length(fieldnames(GeneralEqmEqns)),heteroagentoptions.maxiter);
    itercount=0;
    save pricehistory.mat GEpricepath GEcondnpath itercount
end

%% Solve
if heteroagentoptions.maxiter>0

    if heteroagentoptions.fminalgo~=8 && heteroagentoptions.fminalgo~=3
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_FHorz_PType2L_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
    elseif heteroagentoptions.fminalgo==3
        heteroagentoptions.outputGEform=1;
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_FHorz_PType2L_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
    elseif heteroagentoptions.fminalgo==8
        heteroagentoptions.outputGEform=1;
        weightsbackup=heteroagentoptions.multiGEweights;
        heteroagentoptions.multiGEweights=sqrt(heteroagentoptions.multiGEweights);
        GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_FHorz_PType2L_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
        heteroagentoptions.multiGEweights=weightsbackup;
    end

    minoptions=optimset('TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter);
    p_eqm_index=nan;
    if N_p~=0
        p_gridvals=CreateGridvals(n_p,heteroagentoptions.p_grid,1);
        GeneralEqmConditions=zeros(size(p_gridvals));
        for pp_c=1:N_p
            pvec=p_gridvals(pp_c,:);
            GeneralEqmConditions(pp_c,:)=GeneralEqmConditionsFnOpt(pvec);
        end
        [~,p_eqm_index]=max(sum(GeneralEqmConditions.^2,2));
        p_eqm_vec=p_gridvals(p_eqm_index,:);
        heteroagentoptions.outputGEstruct=0;
    elseif heteroagentoptions.fminalgo==0
        [p_eqm_vec,GeneralEqmConditions]=fzero(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==1
        [p_eqm_vec,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==2
        z=optimvar('z',length(GEparamsvec0));
        optimfun=fcn2optimexpr(GeneralEqmConditionsFnOpt, z);
        prob=optimproblem("Objective",optimfun);
        z0.z=GEparamsvec0;
        [sol,GeneralEqmConditions]=solve(prob,z0);
        p_eqm_vec=sol.z;
    elseif heteroagentoptions.fminalgo==3
        goal=zeros(length(GEparamsvec0),1);
        weight=ones(length(GEparamsvec0),1);
        [p_eqm_vec,GeneralEqmConditionsVec]=fgoalattain(GeneralEqmConditionsFnOpt,GEparamsvec0,goal,weight);
        GeneralEqmConditions=sum(abs(GeneralEqmConditionsVec));
    elseif heteroagentoptions.fminalgo==4
        if ~isfield(heteroagentoptions,'insigma')
            heteroagentoptions.insigma=0.3*abs(GEparamsvec0)+0.1*(GEparamsvec0==0);
        end
        if ~isfield(heteroagentoptions,'inopts')
            heteroagentoptions.inopts=[];
        end
        if isfield(heteroagentoptions,'toleranceGEcondns')
            heteroagentoptions.inopts.StopFitness=heteroagentoptions.toleranceGEcondns;
        end
        if heteroagentoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider citing Hansen & Kern (2004)')
        end
        [p_eqm_vec,GeneralEqmConditions,~,~,~,~]=cmaes_vfitoolkit(GeneralEqmConditionsFnOpt,GEparamsvec0,heteroagentoptions.insigma,heteroagentoptions.inopts);
    elseif heteroagentoptions.fminalgo==5
        heteroagentoptions=setupGEnewprice3_shooting(heteroagentoptions,GeneralEqmEqns,GEPriceParamNames,1,GEpriceindexes');
        p=nan(1,length(GEPriceParamNames));
        for ii=1:length(GEPriceParamNames)
            p(ii)=Parameters.(GEPriceParamNames{ii});
        end
        itercounter=0;
        p_change=Inf;
        GeneralEqmConditions=Inf;
        while (any(p_change>heteroagentoptions.toleranceGEprices) || GeneralEqmConditions>heteroagentoptions.toleranceGEcondns) && itercounter<heteroagentoptions.maxiter
            p_i=GeneralEqmConditionsFnOpt(p);
            GeneralEqmConditionsVec=p_i;
            p_i=p_i(heteroagentoptions.fminalgo5.permute);
            I_makescutoff=(abs(p_i)>heteroagentoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;
            p_new=(p.*heteroagentoptions.fminalgo5.keepold)+heteroagentoptions.fminalgo5.add.*heteroagentoptions.fminalgo5.factor.*p_i-(1-heteroagentoptions.fminalgo5.add).*heteroagentoptions.fminalgo5.factor.*p_i;
            if heteroagentoptions.multiGEcriterion==0
                GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
            elseif heteroagentoptions.multiGEcriterion==1
                GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
            end
            for ii=1:length(GEPriceParamNames)
                Parameters.(GEPriceParamNames{ii})=p_new(ii);
            end
            p_change=abs(p_new-p);
            p=p_new;
            itercounter=itercounter+1;
        end
        if itercounter>=heteroagentoptions.maxiter
            warning('HeteroAgentStationaryEqm stopped due to reaching maximum number of iterations (heteroagentoptions.maxiter)')
        end
        p_eqm_vec=p_new;
    elseif heteroagentoptions.fminalgo==6
        if ~isfield(heteroagentoptions,'lb') || ~isfield(heteroagentoptions,'ub')
            error('When using constrained optimization (heteroagentoptions.fminalgo=6) you must set heteroagentoptions.lb and heteroagentoptions.ub')
        end
        [p_eqm_vec,GeneralEqmConditions]=fmincon(GeneralEqmConditionsFnOpt,GEparamsvec0,[],[],[],[],heteroagentoptions.lb,heteroagentoptions.ub,[],minoptions);
    elseif heteroagentoptions.fminalgo==7
        heteroagentoptions.multiGEcriterion=0;
        [p_eqm_vec,GeneralEqmConditions]=fsolve(GeneralEqmConditionsFnOpt,GEparamsvec0,minoptions);
    elseif heteroagentoptions.fminalgo==8
        minoptions=optimoptions('lsqnonlin','FiniteDifferenceStepSize',1e-2,'TolX',heteroagentoptions.toleranceGEprices,'TolFun',heteroagentoptions.toleranceGEcondns,'MaxFunEvals',heteroagentoptions.maxiter,'MaxIter',heteroagentoptions.maxiter);
        [p_eqm_vec,GeneralEqmConditions]=lsqnonlin(GeneralEqmConditionsFnOpt,GEparamsvec0,[],[],[],[],[],[],[],minoptions);
    end

    [p_eqm_vec,~]=ParameterConstraints_TransformParamsToOriginal(p_eqm_vec,GEpriceindexesB,GEPriceParamNames,heteroagentoptions);

    for pp=1:nGEprices
        p_eqm.(GEPriceParamNames{pp})=p_eqm_vec(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
    end

elseif heteroagentoptions.maxiter==0
    p_eqm_vec=zeros(length(GEparamsvec0),1);
    p_eqm=nan;
    p_eqm_index=nan;
    for pp=1:length(GEPriceParamNames)
        p_eqm_vec(pp)=Parameters.(GEPriceParamNames{pp});
    end
end


%% Final evaluation of GE eqns in user-friendly form
if heteroagentoptions.outputGEstruct==1
    heteroagentoptions.outputGEform=2;
elseif heteroagentoptions.outputGEstruct==2
    heteroagentoptions.outputGEform=1;
end

if heteroagentoptions.outputGEstruct==1 || heteroagentoptions.outputGEstruct==2
    if isfield(heteroagentoptions,'constrainpositive')
        heteroagentoptions.constrainpositive=zeros(length(p_eqm_vec),1);
    end
    if isfield(heteroagentoptions,'constrain0to1')
        heteroagentoptions.constrain0to1=zeros(length(p_eqm_vec),1);
    end
    if isfield(heteroagentoptions,'constrainAtoB')
        heteroagentoptions.constrainAtoB=zeros(length(p_eqm_vec),1);
    end
    GeneralEqmConditionsFnOpt=@(p) HeteroAgentStationaryEqm_FHorz_PType2L_subfn(p, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions);
    GeneralEqmConditions=GeneralEqmConditionsFnOpt(p_eqm_vec);
end
if heteroagentoptions.outputGEstruct==1
    for gg=1:length(GEeqnNames)
        GeneralEqmConditions.(GEeqnNames{gg})=gather(GeneralEqmConditions.(GEeqnNames{gg}));
    end
end


%% pricehistory output
if heteroagentoptions.pricehistory==1
    load pricehistory.mat GEpricepath GEcondnpath itercount
    delete('pricehistory.mat')
    GEpricepath=GEpricepath(:,1:itercount);
    GEcondnpath=GEcondnpath(:,1:itercount);
    PriceHistory.itercount=itercount;
    for pp=1:nGEprices
        PriceHistory.(GEPriceParamNames{pp})=GEpricepath(GEpriceindexes(pp,1):GEpriceindexes(pp,2),:);
    end
    GENames=fieldnames(GeneralEqmEqns);
    for gg=1:length(GENames)
        PriceHistory.(GENames{gg})=GEcondnpath(gg,:);
    end
end


%% varargout
if heteroagentoptions.pricehistory==0
    if nargout==1
        varargout={p_eqm};
    elseif nargout==2
        varargout={p_eqm,GeneralEqmConditions};
    elseif nargout==3
        varargout={p_eqm,p_eqm_index,GeneralEqmConditions};
    end
elseif heteroagentoptions.pricehistory==1
    if nargout==1
        varargout={p_eqm};
    elseif nargout==2
        varargout={p_eqm,GeneralEqmConditions};
    elseif nargout==3
        varargout={p_eqm,GeneralEqmConditions,PriceHistory};
    elseif nargout==4
        varargout={p_eqm,p_eqm_index,GeneralEqmConditions,PriceHistory};
    end
end


end
