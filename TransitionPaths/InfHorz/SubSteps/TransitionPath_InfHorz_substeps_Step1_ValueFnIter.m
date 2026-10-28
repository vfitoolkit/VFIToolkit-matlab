function [VPath,PolicyIndexesPath,aprimeReferencePath,lsdiag]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,aprimeReferencePath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals, a_grid, z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions)
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

%% Local search: the reference policy, carried across GE iterations
% aprimeReferencePath holds, for each period, the aprime index at the centre of that period's search
% window -- the answer the previous sweep found there. It is filled in BOTH modes, so the object means
% the same thing whichever produced it: from the dispatcher's third output under
% vfoptions.localsearch=1, and converted from Policy under a standard sweep.
% The conversion is where the grid interpolation layer is handled. A standard GI sweep answers with a
% FINE grid point while the reference is coarse, so it is compressed to the NEAREST coarse point --
% the same formula the GI local search raw uses for its own third output, so the two agree. It depends
% on gridinterplayer ONLY, not on divideandconquer, because DC returns Policy in the identical format.
% That is what makes "honour whatever divideandconquer is set to" require no code of its own.
%
% lsdiag reports how far the policy moved from the reference it was given. That statistic decides what
% nlocalsearch a model needs, and at an interior reference |move|<nlocalsearch is ALSO exactly the
% condition that the window did not bind, so the restricted answer is the unrestricted one. It is
% therefore the measurement that says whether this scheme could be made exact per iteration, which is
% what Anderson would need and shooting does not.
lsdiag=[];
% localsearchseed is INTERNAL -- the TPath solver sets it on the standard sweeps of a localsearch
% solve -- and it is defaulted HERE, above its first read further down, because THIS FILE reads it
% itself and so cannot rely on the dispatcher's own default. Step1 is reached from every TPath
% solver, including the ones that never touch local search (Anderson, quasi-Newton, the triangular
% Jacobian build), and those pass a vfoptions that has never heard of the field. Without this a
% perfectly ordinary transition path dies on an unrecognised field name.
% Third time this ordering has bitten in this feature, after transpathoptions.localsearchdiagnostic
% in TransitionPath_InfHorz_shooting and this same field in the dispatcher: a new option must be
% defaulted in EVERY file that reads it, above the read, not just in the file where the bug surfaced.
if ~isfield(vfoptions,'localsearchseed')
    vfoptions.localsearchseed=0;
end
if vfoptions.localsearch==1 && isempty(aprimeReferencePath)
    error('vfoptions.localsearch=1 reached Step1 with no aprimeReferencePath: the first sweep of a solve has to be a standard one, which is what builds it')
end

% Semi-exogenous states have their own backward pass (V and Policy carry the composite bothz=(semiz,z) dimension).
% n_semiz is defaulted here, above its read, because Step1 is reached from callers that never set it.
if ~isfield(vfoptions,'n_semiz')
    vfoptions.n_semiz=0;
end
if prod(vfoptions.n_semiz)>0
    [VPath,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter_SemiExo(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_z,N_e,d_gridvals,a_grid,z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions);
    return
end

if vfoptions.experienceasset>=1
    [VPath,PolicyIndexesPath]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter_ExpAsset(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,vfoptions.setup_experienceasset.n_d1,vfoptions.setup_experienceasset.n_d2,vfoptions.setup_experienceasset.n_a1,vfoptions.setup_experienceasset.n_a2,n_z,n_e,N_z,N_e,d_gridvals, vfoptions.setup_experienceasset.d2_gridvals,vfoptions.setup_experienceasset.a1_gridvals,vfoptions.setup_experienceasset.a2_grid, z_gridvals,e_gridvals,pi_z,pi_e,ReturnFn,vfoptions.setup_experienceasset.aprimeFn,DiscountFactorParamNames,ReturnFnParamNames,vfoptions.setup_experienceasset.aprimeFnParamNames,transpathoptions,vfoptions);
    return
end

if N_z==0 && N_e==0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for tt=1:T-1 %so t=T-i
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz(V,n_d,n_a,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,T-tt)=Policy;
    end
elseif N_z>0 && N_e==0
    N_a=prod(n_a);
    % THE REFERENCE IS INDEXED BY WHATEVER THE SEARCH CONDITIONS ON: (a,z) with no d and (d,a,z) with
    % one, so the leading dimension is 1 or N_d. That single difference is the whole of what
    % conditioning on d means here, and getting it wrong is what kept local search out of a general
    % eqm path with a d -- a [1,N_a,N_z] reference reaches the dispatcher, which rejects its shape and
    % complains about aprimeReferencePolicy rather than about d.
    N_dref=max(prod(n_d),1);
    % Allocated ONLY when a sweep is going to fill it. With a d the array is N_d times what it was,
    % so on a big grid it is hundreds of megabytes -- 51*1001*5*50 doubles on the doPart(5) grid --
    % and every ordinary transition path, which never reads it, was paying for that.
    if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
        aprimeReferencePath=zeros(N_dref,N_a,N_z,T-1,'gpuArray');
    end
    if vfoptions.localsearch==1
        lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
    end
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        if transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end

        if vfoptions.localsearch==1
            refslot=aprimeReferencePath(:,:,:,T-ttr);
        else
            refslot=[];
        end
        [V, Policy, aprimeRefNew]=ValueFnIter_InfHorz_TPath_SingleStep(V,n_d,n_a,n_z,d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);

        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,T-ttr)=Policy;

        if vfoptions.localsearch==1
            % THE EDGE TEST, which is what drives the ratchet on nlocalsearch. The window is
            % recomputed here from the reference this sweep was handed, by the same line the raw uses.
            nls=vfoptions.nlocalsearch;
            loweredge=min(max(refslot-nls,1),N_a-2*nls);
            % Tested on aprimeRefNew, NOT on Policy. aprimeRefNew is each d's argmax WITHIN ITS OWN
            % WINDOW -- the coarse argmax under the interpolation layer, the chosen aprime without one
            % -- so this is a single line whether or not there is a d and whether or not there is an
            % interpolation layer. No Policy channels, no fine-point decode, and no dependence on
            % which of the two legal spellings of a coarse node the raw happened to emit.
            % It also tests the right event. The window can only have changed the answer if the argmax
            % it returned sat on a window END; anywhere strictly inside, the unrestricted argmax was
            % inside the window and was taken.
            hitbottom=(aprimeRefNew==loweredge);
            hittop=(aprimeRefNew==loweredge+2*nls);
            % The CORNER EXCLUSIONS. An answer at the GRID's own edge is a corner solution -- the
            % optimum the household wants is outside the grid -- not evidence that the window is too
            % narrow. Without these a top-of-grid saver, or a household at its borrowing constraint,
            % would sit on a window edge on every iteration forever and ratchet nlocalsearch to its
            % cap, where the search covers the whole grid and the restriction is pure overhead.
            hitbottom=hitbottom & (loweredge>1);
            hittop=hittop & (loweredge+2*nls<N_a);
            hitedge=(hitbottom | hittop);
            lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
            lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
            % WITH A d, COUNTED OVER EVERY d AND NOT ONLY THE CHOSEN ONE. A d whose window truncated
            % it might have won with a wider window, and that loss cannot be seen without computing
            % it, so the count has to include the d values this period rejected.
            % THE MEASUREMENT POINTS THE OTHER WAY AND IS NOT THE REASON. On the labour-supply model
            % in doPart(6) the all-d rate came in at 0.16% against 0.64% at the chosen d, so counting
            % over every d DILUTES the signal rather than inflating it: most d values sit well inside
            % their windows and it is the d that wins that presses against its edge. That is an
            % artefact of a model where the optimal aprime varies smoothly and monotonically in d, so
            % the per-d references are all well centred. Where d has no such ordering -- a discrete
            % occupational or participation choice, anything interacting with aprime through a
            % non-convexity -- a rejected d's optimum can sit far from its reference, its window binds,
            % and it is exactly that d which could have won on a wider window. The chosen-d count is
            % blind to it. So the all-d count is kept for the models it protects, not for this one.
            % The chosen-d rate is still accumulated and reported, as the thing to read when asking
            % whether the restriction bound the REALISED policy.
            if N_dref>1
                dstar=Policy(1,:,:);
                linstar=dstar+N_dref*(0:1:N_a-1)+N_dref*N_a*shiftdim((0:1:N_z-1),-1);
                lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                lsdiag.ntotaldstar=lsdiag.ntotaldstar+numel(dstar);
            end
            % movemax is BOUNDED BY THE WINDOW: at most nls at an interior reference and 2*nls at the
            % grid ends. So it reports what the window ALLOWED, not how far the policy would have
            % moved. Reported for information, never used to set nlocalsearch -- a rule fed by a
            % censored measurement can only ratchet downwards and never recover.
            lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
            aprimeReferencePath(:,:,:,T-ttr)=aprimeRefNew;
        elseif vfoptions.localsearchseed==1
            % A SEED SWEEP: unrestricted, but the dispatcher returned a reference for the next
            % restricted sweep. Every branch of it builds that reference the same way -- the coarse
            % argmax of the unrestricted search, per d where there is a d -- so the reference means
            % one thing whichever sweep produced it.
            % This replaces a conversion from Policy, which could only ever recover the chosen d's
            % aprime and which under the interpolation layer rounded a FINE answer to the nearest
            % coarse point, so it meant something subtly different from the restricted sweep's own
            % first-layer argmax.
            aprimeReferencePath(:,:,:,T-ttr)=aprimeRefNew;
        end
    end
elseif N_z==0 && N_e>0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        % The grid is this period's, the distribution is NEXT period's: pi_e_T(:,t) is the distribution of the
        % e realized IN period t (the same convention as pi_e_J in FHorz), and the expectation in V_t is over
        % e_{t+1}. pi_e_T has T columns, so at t=T-1 this reads column T, which exists.
        if transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr+1);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz_e(V,n_d,n_a,n_e,d_gridvals, a_grid, e_gridvals, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    end
elseif N_z>0 && N_e>0
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end

        if transpathoptions.zpathtrivial==0
            z_gridvals=transpathoptions.z_gridvals_T(:,:,T-ttr);
            pi_z=transpathoptions.pi_z_T(:,:,T-ttr);
        end
        % e: this period's grid, NEXT period's distribution (see the noz branch above)
        if transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr+1);
        end

        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
    end
end


