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
if vfoptions.localsearch==1 && isempty(aprimeReferencePath)
    error('vfoptions.localsearch=1 reached Step1 with no aprimeReferencePath: the first sweep of a solve has to be a standard one, which is what builds it')
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

        error('Not yet implemented')
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_noz(V,n_d,n_a,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,T-tt)=Policy;
    end
elseif N_z>0 && N_e==0
    % aprimechannel is the Policy channel the aprime index sits in: 1 with no d, 2 with one, since d
    % comes first. Only used by the standard-sweep conversion above.
    if n_d(1)==0
        aprimechannel=1;
    else
        aprimechannel=1+length(n_d);
    end
    N_a=prod(n_a);
    if isempty(aprimeReferencePath)
        aprimeReferencePath=zeros(1,N_a,N_z,T-1,'gpuArray');
    end
    if vfoptions.localsearch==1
        lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0;
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
            if vfoptions.gridinterplayer==0
                hitbottom=(Policy(aprimechannel,:,:)==loweredge);
                hittop=(Policy(aprimechannel,:,:)==loweredge+2*nls);
            else
                % Only the window's two extreme COARSE points count: an answer strictly inside them
                % is not evidence that the window is too narrow.
                % Compared as the DECODED FINE POINT, never as the raw channels, because the two
                % localsearch GI encodings spell that point differently. The one-pass raw keeps L2 in
                % [1,1+ngridinterp] and always names the window's top point (loweredge+2n, L2=1); the
                % two-layer raw inherits the standard GI spelling where L2 reaches ngridinterp+2, so
                % the same point can come out as (loweredge+2n-1, L2=ngridinterp+2) instead. A channel
                % test would see one spelling and miss the other. The decode is the same number for
                % both, and reduces to the no-GI test above when ngridinterp is 0.
                ngi=vfoptions.ngridinterp;
                finenow=(1+ngi)*(Policy(aprimechannel,:,:)-1)+Policy(aprimechannel+1,:,:);
                hitbottom=(finenow==(1+ngi)*(loweredge-1)+1);
                hittop=(finenow==(1+ngi)*(loweredge+2*nls-1)+1);
            end
            % The CORNER EXCLUSIONS. An answer at the GRID's own edge is a corner solution -- the
            % optimum the household wants is outside the grid -- not evidence that the window is too
            % narrow. Without these a top-of-grid saver, or a household at its borrowing constraint,
            % would sit on a window edge on every iteration forever and ratchet nlocalsearch to its
            % cap, where the search covers the whole grid and the restriction is pure overhead.
            hitbottom=hitbottom & (loweredge>1);
            hittop=hittop & (loweredge+2*nls<N_a);
            lsdiag.nedge=lsdiag.nedge+gather(sum(hitbottom | hittop,'all'));
            % movemax is BOUNDED BY THE WINDOW: at most nls at an interior reference and 2*nls at the
            % grid ends. So it reports what the window ALLOWED, not how far the policy would have
            % moved. Reported for information, never used to set nlocalsearch -- a rule fed by a
            % censored measurement can only ratchet downwards and never recover.
            lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
            lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
            aprimeReferencePath(:,:,:,T-ttr)=aprimeRefNew;
        else
            % Converted from Policy. That channel is the aprime index without the grid interpolation
            % layer, and with it that channel and the next are the lower coarse point and the L2
            % index, which compress to the nearest coarse point.
            if vfoptions.gridinterplayer==0
                aprimeReferencePath(:,:,:,T-ttr)=Policy(aprimechannel,:,:);
            else
                aprimeReferencePath(:,:,:,T-ttr)=Policy(aprimechannel,:,:)+round((Policy(aprimechannel+1,:,:)-1)/(1+vfoptions.ngridinterp));
            end
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

        if transpathoptions.zpathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr);
        end

        error('Not yet implemented')
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
        if transpathoptions.epathtrivial==0
            e_gridvals=transpathoptions.e_gridvals_T(:,:,T-ttr);
            pi_e=transpathoptions.pi_e_T(:,T-ttr);
        end

        error('Not yet implemented')
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,d_gridvals, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
    end
end


