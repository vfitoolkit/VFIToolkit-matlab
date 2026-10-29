function [VPath,PolicyIndexesPath,aprimeReferencePath,lsdiag]=TransitionPath_FHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath,aprimeReferencePath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_j,N_z,N_e,d_gridvals, a_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions)
% VPath is empty, but I am setting it up so that it can be included as an option later on.
VPath=[];

%% Local search: the reference policy, carried across GE iterations
% As in TransitionPath_InfHorz_substeps_Step1_ValueFnIter, with an age dimension. aprimeReferencePath
% holds, for each (j,t), the aprime index at the centre of that window -- the coarse argmax the previous
% sweep found there, per d where there is a d: [N_dref,N_a,N_z,N_j,T-1].
% STORED AS uint16, because with the age dimension it is the one big new array: 4.3 GB as double at
% N_d=9, N_a=1001, N_z=5, N_j=81, T=150, against 1.1 GB as uint16. Every entry is a grid index <= N_a,
% and TransitionPath_Case1_FHorz refuses N_a>65535. It is CAST TO DOUBLE on the one read below: uint16
% arithmetic saturates at 0 and 65535, so ref-n in the window line would silently clip rather than go
% negative.
% lsdiag is the per-sweep edge count that drives the ratchet in TransitionPath_FHorz_shooting.
lsdiag=[];
% Both defaulted HERE, above their first read: Step1 is reached from every FHorz TPath solver, and from
% the PType and MixHorz ones, which pass a vfoptions that has never heard of these fields.
if ~isfield(vfoptions,'localsearch')
    vfoptions.localsearch=0;
end
if ~isfield(vfoptions,'localsearchseed')
    vfoptions.localsearchseed=0;
end
if vfoptions.localsearch==1 && isempty(aprimeReferencePath)
    error('vfoptions.localsearch=1 reached Step1 with no aprimeReferencePath: the first sweep of a solve has to be a standard one, which is what builds it')
end

if vfoptions.experienceasset>=1
    [VPath,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_ExpAsset(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,vfoptions.setup_experienceasset.n_d1,vfoptions.setup_experienceasset.n_d2,vfoptions.setup_experienceasset.n_a1,vfoptions.setup_experienceasset.n_a2,n_z,n_e,N_j,N_z,N_e,d_gridvals, vfoptions.setup_experienceasset.d2_gridvals,vfoptions.setup_experienceasset.a1_gridvals,vfoptions.setup_experienceasset.a2_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,vfoptions.setup_experienceasset.aprimeFn,DiscountFactorParamNames,ReturnFnParamNames,vfoptions.setup_experienceasset.aprimeFnParamNames,transpathoptions,vfoptions);
    return
elseif vfoptions.experienceassetz>=1 % experienceassetz reuses the setup_experienceasset struct
    [VPath,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_ExpAssetz(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,vfoptions.setup_experienceasset.n_d1,vfoptions.setup_experienceasset.n_d2,vfoptions.setup_experienceasset.n_a1,vfoptions.setup_experienceasset.n_a2,n_z,n_e,N_j,N_z,N_e,d_gridvals, vfoptions.setup_experienceasset.d2_gridvals,vfoptions.setup_experienceasset.a1_gridvals,vfoptions.setup_experienceasset.a2_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,vfoptions.setup_experienceasset.aprimeFn,DiscountFactorParamNames,ReturnFnParamNames,vfoptions.setup_experienceasset.aprimeFnParamNames,transpathoptions,vfoptions);
    return
end

if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    [VPath,PolicyIndexesPath]=TransitionPath_FHorz_substeps_Step1_ValueFnIter_QuasiHyperbolic(T,PolicyIndexesPath,V_final,Parameters,PricePathOld,ParamPath,PricePathSizeVec,ParamPathSizeVec,PricePathNames,ParamPathNames,n_d,n_a,n_z,n_e,N_j,N_z,N_e,d_gridvals, a_grid, z_gridvals_J,e_gridvals_J,pi_z_J,pi_e_J,ReturnFn,DiscountFactorParamNames,ReturnFnParamNames,transpathoptions,vfoptions);
    return
end

if transpathoptions.fastOLG==0
    if N_z==0 && N_e==0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_j;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for tt=1:T-1 %so t=T-i
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-tt)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_noz(V,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,T-tt)=Policy;

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-tt)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-tt)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    elseif N_z>0 && N_e==0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_z*N_j;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.zpathtrivial==0
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep(V,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    elseif N_z==0 && N_e>0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_e*N_j;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.zpathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_noz_e(V,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    elseif N_z>0 && N_e>0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_z*N_e*N_j;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.zpathtrivial==0
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
            end
            if transpathoptions.epathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy;

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    end


    %% fastOLG
elseif transpathoptions.fastOLG==1
    if N_z==0 && N_e==0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_j;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 % so tt=T-ttr
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz(V,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            PolicyIndexesPath(:,:,:,T-ttr)=Policy;

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    elseif N_z>0 && N_e==0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_j*N_z;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.zpathtrivial==0
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(V,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy in fastOLG is [N_a,N_j,N_z] and contains the joint-index for (d,aprime)

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    elseif N_z==0 && N_e>0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_j*N_e;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.zpathtrivial==0
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % fastOLG value function uses (j,z',z)
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_e(V,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy in fastOLG is [N_a,N_j,N_e] and contains the joint-index for (d,aprime)

            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    elseif N_z>0 && N_e>0
        % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
        % Since we won't need to keep the value functions for anything later we just store the current one in V
        % LOCAL SEARCH: the reference path, see the top of this file. Stored FLATTENED, [N_dref,N_rest,T-1],
        % in the raws' own Policy order after the leading N_dref: one storage shape and one edge test then
        % serve every layout (the raws reshape it on entry; their dim orders differ between layouts).
        N_a=prod(n_a);
        N_dref=max(prod(n_d),1);
        N_rest=N_a*N_j*N_z*N_e;
        if isempty(aprimeReferencePath) && (vfoptions.localsearch==1 || vfoptions.localsearchseed==1)
            aprimeReferencePath=zeros(N_dref,N_rest,T-1,'uint16','gpuArray'); % allocated only when a sweep will fill it
        end
        if vfoptions.localsearch==1
            lsdiag.movemax=0; lsdiag.nedge=0; lsdiag.ntotal=0; lsdiag.nedgedstar=0; lsdiag.ntotaldstar=0;
        end
        V=V_final;
        for ttr=1:T-1 %so tt=T-ttr

            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end

            if transpathoptions.zpathtrivial==0
                pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr); % [N_j,N_z,l_z,T] (this read transpathoptions.z_gridvals_J(:,:,T-ttr), which does not exist)
            end
            if transpathoptions.epathtrivial==0
                pi_e_J=transpathoptions.pi_e_J_T(:,1,:,T-ttr);
                e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
            end

            if vfoptions.localsearch==1
                refslot=double(aprimeReferencePath(:,:,T-ttr)); % CAST, see top: uint16 arithmetic saturates
            else
                refslot=[];
            end
            [V, Policy, aprimeRefNew]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e(V,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, refslot, vfoptions);
            % The V input is next period value fn, the V output is this period.
            % Policy in fastOLG is [N_a,N_j,N_z,N_e] and contains the joint-index for (d,aprime)

            PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e

            if vfoptions.localsearch==1
                % THE EDGE TEST, which drives the ratchet on nlocalsearch (as in InfHorz). The window is
                % recomputed from the reference this sweep was handed, by the raws' own line, and tested on
                % aprimeRefNew (each d's argmax within its own window), never on Policy. An answer at the
                % GRID's own edge is a corner solution, not a too-narrow window, so it does not count.
                aprimeRefNew=reshape(aprimeRefNew,[N_dref,N_rest]);
                nls=vfoptions.nlocalsearch;
                loweredge=min(max(refslot-nls,1),N_a-2*nls);
                hitbottom=(aprimeRefNew==loweredge) & (loweredge>1);
                hittop=(aprimeRefNew==loweredge+2*nls) & (loweredge+2*nls<N_a);
                hitedge=(hitbottom | hittop);
                lsdiag.nedge=lsdiag.nedge+gather(sum(hitedge,'all'));
                lsdiag.ntotal=lsdiag.ntotal+numel(refslot);
                % Counted over EVERY d (a d whose window truncated it might have won on a wider one); the
                % chosen-d rate is accumulated for the report. Policy is UnKron'd, so with length(n_d)>1
                % the chosen d is spread over the first length(n_d) channels and is re-folded here.
                if N_dref>1
                    dstar=ones(1,N_rest,'gpuArray');
                    dmult=1;
                    for ll=1:length(n_d)
                        dstar=dstar+dmult*(reshape(Policy(ll,:),1,N_rest)-1);
                        dmult=dmult*n_d(ll);
                    end
                    linstar=dstar+N_dref*(0:1:N_rest-1);
                    lsdiag.nedgedstar=lsdiag.nedgedstar+gather(sum(hitedge(linstar),'all'));
                    lsdiag.ntotaldstar=lsdiag.ntotaldstar+N_rest;
                end
                % movemax is BOUNDED BY THE WINDOW: reported, never used to set nlocalsearch
                lsdiag.movemax=max(lsdiag.movemax,gather(max(abs(aprimeRefNew-refslot),[],'all')));
                aprimeReferencePath(:,:,T-ttr)=aprimeRefNew;
            elseif vfoptions.localsearchseed==1
                % A SEED SWEEP: unrestricted, but the dispatcher returned the coarse argmax per d
                aprimeReferencePath(:,:,T-ttr)=reshape(aprimeRefNew,[N_dref,N_rest]);
            end
        end
    end
end
