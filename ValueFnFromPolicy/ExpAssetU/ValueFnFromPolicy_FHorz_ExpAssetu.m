function varargout=ValueFnFromPolicy_FHorz_ExpAssetu(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetu (vfoptions.experienceassetu>=1).
% experienceassetu: a2prime = aprimeFn(d_expasset, a2, u) -- depends on iid (between-period) shock u.
% u does NOT enter Policy; pi_u integrates u out when computing E[V'|policy].

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    V=ValueFnFromPolicy_FHorz_ExpAssetu_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end
%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_ExpAssetu_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Setup
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3,0);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetu you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u')
    error('To use experienceassetu you must define vfoptions.n_u')
end
if ~isfield(vfoptions,'u_grid')
    error('To use experienceassetu you must define vfoptions.u_grid')
end
if ~isfield(vfoptions,'pi_u')
    error('To use experienceassetu you must define vfoptions.pi_u')
end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
pi_u=gpuArray(vfoptions.pi_u);
N_u=prod(n_u);
l_u=length(n_u);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_ExpAssetu: experienceassetu requires at least one decision variable (the one driving a2prime)')
end
l_d=length(n_d);
l_a=length(n_a);

% Split a into a1 (standard) and a2 (experience asset).
% noa1 case (n_a is scalar -- experience asset is the only endogenous state): use n_a1=0, N_a1=0
% (toolkit convention; matches StationaryDist_FHorz_ExpAssetu). Note we have to override l_a1=0
% because length(0)=1, not 0. Downstream, the lookup section has explicit `if N_a1==0` branches.
% vfoptions.experienceassetu is the *count* of EAU dims (1 or 2).
l_a2=vfoptions.experienceassetu;
if length(n_a)<=l_a2
    n_a1=0;
    N_a1=0;
    l_a1=0;
else
    n_a1=n_a(1:end-l_a2);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end-l_a2+1:end); % last l_a2 dims are the experience asset(s)
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
% l_a2 already set from vfoptions.experienceassetu above
l_aprime=l_a1; % Policy stores a1prime only (a2prime implicit); 0 in the noa1 case

% Which d affects the experience asset (default: last d only)
if isfield(vfoptions,'l_dexperienceassetu')
    l_d2=vfoptions.l_dexperienceassetu;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);

% aprimeFnParamNames: first inputs are (d_expasset..., a2, u)
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_u+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+(l_a2>=2)+1:end}};
else
    aprimeFnParamNames={};
end

% Combined shock dim for CreateaprimePolicyExperienceAssetu (uses N_ze in place of N_z)
if N_z==0 && N_e==0
    N_ze=0;
elseif N_z>0 && N_e==0
    N_ze=N_z;
elseif N_z==0 && N_e>0
    N_ze=N_e;
else
    N_ze=N_z*N_e;
end

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetu internally)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); % [N_a, l_daprime, N_j]
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
end

%% Reshape Policy to canonical Kron form
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z*N_e, N_j]);
end

%% Build a1prime joint index
if N_z==0 && N_e==0
    a1prime_idx=ones(N_a, N_j, 'gpuArray');
else
    a1prime_idx=ones(N_a, N_ze, N_j, 'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% Joint z+e gridvals for ReturnFn when both present
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% V allocation
if N_z==0 && N_e==0
    V=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    V=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    V=zeros(N_a, N_z, N_j, 'gpuArray');
else
    V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj); % [l_d+l_a1, N_a]
    else
        Policy_slice=Policy_k(:,:,:,jj); % [l_d+l_a1, N_a, N_ze]
    end

    % Step 1: a2primeIndex, a2primeProbs -- helper adds the u dim
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetu(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    if length(n_a2)==2
        % l_a2==2: the helper returns PER-DIM FACTORED output with the a2-dim index right after N_a.
        % Pull out the two lower indexes and their probs, each shaped exactly as the l_a2==1
        % a2primeIndex/a2primeProbs would be, so each branch below can reshape them as it reshapes
        % a2primeIndex and then Kron-fold the four corners of the a2 lattice.
        if N_ze==0
            loIdx_1=reshape(a2primeIndex(:,1,:),[N_a,N_u]);
            loIdx_2=reshape(a2primeIndex(:,2,:),[N_a,N_u]);
            pr_1=reshape(a2primeProbs(:,1,:),[N_a,N_u]);
            pr_2=reshape(a2primeProbs(:,2,:),[N_a,N_u]);
        else
            loIdx_1=reshape(a2primeIndex(:,1,:,:),[N_a,N_ze,N_u]);
            loIdx_2=reshape(a2primeIndex(:,2,:,:),[N_a,N_ze,N_u]);
            pr_1=reshape(a2primeProbs(:,1,:,:),[N_a,N_ze,N_u]);
            pr_2=reshape(a2primeProbs(:,2,:,:),[N_a,N_ze,N_u]);
        end
    end
    % shape:  N_z==0 && N_e==0 -> [N_a, N_u];  else -> [N_a, N_ze, N_u]

    % Step 2: ReturnFn at policy (u does not enter Return)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_z==0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
    elseif N_z==0 && N_e>0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
    elseif N_z>0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end

    if jj==N_j
        if N_z==0 && N_e==0
            V(:,jj)=F_jj;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=F_jj;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=F_jj;
        else
            V(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3: EVnext indexed by (anext, z_from) -- integrate e' (iid) then z' (markov)
        if N_z==0 && N_e==0
            EVnext=V(:,jj+1); % [N_a]
        elseif N_z==0 && N_e>0
            EVnext=sum(V(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: interpolated lookup. Mirrors standard VFI's skipinterp+isnan
        % so that policies landing on infeasible-on-both-sides next-states (where
        % the standard VFI's argmax zeros the contribution via NaN propagation +
        % isnan clear) give the same finite V here. Order matters: sum over u
        % BEFORE pi_z multiplication and isnan clear, so NaN from skipinterp
        % absorbs -Inf at non-skipinterp positions within the same (a,z,z') cell.
        if N_z==0 && N_e==0
            if length(n_a2)==1
                if N_a1==0
                    aprime_low=a2primeIndex;     % [N_a, N_u]
                    aprime_up =a2primeIndex+1;
                else
                    a1p=a1prime_idx(:,jj); % [N_a]
                    aprime_low=a1p+N_a1*(a2primeIndex-1); % [N_a, N_u]
                    aprime_up =a1p+N_a1*(a2primeIndex);
                end
                Vlower=reshape(EVnext(aprime_low(:)),[N_a,N_u]);
                Vupper=reshape(EVnext(aprime_up(:)), [N_a,N_u]);
                a2pPrb=a2primeProbs;
                a2pPrb(Vlower==Vupper)=0; % skipinterp
                EV=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
                EV(a2pPrb==0)=Vupper(a2pPrb==0); % a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(a2pPrb==1)=Vlower(a2pPrb==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum. The pi_u sum below is unchanged.
                n_a2_1=n_a2(1);
                lo1=loIdx_1; lo2=loIdx_2;
                pq_1=pr_1; pq_2=pr_2;
                a2c_ll=lo1+n_a2_1*(lo2-1);
                a2c_hl=(lo1+1)+n_a2_1*(lo2-1);
                a2c_lh=lo1+n_a2_1*lo2;
                a2c_hh=(lo1+1)+n_a2_1*lo2;
                if N_a1==0
                    ap_ll=a2c_ll;
                    ap_hl=a2c_hl;
                    ap_lh=a2c_lh;
                    ap_hh=a2c_hh;
                else
                    a1p=a1prime_idx(:,jj);
                    ap_ll=a1p+N_a1*(a2c_ll-1);
                    ap_hl=a1p+N_a1*(a2c_hl-1);
                    ap_lh=a1p+N_a1*(a2c_lh-1);
                    ap_hh=a1p+N_a1*(a2c_hh-1);
                end
                V_ll=reshape(EVnext(ap_ll(:)),[N_a,N_u]);
                V_hl=reshape(EVnext(ap_hl(:)),[N_a,N_u]);
                V_lh=reshape(EVnext(ap_lh(:)),[N_a,N_u]);
                V_hh=reshape(EVnext(ap_hh(:)),[N_a,N_u]);
                pp_1=pq_1;
                pp_2=pq_2;
                p1_lo=pp_1; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=pp_1; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                p2=pp_2; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                EV=c_lo+c_hi;
            end
            EVw=EV .* shiftdim(pi_u,-1); EVw(isnan(EVw))=0; % zero the zero-weight terms BEFORE summing (after the sum would destroy it)
            EVnext_atpolicy=sum(EVw,2); % sum over u -> [N_a, 1]
            V(:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z==0 && N_e>0
            if length(n_a2)==1
                if N_a1==0
                    aprime_low=a2primeIndex;     % [N_a, N_e, N_u]
                    aprime_up =a2primeIndex+1;
                else
                    a1p=a1prime_idx(:,:,jj); % [N_a, N_e]
                    aprime_low=a1p+N_a1*(a2primeIndex-1); % broadcast -> [N_a, N_e, N_u]
                    aprime_up =a1p+N_a1*(a2primeIndex);
                end
                % EVnext already has pi_e summed out (e is iid); matches standard VFI noz+e EVpre.
                Vlower=reshape(EVnext(aprime_low(:)),[N_a,N_e,N_u]);
                Vupper=reshape(EVnext(aprime_up(:)), [N_a,N_e,N_u]);
                a2pPrb=a2primeProbs;
                a2pPrb(Vlower==Vupper)=0; % skipinterp on pi_e-collapsed EVnext
                EV=a2pPrb.*Vlower+(1-a2pPrb).*Vupper;
                EV(a2pPrb==0)=Vupper(a2pPrb==0); % a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(a2pPrb==1)=Vlower(a2pPrb==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum. The pi_u sum below is unchanged.
                n_a2_1=n_a2(1);
                lo1=loIdx_1; lo2=loIdx_2;
                pq_1=pr_1; pq_2=pr_2;
                a2c_ll=lo1+n_a2_1*(lo2-1);
                a2c_hl=(lo1+1)+n_a2_1*(lo2-1);
                a2c_lh=lo1+n_a2_1*lo2;
                a2c_hh=(lo1+1)+n_a2_1*lo2;
                if N_a1==0
                    ap_ll=a2c_ll;
                    ap_hl=a2c_hl;
                    ap_lh=a2c_lh;
                    ap_hh=a2c_hh;
                else
                    a1p=a1prime_idx(:,:,jj);
                    ap_ll=a1p+N_a1*(a2c_ll-1);
                    ap_hl=a1p+N_a1*(a2c_hl-1);
                    ap_lh=a1p+N_a1*(a2c_lh-1);
                    ap_hh=a1p+N_a1*(a2c_hh-1);
                end
                V_ll=reshape(EVnext(ap_ll(:)),[N_a,N_e,N_u]);
                V_hl=reshape(EVnext(ap_hl(:)),[N_a,N_e,N_u]);
                V_lh=reshape(EVnext(ap_lh(:)),[N_a,N_e,N_u]);
                V_hh=reshape(EVnext(ap_hh(:)),[N_a,N_e,N_u]);
                pp_1=pq_1;
                pp_2=pq_2;
                p1_lo=pp_1; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=pp_1; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                p2=pp_2; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                EV=c_lo+c_hi;
            end
            EVw=EV .* shiftdim(pi_u,-2); EVw(isnan(EVw))=0; % zero the zero-weight terms BEFORE summing (after the sum would destroy it)
            EVnext_atpolicy=sum(EVw,3); % sum over u -> [N_a, N_e]
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z>0 && N_e==0
            % Per-z' interpolation against V(:,:,jj+1) directly (not EVnext, which has pi_z already summed).
            % Hoisted above the l_a2 guard: both arms look up into Vnext with the same z' offset.
            Vnext=V(:,:,jj+1); % [N_a, N_z']
            zprimeoffset=reshape(N_a*gpuArray(0:N_z-1),[1,1,1,N_z]); % [1,1,1,N_z']
            if length(n_a2)==1
                if N_a1==0
                    aprime_low=a2primeIndex;     % [N_a, N_z, N_u]
                    aprime_up =a2primeIndex+1;
                else
                    a1p=a1prime_idx(:,:,jj);
                    aprime_low=a1p+N_a1*(a2primeIndex-1); % broadcast -> [N_a, N_z, N_u]
                    aprime_up =a1p+N_a1*(a2primeIndex);
                end
                Vlower=reshape(Vnext(aprime_low+zprimeoffset),[N_a,N_z,N_u,N_z]);
                Vupper=reshape(Vnext(aprime_up +zprimeoffset),[N_a,N_z,N_u,N_z]);
                a2pPrb4=repmat(a2primeProbs,[1,1,1,N_z]);
                a2pPrb4(Vlower==Vupper)=0; % skipinterp
                EV4=a2pPrb4.*Vlower+(1-a2pPrb4).*Vupper; % [N_a, N_z, N_u, N_z']
                EV4(a2pPrb4==0)=Vupper(a2pPrb4==0); % a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV4(a2pPrb4==1)=Vlower(a2pPrb4==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum. The pi_u sum below is unchanged.
                n_a2_1=n_a2(1);
                lo1=loIdx_1; lo2=loIdx_2;
                pq_1=pr_1; pq_2=pr_2;
                a2c_ll=lo1+n_a2_1*(lo2-1);
                a2c_hl=(lo1+1)+n_a2_1*(lo2-1);
                a2c_lh=lo1+n_a2_1*lo2;
                a2c_hh=(lo1+1)+n_a2_1*lo2;
                if N_a1==0
                    ap_ll=a2c_ll;
                    ap_hl=a2c_hl;
                    ap_lh=a2c_lh;
                    ap_hh=a2c_hh;
                else
                    a1p=a1prime_idx(:,:,jj);
                    ap_ll=a1p+N_a1*(a2c_ll-1);
                    ap_hl=a1p+N_a1*(a2c_hl-1);
                    ap_lh=a1p+N_a1*(a2c_lh-1);
                    ap_hh=a1p+N_a1*(a2c_hh-1);
                end
                V_ll=reshape(Vnext(ap_ll+zprimeoffset),[N_a,N_z,N_u,N_z]);
                V_hl=reshape(Vnext(ap_hl+zprimeoffset),[N_a,N_z,N_u,N_z]);
                V_lh=reshape(Vnext(ap_lh+zprimeoffset),[N_a,N_z,N_u,N_z]);
                V_hh=reshape(Vnext(ap_hh+zprimeoffset),[N_a,N_z,N_u,N_z]);
                pp_1=repmat(pq_1,[1,1,1,N_z]);
                pp_2=repmat(pq_2,[1,1,1,N_z]);
                p1_lo=pp_1; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=pp_1; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                p2=pp_2; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                EV4=c_lo+c_hi;
            end
            EV4=sum(EV4 .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_z, 1, N_z']
            EV4=EV4 .* reshape(pi_z_J(:,:,jj),[1,N_z,1,N_z]); % weight by pi_z(z, z')
            EV4(isnan(EV4))=0;
            EVnext_atpolicy=reshape(sum(EV4,4),[N_a,N_z]); % sum over z'
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        else
            % Pre-collapse pi_e (e iid), keep z' for per-z' skipinterp (matches standard VFI z+e branch).
            % Hoisted above the l_a2 guard: both arms look up into EVpre with the same z' offset.
            EVpre=reshape(sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1),-2), 3),[N_a,N_z]); % [N_a, N_z']
            zprimeoffset=reshape(N_a*gpuArray(0:N_z-1),[1,1,1,1,N_z]); % [1,1,1,1,N_z']
            if length(n_a2)==1
                % a2pi/pp flat [N_a, N_z*N_e, N_u] -> [N_a, N_z, N_e, N_u]
                a2pIdx=reshape(a2primeIndex,[N_a, N_z, N_e, N_u]);
                a2pPrb=reshape(a2primeProbs,[N_a, N_z, N_e, N_u]);
                if N_a1==0
                    aprime_low=a2pIdx;     % [N_a, N_z, N_e, N_u]
                    aprime_up =a2pIdx+1;
                else
                    a1p=reshape(a1prime_idx(:,:,jj),[N_a, N_z, N_e]);
                    aprime_low=a1p+N_a1*(a2pIdx-1); % broadcast a1p over u -> [N_a, N_z, N_e, N_u]
                    aprime_up =a1p+N_a1*(a2pIdx);
                end
                Vlower=reshape(EVpre(aprime_low+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                Vupper=reshape(EVpre(aprime_up +zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                a2pPrb5=repmat(a2pPrb,[1,1,1,1,N_z]);
                a2pPrb5(Vlower==Vupper)=0; % skipinterp
                EV5=a2pPrb5.*Vlower+(1-a2pPrb5).*Vupper; % [N_a, N_z, N_e, N_u, N_z']
                EV5(a2pPrb5==0)=Vupper(a2pPrb5==0); % a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV5(a2pPrb5==1)=Vlower(a2pPrb5==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each level
                % and per-contribution NaN cleanup so a zero weight against an infinite node (0*(-Inf)=NaN)
                % does not poison the sum. The pi_u sum below is unchanged.
                n_a2_1=n_a2(1);
                lo1=reshape(loIdx_1,[N_a, N_z, N_e, N_u]); lo2=reshape(loIdx_2,[N_a, N_z, N_e, N_u]);
                pq_1=reshape(pr_1,[N_a, N_z, N_e, N_u]); pq_2=reshape(pr_2,[N_a, N_z, N_e, N_u]);
                a2c_ll=lo1+n_a2_1*(lo2-1);
                a2c_hl=(lo1+1)+n_a2_1*(lo2-1);
                a2c_lh=lo1+n_a2_1*lo2;
                a2c_hh=(lo1+1)+n_a2_1*lo2;
                if N_a1==0
                    ap_ll=a2c_ll;
                    ap_hl=a2c_hl;
                    ap_lh=a2c_lh;
                    ap_hh=a2c_hh;
                else
                    a1p=reshape(a1prime_idx(:,:,jj),[N_a, N_z, N_e]);
                    ap_ll=a1p+N_a1*(a2c_ll-1);
                    ap_hl=a1p+N_a1*(a2c_hl-1);
                    ap_lh=a1p+N_a1*(a2c_lh-1);
                    ap_hh=a1p+N_a1*(a2c_hh-1);
                end
                V_ll=reshape(EVpre(ap_ll+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                V_hl=reshape(EVpre(ap_hl+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                V_lh=reshape(EVpre(ap_lh+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                V_hh=reshape(EVpre(ap_hh+zprimeoffset),[N_a,N_z,N_e,N_u,N_z]);
                pp_1=repmat(pq_1,[1,1,1,1,N_z]);
                pp_2=repmat(pq_2,[1,1,1,1,N_z]);
                p1_lo=pp_1; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=pp_1; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                p2=pp_2; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                EV5=c_lo+c_hi;
            end
            EV5=sum(EV5 .* shiftdim(pi_u,-3), 4); % sum over u -> [N_a, N_z, N_e, 1, N_z']
            EV5=EV5 .* reshape(pi_z_J(:,:,jj),[1,N_z,1,1,N_z]); % weight by pi_z(z, z')
            EV5(isnan(EV5))=0;
            EVnext_atpolicy=reshape(sum(EV5,5),[N_a,N_z,N_e]); % sum over z'
            V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
        end
    end
end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_z, N_j]);
else
    V=reshape(V, [n_a, n_z, vfoptions.n_e, N_j]);
end



varargout={V};

end
