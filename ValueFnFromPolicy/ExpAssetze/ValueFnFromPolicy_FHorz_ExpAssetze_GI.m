function varargout=ValueFnFromPolicy_FHorz_ExpAssetze_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy with experienceassetze AND grid interpolation layer (vfoptions.gridinterplayer==1).
% experienceassetze: a2prime = aprimeFn(d_expasset, a2, z, e).
% Under GI, Policy carries an L2 fine-grid index for a1prime; lookup is 2x2.
% Requires both N_z>0 and N_e>0.

%% Setup
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3,0);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetze you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_z==0 || N_e==0
    error('ValueFnFromPolicy_FHorz_ExpAssetze_GI: experienceassetze requires both N_z>0 and N_e>0')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_ExpAssetze_GI: experienceassetze requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
l_e=length(vfoptions.n_e);

% noa1 case (n_a is scalar -- experience asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    V=ValueFnFromPolicy_FHorz_ExpAssetze(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end
n_a1=n_a(1:end-vfoptions.experienceassetze);
N_a1=prod(n_a1);
n_a2=n_a(end-vfoptions.experienceassetze+1:end); % last vfoptions.experienceassetze dims are the experience asset
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1;

if isfield(vfoptions,'l_dexperienceassetze')
    l_d2=vfoptions.l_dexperienceassetze;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e+(l_a2>=2))  % the (l_a2>=2) term is the 'whicha' selector slot, which aprimeFn only takes when there are two experience assets
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+(l_a2>=2)+1:end}}; % the first inputs are (d2,a2,z,e), plus the 'whicha' selector when l_a2>=2
else
    aprimeFnParamNames={};
end

N_ze=N_z*N_e;

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1);
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);

%% Strip trailing L2flag channel if present
size_first=l_d+l_a1+1;
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end

%% Reshape Policy to [size_first, N_a, N_z, N_e, N_j] (helper handles (a,z,e) natively)
Policy_k=reshape(Policy,[size_first, N_a, N_z, N_e, N_j]);

%% Extract a1prime lower grid index and L2
% ValueFnIter converts the midpoint to the lower grid index before returning Policy (the adjust
% block at the end of the GI raws), so this row is the lower index and not the midpoint.
a1_lowerind=shiftdim(Policy_k(l_d+1,:,:,:,:),1);
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1);
w_a1_lower=1-w_a1_upper;
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_lowerind;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_lowerind>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);

%% Joint z+e gridvals for ReturnFn
joint_zegridvals_J=zeros(N_z*N_e, l_z+l_e, N_j, 'gpuArray');
for jj=1:N_j
    joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
end

%% V allocation
V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    Policy_slice=Policy_k(:,:,:,:,jj); % [size_first, N_a, N_z, N_e]

    % Step 1: a2primeIndex, a2primeProbs -- helper handles (a, z, e) natively
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetze(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, vfoptions.n_e, 0,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % helper returns [N_a, N_z*N_e] when l_a2==1, and [N_a, l_a2, N_z*N_e] (per-dim factored) when l_a2==2
    if l_a2==1
        a2primeIndex=reshape(a2primeIndex,[N_a,N_z,N_e]);
        a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_e]);
    else
        a2primeIndex=reshape(a2primeIndex,[N_a,l_a2,N_z,N_e]);
        a2primeProbs=reshape(a2primeProbs,[N_a,l_a2,N_z,N_e]);
    end

    % Step 2: ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);

    if jj==N_j
        V(:,:,:,jj)=F_jj;
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3: EVnext -- integrate e' (iid) then z' (markov)
        EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z, 1]
        EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)';
        EVnext(isnan(EVnext))=0;

        % Step 4: 2x2 corner interpolation
        a1l=a1_lower(:,:,:,jj); a1u=a1_upper(:,:,:,jj);
        wa1l=w_a1_lower(:,:,:,jj); wa1u=w_a1_upper(:,:,:,jj);
        zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);
        if l_a2==1
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z,N_e]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z,N_e]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z,N_e]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z,N_e]);
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
        else
            % l_a2==2: a2primeIndex/a2primeProbs are per-dim factored, so the corner count doubles:
            % 2 (a1) x 2 (a2_1) x 2 (a2_2) = 8. Same FLAT weight-product as the l_a2==1 arm above --
            % this file's convention; its GI checks sit at the ULP floor, not exact zero, and it does
            % not use the raws' nested-skipinterp form. One trailing isnan cleanup, as above.
            n_a2_1=n_a2(1);
            lo1=reshape(a2primeIndex(:,1,:,:),[N_a,N_z,N_e]); lo2=reshape(a2primeIndex(:,2,:,:),[N_a,N_z,N_e]);
            pr1=reshape(a2primeProbs(:,1,:,:),[N_a,N_z,N_e]); pr2=reshape(a2primeProbs(:,2,:,:),[N_a,N_z,N_e]);
            EVnext_atpolicy=zeros(size(a1l),'like',EVnext);
            bits=[0 0; 1 0; 0 1; 1 1];
            for c=1:4
                b1=bits(c,1); b2=bits(c,2);
                a2k=(lo1+b1)+n_a2_1*((lo2+b2)-1);
                w2=pr1; if b1==1, w2=1-w2; end
                q2=pr2; if b2==1, q2=1-q2; end
                lin_l=a1l+N_a1*(a2k-1)+zidxoffset; lin_u=a1u+N_a1*(a2k-1)+zidxoffset;
                EV_l=reshape(EVnext(lin_l(:)),[N_a,N_z,N_e]); EV_u=reshape(EVnext(lin_u(:)),[N_a,N_z,N_e]);
                EVnext_atpolicy=EVnext_atpolicy+wa1l.*w2.*q2.*EV_l+wa1u.*w2.*q2.*EV_u;
            end
            EVnext_atpolicy(isnan(EVnext_atpolicy))=0; % zero corner weights times -Inf next-states give NaN
        end
        V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
    end
end

%% Reshape V out of Kron form
V=reshape(V, [n_a, n_z, vfoptions.n_e, N_j]);



varargout={V};

end
