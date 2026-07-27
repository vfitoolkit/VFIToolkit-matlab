function [V,Policy3]=ValueFnIter_FHorz_ExpAssetsemiz_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% semiz is semi-exog state (drives the asset); no ordinary z, so bothz = semiz
% aprimeFn = aprimeFn(d2, a2, semiz, ...)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_bothz=N_semiz; % no ordinary z

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy storage with separate entries for d1, d2, d3, a1prime
Policy3=zeros(4,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_d23=[n_d2,n_d3];

bothz_gridvals_J=semiz_gridvals_J; % bothz = semiz

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_semiz, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy3(3,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy3(4,:,:,N_j)=ceil(maxindex/N_d); % a1prime
    elseif vfoptions.lowmemory==1
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy3(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,z_c,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,z_c,N_j)=ceil(d_ind/N_d12);
            Policy3(4,:,z_c,N_j)=ceil(maxindex/N_d);
        end
    end
else
    % aprime depends on (d2, a1, a2, current_semiz); independent of d1, d3 -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_semiz]  (N_semiz==N_bothz here)

    aprimeIndex_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1, N_a2, N_bothz]
    aprimeplus1Index_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_full=repmat(a2primeProbs,N_a1,1,1);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,N_j);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                aprime_slice=aprimeIndex_full(:,:,z_c); % bothz==semiz, so z_c indexes semiz directly
                aprimeplus1_slice=aprimeplus1Index_full(:,:,z_c);
                aprimeProbs_slice=aprimeProbs_full(:,:,z_c);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1);

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3 and unpack policy
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,N_j)=V_jj;
    Policy3(3,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy3(4,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index_full=repelem((1:1:N_a1)',N_d2,N_a2,N_semiz)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_full=repmat(a2primeProbs,N_a1,1,1);

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=pi_semiz_J(:,:,d3_c,jj);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                aprime_slice=aprimeIndex_full(:,:,z_c);
                aprimeplus1_slice=aprimeplus1Index_full(:,:,z_c);
                aprimeProbs_slice=aprimeProbs_full(:,:,z_c);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1);

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,jj)=V_jj;
    Policy3(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,jj)=ceil(d12_ind/N_d1);
    Policy3(4,:,:,jj)=ceil(d12a1prime_ind/N_d12);
end


end
