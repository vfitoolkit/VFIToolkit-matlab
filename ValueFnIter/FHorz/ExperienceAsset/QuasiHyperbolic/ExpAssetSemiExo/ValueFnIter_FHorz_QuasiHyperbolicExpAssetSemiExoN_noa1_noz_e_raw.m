function [Vtilde,Policy3,Valt,Policy3alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoN_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetSemiExo_noz_e_raw (d1, noz, e).
% Policy3 stores (d1, d2, d3) -- no a1prime channel since noa1.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Valt=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3alt=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford3_alt=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_semiz, n_e, d123_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy3alt(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3alt(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
        Policy3alt(3,:,:,:,N_j)=ceil(maxindex/N_d12);  % d3
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_semiz, special_n_e, d123_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Valt(:,:,e_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3alt(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3alt(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy3alt(3,:,:,e_c,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, special_n_semiz, special_n_e, d123_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtemp;
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy3alt(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3alt(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy3alt(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d12);
            end
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy3(:,:,:,:,N_j)=Policy3alt(:,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, special_n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                entireRHS_alt=ReturnMatrix_d3e+beta*repelem(entireEV,N_d1,1,1);
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3e+beta0beta*repelem(entireEV,N_d1,1,1);
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,N_j);
                    entireEV_z=entireEV(:,:,z_c);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_alt=ReturnMatrix_d3ze+beta*repelem(entireEV_z,N_d1,1);
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3ze+beta0beta*repelem(entireEV_z,N_d1,1);
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policy3alt(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3alt(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3alt(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy3alt(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3alt(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3alt(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2

end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;

    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            entireRHS_alt=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1);
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, special_n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                entireRHS_alt=ReturnMatrix_d3e+beta*repelem(entireEV,N_d1,1,1);
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3e+beta0beta*repelem(entireEV,N_d1,1,1);
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,jj);
                    entireEV_z=entireEV(:,:,z_c);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_alt=ReturnMatrix_d3ze+beta*repelem(entireEV_z,N_d1,1);
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3ze+beta0beta*repelem(entireEV_z,N_d1,1);
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policy3alt(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_alt((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3alt(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3alt(2,:,:,:,jj)=ceil(d12_ind/N_d1);    % d2

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,jj)=V_jj;
    Policy3(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,jj)=ceil(d12_ind/N_d1);    % d2

end


end
