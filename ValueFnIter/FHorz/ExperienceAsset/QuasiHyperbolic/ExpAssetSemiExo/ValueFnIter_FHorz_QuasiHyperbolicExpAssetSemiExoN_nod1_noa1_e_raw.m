function [Vtilde,Policy2,Valt,Policy2alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoN_nod1_noa1_e_raw(n_d2,n_d3,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetSemiExo_nod1_e_raw (nod1, z, e).
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy2alt=zeros(2,N_a,N_bothz,N_e,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_bothz, n_e, d23_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2alt(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy2alt(2,:,:,:,N_j)=ceil(d_ind/N_d2);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_bothz, special_n_e, d23_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Valt(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2alt(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2alt(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:N_semiz);
            z_val=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, special_n_semiz, special_n_e, d23_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,semizblock,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy2alt(1,:,semizblock,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy2alt(2,:,semizblock,e_c,N_j)=ceil(d_ind/N_d2);
            end
        end
    elseif vfoptions.lowmemory==3
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, special_n_bothz, special_n_e, d23_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy2alt(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy2alt(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
            end
        end
    end
    % Terminal period: no continuation, so the QH-perceived objects equal the exponential ones
    Vtilde(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy2(:,:,:,:,N_j)=Policy2alt(:,:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            entireRHS_alt=ReturnMatrix_d3+beta*entireEV;
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*entireEV;
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, special_n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                entireRHS_alt=ReturnMatrix_d3e+beta*entireEV;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3e+beta0beta*entireEV;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,N_j);
                entireEV_z=entireEV(:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_alt=ReturnMatrix_d3ze+beta*entireEV_z;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3ze+beta0beta*entireEV_z;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,N_j);
                    entireEV_z=entireEV(:,:,z_c);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_bothz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_alt=ReturnMatrix_d3ze+beta*entireEV_z;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=Vtemp_alt;
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=maxindex_alt;
                    entireRHS_tilde=ReturnMatrix_d3ze+beta0beta*entireEV_z;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=Vtemp_tilde;
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=maxindex_tilde;
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policy2alt(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy2alt(1,:,:,:,N_j)=d2_ind;

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,N_j)=V_jj;
    Policy2alt(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy2alt(1,:,:,:,N_j)=d2_ind;

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
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            entireRHS_alt=ReturnMatrix_d3+beta*entireEV;
            [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp_alt,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex_alt,1);
            entireRHS_tilde=ReturnMatrix_d3+beta0beta*entireEV;
            [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp_tilde,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex_tilde,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, special_n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                entireRHS_alt=ReturnMatrix_d3e+beta*entireEV;
                [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex_alt,1);
                entireRHS_tilde=ReturnMatrix_d3e+beta0beta*entireEV;
                [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex_tilde,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,jj);
                entireEV_z=entireEV(:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_alt=ReturnMatrix_d3ze+beta*entireEV_z;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_alt,1);
                    Policy_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(maxindex_alt,1);
                    entireRHS_tilde=ReturnMatrix_d3ze+beta0beta*entireEV_z;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtemp_tilde,1);
                    Policy_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(maxindex_tilde,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,jj);
                    entireEV_z=entireEV(:,:,z_c);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_bothz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_alt=ReturnMatrix_d3ze+beta*entireEV_z;
                    [Vtemp_alt,maxindex_alt]=max(entireRHS_alt,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=Vtemp_alt;
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=maxindex_alt;
                    entireRHS_tilde=ReturnMatrix_d3ze+beta0beta*entireEV_z;
                    [Vtemp_tilde,maxindex_tilde]=max(entireRHS_tilde,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=Vtemp_tilde;
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=maxindex_tilde;
                end
            end
        end
    end

    % Max over d3 (alt)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policy2alt(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy2alt(1,:,:,:,jj)=d2_ind;

    % Max over d3 (tilde)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    Vtilde(:,:,:,jj)=V_jj;
    Policy2(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy2(1,:,:,:,jj)=d2_ind;

end


end
