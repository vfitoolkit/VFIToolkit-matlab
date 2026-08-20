function [Vhat,Policy2,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoS_nod1_noa1_noz_e_raw(n_d2,n_d3,n_a2,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetSemiExo_nod1_noz_e_raw (nod1, noz, e).
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford3_hat=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_semiz, n_e, d23_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy2(2,:,:,:,N_j)=ceil(d_ind/N_d2);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_semiz, special_n_e, d23_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, special_n_semiz, special_n_e, d23_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy2(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy2(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
            end
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
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
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
            entireRHS_hat=ReturnMatrix_d3+beta0beta*entireEV;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            entireRHS_under=ReturnMatrix_d3+beta*entireEV;
            maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_semiz-1),-1)+shiftdim(N_d2*N_a*N_semiz*(0:1:N_e-1),-2);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtemp,1);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            Policy_ford3_hat(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, special_n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                entireRHS_hat=ReturnMatrix_d3e+beta0beta*entireEV;
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                entireRHS_under=ReturnMatrix_d3e+beta*entireEV;
                maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_semiz-1),-1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                Policy_ford3_hat(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*entireEV_z;
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    entireRHS_under=ReturnMatrix_d3ze+beta*entireEV_z;
                    maxindexfull=maxindex+N_d2*(0:1:N_a-1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    Policy_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy2(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy2(1,:,:,:,N_j)=d2_ind;

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3lin-1)),[N_a,N_semiz,N_e]);
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

    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
            entireRHS_hat=ReturnMatrix_d3+beta0beta*entireEV;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            entireRHS_under=ReturnMatrix_d3+beta*entireEV;
            maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_semiz-1),-1)+shiftdim(N_d2*N_a*N_semiz*(0:1:N_e-1),-2);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtemp,1);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            Policy_ford3_hat(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, special_n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                entireRHS_hat=ReturnMatrix_d3e+beta0beta*entireEV;
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                entireRHS_under=ReturnMatrix_d3e+beta*entireEV;
                maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_semiz-1),-1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                Policy_ford3_hat(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
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
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*entireEV_z;
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    entireRHS_under=ReturnMatrix_d3ze+beta*entireEV_z;
                    maxindexfull=maxindex+N_d2*(0:1:N_a-1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                    Policy_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy2(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy2(1,:,:,:,jj)=d2_ind;

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3lin-1)),[N_a,N_semiz,N_e]);
end


end
