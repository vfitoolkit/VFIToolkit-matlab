function [Vhat,Policy3,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetSemiExoS_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetSemiExo_raw (d1, z, noe).
% Policy3 stores (d1, d2, d3) -- no a1prime channel since noa1.

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);

Vhat=zeros(N_a,N_bothz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_bothz,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
    special_n_semiz=[n_semiz,ones(1,length(n_z))]; % semiz vectorised, z scalar (lowmemory=1 split over z)
end

V_ford3_hat=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_bothz,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, n_bothz, d123_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
        Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1);
        Policy3(3,:,:,N_j)=ceil(maxindex/N_d12);
    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for z_c=1:N_z
            zind=(1:1:N_semiz)+N_semiz*(z_c-1);
            z_val=bothz_gridvals_J(zind,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, special_n_semiz, d123_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,zind,N_j)=shiftdim(Vtemp,1);
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3(1,:,zind,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,zind,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,zind,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2
        % joint: loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, special_n_bothz, d123_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,z_c,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,z_c,N_j)=ceil(maxindex/N_d12);
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
            entireRHS_hat=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            entireRHS_under=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_bothz-1),-1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtemp,1);
            V_ford3_under(:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            Policy_ford3_hat(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);
            aprimeProbs_full=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_full(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_full+EV2.*(1-aprimeProbs_full);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,N_j);
                entireEV_z=entireEV(:,:,zind);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, d123_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);
                % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1,1);
                maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_semiz-1),-1);
                V_ford3_hat(:,zind,d3_c)=shiftdim(Vtemp,1);
                V_ford3_under(:,zind,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                Policy_ford3_hat(:,zind,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        % joint: loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, d123_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                EV1=reshape(EV_z(aprimeIndex),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2,N_a2]);

                aprimeProbs_d3z=a2primeProbs;
                skipinterp=(EV1==EV2);
                aprimeProbs_d3z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_d3z+EV2.*(1-aprimeProbs_d3z);

                % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1);
                maxindexfull=maxindex+N_d12*(0:1:N_a-1);
                V_ford3_hat(:,z_c,d3_c)=Vtemp;
                V_ford3_under(:,z_c,d3_c)=entireRHS_under(maxindexfull);
                Policy_ford3_hat(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,N_j)=V_jj;
    Policy3(3,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
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

    EVpre=Vunderbar(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);

            aprimeProbs_d3=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
            entireRHS_hat=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            entireRHS_under=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_bothz-1),-1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtemp,1);
            V_ford3_under(:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            Policy_ford3_hat(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        % split: parallelise over semiz, loop over z
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_bothz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_bothz]);
            aprimeProbs_full=repmat(a2primeProbs,1,1,N_bothz);
            skipinterp=(EV1==EV2);
            aprimeProbs_full(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_full+EV2.*(1-aprimeProbs_full);

            for z_c=1:N_z
                zind=(1:1:N_semiz)+N_semiz*(z_c-1);
                z_val=bothz_gridvals_J(zind,:,jj);
                entireEV_z=entireEV(:,:,zind);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, d123_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);
                % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1,1);
                maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_semiz-1),-1);
                V_ford3_hat(:,zind,d3_c)=shiftdim(Vtemp,1);
                V_ford3_under(:,zind,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                Policy_ford3_hat(:,zind,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        % joint: loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, d123_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                EV1=reshape(EV_z(aprimeIndex),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2,N_a2]);

                aprimeProbs_d3z=a2primeProbs;
                skipinterp=(EV1==EV2);
                aprimeProbs_d3z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_d3z+EV2.*(1-aprimeProbs_d3z);

                % hat: argmax at beta0*beta; under: the beta-RHS gathered at that argmax
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1);
                maxindexfull=maxindex+N_d12*(0:1:N_a-1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtemp,1);
                V_ford3_under(:,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                Policy_ford3_hat(:,z_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,jj)=V_jj;
    Policy3(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy3(1,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,jj)=ceil(d12_ind/N_d1);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
end


end
