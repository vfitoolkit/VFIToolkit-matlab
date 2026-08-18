function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoS_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, z, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

Vhat=zeros(N_a,N_bothz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy=zeros(3,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
Nrow=N_d2; % RHS row dimension, for the under-gather
V_ford3_hat=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy(1,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy(2,:,:,N_j)=ceil(d_ind/N_d2); % d3
        Policy(3,:,:,N_j)=ceil(maxindex/N_d23); % a1prime
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,semizblock,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy(1,:,semizblock,N_j)=rem(d_ind-1,N_d2)+1;
            Policy(2,:,semizblock,N_j)=ceil(d_ind/N_d2);
            Policy(3,:,semizblock,N_j)=ceil(maxindex/N_d23);
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Vhat(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy(2,:,z_c,N_j)=ceil(d_ind/N_d2);
            Policy(3,:,z_c,N_j)=ceil(maxindex/N_d23);
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    % aprime depends on (d2, a1, a2, current_z); independent of d3 and semiz -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_z]

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1, N_a2, N_z]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1, N_a2, N_bothz]
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

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

            % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
            entireRHS_hat=ReturnMatrix_d3+beta0beta*repelem(entireEV,1,N_a1,1);
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_hat(:,:,d3_c)=shiftdim(maxindex,1);

            % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_d3+beta*repelem(entireEV,1,N_a1,1);
            maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_bothz-1),-1);
            V_ford3_under(:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,0,0);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;
                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,1,N_a1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_hat(:,semizblock,d3_c)=shiftdim(maxindex,1);

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,1,N_a1,1);
                maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_semiz-1),-1);
                V_ford3_under(:,semizblock,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=V_Jplus1.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                z_part=ceil(z_c/N_semiz);
                aprime_slice=aprimeIndex(:,:,z_part);
                aprimeplus1_slice=aprimeplus1Index(:,:,z_part);
                aprimeProbs_slice=aprimeProbs_d2a1a2z(:,:,z_part);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,1,N_a1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,z_c,d3_c)=Vtemp;
                Policy_ford3_hat(:,z_c,d3_c)=maxindex;

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,1,N_a1);
                maxindexfull=maxindex+N_d2*(0:1:N_a-1);
                V_ford3_under(:,z_c,d3_c)=entireRHS_under(maxindexfull);

            end
        end
    end

    % Max over d3 and unpack policy
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,N_j)=V_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2a1prime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy(1,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy(3,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
end

%% Iterate backwards through j
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    EVpre=Vunderbar(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

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

            % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
            entireRHS_hat=ReturnMatrix_d3+beta0beta*repelem(entireEV,1,N_a1,1);
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_hat(:,:,d3_c)=shiftdim(maxindex,1);

            % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_d3+beta*repelem(entireEV,1,N_a1,1);
            maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_bothz-1),-1);
            V_ford3_under(:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,0,0);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;
                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,1,N_a1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_hat(:,semizblock,d3_c)=shiftdim(maxindex,1);

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,1,N_a1,1);
                maxindexfull=maxindex+N_d2*(0:1:N_a-1)+shiftdim(N_d2*N_a*(0:1:N_semiz-1),-1);
                V_ford3_under(:,semizblock,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                z_part=ceil(z_c/N_semiz);
                aprime_slice=aprimeIndex(:,:,z_part);
                aprimeplus1_slice=aprimeplus1Index(:,:,z_part);
                aprimeProbs_slice=aprimeProbs_d2a1a2z(:,:,z_part);

                EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_slice;
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,1,N_a1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,z_c,d3_c)=Vtemp;
                Policy_ford3_hat(:,z_c,d3_c)=maxindex;

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3z+beta*repelem(entireEV_z,1,N_a1);
                maxindexfull=maxindex+N_d2*(0:1:N_a-1);
                V_ford3_under(:,z_c,d3_c)=entireRHS_under(maxindexfull);

            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,jj)=V_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2a1prime_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy(1,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1;
    Policy(3,:,:,jj)=ceil(d2a1prime_ind/N_d2);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
end


end
