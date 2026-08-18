function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoS_noa1_e_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetzSemiExo_e_raw (d1, z, e).
% Sophisticated quasi-hyperbolic; no a1prime channel since noa1.
% Vhat/Policy come from the F+beta0*beta*EV argmax; Vunderbar is the F+beta*EV RHS
% GATHERED at that same argmax (not re-maximised), and drives the backward recursion.
% aprimeFn = aprimeFn(d2, a2, z, ...) -- z is the markov z only (not semiz)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

Vhat=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
Nrow=N_d12; % RHS row dimension, for the under-gather
V_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_bothz, n_e, d123_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
        Policy(3,:,:,:,N_j)=ceil(maxindex/N_d12);  % d3
    elseif vfoptions.lowmemory==1 % loop e
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_bothz, special_n_e, d123_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,:,e_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy(3,:,:,e_c,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2 % split: vectorize semiz, outer loop z, inner loop e
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals, a2_grid, z_valblock, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Vhat(:,semizblock,e_c,N_j)=shiftdim(Vtemp,1);
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy(1,:,semizblock,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy(2,:,semizblock,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy(3,:,semizblock,e_c,N_j)=ceil(maxindex/N_d12);
            end
        end
    elseif vfoptions.lowmemory==3 % joint loop over bothz (outer), inner loop e
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, special_n_bothz, special_n_e, d123_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d12);
            end
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    % aprime depends on (d2, a2, current markov z); independent of d3, semiz and e -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_z]

    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;
    % Expand to current_bothz = (semiz, z) with semiz fastest
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2, N_a2, N_bothz]
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz);

    % Integrate over e' first (e is i.i.d. start-of-period); EVpre is [N_a, N_bothz]
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j)); % [current_bothz, bothz_prime]

            EV=EVpre.*shiftdim(pi_bothz',-1); % [N_a, N_bothz_next, N_bothz_current]
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_bothz_current]
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, n_e, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
            entireRHS_hat=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1); % autofills e dimension
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_hat(:,:,:,d3_c)=shiftdim(maxindex,1);

            % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_bothz-1),-1)+shiftdim(N_d12*N_a*N_bothz*(0:1:N_e-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

        end
    elseif vfoptions.lowmemory==1 % loop e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, special_n_e, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3e+beta0beta*repelem(entireEV,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_hat(:,:,e_c,d3_c)=shiftdim(maxindex,1);

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3e+beta*repelem(entireEV,N_d1,1,1);
                maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_bothz-1),-1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

            end
        end
    elseif vfoptions.lowmemory==2 % split: vectorize semiz, outer loop z, inner loop e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

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

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a2_grid, z_valblock, e_val, ReturnFnParamsVec);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*repelem(entireEV_z,N_d1,1,1);
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);

                    % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                    entireRHS_under=ReturnMatrix_d3ze+beta*repelem(entireEV_z,N_d1,1,1);
                    maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_semiz-1),-1);
                    V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint loop over bothz (outer), inner loop e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);

                EV_z=EVpre.*pi_bothz(z_c,:); % [N_a, N_bothz_next]
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2); % [N_a, 1]

                z_part=ceil(z_c/N_semiz); % current z index (bothz = semiz + N_semiz*(z-1))
                EV1=reshape(EV_z(aprimeIndex(:,:,z_part)),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1Index(:,:,z_part)),[N_d2,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=a2primeProbs(:,:,z_part);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, special_n_e, d123_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*repelem(entireEV_z,N_d1,1);
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_hat(:,z_c,e_c,d3_c)=maxindex;

                    % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                    entireRHS_under=ReturnMatrix_d3ze+beta*repelem(entireEV_z,N_d1,1);
                    maxindexfull=maxindex+N_d12*(0:1:N_a-1);
                    V_ford3_under(:,z_c,e_c,d3_c)=entireRHS_under(maxindexfull);

                end
            end
        end
    end

    % Max over d3 and unpack policy
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3lin-1)),[N_a,N_bothz,N_e]);
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

    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz);

    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a, N_bothz]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, n_e, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
            entireRHS_hat=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1); % autofills e dimension
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_hat(:,:,:,d3_c)=shiftdim(maxindex,1);

            % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_bothz-1),-1)+shiftdim(N_d12*N_a*N_bothz*(0:1:N_e-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

        end
    elseif vfoptions.lowmemory==1 % loop e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, special_n_e, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3e+beta0beta*repelem(entireEV,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_hat(:,:,e_c,d3_c)=shiftdim(maxindex,1);

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3e+beta*repelem(entireEV,N_d1,1,1);
                maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_bothz-1),-1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

            end
        end
    elseif vfoptions.lowmemory==2 % split: vectorize semiz, outer loop z, inner loop e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

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

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, [n_semiz,ones(1,length(n_z))], special_n_e, d123_gridvals_val, a2_grid, z_valblock, e_val, ReturnFnParamsVec);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*repelem(entireEV_z,N_d1,1,1);
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    V_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_hat(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);

                    % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                    entireRHS_under=ReturnMatrix_d3ze+beta*repelem(entireEV_z,N_d1,1,1);
                    maxindexfull=maxindex+N_d12*(0:1:N_a-1)+shiftdim(N_d12*N_a*(0:1:N_semiz-1),-1);
                    V_ford3_under(:,semizblock,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);

                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint loop over bothz (outer), inner loop e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                z_part=ceil(z_c/N_semiz);
                EV1=reshape(EV_z(aprimeIndex(:,:,z_part)),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1Index(:,:,z_part)),[N_d2,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=a2primeProbs(:,:,z_part);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, special_n_e, d123_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*repelem(entireEV_z,N_d1,1);
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_hat(:,z_c,e_c,d3_c)=maxindex;

                    % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                    entireRHS_under=ReturnMatrix_d3ze+beta*repelem(entireEV_z,N_d1,1);
                    maxindexfull=maxindex+N_d12*(0:1:N_a-1);
                    V_ford3_under(:,z_c,e_c,d3_c)=entireRHS_under(maxindexfull);

                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12_ind=reshape(Policy_ford3_hat((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy(2,:,:,:,jj)=ceil(d12_ind/N_d1);    % d2

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(d3lin-1)),[N_a,N_bothz,N_e]);
end


end
