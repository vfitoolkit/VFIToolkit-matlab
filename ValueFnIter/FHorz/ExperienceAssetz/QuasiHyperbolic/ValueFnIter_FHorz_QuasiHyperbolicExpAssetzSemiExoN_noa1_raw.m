function [Vtilde,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoN_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetzSemiExo_raw (d1, z, noe).
% Naive quasi-hyperbolic; no a1prime channel since noa1.
% Vtilde/Policy come from the F+beta0*beta*EV argmax; Valt/Policyalt from the F+beta*EV argmax
% (the exponential value, which drives the backward recursion).
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

Valt=zeros(N_a,N_bothz,N_j,'gpuArray');
Vtilde=zeros(N_a,N_bothz,N_j,'gpuArray');
Policyalt=zeros(3,N_a,N_bothz,N_j,'gpuArray');
Policy=zeros(3,N_a,N_bothz,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
V_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, n_bothz, d123_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policyalt(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policyalt(2,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
        Policyalt(3,:,:,N_j)=ceil(maxindex/N_d12);  % d3
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, [n_semiz,ones(1,length(n_z))], d123_gridvals, a2_grid, z_valblock, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,semizblock,N_j)=shiftdim(Vtemp,1);
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policyalt(1,:,semizblock,N_j)=rem(d12_ind-1,N_d1)+1;
            Policyalt(2,:,semizblock,N_j)=ceil(d12_ind/N_d1);
            Policyalt(3,:,semizblock,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d, n_a2, special_n_bothz, d123_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            Valt(:,z_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policyalt(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policyalt(2,:,z_c,N_j)=ceil(d12_ind/N_d1);
            Policyalt(3,:,z_c,N_j)=ceil(maxindex/N_d12);
        end
    end
    % Terminal period: no continuation, so the QH-perceived value equals the exponential one
    Vtilde(:,:,N_j)=Valt(:,:,N_j);
    Policy(:,:,:,N_j)=Policyalt(:,:,:,N_j);
else
    % aprime depends on (d2, a2, current markov z); independent of d3 and semiz -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_z]

    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;
    % Expand to current_bothz = (semiz, z) with semiz fastest
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2, N_a2, N_bothz]
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j)); % [current_bothz, bothz_prime]

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1); % [N_a, N_bothz_next, N_bothz_current]
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_bothz_current]
            EV_2D=reshape(EV,[N_a,N_bothz]);

            % Lookup at aprime positions using linear indexing into EV_2D
            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            % alt (exponential): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex,1);

            % tilde (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex,1);

        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, [n_semiz,ones(1,length(n_z))], d123_gridvals_val, a2_grid, z_valblock, ReturnFnParamsVec);

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

                % alt (exponential): F + beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_alt(:,semizblock,d3_c)=shiftdim(maxindex,1);

                % tilde (QH-perceived): F + beta0*beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_tilde(:,semizblock,d3_c)=shiftdim(maxindex,1);

            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, d123_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_bothz(z_c,:); % [N_a, N_bothz_next]
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2); % [N_a, 1]

                z_part=ceil(z_c/N_semiz); % current z index (bothz = semiz + N_semiz*(z-1))
                EV1=reshape(EV_z(aprimeIndex(:,:,z_part)),[N_d2,N_a2]);
                EV2=reshape(EV_z(aprimeplus1Index(:,:,z_part)),[N_d2,N_a2]);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=a2primeProbs(:,:,z_part);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                % alt (exponential): F + beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex;

                % tilde (QH-perceived): F + beta0*beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex;

            end
        end
    end

    % Max over d3 and unpack policy
    % Max over d3 (alt, exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,N_j)=V_jj;
    Policyalt(3,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policyalt(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policyalt(2,:,:,N_j)=ceil(d12_ind/N_d1);    % d2

    % Max over d3 (tilde, QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy(2,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
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

    EVpre=Valt(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, d123_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec);

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

            % alt (exponential): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta*repelem(entireEV,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_alt(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_alt(:,:,d3_c)=shiftdim(maxindex,1);

            % tilde (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta0beta*repelem(entireEV,N_d1,1,1);
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_tilde(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_tilde(:,:,d3_c)=shiftdim(maxindex,1);

        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, [n_semiz,ones(1,length(n_z))], d123_gridvals_val, a2_grid, z_valblock, ReturnFnParamsVec);

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

                % alt (exponential): F + beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_alt(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_alt(:,semizblock,d3_c)=shiftdim(maxindex,1);

                % tilde (QH-perceived): F + beta0*beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_tilde(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_tilde(:,semizblock,d3_c)=shiftdim(maxindex,1);

            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, d123_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

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

                % alt (exponential): F + beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_alt(:,z_c,d3_c)=Vtemp;
                Policy_ford3_alt(:,z_c,d3_c)=maxindex;

                % tilde (QH-perceived): F + beta0*beta*EV
                entireRHS_d3z=ReturnMatrix_d3z+beta0beta*repelem(entireEV_z,N_d1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_tilde(:,z_c,d3_c)=Vtemp;
                Policy_ford3_tilde(:,z_c,d3_c)=maxindex;

            end
        end
    end

    % Max over d3 (alt, exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],3);
    Valt(:,:,jj)=V_jj;
    Policyalt(3,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policyalt(1,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policyalt(2,:,:,jj)=ceil(d12_ind/N_d1);    % d2

    % Max over d3 (tilde, QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],3);
    Vtilde(:,:,jj)=V_jj;
    Policy(3,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy(1,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy(2,:,:,jj)=ceil(d12_ind/N_d1);    % d2
end


end
