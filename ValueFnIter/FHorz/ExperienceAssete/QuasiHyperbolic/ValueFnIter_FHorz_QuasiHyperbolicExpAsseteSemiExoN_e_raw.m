function [V,Policy4,Valt,Policy4alt]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoN_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Naive quasi-hyperbolic + ExperienceAssete + SemiExo.
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (optional), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, e, ...)   (depends on current e; not on z or semiz)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest; e is separate
%
% Naive QH dual pass over the same argmax axis the exponential SemiExo ze raw maxes over:
%   Valt/Policy4alt maximise  F + beta*EV        (the exponential value)
%   V/Policy4       maximise  F + beta0*beta*EV  (the QH-perceived value)
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Valt (the exponential continuation value).
%
% lowmemory levels {0,1,2,3} implemented (shocks: z markov + semiz + e iid).

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy4=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray');
Valt=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy4alt=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_d23=[n_d2,n_d3];

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate per-d3 (alt=exponential, tilde=QH-perceived)
V_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_alt=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
V_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_tilde=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');



%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_bothz,n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Valt(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy4alt(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy4alt(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy4alt(3,:,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy4alt(4,:,:,:,N_j)=ceil(maxindex/N_d); % a1prime
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Valt(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy4alt(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy4alt(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy4alt(3,:,:,e_c,N_j)=ceil(d_ind/N_d12);
            Policy4alt(4,:,:,e_c,N_j)=ceil(maxindex/N_d);
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,0,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,semizblock,e_c,N_j)=shiftdim(Vtemp,1);
                d_ind=rem(maxindex-1,N_d)+1;
                d12_ind=rem(d_ind-1,N_d12)+1;
                Policy4alt(1,:,semizblock,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4alt(2,:,semizblock,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy4alt(3,:,semizblock,e_c,N_j)=ceil(d_ind/N_d12);
                Policy4alt(4,:,semizblock,e_c,N_j)=ceil(maxindex/N_d);
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Valt(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d)+1;
                d12_ind=rem(d_ind-1,N_d12)+1;
                Policy4alt(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4alt(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy4alt(3,:,z_c,e_c,N_j)=ceil(d_ind/N_d12);
                Policy4alt(4,:,z_c,e_c,N_j)=ceil(maxindex/N_d);
            end
        end
    end
    % Terminal period: no continuation, so QH-perceived value equals exponential value
    V(:,:,:,N_j)=Valt(:,:,:,N_j);
    Policy4(:,:,:,:,N_j)=Policy4alt(:,:,:,:,N_j);
else
    % aprime depends on (d2, a1, a2, current_z, current_e); independent of d3, semiz
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    % Integrate over e' first (e is i.i.d. start-of-period); EVpre is [N_a, N_bothz]
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            % alt (exponential): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta*entireEVpart;
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex,1);

            % tilde (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta0beta*entireEVpart;
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);

                % alt (exponential): F + beta*EV
                entireRHS_d3e=ReturnMatrix_d3e+beta*entireEVpart(:,:,:,e_c);
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex,1);

                % tilde (QH-perceived): F + beta0*beta*EV
                entireRHS_d3e=ReturnMatrix_d3e+beta0beta*entireEVpart(:,:,:,e_c);
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                entireEVpart_z=entireEVpart(:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,0,0);

                    % alt (exponential): F + beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta0beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                entireEVpart_z=entireEVpart(:,:,z_c,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);

                    % alt (exponential): F + beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta0beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    % Max over d3 (dim 4) for alt (exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,N_j)=V_jj;
    Policy4alt(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4alt(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4alt(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy4alt(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime

    % Max over d3 (dim 4) for tilde (QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy4(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy4(2,:,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy4(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    % Continuation value is the exponential value (Valt), integrated over e'
    EVpre=sum(Valt(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a, N_bothz]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            % alt (exponential): F + beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta*entireEVpart;
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_alt(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_alt(:,:,:,d3_c)=shiftdim(maxindex,1);

            % tilde (QH-perceived): F + beta0*beta*EV
            entireRHS_d3=ReturnMatrix_d3+beta0beta*entireEVpart;
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_tilde(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_tilde(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0);

                % alt (exponential): F + beta*EV
                entireRHS_d3e=ReturnMatrix_d3e+beta*entireEVpart(:,:,:,e_c);
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_alt(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_alt(:,:,e_c,d3_c)=shiftdim(maxindex,1);

                % tilde (QH-perceived): F + beta0*beta*EV
                entireRHS_d3e=ReturnMatrix_d3e+beta0beta*entireEVpart(:,:,:,e_c);
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_tilde(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_tilde(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % outer z (markov), inner e, vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                entireEVpart_z=entireEVpart(:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,0,0);

                    % alt (exponential): F + beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_alt(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta0beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_tilde(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3 % joint bothz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_bothz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_bothz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_bothz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,bothzcur,e_cur)
            entireEVpart=repelem(entireEV,N_d1,N_a1,1,1);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                entireEVpart_z=entireEVpart(:,:,z_c,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);

                    % alt (exponential): F + beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_alt(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);

                    % tilde (QH-perceived): F + beta0*beta*EV
                    entireRHS_d3ze=ReturnMatrix_d3ze+beta0beta*entireEVpart_z(:,:,:,e_c);
                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_tilde(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    % Max over d3 (dim 4) for alt (exponential)
    [V_jj,maxindex]=max(V_ford3_alt,[],4);
    Valt(:,:,:,jj)=V_jj;
    Policy4alt(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_alt((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4alt(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy4alt(2,:,:,:,jj)=ceil(d12_ind/N_d1);
    Policy4alt(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12);

    % Max over d3 (dim 4) for tilde (QH-perceived)
    [V_jj,maxindex]=max(V_ford3_tilde,[],4);
    V(:,:,:,jj)=V_jj;
    Policy4(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_tilde((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy4(2,:,:,:,jj)=ceil(d12_ind/N_d1);
    Policy4(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12);
end


end
