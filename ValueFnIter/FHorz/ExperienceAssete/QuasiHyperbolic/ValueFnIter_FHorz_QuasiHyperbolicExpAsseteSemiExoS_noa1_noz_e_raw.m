function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoS_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoS_e_raw (d1, noz, e).
% Sophisticated quasi-hyperbolic + ExperienceAssete + SemiExo (d1, no a1).
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a2 is experience asset; no standard endogenous asset a1
% semiz is semi-exog state, e is i.i.d. start-of-period (required); no z
% aprimeFn = aprimeFn(d2, a2, e, ...)   (depends on current e; NOT on z or semiz)
% semiz only (no markov z); e is separate
%
% Sophisticated QH over the same argmax axis the exponential SemiExo ze noa1 (d1) raw maxes over:
%   Policy (and Vhat) come from the  F + beta0*beta*EV  argmax (QH-perceived).
%   Vunderbar is the  F + beta*EV  RHS GATHERED at that same argmax (NOT re-maximised).
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Vunderbar.
%
% Policy stores (d1, d2, d3) -- no a1prime channel since noa1.
%
% lowmemory levels {0,1,2} implemented (shocks: semiz + e iid).


N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d12=N_d1*N_d2;
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

Vhat=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
n_d=[n_d1,n_d2,n_d3];
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

% Row dimension of the per-d3 ReturnMatrix (d1*d2; noa1 so no a1prime), used for the "gather at argmax"
Nrow=N_d12;

% Preallocate per-d3 (hat=QH-perceived argmax, under=beta-RHS gathered at that argmax)
V_ford3_hat=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_hat=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_semiz, n_e, d123_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
        Policy(3,:,:,:,N_j)=ceil(maxindex/N_d12);  % d3
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_semiz, special_n_e, d123_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,:,e_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
            Policy(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);    % d2
            Policy(3,:,:,e_c,N_j)=ceil(maxindex/N_d12);  % d3
        end
    elseif vfoptions.lowmemory==2 % loop semiz, inner e
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, special_n_semiz, special_n_e, d123_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1; % d1
                Policy(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);    % d2
                Policy(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d12);  % d3
            end
        end
    end
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    % aprime depends on (d2, a2, current_e); independent of d3, semiz, z -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1);
    % Note: a2primeIndex is [N_d2*N_a2*N_e,1], whereas a2primeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    aprimeplus1Index=a2primeIndex+1;

    % Integrate over e' first (e is i.i.d. start-of-period); EVpre is [N_a, N_semiz]
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
            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order
            entireEVpart=repelem(entireEV,N_d1,1,1,1);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
            entireRHS_hat=ReturnMatrix_d3+beta0beta*entireEVpart;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_hat(:,:,:,d3_c)=shiftdim(maxindex,1);

            % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_d3+beta*entireEVpart;
            maxindexfull=maxindex+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_semiz-1),-1)+shiftdim(Nrow*N_a*N_semiz*(0:1:N_e-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order
            entireEVpart=repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, special_n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3e+beta0beta*entireEVpart(:,:,:,e_c);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_hat(:,:,e_c,d3_c)=shiftdim(maxindex,1);

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3e+beta*entireEVpart(:,:,:,e_c);
                maxindexfull=maxindex+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_semiz-1),-1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            end
        end
    elseif vfoptions.lowmemory==2 % loop semiz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order
            entireEVpart=repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*entireEVpart(:,:,z_c,e_c);
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);

                    % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                    entireRHS_under=ReturnMatrix_d3ze+beta*entireEVpart(:,:,z_c,e_c);
                    maxindexfull=maxindex+Nrow*(0:1:N_a-1);
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                end
            end
        end
    end

    % Max over d3 (dim 4) using the hat (QH-perceived) values
    [Vhat_jj,d3maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,N_j)=Vhat_jj;
    Policy(3,:,:,:,N_j)=shiftdim(d3maxindex,-1); % d3
    d3maxindex_lin=reshape(d3maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    Vunderbar(:,:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[N_a,N_semiz,N_e]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,1);
    % Note: a2primeIndex is [N_d2*N_a2*N_e,1], whereas a2primeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    aprimeplus1Index=a2primeIndex+1;

    % Continuation value is Vunderbar, integrated over e'
    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a, N_semiz]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order
            entireEVpart=repelem(entireEV,N_d1,1,1,1);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
            entireRHS_hat=ReturnMatrix_d3+beta0beta*entireEVpart;
            [Vtemp,maxindex]=max(entireRHS_hat,[],1);
            V_ford3_hat(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_hat(:,:,:,d3_c)=shiftdim(maxindex,1);

            % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_d3+beta*entireEVpart;
            maxindexfull=maxindex+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_semiz-1),-1)+shiftdim(Nrow*N_a*N_semiz*(0:1:N_e-1),-2);
            V_ford3_under(:,:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order
            entireEVpart=repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, special_n_e, d123_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                entireRHS_hat=ReturnMatrix_d3e+beta0beta*entireEVpart(:,:,:,e_c);
                [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                V_ford3_hat(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_hat(:,:,e_c,d3_c)=shiftdim(maxindex,1);

                % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_d3e+beta*entireEVpart(:,:,:,e_c);
                maxindexfull=maxindex+Nrow*(0:1:N_a-1)+shiftdim(Nrow*N_a*(0:1:N_semiz-1),-1);
                V_ford3_under(:,:,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            end
        end
    elseif vfoptions.lowmemory==2 % loop semiz, inner e
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order
            entireEVpart=repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    % hat (QH-perceived): argmax of F + beta0*beta*EV over dim 1
                    entireRHS_hat=ReturnMatrix_d3ze+beta0beta*entireEVpart(:,:,z_c,e_c);
                    [Vtemp,maxindex]=max(entireRHS_hat,[],1);
                    V_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_hat(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);

                    % under: F + beta*EV GATHERED at the hat-argmax (not re-maximised)
                    entireRHS_under=ReturnMatrix_d3ze+beta*entireEVpart(:,:,z_c,e_c);
                    maxindexfull=maxindex+Nrow*(0:1:N_a-1);
                    V_ford3_under(:,z_c,e_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                end
            end
        end
    end

    % Max over d3 (dim 4) using the hat (QH-perceived) values
    [Vhat_jj,d3maxindex]=max(V_ford3_hat,[],4);
    Vhat(:,:,:,jj)=Vhat_jj;
    Policy(3,:,:,:,jj)=shiftdim(d3maxindex,-1); % d3
    d3maxindex_lin=reshape(d3maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_hat((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[1,N_a,N_semiz,N_e]);
    Policy(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy(2,:,:,:,jj)=ceil(d12_ind/N_d1);    % d2
    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    Vunderbar(:,:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(d3maxindex_lin-1)),[N_a,N_semiz,N_e]);
end


end
