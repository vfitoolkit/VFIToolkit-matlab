function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_noa1_e_raw(n_d1,n_d2,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated QH + ExpAssetze (z+e dep aprimeFn), baseline (no DC, no GI), no a1.
% Experience asset a2 is the only endogenous state, so d2 is the only choice.
% lowmemory=0 full vectorization; lowmemory=1 loops e; lowmemory=2 nested z+e.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a2=prod(n_a2);
N_a=N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

Vhat=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d, rest of dimensions a,z,e

%%
d2_gridvals=gpuArray(d2_gridvals);
a2_grid=gpuArray(a2_grid);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_z=ones(1,length(n_z));
end

%% j=N_j (terminal)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, special_n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, special_n_e, d_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex is [N_d2*N_a2*N_z*N_e,1], a2primeProbs is [N_d2,N_a2,N_z,N_e]
    % l_a2==2: a2primeIndex/a2primeProbs are [l_a2, N_d2*N_a2*N_z*N_e] (per-dim factored, raveled)

    EVpre=sum(shiftdim(pi_e_J(:,N_j+1),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % Integrate out eprime first

    if length(n_a2)==1
        a2primeProbs=repmat(a2primeProbs,1,1,1,1,N_z);  % [N_d2,N_a2,N_z,N_e,N_z]   (replicate over zprime)

        Vlower=reshape(EVpre(a2primeIndex,:),  [N_d2,N_a2,N_z,N_e,N_z]); % (d2,a2,z_cur,e_cur,zprime)
        Vupper=reshape(EVpre(a2primeIndex+1,:),[N_d2,N_a2,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0;

        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper;
    else
        % l_a2==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a2_1=n_a2(1);
        loIdx_1=a2primeIndex(1,:)';
        loIdx_2=a2primeIndex(2,:)';
        prob_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_z,N_e]);
        prob_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_z,N_e]);

        aprime_ll= loIdx_1   +n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh= loIdx_1   +n_a2_1* loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1* loIdx_2;
        V_ll=reshape(EVpre(aprime_ll,:),[N_d2,N_a2,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl,:),[N_d2,N_a2,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh,:),[N_d2,N_a2,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh,:),[N_d2,N_a2,N_z,N_e,N_z]);

        p1_loy=repmat(prob_1,1,1,1,1,N_z); p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=repmat(prob_1,1,1,1,1,N_z); p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=repmat(prob_2,1,1,1,1,N_z); p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,5),[N_d2,N_a2,N_z,N_e]); % sum zprime -> (d2,a2,z_cur,e_cur) -- already in ReturnMatrix dim order, no permute needed

    entireEV=repelem(EV,N_d1,1,1,1); % aprimeFn only depends on d2, so expand over d1 to match ReturnMatrix

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        entireRHS_hat=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);
        entireRHS_under=ReturnMatrix+beta*entireEV;
        maxindexfull=maxindex+N_d*(0:1:N_a-1)+shiftdim(N_d*N_a*(0:1:N_z-1),-1)+shiftdim(N_d*N_a*N_z*(0:1:N_e-1),-2);
        Vunderbar(:,:,:,N_j)=shiftdim(entireRHS_under(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            entireEV_e=entireEV(:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, special_n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            entireRHS_hat_e=ReturnMatrix_e+beta0beta*entireEV_e;
            [Vtemp,maxindex]=max(entireRHS_hat_e,[],1);
            Vhat(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
            entireRHS_under_e=ReturnMatrix_e+beta*entireEV_e;
            maxindexfull=maxindex+N_d*(0:1:N_a-1)+shiftdim(N_d*N_a*(0:1:N_z-1),-1);
            Vunderbar(:,:,e_c,N_j)=shiftdim(entireRHS_under_e(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                entireEV_ze=entireEV(:,:,z_c,e_c);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, special_n_e, d_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
                entireRHS_hat_ze=ReturnMatrix_ze+beta0beta*entireEV_ze;
                [Vtemp,maxindex]=max(entireRHS_hat_ze,[],1);
                Vhat(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
                entireRHS_under_ze=ReturnMatrix_ze+beta*entireEV_ze;
                maxindexfull=maxindex+N_d*(0:1:N_a-1);
                Vunderbar(:,z_c,e_c,N_j)=entireRHS_under_ze(maxindexfull);
            end
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,1); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex is [N_d2*N_a2*N_z*N_e,1], a2primeProbs is [N_d2,N_a2,N_z,N_e]
    % l_a2==2: a2primeIndex/a2primeProbs are [l_a2, N_d2*N_a2*N_z*N_e] (per-dim factored, raveled)

    EVpre=sum(shiftdim(pi_e_J(:,jj+1),-2).*Vunderbar(:,:,:,jj+1),3); % Integrate out eprime first (sophisticated: continuation is Vunderbar)

    if length(n_a2)==1
        a2primeProbs=repmat(a2primeProbs,1,1,1,1,N_z);  % [N_d2,N_a2,N_z,N_e,N_z]   (replicate over zprime)

        Vlower=reshape(EVpre(a2primeIndex,:),  [N_d2,N_a2,N_z,N_e,N_z]); % (d2,a2,z_cur,e_cur,zprime)
        Vupper=reshape(EVpre(a2primeIndex+1,:),[N_d2,N_a2,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        a2primeProbs(skipinterp)=0;

        EV=a2primeProbs.*Vlower+(1-a2primeProbs).*Vupper;
    else
        % l_a2==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a2_1=n_a2(1);
        loIdx_1=a2primeIndex(1,:)';
        loIdx_2=a2primeIndex(2,:)';
        prob_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_z,N_e]);
        prob_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_z,N_e]);

        aprime_ll= loIdx_1   +n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh= loIdx_1   +n_a2_1* loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1* loIdx_2;
        V_ll=reshape(EVpre(aprime_ll,:),[N_d2,N_a2,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl,:),[N_d2,N_a2,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh,:),[N_d2,N_a2,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh,:),[N_d2,N_a2,N_z,N_e,N_z]);

        p1_loy=repmat(prob_1,1,1,1,1,N_z); p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=repmat(prob_1,1,1,1,1,N_z); p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=repmat(prob_2,1,1,1,1,N_z); p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,5),[N_d2,N_a2,N_z,N_e]); % sum zprime -> (d2,a2,z_cur,e_cur) -- already in ReturnMatrix dim order, no permute needed

    entireEV=repelem(EV,N_d1,1,1,1); % aprimeFn only depends on d2, so expand over d1 to match ReturnMatrix

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
        entireRHS_hat=ReturnMatrix+beta0beta*entireEV;
        [Vtemp,maxindex]=max(entireRHS_hat,[],1);
        Vhat(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);
        entireRHS_under=ReturnMatrix+beta*entireEV;
        maxindexfull=maxindex+N_d*(0:1:N_a-1)+shiftdim(N_d*N_a*(0:1:N_z-1),-1)+shiftdim(N_d*N_a*N_z*(0:1:N_e-1),-2);
        Vunderbar(:,:,:,jj)=shiftdim(entireRHS_under(maxindexfull),1);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            entireEV_e=entireEV(:,:,:,e_c);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, n_z, special_n_e, d_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
            entireRHS_hat_e=ReturnMatrix_e+beta0beta*entireEV_e;
            [Vtemp,maxindex]=max(entireRHS_hat_e,[],1);
            Vhat(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
            entireRHS_under_e=ReturnMatrix_e+beta*entireEV_e;
            maxindexfull=maxindex+N_d*(0:1:N_a-1)+shiftdim(N_d*N_a*(0:1:N_z-1),-1);
            Vunderbar(:,:,e_c,jj)=shiftdim(entireRHS_under_e(maxindexfull),1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                entireEV_ze=entireEV(:,:,z_c,e_c);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn,[n_d1,n_d2], n_a2, special_n_z, special_n_e, d_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec); % with only the experience asset, can just use Case2 command
                entireRHS_hat_ze=ReturnMatrix_ze+beta0beta*entireEV_ze;
                [Vtemp,maxindex]=max(entireRHS_hat_ze,[],1);
                Vhat(:,z_c,e_c,jj)=Vtemp;
                Policy(:,z_c,e_c,jj)=maxindex;
                entireRHS_under_ze=ReturnMatrix_ze+beta*entireEV_ze;
                maxindexfull=maxindex+N_d*(0:1:N_a-1);
                Vunderbar(:,z_c,e_c,jj)=entireRHS_under_ze(maxindexfull);
            end
        end
    end

end

Policy=shiftdim(Policy,-1);

end
