function [V,Policy]=ValueFnIter_FHorz_ExpAssetze_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); %first dim indexes the optimal choice for d and a1prime rest of dimensions a,z

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

if vfoptions.lowmemory==1
    special_n_e=ones(1,length(n_e));
elseif vfoptions.lowmemory==2
    special_n_e=ones(1,length(n_e));
    special_n_z=ones(1,length(n_z));
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
        %Calc the max and its index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        Policy(:,:,:,N_j)=maxindex;
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
            %Calc the max and its index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            Policy(:,:,e_c,N_j)=maxindex;
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2, special_n_z, special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                % Calc the max and its index
                [Vtemp,maxindex]=max(ReturnMatrix_ze);
                V(:,z_c,e_c,N_j)=Vtemp;
                Policy(:,z_c,e_c,N_j)=maxindex;
            end
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EVpre=sum(shiftdim(pi_e_J(:,N_j),-2).*reshape(vfoptions.V_Jplus1,[N_a,N_z,N_e]),3); % Integrate out eprime first

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex/a2primeProbs are [N_d2,N_a2,N_z,N_e] (legacy lower-corner)
    % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z,N_e] (per-dim factored)

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z); % [N_d2*N_a1,N_a2,N_z,N_e,N_z]   (replicate over zprime)

        Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;

        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a2==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);

        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e);
        aprime_ll=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    EV=EV.*reshape(pi_z_J(:,:,N_j),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_z,N_e]); % sum zprime -> (d2*a1prime,a2,z_cur,e_cur)

    DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1,1);

    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0

        entireRHS=ReturnMatrix+DiscountedEV;

        %Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,N_j)=shiftdim(Vtemp,1);
        Policy(:,:,:,N_j)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

            entireRHS=ReturnMatrix_e+DiscountedEV(:,:,:,e_c);

            % Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,N_j)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,N_j)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                DiscountedEV_z=DiscountedEV(:,:,z_c,e_c);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                entireRHS=ReturnMatrix_ze+DiscountedEV_ze;

                % Calc the max and its index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V(:,z_c,e_c,N_j)=shiftdim(Vtemp,1);
                Policy(:,z_c,e_c,N_j)=shiftdim(maxindex,1);
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
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % l_a2==1: a2primeIndex/a2primeProbs are [N_d2,N_a2,N_z,N_e] (legacy lower-corner)
    % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z,N_e] (per-dim factored)

    EVpre=sum(shiftdim(pi_e_J(:,jj),-2).*V(:,:,:,jj+1),3); % Integrate out eprime first

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1,1); % [N_d2*N_a1,N_a2,N_z,N_e]
        aprimeProbs=repmat(a2primeProbs,N_a1,1,1,1,N_z); % [N_d2*N_a1,N_a2,N_z,N_e,N_z]   (replicate over zprime)

        Vlower=reshape(EVpre(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        Vupper=reshape(EVpre(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        skipinterp=(Vlower==Vupper);
        aprimeProbs(skipinterp)=0;

        EV=aprimeProbs.*Vlower+(1-aprimeProbs).*Vupper;
    else
        % l_a2==2: bilinear nested 2-corner interp with per-contribution NaN cleanup
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]);
        prob_1_exp=repmat(reshape(a2primeProbs(1,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);
        prob_2_exp=repmat(reshape(a2primeProbs(2,:,:,:,:),[N_d2,N_a2,N_z,N_e]),N_a1,1,1,1,N_z);

        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2,N_z,N_e);
        aprime_ll=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1,1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat( loIdx_1   +n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1* loIdx_2   -1,N_a1,1,1,1);
        V_ll=reshape(EVpre(aprime_ll(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hl=reshape(EVpre(aprime_hl(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_lh=reshape(EVpre(aprime_lh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);
        V_hh=reshape(EVpre(aprime_hh(:),:),[N_d2*N_a1,N_a2,N_z,N_e,N_z]);

        p1_loy=prob_1_exp; p1_loy(V_ll==V_hl)=0;
        c_ll=p1_loy   .*V_ll; c_ll(isnan(c_ll))=0;
        c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
        EV_loy=c_ll+c_hl;
        p1_hiy=prob_1_exp; p1_hiy(V_lh==V_hh)=0;
        c_lh=p1_hiy   .*V_lh; c_lh(isnan(c_lh))=0;
        c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
        EV_hiy=c_lh+c_hh;
        p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
        c_loy=p2   .*EV_loy; c_loy(isnan(c_loy))=0;
        c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
        EV=c_loy+c_hiy;
    end

    EV=EV.*reshape(pi_z_J(:,:,jj),[1,1,N_z,1,N_z]); % pi[z_cur,z_prime] reshaped to broadcast: z_cur at dim 3, z_prime at dim 5
    EV(isnan(EV))=0; % remove nan created where value fn is -Inf but probability is zero
    EV=reshape(sum(EV,5),[N_d2*N_a1,N_a2,N_z,N_e]); % sum zprime -> (d2*a1prime,a2,z_cur,e_cur)

    DiscountedEV=DiscountFactorParamsVec*repelem(EV,N_d1,N_a1,1,1);

    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0

        entireRHS=ReturnMatrix+DiscountedEV;

        % Calc the max and its index
        [Vtemp,maxindex]=max(entireRHS,[],1);

        V(:,:,:,jj)=shiftdim(Vtemp,1);
        Policy(:,:,:,jj)=shiftdim(maxindex,1);

    elseif vfoptions.lowmemory==1

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

            entireRHS=ReturnMatrix_e+DiscountedEV(:,:,:,e_c);

            % Calc the max and its index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V(:,:,e_c,jj)=shiftdim(Vtemp,1);
            Policy(:,:,e_c,jj)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                DiscountedEV_ze=DiscountedEV(:,:,:,:,:,:,z_c,e_c);
                DiscountedEV_z=DiscountedEV(:,:,z_c,e_c);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d2,n_a1,n_a1,n_a2,special_n_z,special_n_e, d_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                entireRHS=ReturnMatrix_ze+DiscountedEV_ze;

                % Calc the max and its index
                [Vtemp,maxindex]=max(entireRHS,[],1);

                V(:,z_c,e_c,jj)=shiftdim(Vtemp,1);
                Policy(:,z_c,e_c,jj)=shiftdim(maxindex,1);
            end
        end
    end
end

Policy=shiftdim(Policy,-1);



end
