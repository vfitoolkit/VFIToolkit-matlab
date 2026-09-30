function [V,Policy3]=ValueFnIter_FHorz_ExpAssetSemiExo_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

n_bothz=[n_semiz,n_z]; % These are the return function arguments

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz*N_z,N_e,N_j,'gpuArray');
% For semiz it turns out to be easier to go straight to constructing policy that stores d2,d3,a1prime seperately
Policy3=zeros(3,N_a,N_semiz*N_z,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

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

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz*N_z,N_e,N_d3,'gpuArray');


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0

        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_bothz,N_e]; Level=0, Refine=0
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
        Policy3(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy3(2,:,:,:,N_j)=ceil(d_ind/N_d2);
        Policy3(3,:,:,:,N_j)=ceil(maxindex/N_d23);

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0); % [N_d*N_a1,N_a1*N_a2,N_bothz]; Level=0, Refine=0
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1; % Do I need this shiftdim(), can probably delete all these
            Policy3(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy3(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
            Policy3(3,:,:,e_c,N_j)=ceil(maxindex/N_d23);
        end

    elseif vfoptions.lowmemory==2

        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:N_semiz);
            z_val=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,semizblock,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy3(1,:,semizblock,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy3(2,:,semizblock,e_c,N_j)=ceil(d_ind/N_d2);
                Policy3(3,:,semizblock,e_c,N_j)=ceil(maxindex/N_d23);
            end
        end

    elseif vfoptions.lowmemory==3

        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,n_d23,n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                %Calc the max and it's index
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy3(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy3(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
                Policy3(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d23);
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2,N_a2], whereas aprimeProbs is [N_d2,N_a2]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2], per-dim factored. Fold the two
        % per-dim lower indexes into the four corners here, keeping the a1prime offset. prob_1/prob_2
        % stay at [N_d2,N_a2]; each EV block below expands them to its own shape.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1=reshape(a2primeProbs(1,:,:),[N_d2,N_a2]);
        prob_2=reshape(a2primeProbs(2,:,:),[N_d2,N_a2]);
        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);
    end

    % Using V_Jplus1
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j),e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d2,a1prime, a2,z)

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_d3,[],1);

            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d2,a1prime, a2,z)

            DiscountedEV=DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j),e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                % (d,aprime,a,z)

                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV;

                % Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);

                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end

    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d,a1prime, a2,z)

            DiscountedEV=DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,semizblock);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz, special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z;

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d,a1prime, a2,z)

            DiscountedEV=DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,N_j);
                    DiscountedEV_z=DiscountedEV(:,:,z_c);

                    ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val,e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z;

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d2
    V(:,:,:,N_j)=V_jj;
    Policy3(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,N_j)=ceil(d2a1prime_ind/N_d2); % a1prime

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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % Note, is actually aprime_grid (but a_grid is anyway same for all ages)
    % Note: aprimeIndex is [N_d2*N_a2,1], whereas aprimeProbs is [N_d2,N_a2]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat((a2primeIndex-1),N_a1,1); % [N_d2*N_a1,N_a2]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2)+N_a1*repmat(a2primeIndex,N_a1,1); % [N_d2*N_a1,N_a2]
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2], per-dim factored. Fold the two
        % per-dim lower indexes into the four corners here, keeping the a1prime offset. prob_1/prob_2
        % stay at [N_d2,N_a2]; each EV block below expands them to its own shape.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1=reshape(a2primeProbs(1,:,:),[N_d2,N_a2]);
        prob_2=reshape(a2primeProbs(2,:,:),[N_d2,N_a2]);
        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1);
    end

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0); % Level=0, Refine=0
            % (d,aprime,a,z)

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d,a1prime, a2,z)

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);

            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end

    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d,a1prime, a2,z)

            DiscountedEV=DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0
                % (d,aprime,a,z)

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV;

                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_e,[],1);

                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d,a1prime, a2,z)

            DiscountedEV=DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,semizblock);

                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV_z;

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);

                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end

    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            % Note: By definition V_Jplus1 does not depend on d2 (only aprime)
            pi_bothz=kron(pi_z_J(:,:,jj), pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
            EV=sum(EV,2); % sum over z', leaving a singular second dimension

            % Switch EV from being in terms of aprime to being in terms of d and a
            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the lower aprime
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2*N_a1,N_a2,N_bothz]); % (d2,a1prime,a2,z), the upper aprime

                % Skip interpolation when upper and lower are equal (otherwise can cause numerical rounding errors)
                aprimeProbs_d3=repmat(a2primeProbs,N_a1,1,N_bothz); % [N_d2*N_a1,N_a2,N_bothz]
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0; % effectively skips interpolation

                % Apply the aprimeProbs
                EV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % probability of lower grid point+ probability of upper grid point
                EV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                EV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four corners folded above, with skipinterp at
                % each level and per-contribution NaN cleanup for 0*(-Inf). prob_*_exp is expanded to this
                % block's shape, which is what the l_a2==1 arm's aprimeProbs expansion does.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2*N_a1,N_a2,N_bothz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2*N_a1,N_a2,N_bothz]);
                prob_1_exp=repmat(prob_1,N_a1,1,N_bothz);
                prob_2_exp=repmat(prob_2,N_a1,1,N_bothz);
                p1_loy=prob_1_exp; p1_loy(EV_ll==EV_hl)=0;
                c_ll=p1_loy.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_loy).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_loy=c_ll+c_hl;
                p1_hiy=prob_1_exp; p1_hiy(EV_lh==EV_hh)=0;
                c_lh=p1_hiy.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hiy).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hiy=c_lh+c_hh;
                p2=prob_2_exp; p2(EV_loy==EV_hiy)=0;
                c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                EV=c_loy+c_hiy;
            end
            % entireEV is (d,a1prime, a2,z)

            DiscountedEV=DiscountFactorParamsVec*repelem(EV,1,N_a1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);

                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,jj);
                    DiscountedEV_z=DiscountedEV(:,:,z_c);

                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0); % Level=0, Refine=0

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV_z;

                    %Calc the max and it's index
                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);

                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end

    end

    % Now we just max over d3, and keep the policy that corresponded to that (including modify the policy to include the d3 decision)
    [V_jj,maxindex]=max(V_ford3_jj,[],4); % max over d3
    V(:,:,:,jj)=V_jj;
    Policy3(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3 is just maxindex
    maxindex=reshape(maxindex,[N_a*N_semiz*N_z*N_e,1]); % This is the value of d that corresponds, make it this shape for addition just below
    d2a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_z*N_e)'+(N_a*N_semiz*N_z*N_e)*(maxindex-1)),[1,N_a,N_semiz*N_z,N_e]);
    Policy3(1,:,:,:,jj)=rem(d2a1prime_ind-1,N_d2)+1; % d2
    Policy3(3,:,:,:,jj)=ceil(d2a1prime_ind/N_d2); % a1prime

end


%% For experience asset, just output Policy as is and then use Case2 to UnKron

end
