function [V,Policy3]=ValueFnIter_FHorz_ExpAssetzSemiExo_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, z, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest

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

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy storage with separate entries for d1, d2, d3, a1prime
Policy3=zeros(4,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_d23=[n_d2,n_d3];

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_bothz,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_bothz, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
        Policy3(3,:,:,N_j)=ceil(d_ind/N_d12); % d3
        Policy3(4,:,:,N_j)=ceil(maxindex/N_d); % a1prime
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,semizblock,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy3(1,:,semizblock,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,semizblock,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,semizblock,N_j)=ceil(d_ind/N_d12);
            Policy3(4,:,semizblock,N_j)=ceil(maxindex/N_d);
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy3(1,:,z_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,z_c,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,z_c,N_j)=ceil(d_ind/N_d12);
            Policy3(4,:,z_c,N_j)=ceil(maxindex/N_d);
        end
    end
else
    % aprime depends on (d2, a1, a2, current_z); independent of d3 and semiz -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_z]

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1, N_a2, N_z]
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
        aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1); % [N_d2*N_a1, N_a2, N_z]
        % Expand to current_bothz = (semiz, z) with semiz fastest
        aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2*N_a1, N_a2, N_bothz]
        aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
        aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z] per-dim factored. Fold the two
        % per-dim lower indexes into the four corners of the a2 lattice, offset each by a1prime
        % exactly as the l_a2==1 arm does, and interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_z]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_z]);
        prob_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_z]);
        prob_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_z]);
        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2,N_z);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1,1);
        prob_1_d2a1a2z=repmat(prob_1,N_a1,1,1);
        prob_2_d2a1a2z=repmat(prob_2,N_a1,1,1);
        % Expand to current_bothz = (semiz, z) with semiz fastest
        aprime_ll_full=repelem(aprime_ll,1,1,N_semiz); % [N_d2*N_a1, N_a2, N_bothz]
        aprime_hl_full=repelem(aprime_hl,1,1,N_semiz);
        aprime_lh_full=repelem(aprime_lh,1,1,N_semiz);
        aprime_hh_full=repelem(aprime_hh,1,1,N_semiz);
        prob_1_full=repelem(prob_1_d2a1a2z,1,1,N_semiz);
        prob_2_full=repelem(prob_2_d2a1a2z,1,1,N_semiz);
    end

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j)); % [current_bothz, bothz_prime]

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

            % EV[a, current_bothz] = E[V_Jplus1(a, next) | current_bothz]
            EV=V_Jplus1.*shiftdim(pi_bothz',-1); % [N_a, N_bothz_next, N_bothz_current]
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_bothz_current]
            EV_2D=reshape(EV,[N_a,N_bothz]);

            % Lookup at aprime positions using linear indexing into EV_2D
            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset; % [N_d2*N_a1, N_a2, N_bothz]
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=EV_2D(aprime_ll_full+bothz_offset);
                EV_hl=EV_2D(aprime_hl_full+bothz_offset);
                EV_lh=EV_2D(aprime_lh_full+bothz_offset);
                EV_hh=EV_2D(aprime_hh_full+bothz_offset);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_full; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_full; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_full; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,0,0);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                if length(n_a2)==1
                    EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                    EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                    aprimeProbs_z(skipinterp)=0;
                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=EV_2D(aprime_ll_full(:,:,semizblock)+semizblock_offset);
                    EV_hl=EV_2D(aprime_hl_full(:,:,semizblock)+semizblock_offset);
                    EV_lh=EV_2D(aprime_lh_full(:,:,semizblock)+semizblock_offset);
                    EV_hh=EV_2D(aprime_hh_full(:,:,semizblock)+semizblock_offset);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1_full(:,:,semizblock); p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1_full(:,:,semizblock); p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2_full(:,:,semizblock); p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,semizblock,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=V_Jplus1.*pi_bothz(z_c,:); % [N_a, N_bothz_next]
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2); % [N_a, 1]

                z_part=ceil(z_c/N_semiz); % current z index (bothz = semiz + N_semiz*(z-1))

                if length(n_a2)==1
                    aprime_slice=aprimeIndex(:,:,z_part);
                    aprimeplus1_slice=aprimeplus1Index(:,:,z_part);
                    aprimeProbs_slice=aprimeProbs_d2a1a2z(:,:,z_part);
                    EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_slice;
                    aprimeProbs_z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=reshape(EV_z(aprime_ll(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    EV_hl=reshape(EV_z(aprime_hl(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    EV_lh=reshape(EV_z(aprime_lh(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    EV_hh=reshape(EV_z(aprime_hh(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1_d2a1a2z(:,:,z_part); p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1_d2a1a2z(:,:,z_part); p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2_d2a1a2z(:,:,z_part); p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1);

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3 and unpack policy
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,N_j)=V_jj;
    Policy3(3,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,N_j)=ceil(d12_ind/N_d1); % d2
    Policy3(4,:,:,N_j)=ceil(d12a1prime_ind/N_d12); % a1prime
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    if length(n_a2)==1
        aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
        aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
        aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
        aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
        aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
        aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z] per-dim factored. Fold the two
        % per-dim lower indexes into the four corners of the a2 lattice, offset each by a1prime
        % exactly as the l_a2==1 arm does, and interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_z]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_z]);
        prob_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_z]);
        prob_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_z]);
        a1prime_offsets=repelem((1:1:N_a1)',N_d2,N_a2,N_z);
        aprime_ll=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*(loIdx_2-1)-1,N_a1,1,1);
        aprime_hl=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1,N_a1,1,1);
        aprime_lh=a1prime_offsets+N_a1*repmat(loIdx_1+n_a2_1*loIdx_2-1,N_a1,1,1);
        aprime_hh=a1prime_offsets+N_a1*repmat((loIdx_1+1)+n_a2_1*loIdx_2-1,N_a1,1,1);
        prob_1_d2a1a2z=repmat(prob_1,N_a1,1,1);
        prob_2_d2a1a2z=repmat(prob_2,N_a1,1,1);
        % Expand to current_bothz = (semiz, z) with semiz fastest
        aprime_ll_full=repelem(aprime_ll,1,1,N_semiz); % [N_d2*N_a1, N_a2, N_bothz]
        aprime_hl_full=repelem(aprime_hl,1,1,N_semiz);
        aprime_lh_full=repelem(aprime_lh,1,1,N_semiz);
        aprime_hh_full=repelem(aprime_hh,1,1,N_semiz);
        prob_1_full=repelem(prob_1_d2a1a2z,1,1,N_semiz);
        prob_2_full=repelem(prob_2_d2a1a2z,1,1,N_semiz);
    end

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                lin_lower=aprimeIndex_full+bothz_offset;
                lin_upper=aprimeplus1Index_full+bothz_offset;
                EV1=EV_2D(lin_lower);
                EV2=EV_2D(lin_upper);

                skipinterp=(EV1==EV2);
                aprimeProbs_d3=aprimeProbs_full;
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=EV_2D(aprime_ll_full+bothz_offset);
                EV_hl=EV_2D(aprime_hl_full+bothz_offset);
                EV_lh=EV_2D(aprime_lh_full+bothz_offset);
                EV_hh=EV_2D(aprime_hh_full+bothz_offset);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_full; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_full; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_full; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,0,0);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1); % [N_a, N_bothz_next, N_semiz]
                EV(isnan(EV))=0;
                EV=sum(EV,2); % [N_a, 1, N_semiz]
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                if length(n_a2)==1
                    EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                    EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                    aprimeProbs_z(skipinterp)=0;
                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=EV_2D(aprime_ll_full(:,:,semizblock)+semizblock_offset);
                    EV_hl=EV_2D(aprime_hl_full(:,:,semizblock)+semizblock_offset);
                    EV_lh=EV_2D(aprime_lh_full(:,:,semizblock)+semizblock_offset);
                    EV_hh=EV_2D(aprime_hh_full(:,:,semizblock)+semizblock_offset);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1_full(:,:,semizblock); p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1_full(:,:,semizblock); p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2_full(:,:,semizblock); p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1,1);
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,semizblock,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,0,0);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                z_part=ceil(z_c/N_semiz);

                if length(n_a2)==1
                    aprime_slice=aprimeIndex(:,:,z_part);
                    aprimeplus1_slice=aprimeplus1Index(:,:,z_part);
                    aprimeProbs_slice=aprimeProbs_d2a1a2z(:,:,z_part);
                    EV1=reshape(EV_z(aprime_slice),[N_d2*N_a1,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1_slice),[N_d2*N_a1,N_a2]);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=aprimeProbs_slice;
                    aprimeProbs_z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=reshape(EV_z(aprime_ll(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    EV_hl=reshape(EV_z(aprime_hl(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    EV_lh=reshape(EV_z(aprime_lh(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    EV_hh=reshape(EV_z(aprime_hh(:,:,z_part)),[N_d2*N_a1,N_a2]);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1_d2a1a2z(:,:,z_part); p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1_d2a1a2z(:,:,z_part); p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2_d2a1a2z(:,:,z_part); p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,N_a1);

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,jj)=V_jj;
    Policy3(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy3(1,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy3(2,:,:,jj)=ceil(d12_ind/N_d1);
    Policy3(4,:,:,jj)=ceil(d12a1prime_ind/N_d12);
end


end
