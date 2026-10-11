function [V,Policy2]=ValueFnIter_FHorz_ExpAssetzSemiExo_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_semiz,N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetzSemiExo_nod1_raw (nod1, z, noe).
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.
% aprimeFn = aprimeFn(d2, a2, z, ...) -- z is the markov z only (not semiz)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

V=zeros(N_a,N_bothz,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_bothz,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_Case2_Disc* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

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
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, n_bothz, d23_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2(1,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy2(2,:,:,N_j)=ceil(d_ind/N_d2);    % d3
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
            z_valblock=bothz_gridvals_J(semizblock,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, [n_semiz,ones(1,length(n_z))], d23_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,semizblock,N_j)=shiftdim(Vtemp,1);
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,semizblock,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,semizblock,N_j)=ceil(d_ind/N_d2);
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for z_c=1:N_bothz
            z_val=bothz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, special_n_bothz, d23_gridvals, a2_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,z_c,N_j)=ceil(d_ind/N_d2);
        end
    end
else
    % aprime depends on (d2, a2, current markov z); independent of d3 and semiz -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);
    % a2primeIndex, a2primeProbs are both [N_d2, N_a2, N_z]

    if length(n_a2)==1
        aprimeIndex=a2primeIndex;
        aprimeplus1Index=a2primeIndex+1;
        % Expand to current_bothz = (semiz, z) with semiz fastest
        aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz); % [N_d2, N_a2, N_bothz]
        aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
        aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z] per-dim factored. With no a1 the
        % aprime index is just the Kron index in the a2 product space, so fold the two per-dim lower
        % indexes into the four corners of the a2 lattice here, and interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_z]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_z]);
        prob_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_z]);
        prob_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_z]);
        aprime_ll= loIdx_1   +n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh= loIdx_1   +n_a2_1* loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1* loIdx_2;
        % Expand to current_bothz = (semiz, z) with semiz fastest
        aprime_ll_full=repelem(aprime_ll,1,1,N_semiz); % [N_d2, N_a2, N_bothz]
        aprime_hl_full=repelem(aprime_hl,1,1,N_semiz);
        aprime_lh_full=repelem(aprime_lh,1,1,N_semiz);
        aprime_hh_full=repelem(aprime_hh,1,1,N_semiz);
        prob_1_full=repelem(prob_1,1,1,N_semiz);
        prob_2_full=repelem(prob_2,1,1,N_semiz);
    end

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j)); % [current_bothz, bothz_prime]

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_bothz, d23_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_bothz',-1); % [N_a, N_bothz_next, N_bothz_current]
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_bothz_current]
            EV_2D=reshape(EV,[N_a,N_bothz]);

            % Lookup at aprime positions using linear indexing into EV_2D
            if length(n_a2)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

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

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, [n_semiz,ones(1,length(n_z))], d23_gridvals_val, a2_gridvals, z_valblock, ReturnFnParamsVec);

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

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*entireEV_z;
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,semizblock,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_bothz, d23_gridvals_val, a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_bothz(z_c,:); % [N_a, N_bothz_next]
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2); % [N_a, 1]

                z_part=ceil(z_c/N_semiz); % current z index (bothz = semiz + N_semiz*(z-1))
                if length(n_a2)==1
                    EV1=reshape(EV_z(aprimeIndex(:,:,z_part)),[N_d2,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1Index(:,:,z_part)),[N_d2,N_a2]);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=a2primeProbs(:,:,z_part);
                    aprimeProbs_z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=reshape(EV_z(aprime_ll(:,:,z_part)),[N_d2,N_a2]);
                    EV_hl=reshape(EV_z(aprime_hl(:,:,z_part)),[N_d2,N_a2]);
                    EV_lh=reshape(EV_z(aprime_lh(:,:,z_part)),[N_d2,N_a2]);
                    EV_hh=reshape(EV_z(aprime_hh(:,:,z_part)),[N_d2,N_a2]);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1(:,:,z_part); p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1(:,:,z_part); p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2(:,:,z_part); p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*entireEV_z;

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3 and unpack policy
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,N_j)=V_jj;
    Policy2(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy2(1,:,:,N_j)=d2_ind; % d2
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
        aprimeIndex=a2primeIndex;
        aprimeplus1Index=a2primeIndex+1;
        aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
        aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
        aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz);
    else
        % l_a2==2: a2primeIndex/a2primeProbs are [l_a2,N_d2,N_a2,N_z] per-dim factored. With no a1 the
        % aprime index is just the Kron index in the a2 product space, so fold the two per-dim lower
        % indexes into the four corners of the a2 lattice here, and interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:,:),[N_d2,N_a2,N_z]);
        loIdx_2=reshape(a2primeIndex(2,:,:,:),[N_d2,N_a2,N_z]);
        prob_1=reshape(a2primeProbs(1,:,:,:),[N_d2,N_a2,N_z]);
        prob_2=reshape(a2primeProbs(2,:,:,:),[N_d2,N_a2,N_z]);
        aprime_ll= loIdx_1   +n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh= loIdx_1   +n_a2_1* loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1* loIdx_2;
        % Expand to current_bothz = (semiz, z) with semiz fastest
        aprime_ll_full=repelem(aprime_ll,1,1,N_semiz); % [N_d2, N_a2, N_bothz]
        aprime_hl_full=repelem(aprime_hl,1,1,N_semiz);
        aprime_lh_full=repelem(aprime_lh,1,1,N_semiz);
        aprime_hh_full=repelem(aprime_hh,1,1,N_semiz);
        prob_1_full=repelem(prob_1,1,1,N_semiz);
        prob_2_full=repelem(prob_2,1,1,N_semiz);
    end

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_bothz, d23_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            if length(n_a2)==1
                EV1=EV_2D(aprimeIndex_full+bothz_offset);
                EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

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

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, [n_semiz,ones(1,length(n_z))], d23_gridvals_val, a2_gridvals, z_valblock, ReturnFnParamsVec);

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

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*entireEV_z;
                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,semizblock,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,semizblock,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_bothz, d23_gridvals_val, a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                z_part=ceil(z_c/N_semiz);
                if length(n_a2)==1
                    EV1=reshape(EV_z(aprimeIndex(:,:,z_part)),[N_d2,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1Index(:,:,z_part)),[N_d2,N_a2]);

                    skipinterp=(EV1==EV2);
                    aprimeProbs_z=a2primeProbs(:,:,z_part);
                    aprimeProbs_z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);
                    entireEV_z(aprimeProbs_z==0)=EV2(aprimeProbs_z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_z==1)=EV1(aprimeProbs_z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=reshape(EV_z(aprime_ll(:,:,z_part)),[N_d2,N_a2]);
                    EV_hl=reshape(EV_z(aprime_hl(:,:,z_part)),[N_d2,N_a2]);
                    EV_lh=reshape(EV_z(aprime_lh(:,:,z_part)),[N_d2,N_a2]);
                    EV_hh=reshape(EV_z(aprime_hh(:,:,z_part)),[N_d2,N_a2]);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1(:,:,z_part); p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1(:,:,z_part); p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2(:,:,z_part); p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*entireEV_z;

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,jj)=V_jj;
    Policy2(2,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    Policy2(1,:,:,jj)=d2_ind; % d2
end


end
