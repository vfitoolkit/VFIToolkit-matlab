function [V,Policy3]=ValueFnIter_FHorz_ExpAssetSemiExo_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetSemiExo_noz_e_raw (d1, noz, e).
% Policy3 stores (d1, d2, d3) -- no a1prime channel since noa1.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_Case2_Disc* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_semiz, n_e, d123_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy3(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
        Policy3(3,:,:,:,N_j)=ceil(maxindex/N_d12);  % d3
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_semiz, special_n_e, d123_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,:,e_c,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, special_n_semiz, special_n_e, d123_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy3(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy3(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d12);
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    if length(n_a2)==1
        aprimeIndex=a2primeIndex;        % [N_d2,N_a2]
        aprimeplus1Index=a2primeIndex+1; % [N_d2,N_a2]
    else
        % l_a2==2: a2primeIndex and a2primeProbs are [l_a2,N_d2,N_a2], per-dim factored rather
        % than a single lower corner. With no a1, the aprime index is just the Kron index in the
        % a2 product space, so fold the two per-dim lower indexes into the four corners here and
        % do the nested 2-corner interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1=reshape(a2primeProbs(1,:,:),[N_d2,N_a2]);
        prob_2=reshape(a2primeProbs(2,:,:),[N_d2,N_a2]);
        aprime_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh=loIdx_1+n_a2_1*loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
    end

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

                aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2,N_a2,N_semiz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2,N_a2,N_semiz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2,N_a2,N_semiz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2,N_a2,N_semiz]);
                prob_1_d3=repmat(prob_1,1,1,N_semiz);
                prob_2_d3=repmat(prob_2,1,1,N_semiz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, n_e, d123_gridvals_val, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

                aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2,N_a2,N_semiz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2,N_a2,N_semiz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2,N_a2,N_semiz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2,N_a2,N_semiz]);
                prob_1_d3=repmat(prob_1,1,1,N_semiz);
                prob_2_d3=repmat(prob_2,1,1,N_semiz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, special_n_e, d123_gridvals_val, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

                aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2,N_a2,N_semiz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2,N_a2,N_semiz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2,N_a2,N_semiz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2,N_a2,N_semiz]);
                prob_1_d3=repmat(prob_1,1,1,N_semiz);
                prob_2_d3=repmat(prob_2,1,1,N_semiz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,N_j);
                    entireEV_z=entireEV(:,:,z_c);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,1);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy3(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2);
    if length(n_a2)==1
        aprimeIndex=a2primeIndex;        % [N_d2,N_a2]
        aprimeplus1Index=a2primeIndex+1; % [N_d2,N_a2]
    else
        % l_a2==2: a2primeIndex and a2primeProbs are [l_a2,N_d2,N_a2], per-dim factored rather
        % than a single lower corner. With no a1, the aprime index is just the Kron index in the
        % a2 product space, so fold the two per-dim lower indexes into the four corners here and
        % do the nested 2-corner interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:,:),[N_d2,N_a2]);
        loIdx_2=reshape(a2primeIndex(2,:,:),[N_d2,N_a2]);
        prob_1=reshape(a2primeProbs(1,:,:),[N_d2,N_a2]);
        prob_2=reshape(a2primeProbs(2,:,:),[N_d2,N_a2]);
        aprime_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        aprime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        aprime_lh=loIdx_1+n_a2_1*loIdx_2;
        aprime_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
    end

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

                aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2,N_a2,N_semiz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2,N_a2,N_semiz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2,N_a2,N_semiz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2,N_a2,N_semiz]);
                prob_1_d3=repmat(prob_1,1,1,N_semiz);
                prob_2_d3=repmat(prob_2,1,1,N_semiz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, n_e, d123_gridvals_val, a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

                aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2,N_a2,N_semiz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2,N_a2,N_semiz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2,N_a2,N_semiz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2,N_a2,N_semiz]);
                prob_1_d3=repmat(prob_1,1,1,N_semiz);
                prob_2_d3=repmat(prob_2,1,1,N_semiz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_semiz, special_n_e, d123_gridvals_val, a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_semiz]);
                EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_semiz]);

                aprimeProbs_d3=repmat(a2primeProbs,1,1,N_semiz);
                skipinterp=(EV1==EV2);
                aprimeProbs_d3(skipinterp)=0;

                entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
                entireEV(aprimeProbs_d3==0)=EV2(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=EV1(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                EV_ll=reshape(EV(aprime_ll,:),[N_d2,N_a2,N_semiz]);
                EV_hl=reshape(EV(aprime_hl,:),[N_d2,N_a2,N_semiz]);
                EV_lh=reshape(EV(aprime_lh,:),[N_d2,N_a2,N_semiz]);
                EV_hh=reshape(EV(aprime_hh,:),[N_d2,N_a2,N_semiz]);
                prob_1_d3=repmat(prob_1,1,1,N_semiz);
                prob_2_d3=repmat(prob_2,1,1,N_semiz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(EV_ll==EV_hl)=0;
                c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(EV_lh==EV_hh)=0;
                c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,jj);
                    entireEV_z=entireEV(:,:,z_c);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountFactorParamsVec*repelem(entireEV_z,N_d1,1);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy3(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,jj)=ceil(d12_ind/N_d1);    % d2
end


end
