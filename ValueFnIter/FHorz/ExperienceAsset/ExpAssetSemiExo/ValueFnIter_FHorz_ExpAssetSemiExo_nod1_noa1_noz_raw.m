function [V,Policy2]=ValueFnIter_FHorz_ExpAssetSemiExo_nod1_noa1_noz_raw(n_d2,n_d3,n_a2,n_semiz,N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetSemiExo_nod1_noz_raw.
% d2 determines experience asset; d3 determines semi-exog state; a = a2 (experience asset is the only endogenous state); semiz is semi-exog.
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_semiz,N_j,'gpuArray');

%%
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_Case2_Disc* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, n_semiz, d23_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2(1,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy2(2,:,:,N_j)=ceil(d_ind/N_d2);    % d3
    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, special_n_semiz, d23_gridvals, a2_gridvals, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,z_c,N_j)=ceil(d_ind/N_d2);
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec,2); % [N_d2,N_a2]
    % noa1: aprimeIndex/Plus1Index reduce to a2primeIndex(+1) directly (no N_a1*... combination)
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

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_semiz, d23_gridvals_val, a2_gridvals, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=V_Jplus1.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_semiz]

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

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_semiz, d23_gridvals_val, a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=V_Jplus1.*pi_semiz_d3(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                if length(n_a2)==1
                    EV1=reshape(EV_z(aprimeIndex),[N_d2,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1Index),[N_d2,N_a2]);

                    aprimeProbs_d3z=a2primeProbs;
                    skipinterp=(EV1==EV2);
                    aprimeProbs_d3z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_d3z+EV2.*(1-aprimeProbs_d3z);
                    entireEV_z(aprimeProbs_d3z==0)=EV2(aprimeProbs_d3z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_d3z==1)=EV1(aprimeProbs_d3z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=reshape(EV_z(aprime_ll),[N_d2,N_a2]);
                    EV_hl=reshape(EV_z(aprime_hl),[N_d2,N_a2]);
                    EV_lh=reshape(EV_z(aprime_lh),[N_d2,N_a2]);
                    EV_hh=reshape(EV_z(aprime_hh),[N_d2,N_a2]);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1; p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1; p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2; p2(EV_lo==EV_hi)=0;
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

    % Max over d3
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,N_j)=V_jj;
    Policy2(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy2(1,:,:,N_j)=d2_ind; % d2 (no a1prime split)
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

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_semiz, d23_gridvals_val, a2_gridvals, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_semiz]

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

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_semiz, d23_gridvals_val, a2_gridvals, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*(ones(N_a,1,'gpuArray')*pi_semiz_d3(z_c,:));
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                if length(n_a2)==1
                    EV1=reshape(EV_z(aprimeIndex),[N_d2,N_a2]);
                    EV2=reshape(EV_z(aprimeplus1Index),[N_d2,N_a2]);

                    aprimeProbs_d3z=a2primeProbs;
                    skipinterp=(EV1==EV2);
                    aprimeProbs_d3z(skipinterp)=0;

                    entireEV_z=EV1.*aprimeProbs_d3z+EV2.*(1-aprimeProbs_d3z);
                    entireEV_z(aprimeProbs_d3z==0)=EV2(aprimeProbs_d3z==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                    entireEV_z(aprimeProbs_d3z==1)=EV1(aprimeProbs_d3z==1);
                else
                    % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                    % level and per-contribution NaN cleanup so that a zero weight against an infinite
                    % node (0*(-Inf)=NaN) does not poison the sum.
                    EV_ll=reshape(EV_z(aprime_ll),[N_d2,N_a2]);
                    EV_hl=reshape(EV_z(aprime_hl),[N_d2,N_a2]);
                    EV_lh=reshape(EV_z(aprime_lh),[N_d2,N_a2]);
                    EV_hh=reshape(EV_z(aprime_hh),[N_d2,N_a2]);
                    % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                    p1_lo=prob_1; p1_lo(EV_ll==EV_hl)=0;
                    c_ll=p1_lo.*EV_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_lo).*EV_hl; c_hl(isnan(c_hl))=0;
                    EV_lo=c_ll+c_hl;
                    p1_hi=prob_1; p1_hi(EV_lh==EV_hh)=0;
                    c_lh=p1_hi.*EV_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hi).*EV_hh; c_hh(isnan(c_hh))=0;
                    EV_hi=c_lh+c_hh;
                    % outer level: interpolate those two over the a2_2 dimension
                    p2=prob_2; p2(EV_lo==EV_hi)=0;
                    c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                    c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                    entireEV_z=c_lo+c_hi;
                end

                entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*entireEV_z;

                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,z_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    end

    % Max over d3
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,jj)=V_jj;
    Policy2(2,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy2(1,:,:,jj)=d2_ind; % d2

end


end
