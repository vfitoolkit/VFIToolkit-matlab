function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_a3,n_semiz,n_e,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, a3_grid, semiz_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_noz_e_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No z, with e.
%
% a1: standard endogenous state, this is the one the grid interpolation layer refines
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset. EV is integrated over e up front, so DiscountedEV
% carries no e dimension.
%
% Policy is 7-channel: 1=d1, 2=d2, 3=d3, 4=d4, 5=a1prime lower, 6=a2prime, 7=L2ind.
% An 8th channel (the L2flag) is carried alongside and written into row 8.

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_e=prod(n_e);
N_u=prod(n_u);

N_d13=N_d1*N_d3;

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy=zeros(8,N_a,N_semiz,N_e,N_j,'gpuArray'); % (d1,d2,d3,d4,a1prime_low,a2prime,L2,L2flag)

%%
u_grid=gpuArray(u_grid);
a3_grid=gpuArray(a3_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=gpuArray(CreateGridvals([n_d1,n_d3],[d1_grid;d3_grid],1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    special_n_semiz=ones(1,length(n_semiz));
end

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zindB=shiftdim(gpuArray(0:1:N_semiz-1),-1);
zeindB=zindB+N_semiz*shiftdim((0:1:N_e-1),-2);

V_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_semiz,N_e,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_semiz,N_e,N_d4,'gpuArray');
d2index_ford4_jj=ones(N_a,N_semiz,N_e,N_d4,'gpuArray');


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            % [N_d13, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_semiz, N_e]
            [~,maxindex_d4]=max(ReturnMatrix_d4,[],2);

            midpoint_d4=max(min(maxindex_d4,N_a1-1),2);
            a1primeindexesfine=(midpoint_d4+(midpoint_d4-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            % [N_d13*n2long*N_a2, N_a, N_semiz, N_e]
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint_d4(allind)),-1);
            linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zeindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
            d2index_ford4_jj(:,:,:,d4_c)=1;
        end
    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d4e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, special_n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                [~,maxindex_d4e]=max(ReturnMatrix_d4e,[],2);

                midpoint_d4e=max(min(maxindex_d4e,N_a1-1),2);
                a1primeindexesfine=(midpoint_d4e+(midpoint_d4e-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 2);
                % [N_d13*n2long*N_a2, N_a, N_semiz]
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint_d4e(allind)),-1);
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                d2index_ford4_jj(:,:,e_c,d4_c)=1;
            end
        end
    elseif vfoptions.lowmemory>=2
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_semiz, special_n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 1);
                    [~,maxindex_ze]=max(ReturnMatrix_ze,[],2);

                    midpoint_ze=max(min(maxindex_ze,N_a1-1),2);
                    a1primeindexesfine=(midpoint_ze+(midpoint_ze-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_semiz, special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, e_val, ReturnFnParamsVec, 2);
                    % [N_d13*n2long*N_a2, N_a]
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);
                    d_ind       =rem(maxindexL2-1,N_d13)+1;
                    maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                    maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                    allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
                    mid_at=midpoint_ze(allind);
                    linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                    linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                    isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                    Policy_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                    d2index_ford4_jj(:,z_c,e_c,d4_c)=1;
                end
            end
        end
    end
    % Cross-d4 max + final encoding (with d1, with e)
    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,N_j)=Vbest;
    Ncomb=N_a*N_semiz*N_e;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    d13part=rem(polenc-1,N_d13)+1;
    tmp=ceil(polenc/N_d13);
    midpart=rem(tmp-1,N_a1)+1;
    tmp2=ceil(tmp/N_a1);
    a2part=rem(tmp2-1,N_a2)+1;
    L2offset=ceil(tmp2/N_a2);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    d1part=rem(d13part-1,N_d1)+1;
    d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
    Policy(1,:,:,:,N_j)=reshape(d1part,[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(d2winner,[1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(d3part,[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_semiz,N_e]);
    Policy(6,:,:,:,N_j)=reshape(a2part,[1,N_a,N_semiz,N_e]);
    Policy(7,:,:,:,N_j)=reshape(L2ind,[1,N_a,N_semiz,N_e]);
    Policy(8,:,:,:,N_j)=reshape(flagwinner,[1,N_a,N_semiz,N_e]);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]);
    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-2),3); % [N_a,N_semiz]
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    %% Inlined per-period inner (with d1, no z, with e)
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=EVpre.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_semiz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

            entireRHS=ReturnMatrix_d4+DiscountedEV_d13;

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zeindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
            d2index_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        end

    elseif vfoptions.lowmemory>=1
        special_n_e=ones(1,length(n_e));
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=EVpre.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_semiz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, special_n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 1);
                entireRHS_e=ReturnMatrix_e+DiscountedEV_d13;
                [~,maxindex]=max(entireRHS_e,[],2);

                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint(allind)),-1);
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1opt_mid=midpoint(allind);
                zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
                d2index_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
            end
        end
    end

    % Cross-d4 max + final encoding (with d1, with e)
    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,N_j)=Vbest;
    Ncomb=N_a*N_semiz*N_e;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    d13part=rem(polenc-1,N_d13)+1;
    tmp=ceil(polenc/N_d13);
    midpart=rem(tmp-1,N_a1)+1;
    tmp2=ceil(tmp/N_a1);
    a2part=rem(tmp2-1,N_a2)+1;
    L2offset=ceil(tmp2/N_a2);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    d1part=rem(d13part-1,N_d1)+1;
    d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
    Policy(1,:,:,:,N_j)=reshape(d1part,[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,N_j)=reshape(d2winner,[1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(d3part,[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(d4winner,[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_semiz,N_e]);
    Policy(6,:,:,:,N_j)=reshape(a2part,[1,N_a,N_semiz,N_e]);
    Policy(7,:,:,:,N_j)=reshape(L2ind,[1,N_a,N_semiz,N_e]);
    Policy(8,:,:,:,N_j)=reshape(flagwinner,[1,N_a,N_semiz,N_e]);
end


%% Iterate backwards
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a,N_semiz]

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    %% Inlined per-period inner (with d1, no z, with e)
    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=EVpre.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_semiz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

            entireRHS=ReturnMatrix_d4+DiscountedEV_d13;

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 3);
            aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_semiz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_semiz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zeindB;
            mid_at=shiftdim(squeeze(midpoint(allind)),-1);
            linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zeindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zeindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            Policy_ford4_jj(:,:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
            d2index_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        end

    elseif vfoptions.lowmemory>=1
        special_n_e=ones(1,length(n_e));
        for d4_c=1:N_d4
            pi_semizd4=pi_semiz(:,:,d4_c);
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            EV=EVpre.*shiftdim(pi_semizd4',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_semiz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_semiz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_semiz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_semiz)-1)),[N_d23*N_a12,N_u,N_semiz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_semiz]);

            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_semiz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_semiz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_semiz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_semiz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_semiz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, special_n_e, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 1);
                entireRHS_e=ReturnMatrix_e+DiscountedEV_d13;
                [~,maxindex]=max(entireRHS_e,[],2);

                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_semiz, special_n_e, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_semiz-1),-5);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
                mid_at=shiftdim(squeeze(midpoint(allind)),-1);
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,:,e_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1opt_mid=midpoint(allind);
                zlin=shiftdim(gpuArray(0:N_semiz-1),-1);
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
                d2index_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
            end
        end
    end

    % Cross-d4 max + final encoding (with d1, with e)
    [Vbest,d4winner]=max(V_ford4_jj,[],4);
    V(:,:,:,jj)=Vbest;
    Ncomb=N_a*N_semiz*N_e;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_semiz,N_e]);
    d13part=rem(polenc-1,N_d13)+1;
    tmp=ceil(polenc/N_d13);
    midpart=rem(tmp-1,N_a1)+1;
    tmp2=ceil(tmp/N_a1);
    a2part=rem(tmp2-1,N_a2)+1;
    L2offset=ceil(tmp2/N_a2);
    adjust=(L2offset<1+n2short+1);
    a1prime_low=midpart-adjust;
    L2ind=adjust.*L2offset+(1-adjust).*(L2offset-n2short-1);
    d1part=rem(d13part-1,N_d1)+1;
    d3part=rem(ceil(d13part/N_d1)-1,N_d3)+1;
    Policy(1,:,:,:,jj)=reshape(d1part,[1,N_a,N_semiz,N_e]);
    Policy(2,:,:,:,jj)=reshape(d2winner,[1,N_a,N_semiz,N_e]);
    Policy(3,:,:,:,jj)=reshape(d3part,[1,N_a,N_semiz,N_e]);
    Policy(4,:,:,:,jj)=reshape(d4winner,[1,N_a,N_semiz,N_e]);
    Policy(5,:,:,:,jj)=reshape(a1prime_low,[1,N_a,N_semiz,N_e]);
    Policy(6,:,:,:,jj)=reshape(a2part,[1,N_a,N_semiz,N_e]);
    Policy(7,:,:,:,jj)=reshape(L2ind,[1,N_a,N_semiz,N_e]);
    Policy(8,:,:,:,jj)=reshape(flagwinner,[1,N_a,N_semiz,N_e]);
end


end
