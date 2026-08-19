function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_a3,n_semiz,n_z,n_u,N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, a3_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_raw.
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
%
% Plain GI + d4 outer loop. Inside each d4: refine d2 out of EV, then GI midpoint+L2 over a1 with d1+d3+a1prime.
% After d4 loop: max over d4 and look up per-d4 slabs.
%
% a1: standard endogenous state, this is the one the grid interpolation layer refines
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset.
%
% Policy: rows (d1,d2,d3,d4,a1prime_low,a2prime,L2ind,L2flag)

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_u=prod(n_u);

N_d13=N_d1*N_d3;

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

% For aprimeFn (d2,d3)
n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

% Variant of d4 (single slice) and d4 gridvals
special_n_d4=ones(1,length(n_d4));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

V=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy: rows (d1,d2,d3,d4,a1prime_low,a2prime,L2ind,L2flag)
Policy=zeros(8,N_a,N_bothz,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a3_grid=gpuArray(a3_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d13_gridvals=gpuArray(CreateGridvals([n_d1,n_d3],[d1_grid;d3_grid],1));

pi_u_col=pi_u(:);

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zindB=shiftdim(gpuArray(0:1:N_bothz-1),-1);

V_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
Policy_ford4_jj=zeros(N_a,N_bothz,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_bothz,N_d4,'gpuArray');
d2index_ford4_jj=ones(N_a,N_bothz,N_d4,'gpuArray');


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
            % [N_d13, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_bothz]
            [~,maxindex_d4]=max(ReturnMatrix_d4,[],2);

            midpoint_d4=max(min(maxindex_d4,N_a1-1),2);
            a1primeindexesfine=(midpoint_d4+(midpoint_d4-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
            % [N_d13*n2long*N_a2, N_a, N_bothz]
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
            d_ind       =rem(maxindexL2-1,N_d13)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
            allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*zindB;
            mid_at=shiftdim(squeeze(midpoint_d4(allind)),-1);
            linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
            linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*zindB;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford4_jj(:,:,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
            Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
            d2index_ford4_jj(:,:,d4_c)=1;
        end
    else
        for d4_c=1:N_d4
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);
                [~,maxindex_d4z]=max(ReturnMatrix_d4z,[],2);

                midpoint_d4z=max(min(maxindex_d4z,N_a1-1),2);
                a1primeindexesfine=(midpoint_d4z+(midpoint_d4z-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 2);
                % [N_d13*n2long*N_a2, N_a]
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_z,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
                mid_at=midpoint_d4z(allind);
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);
                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                d2index_ford4_jj(:,z_c,d4_c)=1;
            end
        end
    end
    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,N_j)=Vbest;
    Ncomb=N_a*N_bothz;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_bothz]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_bothz]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_bothz]);
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
    Policy(1,:,:,N_j)=reshape(d1part,[1,N_a,N_bothz]);
    Policy(2,:,:,N_j)=reshape(d2winner,[1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(d3part,[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(d4winner,[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_bothz]);
    Policy(6,:,:,N_j)=reshape(a2part,[1,N_a,N_bothz]);
    Policy(7,:,:,N_j)=reshape(L2ind,[1,N_a,N_bothz]);
    Policy(8,:,:,N_j)=reshape(flagwinner,[1,N_a,N_bothz]);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    EVnext=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    pi_z=pi_z_J(:,:,N_j);
    bothz_gridvals=bothz_gridvals_J(:,:,N_j);

    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % EV integrated over bothz'
            EV=EVnext.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_bothz]);

            % Refine d2
            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_bothz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            % Level-1 Return at coarse a1prime grid
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, ReturnFnParamsVec, 1);

            entireRHS=ReturnMatrix_d4+DiscountedEV_d13; % broadcast a3

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, ReturnFnParamsVec, 3);
            aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
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
            flag_ford4_jj(:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            % Per-d4 policy encoding: d13 + N_d13*(mid-1) + N_d13*N_a1*(a2prime-1) + N_d13*N_a1*N_a2*(L2offset-1)
            Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
            % d2 lookup per (a,bothz) — uses the COARSE midpoint (matches Plain GI convention)
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_bothz-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
            d2index_ford4_jj(:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        end

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % EV integrated over bothz'
            EV=EVnext.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_bothz]);

            % Refine d2
            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_bothz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                semizind=shiftdim(gpuArray(0:1:N_semiz-1),-5);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEV_d13block=DiscountedEV_d13(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_d13block=DiscountedEVinterp_d13(:,:,:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,:,semizblock);

                % Level-1 Return at coarse a1prime grid
                ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, ReturnFnParamsVec, 1);

                entireRHS=ReturnMatrix_d4+DiscountedEV_d13block; % broadcast a3

                [~,maxindex]=max(entireRHS,[],2);

                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*semizind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13block(aprime),[N_d13*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,semizblock,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*semizBind;
                mid_at=shiftdim(squeeze(midpoint(allind)),-1); % [1,N_a,N_semiz]
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*semizBind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*semizBind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,semizblock,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                % Per-d4 policy encoding: d13 + N_d13*(mid-1) + N_d13*N_a1*(a2prime-1) + N_d13*N_a1*N_a2*(L2offset-1)
                Policy_ford4_jj(:,semizblock,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                % d2 lookup per (a,semiz) — uses the COARSE midpoint (matches Plain GI convention)
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_semiz]
                a1opt_mid=midpoint(allind); % [1,N_a,N_semiz]
                zlin=shiftdim(gpuArray(0:N_semiz-1),-1); % [1,1,N_semiz]
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
                d2index_ford4_jj(:,semizblock,d4_c)=shiftdim(d2index_reshblock(lin),1);
            end
        end

    elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
        special_n_bothz=ones(1,length(n_bothz));
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);

                EV_z=EVnext.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a3primeProbs,N_a12,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

                EVres=reshape(EV_z,[N_d2,N_d3*N_a12]);
                [EV_onlyd3,d2index]=max(EVres,[],1);
                EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1,N_a2]);

                DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2]);
                DiscountedEVinterp_z=permute(interp1(a1_grid,permute(DiscountedEV_z,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d3,N_a1fine,N_a2]
                DiscountedEV_d13_z=repelem(DiscountedEV_z,N_d1,1);
                DiscountedEVinterp_d13_z=repelem(DiscountedEVinterp_z,N_d1,1);

                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

                entireRHS_z=ReturnMatrix_d4z+DiscountedEV_d13_z;

                [~,maxindex]=max(entireRHS_z,[],2);

                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_d13_z(aprime),[N_d13*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
                mid_at=midpoint(allind);
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1opt_mid=midpoint(allind);
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index_z(lin),1);
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,N_j)=Vbest;
    Ncomb=N_a*N_bothz;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_bothz]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_bothz]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_bothz]);
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
    Policy(1,:,:,N_j)=reshape(d1part,[1,N_a,N_bothz]);
    Policy(2,:,:,N_j)=reshape(d2winner,[1,N_a,N_bothz]);
    Policy(3,:,:,N_j)=reshape(d3part,[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(d4winner,[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(a1prime_low,[1,N_a,N_bothz]);
    Policy(6,:,:,N_j)=reshape(a2part,[1,N_a,N_bothz]);
    Policy(7,:,:,N_j)=reshape(L2ind,[1,N_a,N_bothz]);
    Policy(8,:,:,N_j)=reshape(flagwinner,[1,N_a,N_bothz]);
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

    EVnext=V(:,:,jj+1);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    pi_z=pi_z_J(:,:,jj);
    bothz_gridvals=bothz_gridvals_J(:,:,jj);

    aprimeIndex     =repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    if vfoptions.lowmemory==0
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % EV integrated over bothz'
            EV=EVnext.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_bothz]);

            % Refine d2
            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_bothz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            % Level-1 Return at coarse a1prime grid
            ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, ReturnFnParamsVec, 1);

            entireRHS=ReturnMatrix_d4+DiscountedEV_d13; % broadcast a3

            [~,maxindex]=max(entireRHS,[],2);

            midpoint=max(min(maxindex,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, ReturnFnParamsVec, 3);
            aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*shiftdim((0:1:N_bothz-1),-5);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13(aprime),[N_d13*n2long*N_a2,N_a,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,d4_c)=shiftdim(Vtempii,1);
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
            flag_ford4_jj(:,:,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            % Per-d4 policy encoding: d13 + N_d13*(mid-1) + N_d13*N_a1*(a2prime-1) + N_d13*N_a1*N_a2*(L2offset-1)
            Policy_ford4_jj(:,:,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
            % d2 lookup per (a,bothz) — uses the COARSE midpoint (matches Plain GI convention)
            d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
            a1opt_mid=midpoint(allind);
            zlin=shiftdim(gpuArray(0:N_bothz-1),-1);
            lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
            d2index_ford4_jj(:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        end

    elseif vfoptions.lowmemory==1
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            % EV integrated over bothz'
            EV=EVnext.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV=reshape(EV,[N_a,N_bothz]);

            skipinterp=logical(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1))==EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)));
            aprimeProbs=repmat(a3primeProbs,N_a12,N_bothz);
            aprimeProbs(skipinterp)=0;
            aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u,N_bothz]);

            EV1=reshape(EV(aprimeIndex(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*aprimeProbs;
            EV2=reshape(EV(aprimeplus1Index(:)+N_a*((1:1:N_bothz)-1)),[N_d23*N_a12,N_u,N_bothz]).*(1-aprimeProbs);
            EV=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);
            EV=reshape(EV,[N_d23*N_a12,N_bothz]);

            % Refine d2
            EVres=reshape(EV,[N_d2,N_d3*N_a12,N_bothz]);
            [EV_onlyd3,d2index]=max(EVres,[],1);
            EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_bothz]);
            d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_bothz]);

            DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_bothz]);
            DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_bothz]
            DiscountedEV_d13=repelem(DiscountedEV,N_d1,1);
            DiscountedEVinterp_d13=repelem(DiscountedEVinterp,N_d1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                semizind=shiftdim(gpuArray(0:1:N_semiz-1),-5);
                semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEV_d13block=DiscountedEV_d13(:,:,:,:,:,:,semizblock);
                DiscountedEVinterp_d13block=DiscountedEVinterp_d13(:,:,:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,:,semizblock);

                % Level-1 Return at coarse a1prime grid
                ReturnMatrix_d4=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, ReturnFnParamsVec, 1);

                entireRHS=ReturnMatrix_d4+DiscountedEV_d13block; % broadcast a3

                [~,maxindex]=max(entireRHS,[],2);

                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1) + N_d13*N_a1fine*N_a2*semizind;
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_d13block(aprime),[N_d13*n2long*N_a2,N_a,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,semizblock,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind + N_d13*N_a2*N_a*semizBind;
                mid_at=shiftdim(squeeze(midpoint(allind)),-1); % [1,N_a,N_semiz]
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*semizBind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind + N_d13*n2long*N_a2*N_a*semizBind;
                isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,semizblock,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                % Per-d4 policy encoding: d13 + N_d13*(mid-1) + N_d13*N_a1*(a2prime-1) + N_d13*N_a1*N_a2*(L2offset-1)
                Policy_ford4_jj(:,semizblock,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                % d2 lookup per (a,semiz) — uses the COARSE midpoint (matches Plain GI convention)
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1; % [1,N_a,N_semiz]
                a1opt_mid=midpoint(allind); % [1,N_a,N_semiz]
                zlin=shiftdim(gpuArray(0:N_semiz-1),-1); % [1,1,N_semiz]
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zlin;
                d2index_ford4_jj(:,semizblock,d4_c)=shiftdim(d2index_reshblock(lin),1);
            end
        end

    elseif vfoptions.lowmemory>=2 % lm2 already does the most-looped variant, so it also serves the higher lowmemory values
        special_n_bothz=ones(1,length(n_bothz));
        for d4_c=1:N_d4
            pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
            d13_with_d4=[d13_gridvals,repmat(d4_gridvals(d4_c,:),N_d13,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals(z_c,:);

                EV_z=EVnext.*pi_bothz(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);
                EV_z=reshape(EV_z,[N_a,1]);

                skipinterp=logical(EV_z(aprimeIndex(:))==EV_z(aprimeplus1Index(:)));
                aprimeProbs=repmat(a3primeProbs,N_a12,1);
                aprimeProbs(skipinterp)=0;
                aprimeProbs=reshape(aprimeProbs,[N_d23*N_a12,N_u]);

                EV1=reshape(EV_z(aprimeIndex(:)),[N_d23*N_a12,N_u]).*aprimeProbs;
                EV2=reshape(EV_z(aprimeplus1Index(:)),[N_d23*N_a12,N_u]).*(1-aprimeProbs);
                EV_z=sum(EV1.*pi_u_col',2)+sum(EV2.*pi_u_col',2);

                EVres=reshape(EV_z,[N_d2,N_d3*N_a12]);
                [EV_onlyd3,d2index]=max(EVres,[],1);
                EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,1]);
                d2index_z=reshape(d2index,[N_d3,N_a1,N_a2]);

                DiscountedEV_z=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2]);
                DiscountedEVinterp_z=permute(interp1(a1_grid,permute(DiscountedEV_z,[2,1,3]),a1prime_grid),[2,1,3]); % [N_d3,N_a1fine,N_a2]
                DiscountedEV_d13_z=repelem(DiscountedEV_z,N_d1,1);
                DiscountedEVinterp_d13_z=repelem(DiscountedEVinterp_z,N_d1,1);

                ReturnMatrix_d4z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, d13_with_d4, a1_grid, a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 1);

                entireRHS_z=ReturnMatrix_d4z+DiscountedEV_d13_z;

                [~,maxindex]=max(entireRHS_z,[],2);

                midpoint=max(min(maxindex,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_z=CreateReturnFnMatrix_ExpAsset_Disc_DC2A(ReturnFn, n_d1,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, d13_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_val, ReturnFnParamsVec, 3);
                aprime=(1:1:N_d13)' + N_d13*(a1primeindexesfine-1) + N_d13*N_a1fine*shiftdim((0:1:N_a2-1),-1);
                entireRHS_ii_z=reshape(ReturnMatrix_ii_z+DiscountedEVinterp_d13_z(aprime),[N_d13*n2long*N_a2,N_a]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_z,[],1);
                V_ford4_jj(:,z_c,d4_c)=shiftdim(Vtempii,1);
                d_ind       =rem(maxindexL2-1,N_d13)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d13),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d13*n2long))+1;
                allind=d_ind + N_d13*(maxindexL2a2-1) + N_d13*N_a2*aind;
                mid_at=midpoint(allind);
                linidx_lower  = d_ind                    + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                linidx_upper  = d_ind + N_d13*(n2long-1) + N_d13*n2long*(maxindexL2a2-1) + N_d13*n2long*N_a2*aind;
                isInfLower    = (ReturnMatrix_ii_z(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_z(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,z_c,d4_c)=shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                Policy_ford4_jj(:,z_c,d4_c)=shiftdim(d_ind,1)+N_d13*(shiftdim(mid_at,1)-1)+N_d13*N_a1*(shiftdim(maxindexL2a2,1)-1)+N_d13*N_a1*N_a2*(shiftdim(maxindexL2a1,1)-1);
                d3opt=rem(ceil(d_ind/N_d1)-1,N_d3)+1;
                a1opt_mid=midpoint(allind);
                lin=d3opt+N_d3*(a1opt_mid-1)+N_d3*N_a1*(maxindexL2a2-1);
                d2index_ford4_jj(:,z_c,d4_c)=shiftdim(d2index_z(lin),1);
            end
        end
    end

    [Vbest,d4winner]=max(V_ford4_jj,[],3);
    V(:,:,jj)=Vbest;
    Ncomb=N_a*N_bothz;
    linidx=(1:1:Ncomb)'+Ncomb*(reshape(d4winner,[Ncomb,1])-1);
    polenc=reshape(Policy_ford4_jj(linidx),[N_a,N_bothz]);
    d2winner=reshape(d2index_ford4_jj(linidx),[N_a,N_bothz]);
    flagwinner=reshape(flag_ford4_jj(linidx),[N_a,N_bothz]);
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
    Policy(1,:,:,jj)=reshape(d1part,[1,N_a,N_bothz]);
    Policy(2,:,:,jj)=reshape(d2winner,[1,N_a,N_bothz]);
    Policy(3,:,:,jj)=reshape(d3part,[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(d4winner,[1,N_a,N_bothz]);
    Policy(5,:,:,jj)=reshape(a1prime_low,[1,N_a,N_bothz]);
    Policy(6,:,:,jj)=reshape(a2part,[1,N_a,N_bothz]);
    Policy(7,:,:,jj)=reshape(L2ind,[1,N_a,N_bothz]);
    Policy(8,:,:,jj)=reshape(flagwinner,[1,N_a,N_bothz]);
end


end
