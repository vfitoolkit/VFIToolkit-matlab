function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_GI2A_nod1_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_a3,n_semiz,n_z,n_e,n_u,N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, a3_grid, semiz_gridvals_J, z_gridvals_J, e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Two standard endogenous assets version of ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_nod1_e_raw.
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions
% No d1; e iid.
%
% a1: standard endogenous state, this is the one divide-and-conquer (and then the grid interp layer) is applied to
% a2: standard endogenous state, this one is folded (kept whole inside the return matrix)
% a3: the riskyasset, a3prime=aprimeFn(d2,d3,u)
%
% The EV pipeline is unchanged from the DC1_GI1 version except that the "carried forward
% directly" block is now N_a1*N_a2 rather than N_a1, so that is the stride against which
% the riskyasset index is offset. DiscountedEV is (d3,a1prime,a2prime) with no a3 term and
% no e term (EV is integrated over e), so it broadcasts against the trailing e dimension.
%
% Policy3 rows: (1)=d3, (2)=joint(a1prime midpoint,a2prime), (3)=a1prime L2.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_d4=prod(n_d4);
special_n_d4=ones(1,length(n_d4));
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a3=prod(n_a3);
N_a=N_a1*N_a2*N_a3;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);
N_u=prod(n_u);

N_a12=N_a1*N_a2; % the two standard assets, carried forward directly

n_d23=[n_d2,n_d3];
N_d23=N_d2*N_d3;
d23_grid=[d2_grid; d3_grid];

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_bothz,N_e,N_j,'gpuArray'); % (1)=d3, (2)=joint(a1prime midpoint,a2prime), (3)=L2ind
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');
d2Policy=ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');
d4Policy=ones(1,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
u_grid=gpuArray(u_grid);
a3_grid=gpuArray(a3_grid);
a2_grid=gpuArray(a2_grid);
a1_grid=gpuArray(a1_grid);
d23_grid=gpuArray(d23_grid);
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
d3_gridvals=gpuArray(CreateGridvals(n_d3,d3_grid,1));
d3d4a1a2_gridvals=gpuArray(CreateGridvals([n_d3,n_d4,n_a1,n_a2],[d3_grid;d4_grid;a1_grid;a2_grid],1));
a1a2a3_gridvals=gpuArray(CreateGridvals([n_a1,n_a2,n_a3],[a1_grid;a2_grid;a3_grid],1));
d4_gridvals=CreateGridvals(n_d4,d4_grid,1);

pi_u_col=pi_u(:);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:N_a1,a1_grid,linspace(1,N_a1,N_a1+(N_a1-1)*n2short))';
N_a1fine=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
zBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2);
d3col=(1:1:N_d3)';                     % [N_d3,1]
a2pcol=reshape(0:1:N_a2-1,[1,1,N_a2]); % [1,1,N_a2prime]

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    special_n_semiz=ones(1,length(n_semiz));
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford4_jj=zeros(N_a,N_bothz,N_e,N_d4,'gpuArray');
Policy3_ford4_jj=zeros(3,N_a,N_bothz,N_e,N_d4,'gpuArray');
flag_ford4_jj=2*ones(N_a,N_bothz,N_e,N_d4,'gpuArray');
d2_ford4_jj=ones(N_a,N_bothz,N_e,N_d4,'gpuArray');


%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d3,n_d4,n_a1,n_a2], [n_a1,n_a2,n_a3], n_bothz, n_e, d3d4a1a2_gridvals, a1a2a3_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=shiftdim(Vtemp,1);
    dindex=rem(maxindex-1,N_d3*N_d4)+1;
    d3part=rem(dindex-1,N_d3)+1;
    d4part=ceil(dindex/N_d3);
    a12primepart=ceil(maxindex/(N_d3*N_d4)); % joint(a1prime,a2prime)
    Policy3(1,:,:,:,N_j)=shiftdim(d3part,-1);
    Policy3(2,:,:,:,N_j)=shiftdim(a12primepart,-1);
    Policy3(3,:,:,:,N_j)=n2short+2;
    d4Policy(1,:,:,:,N_j)=shiftdim(d4part,-1);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]);
    EVpre=sum(V_Jplus1.*shiftdim(pi_e_J(:,N_j+1),-2),3);
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a3primeIndex,a3primeProbs]=CreateRiskyAssetFnMatrix(aprimeFn, n_d23, n_a3, n_u, d23_grid, a3_grid, u_grid, aprimeFnParamsVec,2);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(N_j)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,N_j);
    end

    % Local aliases so the inlined per-period body is otherwise verbatim
    EVnext=EVpre;
    bothz_gridvals=bothz_gridvals_J(:,:,N_j);
    e_gridvals=e_gridvals_J(:,:,N_j);
    pi_z=pi_z_J(:,:,N_j);

    V_jj=zeros(N_a,N_bothz,N_e,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_bothz,N_e,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_bothz,N_e,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_bothz,N_e,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_bothz,N_e,'gpuArray');

    aprimeIndex=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

        % EV / d2index / DiscountedEV are independent of e — compute once per d4
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

        EVres=reshape(EV,[N_d2,N_d3*N_a12,N_bothz]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_bothz]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_bothz]);

        % DiscountedEV: (d3,a1prime,a2prime,-,-,-,bothz), no a3 term and no e term
        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_bothz]);
        DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]); % [N_d3,N_a1fine,N_a2,1,1,1,N_bothz]

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec, 1);
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii)); % [N_d3,1,N_a2prime,1,N_a2,N_a3,N_bothz,N_e]
                    a1primeindexes=loweredge+(0:1:maxgap(ii));                  % [N_d3,maxgap+1,N_a2prime,1,N_a2,N_a3,N_bothz,N_e]
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec, 3);
                    d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4); % linear index into DiscountedEV [N_d3,N_a1,N_a2,1,1,1,N_bothz]
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec, 3);
            aprimez=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBind,-4); % linear index into DiscountedEVinterp [N_d3,N_a1fine,N_a2,1,1,1,N_bothz]
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d3)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

            allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zBind + N_d3*N_a2*N_a*N_bothz*eBind;
            a1mid=midpoint(allind);
            Policy3_ford4_jj(1,:,:,:,d4_c)=d_ind;
            Policy3_ford4_jj(2,:,:,:,d4_c)=a1mid+N_a1*(maxindexL2a2-1); % joint(a1prime midpoint,a2prime)
            Policy3_ford4_jj(3,:,:,:,d4_c)=maxindexL2a1;

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind + N_d3*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind + N_d3*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBind;
            d2_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        elseif vfoptions.lowmemory==1
            % Loop over e inside d4 to reduce memory footprint
            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,N_bothz,'gpuArray');

                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, special_n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, special_n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals, e_val, ReturnFnParamsVec, 3);
                        d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, e_val, ReturnFnParamsVec, 3);
                aprimez=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);

                d_ind       =rem(maxindexL2-1,N_d3)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zBind;
                a1mid=midpoint(allind);
                Policy3_ford4_jj(1,:,:,e_c,d4_c)=d_ind;
                Policy3_ford4_jj(2,:,:,e_c,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
                Policy3_ford4_jj(3,:,:,e_c,d4_c)=maxindexL2a1;

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a,N_bothz]);
                linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind;
                linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind;
                isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBind;
                d2_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
            end
        elseif vfoptions.lowmemory==2
            % Loop over z (semiz blocks, outer) and e (inner) to reduce memory footprint
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                zBindblock=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,:,:,semizblock);
                DiscountedEVinterpblock=DiscountedEVinterp(:,:,:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEVblock;

                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBindblock,-4);
                            entireRHS_ii=ReturnMatrix_ii+DiscountedEVblock(d3aprimez);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end

                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprimez=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBindblock,-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterpblock(aprimez),[N_d3*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(Vtempii,1);

                    d_ind       =rem(maxindexL2-1,N_d3)+1;
                    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                    allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zBindblock;
                    a1mid=midpoint(allind);
                    Policy3_ford4_jj(1,:,semizblock,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,semizblock,e_c,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
                    Policy3_ford4_jj(3,:,semizblock,e_c,d4_c)=maxindexL2a1;

                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBindblock;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBindblock;
                    isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
                    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                    flag_ford4_jj(:,semizblock,e_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBindblock;
                    d2_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d2index_reshblock(lin),1);
                end
            end
        elseif vfoptions.lowmemory==3
            % Loop over bothz (outer) and e (inner) to reduce memory footprint
            for z_c=1:N_bothz
                bothz_val=bothz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                d2index_resh_zc=d2index_resh(:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, special_n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_zc;

                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, special_n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_val, e_val, ReturnFnParamsVec, 3);
                            d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_zc(d3aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end

                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol;
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_zc(aprime),[N_d3*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);

                    d_ind       =rem(maxindexL2-1,N_d3)+1;
                    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                    allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind;
                    a1mid=midpoint(allind);
                    Policy3_ford4_jj(1,:,z_c,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,z_c,e_c,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
                    Policy3_ford4_jj(3,:,z_c,e_c,d4_c)=maxindexL2a1;

                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a]);
                    linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
                    isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
                    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
                    d2_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d2index_resh_zc(lin),1);
                end
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],4);
    N=N_a*N_bothz*N_e;
    P1=reshape(Policy3_ford4_jj(1,:,:,:,:),[N,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:,:),[N,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:,:),[N,N_d4]);
    F =reshape(flag_ford4_jj,[N,N_d4]);
    D2=reshape(d2_ford4_jj,[N,N_d4]);
    rowidx=(1:1:N)';
    gather_idx=rowidx+N*(reshape(d4winner,[N,1])-1);
    Policy3_jj(1,:,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_bothz,N_e]),-1);
    Policy3_jj(2,:,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_bothz,N_e]),-1);
    Policy3_jj(3,:,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_bothz,N_e]),-1);
    PolicyL2flag_jj(1,:,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_bothz,N_e]),-1);
    d2Policy_jj(1,:,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_bothz,N_e]),-1);
    d4Policy_jj(1,:,:,:)=shiftdim(d4winner,-1);

    V(:,:,:,N_j)=V_jj;
    Policy3(:,:,:,:,N_j)=Policy3_jj;
    PolicyL2flag(:,:,:,:,N_j)=PolicyL2flag_jj;
    d2Policy(:,:,:,:,N_j)=d2Policy_jj;
    d4Policy(:,:,:,:,N_j)=d4Policy_jj;
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

    EVnext=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);

    if isstruct(pi_semiz_J)
        pi_semiz=gpuArray(reshape(full(pi_semiz_J.(['j',num2str(jj)])),[N_semiz,N_semiz,N_d4]));
    else
        pi_semiz=pi_semiz_J(:,:,:,jj);
    end

    % Local aliases so the inlined per-period body is otherwise verbatim
    bothz_gridvals=bothz_gridvals_J(:,:,jj);
    e_gridvals=e_gridvals_J(:,:,jj);
    pi_z=pi_z_J(:,:,jj);

    V_jj=zeros(N_a,N_bothz,N_e,'gpuArray');
    Policy3_jj=zeros(3,N_a,N_bothz,N_e,'gpuArray');
    PolicyL2flag_jj=2*ones(1,N_a,N_bothz,N_e,'gpuArray');
    d2Policy_jj=ones(1,N_a,N_bothz,N_e,'gpuArray');
    d4Policy_jj=ones(1,N_a,N_bothz,N_e,'gpuArray');

    aprimeIndex=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex-1,N_a12,1);
    aprimeplus1Index=repelem((1:1:N_a12)',N_d23,N_u)+N_a12*repmat(a3primeIndex,N_a12,1);

    for d4_c=1:N_d4
        pi_bothz=kron(pi_z, pi_semiz(:,:,d4_c));
        d3_with_d4=[d3_gridvals,repmat(d4_gridvals(d4_c,:),N_d3,1)];

        % EV / d2index / DiscountedEV are independent of e — compute once per d4
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

        EVres=reshape(EV,[N_d2,N_d3*N_a12,N_bothz]);
        [EV_onlyd3,d2index]=max(EVres,[],1);
        EV_onlyd3=reshape(EV_onlyd3,[N_d3*N_a12,N_bothz]);
        d2index_resh=reshape(d2index,[N_d3,N_a1,N_a2,N_bothz]);

        DiscountedEV=DiscountFactorParamsVec*reshape(EV_onlyd3,[N_d3,N_a1,N_a2,1,1,1,N_bothz]);
        DiscountedEVinterp=permute(interp1(a1_grid,permute(DiscountedEV,[2,1,3,4,5,6,7]),a1prime_grid),[2,1,3,4,5,6,7]);

        if vfoptions.lowmemory==0
            midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,N_bothz,N_e,'gpuArray');

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec, 1);
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,:,level1ii,:,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:,:), [],8),[],7),[],6),[],5),[],3),[],1));
            for ii=1:(vfoptions.level1n-1)
                curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,:,ii,:,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec, 3);
                    d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                    [~,maxindex_inner]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=maxindex_inner+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,:,ii,:,:,:,:);
                    midpoint(:,1,:,curra1inner,:,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1,1);
                end
            end

            midpoint=max(min(midpoint,N_a1-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, e_gridvals, ReturnFnParamsVec, 3);
            aprimez=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBind,-4);
            entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_bothz,N_e]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V_ford4_jj(:,:,:,d4_c)=shiftdim(Vtempii,1);

            d_ind       =rem(maxindexL2-1,N_d3)+1;
            maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
            maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

            allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zBind + N_d3*N_a2*N_a*N_bothz*eBind;
            a1mid=midpoint(allind);
            Policy3_ford4_jj(1,:,:,:,d4_c)=d_ind;
            Policy3_ford4_jj(2,:,:,:,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
            Policy3_ford4_jj(3,:,:,:,d4_c)=maxindexL2a1;

            ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a,N_bothz,N_e]);
            linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind + N_d3*n2long*N_a2*N_a*N_bothz*eBind;
            linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind + N_d3*n2long*N_a2*N_a*N_bothz*eBind;
            isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
            inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
            inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
            flag_ford4_jj(:,:,:,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

            lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBind;
            d2_ford4_jj(:,:,:,d4_c)=shiftdim(d2index_resh(lin),1);
        elseif vfoptions.lowmemory==1
            % Loop over e inside d4 to reduce memory footprint
            for e_c=1:N_e
                e_val=e_gridvals(e_c,:);
                midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,N_bothz,'gpuArray');

                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, special_n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_gridvals, e_val, ReturnFnParamsVec, 1);
                entireRHS_ii=ReturnMatrix_ii+DiscountedEV;

                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, special_n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_gridvals, e_val, ReturnFnParamsVec, 3);
                        d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBind,-4);
                        entireRHS_ii=ReturnMatrix_ii+DiscountedEV(d3aprimez);
                        [~,maxindex_inner]=max(entireRHS_ii,[],2);
                        midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,:,ii,:,:,:);
                        midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                    end
                end

                midpoint=max(min(midpoint,N_a1-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, n_bothz, special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_gridvals, e_val, ReturnFnParamsVec, 3);
                aprimez=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBind,-4);
                entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp(aprimez),[N_d3*n2long*N_a2,N_a,N_bothz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                V_ford4_jj(:,:,e_c,d4_c)=shiftdim(Vtempii,1);

                d_ind       =rem(maxindexL2-1,N_d3)+1;
                maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zBind;
                a1mid=midpoint(allind);
                Policy3_ford4_jj(1,:,:,e_c,d4_c)=d_ind;
                Policy3_ford4_jj(2,:,:,e_c,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
                Policy3_ford4_jj(3,:,:,e_c,d4_c)=maxindexL2a1;

                ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a,N_bothz]);
                linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind;
                linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBind;
                isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
                inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                flag_ford4_jj(:,:,e_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBind;
                d2_ford4_jj(:,:,e_c,d4_c)=shiftdim(d2index_resh(lin),1);
            end
        elseif vfoptions.lowmemory==2
            % Loop over z (semiz blocks, outer) and e (inner) to reduce memory footprint
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals(semizblock,:);
                zBindblock=shiftdim(gpuArray(0:1:N_semiz-1),-1);
                DiscountedEVblock=DiscountedEV(:,:,:,:,:,:,semizblock);
                DiscountedEVinterpblock=DiscountedEVinterp(:,:,:,:,:,:,semizblock);
                d2index_reshblock=d2index_resh(:,:,:,semizblock);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,N_semiz,'gpuArray');

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEVblock;

                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(max(max( maxindex1(:,1,:,2:end,:,:,:)-maxindex1(:,1,:,1:end-1,:,:,:), [],7),[],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 3);
                            d3aprimez=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol + N_d3*N_a1*N_a2*shiftdim(zBindblock,-4);
                            entireRHS_ii=ReturnMatrix_ii+DiscountedEVblock(d3aprimez);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:,:);
                            midpoint(:,1,:,curra1inner,:,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1,1);
                        end
                    end

                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, [n_semiz,ones(1,length(n_z))], special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, z_valblock, e_val, ReturnFnParamsVec, 3);
                    aprimez=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol + N_d3*N_a1fine*N_a2*shiftdim(zBindblock,-4);
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterpblock(aprimez),[N_d3*n2long*N_a2,N_a,N_semiz]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(Vtempii,1);

                    d_ind       =rem(maxindexL2-1,N_d3)+1;
                    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                    allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind + N_d3*N_a2*N_a*zBindblock;
                    a1mid=midpoint(allind);
                    Policy3_ford4_jj(1,:,semizblock,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,semizblock,e_c,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
                    Policy3_ford4_jj(3,:,semizblock,e_c,d4_c)=maxindexL2a1;

                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a,N_semiz]);
                    linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBindblock;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind + N_d3*n2long*N_a2*N_a*zBindblock;
                    isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
                    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                    flag_ford4_jj(:,semizblock,e_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1)+N_d3*N_a1*N_a2*zBindblock;
                    d2_ford4_jj(:,semizblock,e_c,d4_c)=shiftdim(d2index_reshblock(lin),1);
                end
            end
        elseif vfoptions.lowmemory==3
            % Loop over bothz (outer) and e (inner) to reduce memory footprint
            for z_c=1:N_bothz
                bothz_val=bothz_gridvals(z_c,:);
                DiscountedEV_zc=DiscountedEV(:,:,:,:,:,:,z_c);
                DiscountedEVinterp_zc=DiscountedEVinterp(:,:,:,:,:,:,z_c);
                d2index_resh_zc=d2index_resh(:,:,:,z_c);
                for e_c=1:N_e
                    e_val=e_gridvals(e_c,:);
                    midpoint=zeros(N_d3,1,N_a2,N_a1,N_a2,N_a3,'gpuArray');

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, special_n_e, d3_with_d4, a1_grid, a2_gridvals, a1_grid(level1ii), a2_gridvals, a3_grid, bothz_val, e_val, ReturnFnParamsVec, 1);
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_zc;

                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,:,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(max( maxindex1(:,1,:,2:end,:,:)-maxindex1(:,1,:,1:end-1,:,:), [],6),[],5),[],3),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curra1inner=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,:,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, special_n_e, d3_with_d4, a1_grid(a1primeindexes), a2_gridvals, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, a3_grid, bothz_val, e_val, ReturnFnParamsVec, 3);
                            d3aprime=d3col + N_d3*(a1primeindexes-1) + N_d3*N_a1*a2pcol;
                            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_zc(d3aprime);
                            [~,maxindex_inner]=max(entireRHS_ii,[],2);
                            midpoint(:,1,:,curra1inner,:,:)=maxindex_inner+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,:,ii,:,:);
                            midpoint(:,1,:,curra1inner,:,:)=repelem(loweredge,1,1,1,level1iidiff(ii),1,1);
                        end
                    end

                    midpoint=max(min(midpoint,N_a1-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, 0,[n_d3,special_n_d4], n_a2, n_a3, special_n_bothz, special_n_e, d3_with_d4, a1prime_grid(a1primeindexesfine), a2_gridvals, a1_grid, a2_gridvals, a3_grid, bothz_val, e_val, ReturnFnParamsVec, 3);
                    aprime=d3col + N_d3*(a1primeindexesfine-1) + N_d3*N_a1fine*a2pcol;
                    entireRHS_ii=reshape(ReturnMatrix_ii+DiscountedEVinterp_zc(aprime),[N_d3*n2long*N_a2,N_a]);
                    [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
                    V_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(Vtempii,1);

                    d_ind       =rem(maxindexL2-1,N_d3)+1;
                    maxindexL2a1=rem(floor((maxindexL2-1)/N_d3),n2long)+1;
                    maxindexL2a2=floor((maxindexL2-1)/(N_d3*n2long))+1;

                    allind=d_ind + N_d3*(maxindexL2a2-1) + N_d3*N_a2*aind;
                    a1mid=midpoint(allind);
                    Policy3_ford4_jj(1,:,z_c,e_c,d4_c)=d_ind;
                    Policy3_ford4_jj(2,:,z_c,e_c,d4_c)=a1mid+N_a1*(maxindexL2a2-1);
                    Policy3_ford4_jj(3,:,z_c,e_c,d4_c)=maxindexL2a1;

                    ReturnMatrix_ii_flat=reshape(ReturnMatrix_ii,[N_d3*n2long*N_a2,N_a]);
                    linidx_lower  = d_ind                   + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
                    linidx_upper  = d_ind + N_d3*(n2long-1) + N_d3*n2long*(maxindexL2a2-1) + N_d3*n2long*N_a2*aind;
                    isInfLower    = (ReturnMatrix_ii_flat(linidx_lower) == -Inf);
                    isInfUpper    = (ReturnMatrix_ii_flat(linidx_upper) == -Inf);
                    inLowerStrict = (maxindexL2a1 >= 2)         & (maxindexL2a1 <= n2short+1);
                    inUpperStrict = (maxindexL2a1 >= n2short+3) & (maxindexL2a1 <= n2long-1);
                    flag_ford4_jj(:,z_c,e_c,d4_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),1);

                    lin=d_ind+N_d3*(a1mid-1)+N_d3*N_a1*(maxindexL2a2-1);
                    d2_ford4_jj(:,z_c,e_c,d4_c)=shiftdim(d2index_resh_zc(lin),1);
                end
            end
        end
    end

    % Cross-d4 max
    [V_jj,d4winner]=max(V_ford4_jj,[],4);
    N=N_a*N_bothz*N_e;
    P1=reshape(Policy3_ford4_jj(1,:,:,:,:),[N,N_d4]);
    P2=reshape(Policy3_ford4_jj(2,:,:,:,:),[N,N_d4]);
    P3=reshape(Policy3_ford4_jj(3,:,:,:,:),[N,N_d4]);
    F =reshape(flag_ford4_jj,[N,N_d4]);
    D2=reshape(d2_ford4_jj,[N,N_d4]);
    rowidx=(1:1:N)';
    gather_idx=rowidx+N*(reshape(d4winner,[N,1])-1);
    Policy3_jj(1,:,:,:)=shiftdim(reshape(P1(gather_idx),[N_a,N_bothz,N_e]),-1);
    Policy3_jj(2,:,:,:)=shiftdim(reshape(P2(gather_idx),[N_a,N_bothz,N_e]),-1);
    Policy3_jj(3,:,:,:)=shiftdim(reshape(P3(gather_idx),[N_a,N_bothz,N_e]),-1);
    PolicyL2flag_jj(1,:,:,:)=shiftdim(reshape(F(gather_idx),[N_a,N_bothz,N_e]),-1);
    d2Policy_jj(1,:,:,:)=shiftdim(reshape(D2(gather_idx),[N_a,N_bothz,N_e]),-1);
    d4Policy_jj(1,:,:,:)=shiftdim(d4winner,-1);

    V(:,:,:,jj)=V_jj;
    Policy3(:,:,:,:,jj)=Policy3_jj;
    PolicyL2flag(:,:,:,:,jj)=PolicyL2flag_jj;
    d2Policy(:,:,:,:,jj)=d2Policy_jj;
    d4Policy(:,:,:,:,jj)=d4Policy_jj;
end


%% Switch Policy3(2,:) from 'midpoint' to 'lower grid index'
% Policy3(2,:) is joint(a1prime midpoint,a2prime); decrementing the a1prime component keeps it
% within the same a2prime block because midpoint>=2.
adjust=(Policy3(3,:,:,:,:)<1+n2short+1);
Policy3(2,:,:,:,:)=Policy3(2,:,:,:,:)-adjust;
Policy3(3,:,:,:,:)=adjust.*Policy3(3,:,:,:,:)+(1-adjust).*(Policy3(3,:,:,:,:)-n2short-1);

%% Component-form Policy (no d1, with e): rows (d2,d3,d4,a1prime,a2prime,L2,L2flag)
Policy=zeros(7,N_a,N_bothz,N_e,N_j,'gpuArray');
Policy(1,:,:,:,:)=d2Policy;
Policy(2,:,:,:,:)=Policy3(1,:,:,:,:);
Policy(3,:,:,:,:)=d4Policy;
Policy(4,:,:,:,:)=rem(Policy3(2,:,:,:,:)-1,N_a1)+1; % a1prime (lower grid index)
Policy(5,:,:,:,:)=floor((Policy3(2,:,:,:,:)-1)/N_a1)+1; % a2prime
Policy(6,:,:,:,:)=Policy3(3,:,:,:,:);
Policy(7,:,:,:,:)=PolicyL2flag;

end
