function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteSemiExoS_DC1_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic + ExperienceAssete + SemiExo + DivideConquer + GridInterpLayer (no d1).
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (optional), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, e, ...)   (depends on current e; not on z or semiz)
%
% Sophisticated QH over the divide-conquer + grid-interp argmax:
%   Policy (and Vhat) come from the  F + beta0*beta*EV  argmax (QH-perceived).
%   Vunderbar is the  F + beta*EV  value GATHERED at that same argmax (not re-maximised),
%   computed as  Vhat + (beta-beta0beta)*EV_at_policy.
% beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj).
% Backward EVpre uses Vunderbar.
% Policy rows: (d2, d3, midpoint, L2ind) + L2flag appended.
%
% lowmemory levels {0,1,2,3} implemented (shocks: z markov + semiz + e iid).

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;
N_e=prod(n_e);

Vhat=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy=zeros(4,N_a,N_bothz,N_e,N_j,'gpuArray'); % (d2, d3, midpoint, L2ind)
PolicyL2flag=2*ones(1,N_a,N_bothz,N_e,N_j,'gpuArray'); % L2 flag: 1=all to lower, 2=usual, 3=all to upper

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

% Preallocate
if vfoptions.lowmemory==0
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_bothz,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_bothz,'gpuArray');
elseif vfoptions.lowmemory==2
    midpoint=zeros(N_d2,1,N_a1,N_a2,N_semiz,'gpuArray');
elseif vfoptions.lowmemory==3
    midpoint=zeros(N_d2,1,N_a1,N_a2,'gpuArray');
end

% Per-d3 arrays (hat=QH-perceived argmax; under=beta-value gathered at that argmax)
Vhat_ford3=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Vunderbar_ford3=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy3_ford3=zeros(3,N_a,N_bothz,N_e,N_d3,'gpuArray'); % (d2, midpoint, L2ind)
flag_ford3=2*ones(1,N_a,N_bothz,N_e,N_d3,'gpuArray');

% n-Monotonicity
level1ii=round(linspace(1,n_a1,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
eBind=shiftdim(gpuArray(0:1:N_e-1),-2); % already includes -1
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-3); % already includes -1
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1); % already includes -1
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            midpoint(:,1,level1ii,:,:,:)=maxindex1;

            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end

            midpoint=max(min(midpoint,n_a1(1)-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Vhat_ford3(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);

                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);

                [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                midpoint(:,1,level1ii,:,:)=maxindex1;

                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        [~,maxindex]=max(ReturnMatrix_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end

                midpoint=max(min(midpoint,n_a1(1)-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                Vhat_ford3(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                    midpoint(:,1,level1ii,:,:)=maxindex1;

                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford3(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);

                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);

                    [~,maxindex1]=max(ReturnMatrix_ii,[],2);

                    midpoint(:,1,level1ii,:)=maxindex1;

                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            [~,maxindex]=max(ReturnMatrix_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end

                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(aprimeindexes), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
                    Vhat_ford3(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    % L2 flag to later avoid -Inf ReturnFn (1=all to lower, 2=usual, 3=all to upper)
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                   + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_ii(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_ii(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                end
            end
        end
    end

    % Max over d3; terminal period has no continuation, so Vunderbar equals Vhat
    [V_jj,maxindex]=max(Vhat_ford3,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    temp=3*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3(flat_idx),[1,N_a,N_bothz,N_e]);
    Vunderbar(:,:,:,N_j)=Vhat(:,:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeProbs_d2a1a2e=reshape(aprimeProbs_d2a1a2e,[N_d2*N_a1,N_a2,1,N_e]);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;


    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3,0);
                    d2aprimeze=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape(d2aprimeze);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind+N_d2*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            EVfine=reshape(EVinterp(d2a1primea2bothze(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vhat_ford3(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
            linidx=reshape(maxindexL2,[1,N_a*N_bothz*N_e])+N_d2*n2long*(0:N_a*N_bothz*N_e-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz,N_e]);
            Vunderbar_ford3(:,:,:,d3_c)=Vhat_ford3(:,:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                EVreshape_e=EVreshape(:,:,:,:,:,e_c);
                EVinterp_e=EVinterp(:,:,:,:,:,e_c);

                % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,1,0);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                EVfine=reshape(EVinterp_e(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                Vhat_ford3(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                
                % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
                linidx=reshape(maxindexL2,[1,N_a*N_bothz])+N_d2*n2long*(0:N_a*N_bothz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz]);
                Vunderbar_ford3(:,:,e_c,d3_c)=Vhat_ford3(:,:,e_c,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                EVreshape_zb=EVreshape(:,:,:,:,semizblock,:);
                EVinterp_zb=EVinterp(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    EVreshape_zbe=EVreshape_zb(:,:,:,:,:,e_c);
                    EVinterp_zbe=EVinterp_zb(:,:,:,:,:,e_c);

                    % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_zbe;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_zbe(d2aprimez);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    EVfine=reshape(EVinterp_zbe(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    Vhat_ford3(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                    
                    % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
                    linidx=reshape(maxindexL2,[1,N_a*N_semiz])+N_d2*n2long*(0:N_a*N_semiz-1);
                    EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                    Vunderbar_ford3(:,semizblock,e_c,d3_c)=Vhat_ford3(:,semizblock,e_c,d3_c)+(beta-beta0beta)*EV_at_policy;
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    EVreshape_ze=EVreshape(:,:,:,:,z_c,e_c);
                    EVinterp_ze=EVinterp(:,:,:,:,z_c,e_c);

                    % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze(d2aprime);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    EVfine=reshape(EVinterp_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    Vhat_ford3(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                   + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                    
                    % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
                    linidx=reshape(maxindexL2,[1,N_a])+N_d2*n2long*(0:N_a-1);
                    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                    Vunderbar_ford3(:,z_c,e_c,d3_c)=Vhat_ford3(:,z_c,e_c,d3_c)+(beta-beta0beta)*EV_at_policy;
                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(Vhat_ford3,[],4);
    Vhat(:,:,:,N_j)=V_jj;
    Policy(2,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,N_j)=reshape(Vunderbar_ford3((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[N_a,N_bothz,N_e]);
    temp=3*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,N_j)=reshape(Policy3_ford3(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,N_j)=reshape(Policy3_ford3(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,N_j)=reshape(Policy3_ford3(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,N_j)=reshape(flag_ford3(flat_idx),[1,N_a,N_bothz,N_e]);
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex=reshape(aprimeIndex,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeplus1Index=reshape(aprimeplus1Index,[N_d2*N_a1,N_a2,1,N_e]);
    aprimeProbs_d2a1a2e=reshape(aprimeProbs_d2a1a2e,[N_d2*N_a1,N_a2,1,N_e]);

    % Continuation value is Vunderbar, integrated over e'
    EVpre=sum(Vunderbar(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3);


    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape;
            [~,maxindex1]=max(entireRHS_ii,[],2);
            midpoint(:,1,level1ii,:,:,:)=maxindex1;
            maxgap=squeeze(max(max(max(max(maxindex1(:,1,2:end,:,:,:)-maxindex1(:,1,1:end-1,:,:,:),[],6),[],5),[],4),[],1));
            for ii=1:(vfoptions.level1n-1)
                curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii,:,:,:),N_a1-maxgap(ii));
                    a1primeindexes=loweredge+(0:1:maxgap(ii));
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3,0);
                    d2aprimeze=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind+N_d2*N_a1*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape(d2aprimeze);
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoint(:,1,curraindex,:,:,:)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii,:,:,:);
                    midpoint(:,1,curraindex,:,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                end
            end
            midpoint=max(min(midpoint,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d2a1primea2bothze=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind+N_d2*N_a1prime*N_a2*N_bothz*shiftdim((0:1:N_e-1),-4);
            EVfine=reshape(EVinterp(d2a1primea2bothze(:)),[N_d2*n2long,N_a1*N_a2,N_bothz,N_e]);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vhat_ford3(:,:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d2)+1;
            allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind+N_d2*N_a*N_bothz*eBind;
            Policy3_ford3(1,:,:,:,d3_c)=d_ind;
            Policy3_ford3(2,:,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy3_ford3(3,:,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
            L2offset = ceil(maxindexL2/N_d2);
            linidx_lower = d_ind                  + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind + N_d2*n2long*N_a*N_bothz*eBind;
            isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
            isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford3(1,:,:,:,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);

            % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
            linidx=reshape(maxindexL2,[1,N_a*N_bothz*N_e])+N_d2*n2long*(0:N_a*N_bothz*N_e-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz,N_e]);
            Vunderbar_ford3(:,:,:,d3_c)=Vhat_ford3(:,:,:,d3_c)+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex+bothz_offset;
            lin_upper=aprimeplus1Index+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                EVreshape_e=EVreshape(:,:,:,:,:,e_c);
                EVinterp_e=EVinterp(:,:,:,:,:,e_c);

                % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
                ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,1,0);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e;
                [~,maxindex1]=max(entireRHS_ii,[],2);
                midpoint(:,1,level1ii,:,:)=maxindex1;
                maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                for ii=1:(vfoptions.level1n-1)
                    curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                    if maxgap(ii)>0
                        loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                        a1primeindexes=loweredge+(0:1:maxgap(ii));
                        ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,3,0);
                        d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*bothzind;
                        entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_e(d2aprimez);
                        [~,maxindex]=max(entireRHS_ii,[],2);
                        midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                    else
                        loweredge=maxindex1(:,1,ii,:,:);
                        midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                    end
                end
                midpoint=max(min(midpoint,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,2,0);
                d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*bothzind;
                EVfine=reshape(EVinterp_e(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_bothz]);
                entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                Vhat_ford3(:,:,e_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d2)+1;
                allind=d_ind+N_d2*aind+N_d2*N_a*bothzBind;
                Policy3_ford3(1,:,:,e_c,d3_c)=d_ind;
                Policy3_ford3(2,:,:,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy3_ford3(3,:,:,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                L2offset = ceil(maxindexL2/N_d2);
                linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*bothzBind;
                isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford3(1,:,:,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                
                % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
                linidx=reshape(maxindexL2,[1,N_a*N_bothz])+N_d2*n2long*(0:N_a*N_bothz-1);
                EV_at_policy=reshape(EVfine(linidx),[N_a,N_bothz]);
                Vunderbar_ford3(:,:,e_c,d3_c)=Vhat_ford3(:,:,e_c,d3_c)+(beta-beta0beta)*EV_at_policy;
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);
                EVreshape_zb=EVreshape(:,:,:,:,semizblock,:);
                EVinterp_zb=EVinterp(:,:,:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    EVreshape_zbe=EVreshape_zb(:,:,:,:,:,e_c);
                    EVinterp_zbe=EVinterp_zb(:,:,:,:,:,e_c);

                    % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_zbe;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:,:)=maxindex1;
                    maxgap=squeeze(max(max(max(maxindex1(:,1,2:end,:,:)-maxindex1(:,1,1:end-1,:,:),[],5),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,3,0);
                            d2aprimez=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind+N_d2*N_a1*N_a2*semizind;
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_zbe(d2aprimez);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:,:);
                            midpoint(:,1,curraindex,:,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))],special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2bothz=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind+N_d2*N_a1prime*N_a2*semizind;
                    EVfine=reshape(EVinterp_zbe(d2a1primea2bothz(:)),[N_d2*n2long,N_a1*N_a2,N_semiz]);
                    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    Vhat_ford3(:,semizblock,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind+N_d2*N_a*semizBind;
                    Policy3_ford3(1,:,semizblock,e_c,d3_c)=d_ind;
                    Policy3_ford3(2,:,semizblock,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3(3,:,semizblock,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                   + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind + N_d2*n2long*N_a*semizBind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3(1,:,semizblock,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                    
                    % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
                    linidx=reshape(maxindexL2,[1,N_a*N_semiz])+N_d2*n2long*(0:N_a*N_semiz-1);
                    EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
                    Vunderbar_ford3(:,semizblock,e_c,d3_c)=Vhat_ford3(:,semizblock,e_c,d3_c)+(beta-beta0beta)*EV_at_policy;
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);
            EV1=EV_2D(aprimeIndex+bothz_offset);
            EV2=EV_2D(aprimeplus1Index+bothz_offset);
            skipinterp=(EV1==EV2);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,N_bothz,1);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            EVreshape=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz,N_e]);
            EVinterp=permute(interp1(a1_gridvals,permute(EVreshape,[2,1,3,4,5,6]),a1prime_grid),[2,1,3,4,5,6]);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    EVreshape_ze=EVreshape(:,:,:,:,z_c,e_c);
                    EVinterp_ze=EVinterp(:,:,:,:,z_c,e_c);

                    % hat (QH-perceived): F + beta0*beta*EV, full divide-conquer + grid-interp
                    ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n_a1,vfoptions.level1n,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals, a1_gridvals(level1ii), a2_gridvals, z_val, e_val, ReturnFnParamsVec,1,0);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze;
                    [~,maxindex1]=max(entireRHS_ii,[],2);
                    midpoint(:,1,level1ii,:)=maxindex1;
                    maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1));
                    for ii=1:(vfoptions.level1n-1)
                        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1);
                        if maxgap(ii)>0
                            loweredge=min(maxindex1(:,1,ii,:),N_a1-maxgap(ii));
                            a1primeindexes=loweredge+(0:1:maxgap(ii));
                            ReturnMatrix_ii=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],maxgap(ii)+1,level1iidiff(ii),n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1_gridvals(a1primeindexes), a1_gridvals(level1ii(ii)+1:level1ii(ii+1)-1), a2_gridvals, z_val, e_val, ReturnFnParamsVec,3,0);
                            d2aprime=(1:1:N_d2)'+N_d2*(a1primeindexes-1)+N_d2*N_a1*a2ind;
                            entireRHS_ii=ReturnMatrix_ii+beta0beta*EVreshape_ze(d2aprime);
                            [~,maxindex]=max(entireRHS_ii,[],2);
                            midpoint(:,1,curraindex,:)=maxindex+(loweredge-1);
                        else
                            loweredge=maxindex1(:,1,ii,:);
                            midpoint(:,1,curraindex,:)=repelem(loweredge,1,1,level1iidiff(ii),1);
                        end
                    end
                    midpoint=max(min(midpoint,n_a1(1)-1),2);
                    a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                    ReturnMatrix_L2=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, 0,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz,special_n_e, d23_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,2,0);
                    d2a1primea2=(1:1:N_d2)'+N_d2*(a1primeindexesfine-1)+N_d2*N_a1prime*a2ind;
                    EVfine=reshape(EVinterp_ze(d2a1primea2(:)),[N_d2*n2long,N_a1*N_a2]);
                    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
                    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
                    Vhat_ford3(:,z_c,e_c,d3_c)=shiftdim(Vtempii,1);
                    d_ind=rem(maxindexL2-1,N_d2)+1;
                    allind=d_ind+N_d2*aind;
                    Policy3_ford3(1,:,z_c,e_c,d3_c)=d_ind;
                    Policy3_ford3(2,:,z_c,e_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                    Policy3_ford3(3,:,z_c,e_c,d3_c)=shiftdim(ceil(maxindexL2/N_d2),-1);
                    L2offset = ceil(maxindexL2/N_d2);
                    linidx_lower = d_ind                   + N_d2*n2long*aind;
                    linidx_upper = d_ind + N_d2*(n2long-1) + N_d2*n2long*aind;
                    isInfLower = (ReturnMatrix_L2(linidx_lower) == -Inf);
                    isInfUpper = (ReturnMatrix_L2(linidx_upper) == -Inf);
                    inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                    inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                    flag_ford3(1,:,z_c,e_c,d3_c) = shiftdim(squeeze(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper)),-1);
                    
                    % under: F + beta*EV gathered at the hat-argmax (= Vhat + (beta-beta0beta)*EV_at_policy)
                    linidx=reshape(maxindexL2,[1,N_a])+N_d2*n2long*(0:N_a-1);
                    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
                    Vunderbar_ford3(:,z_c,e_c,d3_c)=Vhat_ford3(:,z_c,e_c,d3_c)+(beta-beta0beta)*EV_at_policy;
                end
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(Vhat_ford3,[],4);
    Vhat(:,:,:,jj)=V_jj;
    Policy(2,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    Vunderbar(:,:,:,jj)=reshape(Vunderbar_ford3((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[N_a,N_bothz,N_e]);
    temp=3*((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)-1);
    Policy(1,:,:,:,jj)=reshape(Policy3_ford3(1+temp),[1,N_a,N_bothz,N_e]);
    Policy(3,:,:,:,jj)=reshape(Policy3_ford3(2+temp),[1,N_a,N_bothz,N_e]);
    Policy(4,:,:,:,jj)=reshape(Policy3_ford3(3+temp),[1,N_a,N_bothz,N_e]);
    flat_idx=(1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1);
    PolicyL2flag(1,:,:,:,jj)=reshape(flag_ford3(flat_idx),[1,N_a,N_bothz,N_e]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(4,:,:,:,:)<1+n2short+1);
Policy(3,:,:,:,:)=Policy(3,:,:,:,:)-adjust;
Policy(4,:,:,:,:)=adjust.*Policy(4,:,:,:,:)+(1-adjust).*(Policy(4,:,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
