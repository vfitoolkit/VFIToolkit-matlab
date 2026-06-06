function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_nod_raw(V,n_a,n_z,N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated QH: V carries Vunderbar; Policy = QH choice; Vhat is the agent's-perspective (beta0*beta) value.

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(3,N_a,N_z,N_j,'gpuArray'); % [midpoint; aprimeL2ind; L2flag]
Vhat=zeros(N_a,N_z,N_j,'gpuArray');

%%
if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a,N_z,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(1,N_a,'gpuArray');
    special_n_z=ones(1,length(n_z));
elseif vfoptions.lowmemory>=2
    error('vfoptions.lowmemory>=2 not supported')
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1);

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);


%% j=N_j: terminal age has no continuation in TPath
Vtemp_j=V(:,:,N_j);

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

    [~,maxindex1]=max(ReturnMatrix_ii,[],1);
    midpoints_jj(1,level1ii,:)=maxindex1;

    maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            [~,maxindex]=max(ReturnMatrix_ii,[],1);
            midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
        else
            loweredge=maxindex1(1,ii,:);
            midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
        end
    end

    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    V(:,:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);
    isInfLower    = (ReturnMatrix_ii(1,     :,:) == -Inf);
    isInfUpper    = (ReturnMatrix_ii(n2long,:,:) == -Inf);
    inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
    inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
    Policy(3,:,:,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals_J(z_c,:,N_j);
        ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

        [~,maxindex1]=max(ReturnMatrix_ii_z,[],1);
        midpoints_jj(1,level1ii)=maxindex1;

        maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_z=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                [~,maxindex]=max(ReturnMatrix_ii_z,[],1);
                midpoints_jj(1,curraindex)=shiftdim(maxindex+(loweredge-1),1);
            else
                loweredge=maxindex1(1,ii);
                midpoints_jj(1,curraindex)=repelem(loweredge,length(curraindex),1);
            end
        end

        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,z_c,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,z_c,N_j)=shiftdim(squeeze(midpoints_jj),-1);
        Policy(2,:,z_c,N_j)=shiftdim(maxindexL2,-1);
        isInfLower    = (ReturnMatrix_ii(1,     :) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(n2long,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        Policy(3,:,z_c,N_j) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);
    end
end
Vhat(:,:,N_j)=V(:,:,N_j); % terminal: Vhat coincides with V (Vunderbar)


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vtemp_j;
    Vtemp_j=V(:,:,jj);

    EV=VKronNext_j.*shiftdim(pi_z_J(:,:,jj)',-1);
    EV(isnan(EV))=0;
    EV=sum(EV,2);

    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        %% Vhat (beta0*beta) — find QH-optimal Policy
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [~,maxindex1]=max(entireRHS_ii,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                aprimez=aprimeindexes+N_a*zind;
                entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimez),[(maxgap(ii)+1),1,N_z]);
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,n_z,aprime_grid(aprimeindexes),a_grid,z_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        aprimez=aprimeindexes+n2aprime*zind;
        EVfine=reshape(EVinterp(aprimez(:)),[n2long,N_a,N_z]);
        entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vhat_jj=shiftdim(Vtempii,1);
        Vhat(:,:,jj)=Vhat_jj;
        Policy(1,:,:,jj)=shiftdim(squeeze(midpoints_jj),-1);
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1);
        isInfLower    = (ReturnMatrix_L2(1,     :,:) == -Inf);
        isInfUpper    = (ReturnMatrix_L2(n2long,:,:) == -Inf);
        inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
        inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
        Policy(3,:,:,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

        linidx=reshape(maxindexL2,[1,N_a*N_z])+n2long*(0:N_a*N_z-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_z]);
        V(:,:,jj)=Vhat_jj+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for z_c=1:N_z
            z_val=z_gridvals_J(z_c,:,jj);
            EV_z=EV(:,:,z_c);
            EVinterp_z=EVinterp(:,:,z_c);

            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid, a_grid(level1ii), z_val, ReturnFnParamsVec,1);

            %% Vhat (beta0*beta) — find QH-optimal Policy
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV_z;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            midpoints_jj(1,level1ii)=maxindex1;
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn, special_n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV_z(aprimeindexes);
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex)=shiftdim(maxindex+(loweredge-1),1);
                else
                    loweredge=maxindex1(1,ii);
                    midpoints_jj(1,curraindex)=repelem(loweredge,length(curraindex),1);
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2=CreateReturnFnMatrix_Disc_DC1_nod(ReturnFn,special_n_z,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            EVfine_z=reshape(EVinterp_z(aprimeindexes(:)),[n2long,N_a]);
            entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine_z;
            [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
            Vhat_jj_z=shiftdim(Vtempii,1);
            Vhat(:,z_c,jj)=Vhat_jj_z;
            Policy(1,:,z_c,jj)=shiftdim(squeeze(midpoints_jj),-1);
            Policy(2,:,z_c,jj)=shiftdim(maxindexL2,-1);
            isInfLower    = (ReturnMatrix_L2(1,     :) == -Inf);
            isInfUpper    = (ReturnMatrix_L2(n2long,:) == -Inf);
            inLowerStrict = (maxindexL2 >= 2)         & (maxindexL2 <= n2short+1);
            inUpperStrict = (maxindexL2 >= n2short+3) & (maxindexL2 <= n2long-1);
            Policy(3,:,z_c,jj) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper),-1);

            linidx_z=reshape(maxindexL2,[1,N_a])+n2long*(0:N_a-1);
            EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
            V(:,z_c,jj)=Vhat_jj_z+(beta-beta0beta)*EV_at_policy_z;
        end
    end
end

%% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

end
