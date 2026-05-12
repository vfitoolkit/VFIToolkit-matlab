function varargout=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_GI1_nod_noz_e_raw(n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_GI_nod_noz_e_raw.
% No d variables. No z variable. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j     = max_{a'} u + beta_0*beta*EV_{j+1}
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVinterp_at_policy
%                EVsource = Vunderbar

N_a=prod(n_a);
N_e=prod(n_e);

Vhat=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(2,N_a,N_e,N_j,'gpuArray');

if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a,N_e,'gpuArray');
elseif vfoptions.lowmemory==1
    midpoints_jj=zeros(1,N_a,'gpuArray');
    special_n_e=ones(1,length(n_e));
end

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

pi_e_J=shiftdim(pi_e_J,-1);

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);
        [~,maxindex1]=max(ReturnMatrix_ii,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                [~,maxindex]=max(ReturnMatrix_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                midpoints_jj(1,curraindex,:)=repelem(maxindex1(1,ii,:),1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);

        Vunderbar=Vhat;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec, 1);
            [~,maxindex1]=max(ReturnMatrix_ii,[],1);
            midpoints_jj(1,level1ii)=maxindex1;
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec, 2);
                    [~,maxindex]=max(ReturnMatrix_ii,[],1);
                    midpoints_jj(1,curraindex)=maxindex+(loweredge-1);
                else
                    midpoints_jj(1,curraindex)=repelem(maxindex1(1,ii),1,length(curraindex));
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Vhat(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj),-1);
            Policy(2,:,e_c,N_j)=shiftdim(maxindexL2,-1);
        end
        Vunderbar=Vhat;
    end

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames, N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j),2);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vunderbar=zeros(N_a,N_e,N_j,'gpuArray');

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 1);

        % --- Vhat search (beta0beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [~,maxindex1]=max(entireRHS_ii,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,N_j), ReturnFnParamsVec, 2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(reshape(aprimeindexes(:),[maxgap(ii)+1,1,N_e]));
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                midpoints_jj(1,curraindex,:)=repelem(maxindex1(1,ii,:),1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        EVfine=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a,N_e]);
        entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1);
        linidx=double(reshape(maxindexL2,[1,N_a*N_e]))+n2long*(0:N_a*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_e]);
        Vunderbar(:,:,N_j)=Vhat(:,:,N_j)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec, 1);

            % --- Vhat search (beta0beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            midpoints_jj(1,level1ii)=maxindex1;
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec, 2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(aprimeindexes(:));
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex)=maxindex+(loweredge-1);
                else
                    midpoints_jj(1,curraindex)=repelem(maxindex1(1,ii),1,length(curraindex));
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            EVfine_e=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
            entireRHS_L2_e=ReturnMatrix_L2_e+beta0beta*EVfine_e;
            [Vtempii,maxindexL2]=max(entireRHS_L2_e,[],1);
            Vhat(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj),-1);
            Policy(2,:,e_c,N_j)=shiftdim(maxindexL2,-1);
            linidx_e=double(reshape(maxindexL2,[1,N_a]))+n2long*(0:N_a-1);
            EV_at_policy_e=reshape(EVfine_e(linidx_e),[N_a,1]);
            Vunderbar(:,e_c,N_j)=Vhat(:,e_c,N_j)+(beta-beta0beta)*EV_at_policy_e;
        end
    end
end

%% Iterate backwards through j.
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

    EVsource=Vunderbar(:,:,jj+1);
    EV=sum(EVsource.*pi_e_J(1,:,jj),2);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 1);

        % --- Vhat search (beta0beta) ---
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
        [~,maxindex1]=max(entireRHS_ii,[],1);
        midpoints_jj(1,level1ii,:)=maxindex1;
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii));
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(curraindex), e_gridvals_J(:,:,jj), ReturnFnParamsVec, 2);
                entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(reshape(aprimeindexes(:),[maxgap(ii)+1,1,N_e]));
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                midpoints_jj(1,curraindex,:)=repelem(maxindex1(1,ii,:),1,length(curraindex),1);
            end
        end
        midpoints_jj=max(min(midpoints_jj,n_a-1),2);
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
        ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        EVfine=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a,N_e]);
        entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
        [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
        Vhat(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=shiftdim(squeeze(midpoints_jj),-1);
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1);
        linidx=double(reshape(maxindexL2,[1,N_a*N_e]))+n2long*(0:N_a*N_e-1);
        EV_at_policy=reshape(EVfine(linidx),[N_a,N_e]);
        Vunderbar(:,:,jj)=Vhat(:,:,jj)+(beta-beta0beta)*EV_at_policy;

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);

            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec, 1);

            % --- Vhat search (beta0beta) ---
            entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
            [~,maxindex1]=max(entireRHS_ii,[],1);
            midpoints_jj(1,level1ii)=maxindex1;
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii));
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), e_val, ReturnFnParamsVec, 2);
                    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(aprimeindexes(:));
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex)=maxindex+(loweredge-1);
                else
                    midpoints_jj(1,curraindex)=repelem(maxindex1(1,ii),1,length(curraindex));
                end
            end
            midpoints_jj=max(min(midpoints_jj,n_a-1),2);
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
            ReturnMatrix_L2_e=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            EVfine_e=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
            entireRHS_L2_e=ReturnMatrix_L2_e+beta0beta*EVfine_e;
            [Vtempii,maxindexL2]=max(entireRHS_L2_e,[],1);
            Vhat(:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,jj)=shiftdim(squeeze(midpoints_jj),-1);
            Policy(2,:,e_c,jj)=shiftdim(maxindexL2,-1);
            linidx_e=double(reshape(maxindexL2,[1,N_a]))+n2long*(0:N_a-1);
            EV_at_policy_e=reshape(EVfine_e(linidx_e),[N_a,1]);
            Vunderbar(:,e_c,jj)=Vhat(:,e_c,jj)+(beta-beta0beta)*EV_at_policy_e;
        end
    end
end

%% Post-process Policy: convert [midpoint; aprimeL2ind] to canonical combined index
adjust=(Policy(2,:,:,:)<1+n2short+1);
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust;
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:,:)+N_a*(Policy(2,:,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vhat,Policy};
elseif nOutputs==3
    varargout={Vhat,Policy,Vunderbar};
end

end
