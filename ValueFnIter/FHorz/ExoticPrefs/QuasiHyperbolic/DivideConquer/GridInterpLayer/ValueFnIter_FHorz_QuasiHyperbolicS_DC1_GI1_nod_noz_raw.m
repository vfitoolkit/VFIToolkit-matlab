function varargout=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_GI1_nod_noz_raw(n_a,N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_GI_nod_noz_raw.
% No d variables. No z variable. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j   = max_{a'} u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EVinterp_at_optimal_aprime

N_a=prod(n_a);

Vhat=zeros(N_a,N_j,'gpuArray');
Policy=zeros(2,N_a,N_j,'gpuArray'); % [midpoint; aprimeL2ind]

midpoints_jj=zeros(1,N_a,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    [~,maxindex]=max(ReturnMatrix_ii,[],1);
    midpoints_jj(1,level1ii)=maxindex;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        [~,maxindex]=max(ReturnMatrix_ii,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
    Vhat(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1);

    Vunderbar=Vhat;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);
    EVinterp=interp1(a_grid,EV,aprime_grid);

    Vunderbar=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    % --- Vhat search (beta0beta) ---
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [~,maxindex]=max(entireRHS_ii,[],1);
    midpoints_jj(1,level1ii)=maxindex;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii_g,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    EVfine=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vhat(:,N_j)=shiftdim(Vtempii,1);
    Policy(1,:,N_j)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,N_j)=shiftdim(maxindexL2,-1);
    linidx=double(reshape(maxindexL2,[1,N_a]))+n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vunderbar(:,N_j)=Vhat(:,N_j)+(beta-beta0beta)*EV_at_policy;
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

    EVsource=Vunderbar(:,jj+1);
    EV=EVsource;
    EVinterp=interp1(a_grid,EV,aprime_grid);

    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);

    % --- Vhat search (beta0beta) ---
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [~,maxindex]=max(entireRHS_ii,[],1);
    midpoints_jj(1,level1ii)=maxindex;
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii_g=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, a_grid(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1))), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii_g=ReturnMatrix_ii_g+beta0beta*EV(midpoints_jj(level1ii(ii)):midpoints_jj(level1ii(ii+1)));
        [~,maxindex]=max(entireRHS_ii_g,[],1);
        midpoints_jj(1,curraindex)=maxindex+midpoints_jj(level1ii(ii))-1;
    end
    midpoints_jj=max(min(midpoints_jj,n_a-1),2);
    aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)';
    ReturnMatrix_L2=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn,aprime_grid(aprimeindexes),a_grid,ReturnFnParamsVec);
    EVfine=reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
    entireRHS_L2=ReturnMatrix_L2+beta0beta*EVfine;
    [Vtempii,maxindexL2]=max(entireRHS_L2,[],1);
    Vhat(:,jj)=shiftdim(Vtempii,1);
    Policy(1,:,jj)=shiftdim(squeeze(midpoints_jj),-1);
    Policy(2,:,jj)=shiftdim(maxindexL2,-1);
    linidx=double(reshape(maxindexL2,[1,N_a]))+n2long*(0:N_a-1);
    EV_at_policy=reshape(EVfine(linidx),[N_a,1]);
    Vunderbar(:,jj)=Vhat(:,jj)+(beta-beta0beta)*EV_at_policy;
end

%% Post-process Policy: convert [midpoint, aprimeL2ind] to canonical combined index
adjust=(Policy(2,:,:)<1+n2short+1);
Policy(1,:,:)=Policy(1,:,:)-adjust;
Policy(2,:,:)=adjust.*Policy(2,:,:)+(1-adjust).*(Policy(2,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:)+N_a*(Policy(2,:,:)-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vhat,Policy};
elseif nOutputs==3
    varargout={Vhat,Policy,Vunderbar};
end

end
