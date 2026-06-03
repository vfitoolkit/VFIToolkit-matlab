function [V,Policy,Vhat]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_noz_raw(V,n_d,n_a,N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The V input is next period value fn (across all ages), the V output is this period.
% Sophisticated quasi-hyperbolic: V carries Vunderbar (realised value under QH policies); Vhat is the agent's-perspective (beta0*beta) value.

N_d=prod(n_d);
N_a=prod(n_a);

Policy=zeros(N_a,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a
Vhat=zeros(N_a,N_j,'gpuArray'); % agent's-perspective value (beta0*beta-discounted), before the Vunderbar transform

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% j=N_j: terminal age has no continuation in TPath
% Temporarily save the time period of V that is being replaced
Vtemp_j=V(:,N_j);

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

% n-Monotonicity
ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

[~,maxindex1]=max(ReturnMatrix_ii,[],2);

[Vtempii,maxindex2]=max(reshape(ReturnMatrix_ii,[N_d*N_a,vfoptions.level1n]),[],1);

V(level1ii,N_j)=shiftdim(Vtempii,1);
Policy(level1ii,N_j)=shiftdim(maxindex2,1);

maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
for ii=1:(vfoptions.level1n-1)
    curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
    if maxgap(ii)>0
        loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
        aprimeindexes=loweredge+(0:1:maxgap(ii));
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
    else
        loweredge=maxindex1(:,1,ii);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        V(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
    end
end
Vhat(:,N_j)=V(:,N_j); % terminal: Vhat coincides with V (no Vunderbar transform at terminal)


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
    beta0beta=beta0*beta;

    VKronNext_j=Vtemp_j; % Has been presaved before it was replaced
    Vtemp_j=V(:,jj); % Grab this before it is replaced/updated

    EV=VKronNext_j;

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid, a_grid(level1ii), ReturnFnParamsVec,1);

    entireRHS_ii=ReturnMatrix_ii+beta0beta*shiftdim(EV,-1);
    [~,maxindex1]=max(entireRHS_ii,[],2);
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_d*N_a,vfoptions.level1n]),[],1);
    V(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex2,1);
    maxgap=max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii));
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*reshape(EV(aprimeindexes),[N_d*(maxgap(ii)+1),1]);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        else
            loweredge=maxindex1(:,1,ii);
            ReturnMatrix_ii_dc=CreateReturnFnMatrix_Disc_DC1_noz(ReturnFn, n_d, d_gridvals, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), ReturnFnParamsVec,2);
            entireRHS_ii=ReturnMatrix_ii_dc+beta0beta*EV(loweredge);
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,jj)=shiftdim(Vtempii,1);
            Policy(curraindex,jj)=shiftdim(maxindex,1)+N_d*(loweredge(rem(maxindex-1,N_d)+1)-1);
        end
    end
    % Vunderbar = Vhat + (beta - beta0*beta)*EV_at_optimal_aprime
    Vhat(:,jj)=V(:,jj); % Save Vhat before applying the Vunderbar transform
    aprime_ind=ceil(Policy(:,jj)/N_d);
    EV_at_policy=EV(aprime_ind);
    V(:,jj)=V(:,jj)+(beta-beta0beta)*EV_at_policy;

end

%% Output shape for policy
Policy=shiftdim(Policy,-1);

end
