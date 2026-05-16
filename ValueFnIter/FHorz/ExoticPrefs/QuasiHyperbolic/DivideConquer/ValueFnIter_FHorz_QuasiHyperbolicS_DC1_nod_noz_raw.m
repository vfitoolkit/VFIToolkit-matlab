function [Vunderbar,Policy,Vhat]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_nod_noz_raw.
% No d variables. No z variable. No e variable. GPU (parallel==2 only).
%
% Sophisticated: Vhat_j = max u + beta_0*beta*E[Vunderbar_{j+1}]
%                Vunderbar_j = Vhat_j + (beta - beta_0*beta)*EV_at_optimal_aprime

N_a=prod(n_a);

Vhat=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray');

level1ii=round(linspace(1,n_a,vfoptions.level1n));

%% j=N_j (terminal period)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

if ~isfield(vfoptions,'V_Jplus1')
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
    Vhat(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(Policy(level1ii(ii),N_j):Policy(level1ii(ii+1),N_j)), a_grid(curraindex), ReturnFnParamsVec);
        [Vtempii,maxindex]=max(ReturnMatrix_ii,[],1);
        Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+Policy(level1ii(ii),N_j)-1;
    end
    Vunderbar=Vhat;

else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EV=reshape(vfoptions.V_Jplus1,[N_a,1]);

    Vunderbar=zeros(N_a,N_j,'gpuArray');

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    Vhat(level1ii,N_j)=shiftdim(Vtempii,1);
    Policy(level1ii,N_j)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        a_range=Policy(level1ii(ii),N_j):Policy(level1ii(ii+1),N_j);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(a_range), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(a_range);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        Vhat(curraindex,N_j)=shiftdim(Vtempii,1);
        Policy(curraindex,N_j)=shiftdim(maxindex,1)+Policy(level1ii(ii),N_j)-1;
    end
    aprime_ind=Policy(:,N_j);
    EV_at_policy=EV(aprime_ind);
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

    EV=Vunderbar(:,jj+1);

    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid, a_grid(level1ii), ReturnFnParamsVec);
    entireRHS_ii=ReturnMatrix_ii+beta0beta*EV;
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1);
    Vhat(level1ii,jj)=shiftdim(Vtempii,1);
    Policy(level1ii,jj)=shiftdim(maxindex1,1);
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        a_range=Policy(level1ii(ii),jj):Policy(level1ii(ii+1),jj);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, a_grid(a_range), a_grid(curraindex), ReturnFnParamsVec);
        entireRHS_ii=ReturnMatrix_ii+beta0beta*EV(a_range);
        [Vtempii,maxindex]=max(entireRHS_ii,[],1);
        Vhat(curraindex,jj)=shiftdim(Vtempii,1);
        Policy(curraindex,jj)=shiftdim(maxindex,1)+Policy(level1ii(ii),jj)-1;
    end
    aprime_ind=Policy(:,jj);
    EV_at_policy=EV(aprime_ind);
    Vunderbar(:,jj)=Vhat(:,jj)+(beta-beta0beta)*EV_at_policy;
end

end
