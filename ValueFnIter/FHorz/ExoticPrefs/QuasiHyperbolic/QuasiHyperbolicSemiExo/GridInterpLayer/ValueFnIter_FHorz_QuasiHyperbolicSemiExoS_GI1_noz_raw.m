function varargout=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Sophisticated QH + SemiExo + GI: with d1, no z (only semiz).

n_d=[n_d1,n_d2];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d=N_d1*N_d2;
N_a=prod(n_a);
N_semiz=prod(n_semiz);

Vhat=zeros(N_a,N_semiz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy=zeros(4,N_a,N_semiz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_semiz,N_j,'gpuArray');

%%
special_n_d=[n_d1,ones(1,length(n_d2))];
d_gridvals=[repmat(d1_gridvals,N_d2,1),repelem(d2_gridvals,N_d1,1)];
d12_gridvals=permute(reshape(d_gridvals,[N_d1,N_d2,length(n_d1)+length(n_d2)]),[1,3,2]);

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

aind=gpuArray(0:1:N_a-1);
semizind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-2);

Vhat_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Vunderbar_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
Policy_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
midpoint_ford2_jj=zeros(N_a,N_semiz,N_d2,'gpuArray');
flag_ford2_jj=2*ones(N_a,N_semiz,N_d2,'gpuArray');

n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, n_semiz, d_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);
        [~,maxindex]=max(ReturnMatrix,[],2);
        midpoint=max(min(maxindex,n_a-1),2);
        aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,n_semiz,d_gridvals,aprime_grid(aprimeindexes),a_grid,semiz_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        Vhat(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*semizind;
        L2offset      = ceil(maxindexL2/N_d);
        linidx_lower  = d_ind                  + N_d*n2long*aind + N_d*n2long*N_a*semizind;
        linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind + N_d*n2long*N_a*semizind;
        isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
        isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
        inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
        inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
        PolicyL2flag(1,:,:,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
        Policy(1,:,:,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
        Policy(2,:,:,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
        Policy(3,:,:,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
        Policy(4,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn, n_d, n_a, special_n_semiz, d_gridvals, a_grid, z_val, ReturnFnParamsVec,1);
            [~,maxindex]=max(ReturnMatrix_z,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn,n_d,special_n_semiz,d_gridvals,aprime_grid(aprimeindexes),a_grid,z_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            Vhat(:,z_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind;
            L2offset      = ceil(maxindexL2/N_d);
            linidx_lower  = d_ind                  + N_d*n2long*aind;
            linidx_upper  = d_ind + N_d*(n2long-1) + N_d*n2long*aind;
            isInfLower    = (ReturnMatrix_ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            PolicyL2flag(1,:,z_c,N_j) = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);
            Policy(1,:,z_c,N_j)=shiftdim(rem(d_ind-1,N_d1)+1,-1);
            Policy(2,:,z_c,N_j)=shiftdim(ceil(d_ind/N_d1),-1);
            Policy(3,:,z_c,N_j)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy(4,:,z_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1);
        end
    end
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

            entireRHS=ReturnMatrix_d2+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*semizBind;
            EVfine=reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_semiz]);
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*EVfine;
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            Vhat_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind;
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));

            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            linidx=double(reshape(maxindex,[1,N_a*N_semiz]))+N_d1*n2long*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            Vunderbar_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1)+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,N_j);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);

                EV_d2z=EVpre.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                EVinterp_z=interp1(a_grid,EV_d2z,aprime_grid);

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

                entireRHS_z=ReturnMatrix_d2z+beta0beta*shiftdim(EV_d2z,-1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                EVfine_z=reshape(EVinterp_z(aprimeindexes),[N_d1*n2long,N_a]);
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*EVfine_z;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                Vhat_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind;
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoint(allind));

                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                linidx_z=double(reshape(maxindex,[1,N_a]))+N_d1*n2long*(0:N_a-1);
                EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
                Vunderbar_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1)+(beta-beta0beta)*EV_at_policy_z;
            end
        end
    end

    [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,N_j)=Vhat_jj;
    Policy(2,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Policy(1,:,:,N_j)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1);
    Policy(4,:,:,N_j)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1);
    Policy(3,:,:,N_j)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Vunderbar(:,:,N_j)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
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

    EVpre=Vunderbar(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            EV_d2=EVpre.*shiftdim(pi_semiz',-1);
            EV_d2(isnan(EV_d2))=0;
            EV_d2=sum(EV_d2,2);

            EVinterp=interp1(a_grid,EV_d2,aprime_grid);

            ReturnMatrix_d2=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, n_semiz, d12c_gridvals, a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

            entireRHS=ReturnMatrix_d2+beta0beta*shiftdim(EV_d2,-1);
            [~,maxindex]=max(entireRHS,[],2);
            midpoint=max(min(maxindex,n_a-1),2);
            aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
            aprimez=aprimeindexes+n2aprime*semizBind;
            EVfine=reshape(EVinterp(aprimez),[N_d1*n2long,N_a,N_semiz]);
            entireRHS_ii=ReturnMatrix_d2ii+beta0beta*EVfine;
            [Vtemp,maxindex]=max(entireRHS_ii,[],1);

            Vhat_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1);
            Policy_ford2_jj(:,:,d2_c)=shiftdim(maxindex,1);

            d1_ind=rem(maxindex-1,N_d1)+1;
            allind=d1_ind+N_d1*aind+N_d1*N_a*semizind;
            midpoint_ford2_jj(:,:,d2_c)=squeeze(midpoint(allind));

            L2offset      = ceil(maxindex/N_d1);
            linidx_lower  = d1_ind                   + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
            linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind + N_d1*n2long*N_a*semizind;
            isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
            isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
            inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
            inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
            flag_ford2_jj(:,:,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

            linidx=double(reshape(maxindex,[1,N_a*N_semiz]))+N_d1*n2long*(0:N_a*N_semiz-1);
            EV_at_policy=reshape(EVfine(linidx),[N_a,N_semiz]);
            Vunderbar_ford2_jj(:,:,d2_c)=shiftdim(Vtemp,1)+(beta-beta0beta)*EV_at_policy;
        end
    elseif vfoptions.lowmemory==1
        for d2_c=1:N_d2
            pi_semiz=pi_semiz_J(:,:,d2_c,jj);
            d12c_gridvals=d12_gridvals(:,:,d2_c);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);

                EV_d2z=EVpre.*pi_semiz(z_c,:);
                EV_d2z(isnan(EV_d2z))=0;
                EV_d2z=sum(EV_d2z,2);

                EVinterp_z=interp1(a_grid,EV_d2z,aprime_grid);

                ReturnMatrix_d2z=CreateReturnFnMatrix_Disc(ReturnFn, special_n_d, n_a, special_n_semiz, d12c_gridvals, a_grid, z_val, ReturnFnParamsVec,1);

                entireRHS_z=ReturnMatrix_d2z+beta0beta*shiftdim(EV_d2z,-1);
                [~,maxindex]=max(entireRHS_z,[],2);
                midpoint=max(min(maxindex,n_a-1),2);
                aprimeindexes=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_d2ii=CreateReturnFnMatrix_Disc_DC1(ReturnFn, special_n_d, special_n_semiz, d12c_gridvals, aprime_grid(aprimeindexes), a_grid, z_val, ReturnFnParamsVec,2);
                EVfine_z=reshape(EVinterp_z(aprimeindexes),[N_d1*n2long,N_a]);
                entireRHS_ii=ReturnMatrix_d2ii+beta0beta*EVfine_z;
                [Vtemp,maxindex]=max(entireRHS_ii,[],1);

                Vhat_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1);
                Policy_ford2_jj(:,z_c,d2_c)=shiftdim(maxindex,1);

                d1_ind=rem(maxindex-1,N_d1)+1;
                allind=d1_ind+N_d1*aind;
                midpoint_ford2_jj(:,z_c,d2_c)=squeeze(midpoint(allind));

                L2offset      = ceil(maxindex/N_d1);
                linidx_lower  = d1_ind                   + N_d1*n2long*aind;
                linidx_upper  = d1_ind + N_d1*(n2long-1) + N_d1*n2long*aind;
                isInfLower    = (ReturnMatrix_d2ii(linidx_lower) == -Inf);
                isInfUpper    = (ReturnMatrix_d2ii(linidx_upper) == -Inf);
                inLowerStrict = (L2offset >= 2)         & (L2offset <= n2short+1);
                inUpperStrict = (L2offset >= n2short+3) & (L2offset <= n2long-1);
                flag_ford2_jj(:,z_c,d2_c) = shiftdim(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), 1);

                linidx_z=double(reshape(maxindex,[1,N_a]))+N_d1*n2long*(0:N_a-1);
                EV_at_policy_z=reshape(EVfine_z(linidx_z),[N_a,1]);
                Vunderbar_ford2_jj(:,z_c,d2_c)=shiftdim(Vtemp,1)+(beta-beta0beta)*EV_at_policy_z;
            end
        end
    end

    [Vhat_jj,maxindex]=max(Vhat_ford2_jj,[],3);
    Vhat(:,:,jj)=Vhat_jj;
    Policy(2,:,:,jj)=shiftdim(maxindex,-1);
    maxindex_lin=reshape(maxindex,[N_a*N_semiz,1]);
    d1aprimeL2_ind=reshape(Policy_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Policy(1,:,:,jj)=shiftdim(rem(d1aprimeL2_ind-1,N_d1)+1,-1);
    Policy(4,:,:,jj)=shiftdim(ceil(d1aprimeL2_ind/N_d1),-1);
    Policy(3,:,:,jj)=reshape(midpoint_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[1,N_a,N_semiz]);
    Vunderbar(:,:,jj)=reshape(Vunderbar_ford2_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex_lin-1)),[N_a,N_semiz]);
end

%% Post-process Policy
adjust=(Policy(4,:,:,:)<1+n2short+1);
Policy(3,:,:,:)=Policy(3,:,:,:)-adjust;
Policy(4,:,:,:)=adjust.*Policy(4,:,:,:)+(1-adjust).*(Policy(4,:,:,:)-n2short-1);

Policy=squeeze(Policy(1,:,:,:)+N_d1*(Policy(2,:,:,:)-1)+N_d*(Policy(3,:,:,:)-1)+N_d*N_a*(Policy(4,:,:,:)-1)+N_d*N_a*(n2short+2)*(PolicyL2flag-1));

%%
nOutputs=nargout;
if nOutputs==2
    varargout={Vunderbar,Policy};
elseif nOutputs==3
    varargout={Vunderbar,Policy,Vhat};
end

end
