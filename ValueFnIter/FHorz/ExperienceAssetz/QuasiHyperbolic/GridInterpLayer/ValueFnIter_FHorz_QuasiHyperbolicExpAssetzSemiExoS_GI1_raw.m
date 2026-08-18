function [Vhat,Policy,Vunderbar]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzSemiExoS_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, z, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=N_semiz*N_z;

Vhat=zeros(N_a,N_bothz,N_j,'gpuArray');
Vunderbar=zeros(N_a,N_bothz,N_j,'gpuArray');
% Policy storage with d1, d2, d3, a1prime_midpoint, a1primeL2ind
Policy=zeros(5,N_a,N_bothz,N_j,'gpuArray');
PolicyL2flag=2*ones(1,N_a,N_bothz,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);

bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

if vfoptions.lowmemory>0
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford3_hat=zeros(N_a,N_bothz,N_d3,'gpuArray');
V_ford3_under=zeros(N_a,N_bothz,N_d3,'gpuArray');
Policy4_ford3_hat=zeros(4,N_a,N_bothz,N_d3,'gpuArray');
flag_ford3_hat=2*ones(N_a,N_bothz,N_d3,'gpuArray');

% Grid interpolation
n2short=vfoptions.ngridinterp;
n2long=vfoptions.ngridinterp*2+3;
a1prime_grid=interp1(1:1:n_a1(1),a1_gridvals,linspace(1,n_a1(1),n_a1(1)+(n_a1(1)-1)*n2short));
N_a1prime=length(a1prime_grid);

aind=gpuArray(0:1:N_a-1);
a2ind=shiftdim(gpuArray(0:1:N_a2-1),-2);
bothzind=shiftdim(gpuArray(0:1:N_bothz-1),-3);
bothzBind=shiftdim(gpuArray(0:1:N_bothz-1),-1);

bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);
            [~,maxindex]=max(ReturnMatrix_d3,[],2);

            midpoint_hat=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3,[],1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
            Policy4_ford3_hat(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_hat(2,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_hat(3,:,:,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
            Policy4_ford3_hat(4,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
            isInfLower=(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_d3z,[],2);

                midpoint_hat=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3z,[],1);
                V_ford3_hat(:,semizblock,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_hat(1,:,semizblock,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,semizblock,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                Policy4_ford3_hat(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);
                [~,maxindex]=max(ReturnMatrix_d3z,[],2);

                midpoint_hat=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint_hat+(midpoint_hat-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                [Vtempii,maxindexL2]=max(ReturnMatrix_ii_d3z,[],1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind;
                Policy4_ford3_hat(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,z_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint_hat(allind)),-1);
                Policy4_ford3_hat(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,z_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    end
    % Max over d3 and unpack
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policy(1,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz]);
    Policy(2,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);
    % Terminal period: no continuation, so Vunderbar equals Vhat
    Vunderbar(:,:,N_j)=Vhat(:,:,N_j);
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_bothz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    beta=prod(DiscountFactorParamsVec);
    beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,N_j);
    beta0beta=beta0*beta;

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_under=repelem(DiscountedEV_under,N_d1,1);
            DiscountedEVinterp_under=repelem(DiscountedEVinterp_under,N_d1,1);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_hat=repelem(DiscountedEV_hat,N_d1,1);
            DiscountedEVinterp_hat=repelem(DiscountedEVinterp_hat,N_d1,1);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,1,0);

            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_hat;

            [~,maxindex]=max(entireRHS_d3,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), ReturnFnParamsVec,2,0);
            d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_hat(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_under(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_bothz)-1),-1);
            V_ford3_under(:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
            Policy4_ford3_hat(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_hat(2,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_hat(3,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_hat(4,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
            isInfLower=(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
        semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,N_j);

                EV=V_Jplus1.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEV_under=beta*EVbase_qh;
                DiscountedEV_hat=beta0beta*EVbase_qh;
                DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
                DiscountedEV_under=repelem(DiscountedEV_under,N_d1,1);
                DiscountedEVinterp_under=repelem(DiscountedEVinterp_under,N_d1,1);

                DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
                DiscountedEV_hat=repelem(DiscountedEV_hat,N_d1,1);
                DiscountedEVinterp_hat=repelem(DiscountedEVinterp_hat,N_d1,1);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_hat;

                [~,maxindex]=max(entireRHS_d3z,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d12a1primea2semiz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_hat(d12a1primea2semiz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_under(d12a1primea2semiz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_semiz)-1),-1);
                V_ford3_under(:,semizblock,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,semizblock,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_hat(1,:,semizblock,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,semizblock,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_hat(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=V_Jplus1.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_under=repelem(DiscountedEV_under,N_d1,1);
            DiscountedEVinterp_under=repelem(DiscountedEVinterp_under,N_d1,1);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_hat=repelem(DiscountedEV_hat,N_d1,1);
            DiscountedEVinterp_hat=repelem(DiscountedEVinterp_hat,N_d1,1);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,z_c);
                DiscountedEVinterp_z_hat=DiscountedEVinterp_hat(:,:,:,:,z_c);
                DiscountedEVinterp_z_under=DiscountedEVinterp_under(:,:,:,:,z_c);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z_hat;

                [~,maxindex]=max(entireRHS_d3z,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d12a1primea2=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z_hat(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z_under(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1);
            V_ford3_under(:,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind;
                Policy4_ford3_hat(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,z_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_hat(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,z_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    end

    % Max over d3 and unpack
    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,N_j)=V_jj;
    Policy(3,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policy(1,:,:,N_j)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz]);
    Policy(2,:,:,N_j)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,N_j)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,N_j)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,N_j)=reshape(flag_ford3_hat((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,N_j)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzFnMatrix(aprimeFn, n_d2, n_a2, n_z, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem(gpuArray(1:1:N_a1)',N_d2,N_a2,N_z)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2z=repmat(a2primeProbs,N_a1,1,1);
    aprimeIndex_full=repelem(aprimeIndex,1,1,N_semiz);
    aprimeplus1Index_full=repelem(aprimeplus1Index,1,1,N_semiz);
    aprimeProbs_full=repelem(aprimeProbs_d2a1a2z,1,1,N_semiz);

    EVpre=Vunderbar(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_under=repelem(DiscountedEV_under,N_d1,1);
            DiscountedEVinterp_under=repelem(DiscountedEVinterp_under,N_d1,1);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_hat=repelem(DiscountedEV_hat,N_d1,1);
            DiscountedEVinterp_hat=repelem(DiscountedEVinterp_hat,N_d1,1);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,1,0);

            entireRHS_d3=ReturnMatrix_d3+DiscountedEV_hat;

            [~,maxindex]=max(entireRHS_d3,[],2);

            midpoint=max(min(maxindex,n_a1(1)-1),2);
            a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
            ReturnMatrix_ii_d3=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, bothz_gridvals_J(:,:,jj), ReturnFnParamsVec,2,0);
            d12a1primea2bothz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*bothzind;
            entireRHS_ii_d3=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_hat(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_d3,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3+reshape(DiscountedEVinterp_under(d12a1primea2bothz(:)),[N_d12*n2long,N_a1*N_a2,N_bothz]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_bothz)-1),-1);
            V_ford3_under(:,:,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
            V_ford3_hat(:,:,d3_c)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d12)+1;
            allind=d_ind+N_d12*aind+N_d12*N_a*bothzBind;
            Policy4_ford3_hat(1,:,:,d3_c)=rem(d_ind-1,N_d1)+1;
            Policy4_ford3_hat(2,:,:,d3_c)=ceil(d_ind/N_d1);
            Policy4_ford3_hat(3,:,:,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
            Policy4_ford3_hat(4,:,:,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
            L2offset=ceil(maxindexL2/N_d12);
            linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
            linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*bothzBind;
            isInfLower=(ReturnMatrix_ii_d3(linidx_lower)==-Inf);
            isInfUpper=(ReturnMatrix_ii_d3(linidx_upper)==-Inf);
            inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
            inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
            flag_ford3_hat(:,:,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
        end
    elseif vfoptions.lowmemory==1 % split: loop z (markov), vectorize semiz
        semizBind=shiftdim(gpuArray(0:1:N_semiz-1),-1);
        semizind=shiftdim(gpuArray(0:1:N_semiz-1),-3);
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));
            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:1:N_semiz);
                z_valblock=bothz_gridvals_J(semizblock,:,jj);

                EV=EVpre.*shiftdim(pi_bothz(semizblock,:)',-1);
                EV(isnan(EV))=0;
                EV=sum(EV,2);
                EV_2D=reshape(EV,[N_a,N_semiz]);

                semizblock_offset=N_a*reshape(0:N_semiz-1,[1,1,N_semiz]);
                EV1=EV_2D(aprimeIndex_full(:,:,semizblock)+semizblock_offset);
                EV2=EV_2D(aprimeplus1Index_full(:,:,semizblock)+semizblock_offset);

                skipinterp=(EV1==EV2);
                aprimeProbs_z=aprimeProbs_full(:,:,semizblock);
                aprimeProbs_z(skipinterp)=0;

                entireEV_z=EV1.*aprimeProbs_z+EV2.*(1-aprimeProbs_z);

                EVbase_qh=reshape(entireEV_z,[N_d2,N_a1,1,N_a2,N_semiz]);
                DiscountedEV_under=beta*EVbase_qh;
                DiscountedEV_hat=beta0beta*EVbase_qh;
                DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
                DiscountedEV_under=repelem(DiscountedEV_under,N_d1,1);
                DiscountedEVinterp_under=repelem(DiscountedEVinterp_under,N_d1,1);

                DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
                DiscountedEV_hat=repelem(DiscountedEV_hat,N_d1,1);
                DiscountedEVinterp_hat=repelem(DiscountedEVinterp_hat,N_d1,1);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,1,0);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_hat;

                [~,maxindex]=max(entireRHS_d3z,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,[n_semiz,ones(1,length(n_z))], d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_valblock, ReturnFnParamsVec,2,0);
                d12a1primea2semiz=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind+N_d12*N_a1prime*N_a2*semizind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_hat(d12a1primea2semiz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
                % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
                entireRHS_under=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_under(d12a1primea2semiz(:)),[N_d12*n2long,N_a1*N_a2,N_semiz]);
                maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1)+shiftdim((N_d12*n2long)*(N_a1*N_a2)*(0:1:(N_semiz)-1),-1);
                V_ford3_under(:,semizblock,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,semizblock,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind+N_d12*N_a*semizBind;
                Policy4_ford3_hat(1,:,semizblock,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,semizblock,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,semizblock,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_hat(4,:,semizblock,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind+N_d12*n2long*N_a*semizBind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,semizblock,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    elseif vfoptions.lowmemory==2 % joint loop over bothz
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            lin_lower=aprimeIndex_full+bothz_offset;
            lin_upper=aprimeplus1Index_full+bothz_offset;
            EV1=EV_2D(lin_lower);
            EV2=EV_2D(lin_upper);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            EVbase_qh=reshape(entireEV,[N_d2,N_a1,1,N_a2,N_bothz]);
            DiscountedEV_under=beta*EVbase_qh;
            DiscountedEV_hat=beta0beta*EVbase_qh;
            DiscountedEVinterp_under=permute(interp1(a1_gridvals,permute(DiscountedEV_under,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_under=repelem(DiscountedEV_under,N_d1,1);
            DiscountedEVinterp_under=repelem(DiscountedEVinterp_under,N_d1,1);

            DiscountedEVinterp_hat=permute(interp1(a1_gridvals,permute(DiscountedEV_hat,[2,1,3,4,5]),a1prime_grid),[2,1,3,4,5]);
            DiscountedEV_hat=repelem(DiscountedEV_hat,N_d1,1);
            DiscountedEVinterp_hat=repelem(DiscountedEVinterp_hat,N_d1,1);

            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,jj);
                DiscountedEV_z_hat=DiscountedEV_hat(:,:,:,:,z_c);
                DiscountedEV_z_under=DiscountedEV_under(:,:,:,:,z_c);
                DiscountedEVinterp_z_hat=DiscountedEVinterp_hat(:,:,:,:,z_c);
                DiscountedEVinterp_z_under=DiscountedEVinterp_under(:,:,:,:,z_c);

                ReturnMatrix_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,1,0);

                entireRHS_d3z=ReturnMatrix_d3z+DiscountedEV_z_hat;

                [~,maxindex]=max(entireRHS_d3z,[],2);

                midpoint=max(min(maxindex,n_a1(1)-1),2);
                a1primeindexesfine=(midpoint+(midpoint-1)*n2short)+(-n2short-1:1:1+n2short);
                ReturnMatrix_ii_d3z=CreateReturnFnMatrix_ExpAsset_Disc(ReturnFn, n_d1,[n_d2,1],n2long,n_a1,n_a2,special_n_bothz, d123_gridvals_val, a1prime_grid(a1primeindexesfine), a1_gridvals, a2_gridvals, z_val, ReturnFnParamsVec,2,0);
                d12a1primea2=(1:1:N_d12)'+N_d12*(a1primeindexesfine-1)+N_d12*N_a1prime*a2ind;
                entireRHS_ii_d3z=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z_hat(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
                [Vtempii,maxindexL2]=max(entireRHS_ii_d3z,[],1);
            % Vunderbar: the beta fine-RHS gathered at the hat argmax (not re-maximised)
            entireRHS_under=ReturnMatrix_ii_d3z+reshape(DiscountedEVinterp_z_under(d12a1primea2(:)),[N_d12*n2long,N_a1*N_a2]);
            maxindexfull=maxindexL2+(N_d12*n2long)*(0:1:(N_a1*N_a2)-1);
            V_ford3_under(:,z_c,d3_c)=shiftdim(entireRHS_under(maxindexfull),1);
                V_ford3_hat(:,z_c,d3_c)=shiftdim(Vtempii,1);
                d_ind=rem(maxindexL2-1,N_d12)+1;
                allind=d_ind+N_d12*aind;
                Policy4_ford3_hat(1,:,z_c,d3_c)=rem(d_ind-1,N_d1)+1;
                Policy4_ford3_hat(2,:,z_c,d3_c)=ceil(d_ind/N_d1);
                Policy4_ford3_hat(3,:,z_c,d3_c)=shiftdim(squeeze(midpoint(allind)),-1);
                Policy4_ford3_hat(4,:,z_c,d3_c)=shiftdim(ceil(maxindexL2/N_d12),-1);
                L2offset=ceil(maxindexL2/N_d12);
                linidx_lower=d_ind+N_d12*n2long*aind;
                linidx_upper=d_ind+N_d12*(n2long-1)+N_d12*n2long*aind;
                isInfLower=(ReturnMatrix_ii_d3z(linidx_lower)==-Inf);
                isInfUpper=(ReturnMatrix_ii_d3z(linidx_upper)==-Inf);
                inLowerStrict=(L2offset>=2)&(L2offset<=n2short+1);
                inUpperStrict=(L2offset>=n2short+3)&(L2offset<=n2long-1);
                flag_ford3_hat(:,z_c,d3_c)=shiftdim(2+(inLowerStrict&isInfLower)-(inUpperStrict&isInfUpper),1);
            end
        end
    end

    % Max over d3 using the hat (QH-perceived) values
    [V_jj,maxindex]=max(V_ford3_hat,[],3);
    Vhat(:,:,jj)=V_jj;
    Policy(3,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_bothz,1]);
    temp=4*((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)-1);
    Policy(1,:,:,jj)=reshape(Policy4_ford3_hat(1+temp),[1,N_a,N_bothz]);
    Policy(2,:,:,jj)=reshape(Policy4_ford3_hat(2+temp),[1,N_a,N_bothz]);
    Policy(4,:,:,jj)=reshape(Policy4_ford3_hat(3+temp),[1,N_a,N_bothz]);
    Policy(5,:,:,jj)=reshape(Policy4_ford3_hat(4+temp),[1,N_a,N_bothz]);
    PolicyL2flag(1,:,:,jj)=reshape(flag_ford3_hat((1:N_a*N_bothz)'+(N_a*N_bothz)*(maxindex-1)),[1,N_a,N_bothz]);

    % Vunderbar: gather the beta-RHS (already inner-gathered) at the same chosen d3
    d3lin=reshape(maxindex,[N_a*N_bothz,1]);
    Vunderbar(:,:,jj)=reshape(V_ford3_under((1:1:N_a*N_bothz)'+(N_a*N_bothz)*(d3lin-1)),[N_a,N_bothz]);
end


%% Switch from midpoint to lower grid index
adjust=(Policy(5,:,:,:)<1+n2short+1);
Policy(4,:,:,:)=Policy(4,:,:,:)-adjust;
Policy(5,:,:,:)=adjust.*Policy(5,:,:,:)+(1-adjust).*(Policy(5,:,:,:)-n2short-1);

Policy=[Policy; PolicyL2flag];


end
