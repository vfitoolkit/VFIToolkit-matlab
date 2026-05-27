function [V,Policy4]=ValueFnIter_FHorz_ExpAsseteSemiExo_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% semiz is semi-exog state, e is i.i.d. start-of-period (required); no z
% aprimeFn = aprimeFn(d2, a2, e, ...)

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy4=zeros(4,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
a2_gridvals=CreateGridvals(n_a2,a2_grid,1);
n_d23=[n_d2,n_d3];

n_d=[n_d1,n_d2,n_d3];
N_d=prod(n_d);
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>1
    special_n_semiz=ones(1,length(n_semiz));
end

V_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_e,N_d3,'gpuArray');


%% j=N_j

ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_semiz,n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d)+1;
        d12_ind=rem(d_ind-1,N_d12)+1;
        Policy4(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
        Policy4(2,:,:,:,N_j)=ceil(d12_ind/N_d1);
        Policy4(3,:,:,:,N_j)=ceil(d_ind/N_d12);
        Policy4(4,:,:,:,N_j)=ceil(maxindex/N_d);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d)+1;
            d12_ind=rem(d_ind-1,N_d12)+1;
            Policy4(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy4(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy4(3,:,:,e_c,N_j)=ceil(d_ind/N_d12);
            Policy4(4,:,:,e_c,N_j)=ceil(maxindex/N_d);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,n_d23,n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d)+1;
                d12_ind=rem(d_ind-1,N_d12)+1;
                Policy4(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy4(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy4(3,:,z_c,e_c,N_j)=ceil(d_ind/N_d12);
                Policy4(4,:,z_c,e_c,N_j)=ceil(maxindex/N_d);
            end
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1); % [N_d2*N_a1,N_a2,N_e]
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j),-2),3); % [N_a, N_semiz]

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2*a1prime,a2,semizcur,e_cur)

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec,0,0);

                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,1);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,z_c,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);

                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_z(:,:,:,e_c);

                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy4(3,:,:,:,N_j)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1;
    Policy4(2,:,:,:,N_j)=ceil(d12_ind/N_d1);
    Policy4(4,:,:,:,N_j)=ceil(d12a1prime_ind/N_d12);
end

%% Iterate backwards through j
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,2);

    aprimeIndex=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex-1,N_a1,1,1);
    aprimeplus1Index=repelem((1:1:N_a1)',N_d2,N_a2,N_e)+N_a1*repmat(a2primeIndex,N_a1,1,1);
    aprimeProbs_d2a1a2e=repmat(a2primeProbs,N_a1,1,1);

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj),-2),3);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec,0,0);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,1);

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec,0,0);

                entireRHS_d3e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_d3e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_semi_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semi_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_byzcur=reshape(EV,[N_a,N_semiz]);

            Vlower=reshape(EV_byzcur(aprimeIndex(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV_byzcur(aprimeplus1Index(:),:),[N_d2*N_a1,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(aprimeProbs_d2a1a2e,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]);
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,N_a1,1,1);

            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,z_c,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_ExpAsset_Disc_e(ReturnFn, n_d1,[n_d2,1],n_a1,n_a1,n_a2,special_n_semiz,special_n_e, d123_gridvals_val, a1_gridvals, a1_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec,0,0);

                    entireRHS_d3ze=ReturnMatrix_d3ze+DiscountedEV_z(:,:,:,e_c);

                    [Vtemp,maxindex]=max(entireRHS_d3ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy4(3,:,:,:,jj)=shiftdim(maxindex,-1);
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d12a1prime_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    d12_ind=rem(d12a1prime_ind-1,N_d12)+1;
    Policy4(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1;
    Policy4(2,:,:,:,jj)=ceil(d12_ind/N_d1);
    Policy4(4,:,:,:,jj)=ceil(d12a1prime_ind/N_d12);
end


end
