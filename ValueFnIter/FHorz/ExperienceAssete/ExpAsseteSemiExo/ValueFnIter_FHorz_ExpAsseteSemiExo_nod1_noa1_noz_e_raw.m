function [V,Policy2]=ValueFnIter_FHorz_ExpAsseteSemiExo_nod1_noa1_noz_e_raw(n_d2,n_d3,n_a2,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, e_gridvals_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAsseteSemiExo_nod1_noz_e_raw (nod1, noz, e).
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.
% aprimeFn = aprimeFn(d2, a2, e, ...)

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_e=prod(n_e);

V=zeros(N_a,N_semiz,N_e,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_semiz,N_e,N_j,'gpuArray');

%%
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

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
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_semiz, n_e, d23_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1;
        Policy2(2,:,:,:,N_j)=ceil(d_ind/N_d2);
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_semiz, special_n_e, d23_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
        end
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, special_n_semiz, special_n_e, d23_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy2(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy2(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
            end
        end
    end
else
    % aprime depends on (d2, a2, current_e); independent of d3 and semiz -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1);
    % Note: a2primeIndex is [N_d2*N_a2*N_e,1], whereas a2primeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    aprimeplus1Index=a2primeIndex+1;

    % Integrate out eprime first (e is i.i.d. start-of-period); EVpre is [N_a, N_semiz]
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_semiz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_e,N_semiz]
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2,a2,e_cur,semizcur)
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur) -- match return-fn dim order

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, special_n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV(:,:,z_c,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy2(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy2(1,:,:,:,N_j)=d2_ind; % d2
end


%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end

    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,jj), aprimeFnParamsVec,1);
    % Note: a2primeIndex is [N_d2*N_a2*N_e,1], whereas a2primeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    aprimeplus1Index=a2primeIndex+1;

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a, N_semiz]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]); % (d2,a2,e_cur,semizcur)
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur)

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_semiz, special_n_e, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_semiz]);
            Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_semiz]);
            skipinterp=(Vlower==Vupper);
            aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_semiz);
            aprimeProbs_d3(skipinterp)=0;
            entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,semizcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_semiz
                    z_val=semiz_gridvals_J(z_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV(:,:,z_c,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy2(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz*N_e,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz*N_e)'+(N_a*N_semiz*N_e)*(maxindex-1)),[1,N_a,N_semiz,N_e]);
    Policy2(1,:,:,:,jj)=d2_ind; % d2
end


end
