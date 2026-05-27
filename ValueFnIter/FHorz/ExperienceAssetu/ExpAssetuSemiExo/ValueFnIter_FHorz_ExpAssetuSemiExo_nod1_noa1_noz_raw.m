function [V,Policy2]=ValueFnIter_FHorz_ExpAssetuSemiExo_nod1_noa1_noz_raw(n_d2,n_d3,n_a2,n_semiz,n_u,N_j, d2_gridvals, d3_grid, a2_grid, semiz_gridvals_J, u_gridvals, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetuSemiExo_nod1_noz_raw.
% d2 determines experience asset; d3 determines semi-exog state; a = a2; semiz is semi-exog; u is iid.
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_u=prod(n_u);

V=zeros(N_a,N_semiz,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_semiz,N_j,'gpuArray');

pi_u=shiftdim(pi_u,-2); % put it into third dimension

%%
n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_semiz=ones(1,length(n_semiz));
end

% Preallocate
V_ford3_jj=zeros(N_a,N_semiz,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_semiz,N_d3,'gpuArray');

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, n_semiz, d23_gridvals, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2(1,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy2(2,:,:,N_j)=ceil(d_ind/N_d2);    % d3
    elseif vfoptions.lowmemory==1
        for z_c=1:N_semiz
            z_val=semiz_gridvals_J(z_c,:,N_j);
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, n_d23, n_a2, special_n_semiz, d23_gridvals, a2_grid, z_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
            V(:,z_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,z_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,z_c,N_j)=ceil(d_ind/N_d2);
        end
    end
else
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % [N_d2,N_a2,N_u]
    aprimeIndex=a2primeIndex;        % [N_d2,N_a2,N_u]
    aprimeplus1Index=a2primeIndex+1; % [N_d2,N_a2,N_u]
    if vfoptions.lowmemory==0
        aprimeProbs=repmat(a2primeProbs,1,1,1,N_semiz); % [N_d2,N_a2,N_u,N_semiz]
    else
        aprimeProbs=a2primeProbs; % [N_d2,N_a2,N_u]
    end

    EVpre=reshape(vfoptions.V_Jplus1,[N_a,N_semiz]);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_semiz, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_semiz]

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_u,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_u,N_semiz]);

            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0;

            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs);
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a2,semiz)
            EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

            entireRHS_d3=ReturnMatrix_d3+DiscountFactorParamsVec*EV;

            [Vtemp,maxindex]=max(entireRHS_d3,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,N_j);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_semiz, d23_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*pi_semiz_d3(z_c,:);
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                EV1=reshape(EV_z(aprimeIndex),[N_d2,N_a2,N_u]);
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2,N_a2,N_u]);

                skipinterp=(EV1==EV2);
                aprimeProbs(skipinterp)=0;

                EV_z=EV1.*aprimeProbs+EV2.*(1-aprimeProbs);
                EV_z=sum((EV_z.*pi_u),3); % (d2,a2)
                EV_z(isnan(EV_z))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

                entireRHS_d3z=ReturnMatrix_d3z+DiscountFactorParamsVec*EV_z;

                [Vtemp,maxindex]=max(entireRHS_d3z,[],1);
                V_ford3_jj(:,z_c,d3_c)=Vtemp;
                Policy_ford3_jj(:,z_c,d3_c)=maxindex;
            end
        end
    end

    % Max over d3
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,N_j)=V_jj;
    Policy2(2,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy2(1,:,:,N_j)=d2_ind; % d2
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetuFnMatrix(aprimeFn, n_d2, n_a2, n_u, d2_gridvals, a2_grid, u_gridvals, aprimeFnParamsVec,2); % [N_d2,N_a2,N_u]
    aprimeIndex=a2primeIndex;
    aprimeplus1Index=a2primeIndex+1;
    if vfoptions.lowmemory==0
        aprimeProbs=repmat(a2primeProbs,1,1,1,N_semiz);
    else
        aprimeProbs=a2primeProbs;
    end

    EVpre=V(:,:,jj+1);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, n_semiz, d23_gridvals_val, a2_grid, semiz_gridvals_J(:,:,jj), ReturnFnParamsVec);

            EV=EVpre.*shiftdim(pi_semiz_d3',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2); % [N_a, 1, N_semiz]

            EV1=reshape(EV(aprimeIndex,:),[N_d2,N_a2,N_u,N_semiz]);
            EV2=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_u,N_semiz]);

            skipinterp=(EV1==EV2);
            aprimeProbs(skipinterp)=0;

            EV=EV1.*aprimeProbs+EV2.*(1-aprimeProbs);
            EV=squeeze(sum((EV.*pi_u),3)); % (d2,a2,semiz)
            EV(isnan(EV))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*EV;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_semiz_d3=pi_semiz_J(:,:,d3_c,jj);
            for z_c=1:N_semiz
                z_val=semiz_gridvals_J(z_c,:,jj);
                ReturnMatrix_d3z=CreateReturnFnMatrix_Case2_Disc(ReturnFn, [n_d2,1], n_a2, special_n_semiz, d23_gridvals_val, a2_grid, z_val, ReturnFnParamsVec);

                EV_z=EVpre.*(ones(N_a,1,'gpuArray')*pi_semiz_d3(z_c,:));
                EV_z(isnan(EV_z))=0;
                EV_z=sum(EV_z,2);

                EV1=reshape(EV_z(aprimeIndex),[N_d2,N_a2,N_u]);
                EV2=reshape(EV_z(aprimeplus1Index),[N_d2,N_a2,N_u]);

                skipinterp=(EV1==EV2);
                aprimeProbs(skipinterp)=0;

                EV_z=EV1.*aprimeProbs+EV2.*(1-aprimeProbs);
                EV_z=sum((EV_z.*pi_u),3); % (d2,a2)
                EV_z(isnan(EV_z))=0; % NaN from 0*(-Inf) at skipinterp positions; treat as zero contribution

                entireRHS_z=ReturnMatrix_d3z+DiscountFactorParamsVec*EV_z;

                [Vtemp,maxindex]=max(entireRHS_z,[],1);
                V_ford3_jj(:,z_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,z_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    end

    % Max over d3
    [V_jj,maxindex]=max(V_ford3_jj,[],3);
    V(:,:,jj)=V_jj;
    Policy2(2,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_semiz,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_semiz)'+(N_a*N_semiz)*(maxindex-1)),[1,N_a,N_semiz]);
    Policy2(1,:,:,jj)=d2_ind; % d2

end


end
