function [V,Policy2]=ValueFnIter_FHorz_ExpAssetzeSemiExo_nod1_noa1_e_raw(n_d2,n_d3,n_a2,n_z,n_semiz,n_e,N_j, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAssetzeSemiExo_nod1_e_raw (nod1, z, e).
% d2 determines experience asset, d3 determines semi-exog state (no d1)
% a2 is experience asset; no standard endogenous asset a1
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. start-of-period (required)
% aprimeFn = aprimeFn(d2, a2, z, e, ...)   (depends on BOTH current markov z and current e; NOT on semiz)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest; e is separate
% Policy2 stores (d2, d3) -- no a1prime channel since noa1.

n_bothz=[n_semiz,n_z];

N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy2=zeros(2,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d23=[n_d2,n_d3];
N_d23=prod(n_d23);
d23_gridvals=[repmat(d2_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d2,1)];

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory==2
    special_n_semiz=[n_semiz,ones(1,length(n_z))];
elseif vfoptions.lowmemory==3
    special_n_bothz=ones(1,length(n_semiz)+length(n_z));
end

V_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');
Policy_ford3_jj=zeros(N_a,N_bothz,N_e,N_d3,'gpuArray');

% Offset for linear indexing into [N_a, N_bothz]
bothz_offset=N_a*reshape(0:N_bothz-1,[1,1,N_bothz]);

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_bothz, n_e, d23_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d_ind=rem(maxindex-1,N_d23)+1;
        Policy2(1,:,:,:,N_j)=rem(d_ind-1,N_d2)+1; % d2
        Policy2(2,:,:,:,N_j)=ceil(d_ind/N_d2);    % d3
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, n_bothz, special_n_e, d23_gridvals, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d_ind=rem(maxindex-1,N_d23)+1;
            Policy2(1,:,:,e_c,N_j)=rem(d_ind-1,N_d2)+1;
            Policy2(2,:,:,e_c,N_j)=ceil(d_ind/N_d2);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:N_semiz);
            z_val=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, special_n_semiz, special_n_e, d23_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,semizblock,e_c,N_j)=shiftdim(Vtemp,1);
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy2(1,:,semizblock,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy2(2,:,semizblock,e_c,N_j)=ceil(d_ind/N_d2);
            end
        end
    elseif vfoptions.lowmemory==3
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d23, n_a2, special_n_bothz, special_n_e, d23_gridvals, a2_grid, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d_ind=rem(maxindex-1,N_d23)+1;
                Policy2(1,:,z_c,e_c,N_j)=rem(d_ind-1,N_d2)+1;
                Policy2(2,:,z_c,e_c,N_j)=ceil(d_ind/N_d2);
            end
        end
    end
else
    % aprime depends on (d2, a2, current_z, current_e); independent of d3, semiz
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), aprimeFnParamsVec,2); % [N_d2,N_a2,N_z,N_e]

    % Expand z->bothz=(semiz,z) with semiz fastest
    aprimeIndex_full=repelem(a2primeIndex,1,1,N_semiz,1); % [N_d2,N_a2,N_bothz,N_e]
    aprimeplus1Index_full=repelem(a2primeIndex+1,1,1,N_semiz,1);
    aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz,1);

    % Integrate over e' first (e is i.i.d. start-of-period); EVpre is [N_a, N_bothz]
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset); % [N_d2,N_a2,N_bothz,N_e]
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3); % [N_d2,N_a2,N_bothz,N_e]

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, special_n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV_z(:,:,:,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_bothz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV(:,:,z_c,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,N_j)=V_jj;
    Policy2(2,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
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
    [a2primeIndex,a2primeProbs]=CreateExperienceAssetzeFnMatrix(aprimeFn, n_d2, n_a2, n_z, n_e, d2_gridvals, a2_grid, z_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), aprimeFnParamsVec,2); % [N_d2,N_a2,N_z,N_e]

    % Expand z->bothz=(semiz,z) with semiz fastest
    aprimeIndex_full=repelem(a2primeIndex,1,1,N_semiz,1); % [N_d2,N_a2,N_bothz,N_e]
    aprimeplus1Index_full=repelem(a2primeIndex+1,1,1,N_semiz,1);
    aprimeProbs_full=repelem(a2primeProbs,1,1,N_semiz,1);

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a, N_bothz]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*entireEV;

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, n_bothz, special_n_e, d23_gridvals_val, a2_grid, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_semiz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV_z(:,:,:,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d23_gridvals_val=[d2_gridvals,repelem(d3_grid(d3_c),N_d2,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);
            EV_2D=reshape(EV,[N_a,N_bothz]);

            EV1=EV_2D(aprimeIndex_full+bothz_offset);
            EV2=EV_2D(aprimeplus1Index_full+bothz_offset);

            skipinterp=(EV1==EV2);
            aprimeProbs_d3=aprimeProbs_full;
            aprimeProbs_d3(skipinterp)=0;

            entireEV=EV1.*aprimeProbs_d3+EV2.*(1-aprimeProbs_d3);
            DiscountedEV=DiscountFactorParamsVec*entireEV;

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d2,1], n_a2, special_n_bothz, special_n_e, d23_gridvals_val, a2_grid, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV(:,:,z_c,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,z_c,e_c,d3_c)=Vtemp;
                    Policy_ford3_jj(:,z_c,e_c,d3_c)=maxindex;
                end
            end
        end
    end

    [V_jj,maxindex]=max(V_ford3_jj,[],4);
    V(:,:,:,jj)=V_jj;
    Policy2(2,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d2_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy2(1,:,:,:,jj)=d2_ind; % d2
end


end
