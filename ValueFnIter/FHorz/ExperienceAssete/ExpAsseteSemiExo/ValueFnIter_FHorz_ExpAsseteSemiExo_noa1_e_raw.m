function [V,Policy3]=ValueFnIter_FHorz_ExpAsseteSemiExo_noa1_e_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_semiz,n_e,N_j, d12_gridvals, d2_gridvals, d3_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, e_gridvals_J, pi_z_J, pi_semiz_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% noa1 version of ValueFnIter_FHorz_ExpAsseteSemiExo_e_raw (d1, z, e).
% Policy3 stores (d1, d2, d3) -- no a1prime channel since noa1.
% aprimeFn = aprimeFn(d2, a2, e, ...)
% Joint exogenous ordering: bothz = [semiz, z], semiz fastest; e is separate

n_bothz=[n_semiz,n_z];

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d12=N_d1*N_d2;
N_d3=prod(n_d3);
N_a2=prod(n_a2);
N_a=N_a2;
a2_gridvals=CreateGridvals(n_a2,a2_grid,1); % the CreateReturnFnMatrix_* commands want gridvals ([N_a2-by-l_a2]), not the stacked a2_grid.
% (These are the same array when there is only one experience asset, which is why passing a2_grid worked until l_a2=2.)
N_semiz=prod(n_semiz);
N_z=prod(n_z);
N_bothz=prod(n_bothz);
N_e=prod(n_e);

V=zeros(N_a,N_bothz,N_e,N_j,'gpuArray');
Policy3=zeros(3,N_a,N_bothz,N_e,N_j,'gpuArray');

%%
bothz_gridvals_J=[repmat(semiz_gridvals_J,N_z,1,1),repelem(z_gridvals_J,N_semiz,1,1)];

n_d=[n_d1,n_d2,n_d3];
d123_gridvals=[repmat(d12_gridvals,N_d3,1),repelem(CreateGridvals(n_d3,d3_grid,1),N_d12,1)];

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

%% j=N_j
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_bothz, n_e, d123_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);
        [Vtemp,maxindex]=max(ReturnMatrix,[],1);
        V(:,:,:,N_j)=Vtemp;
        d12_ind=rem(maxindex-1,N_d12)+1;
        Policy3(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
        Policy3(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
        Policy3(3,:,:,:,N_j)=ceil(maxindex/N_d12);  % d3
    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            ReturnMatrix_e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, n_bothz, special_n_e, d123_gridvals, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);
            [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
            V(:,:,e_c,N_j)=Vtemp;
            d12_ind=rem(maxindex-1,N_d12)+1;
            Policy3(1,:,:,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
            Policy3(2,:,:,e_c,N_j)=ceil(d12_ind/N_d1);
            Policy3(3,:,:,e_c,N_j)=ceil(maxindex/N_d12);
        end
    elseif vfoptions.lowmemory==2
        for z_c=1:N_z
            semizblock=(z_c-1)*N_semiz+(1:N_semiz);
            z_val=bothz_gridvals_J(semizblock,:,N_j);
            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, special_n_semiz, special_n_e, d123_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,semizblock,e_c,N_j)=shiftdim(Vtemp,1);
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy3(1,:,semizblock,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3(2,:,semizblock,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy3(3,:,semizblock,e_c,N_j)=ceil(maxindex/N_d12);
            end
        end
    elseif vfoptions.lowmemory==3
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            for z_c=1:N_bothz
                z_val=bothz_gridvals_J(z_c,:,N_j);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, n_d, n_a2, special_n_bothz, special_n_e, d123_gridvals, a2_gridvals, z_val, e_val, ReturnFnParamsVec);
                [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
                V(:,z_c,e_c,N_j)=Vtemp;
                d12_ind=rem(maxindex-1,N_d12)+1;
                Policy3(1,:,z_c,e_c,N_j)=rem(d12_ind-1,N_d1)+1;
                Policy3(2,:,z_c,e_c,N_j)=ceil(d12_ind/N_d1);
                Policy3(3,:,z_c,e_c,N_j)=ceil(maxindex/N_d12);
            end
        end
    end
else
    % aprime depends on (d2, a2, current_e); independent of d3, semiz, z -- compute once
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [a2primeIndex,a2primeProbs]=CreateExperienceAsseteFnMatrix(aprimeFn, n_d2, n_a2, n_e, d2_gridvals, a2_grid, e_gridvals_J(:,:,N_j), aprimeFnParamsVec,1);
    % Note: a2primeIndex is [N_d2*N_a2*N_e,1], whereas a2primeProbs is [N_d2,N_a2,N_e]   (N_e here is the current e)
    if length(n_a2)==1
        aprimeplus1Index=a2primeIndex+1;
    else
        % l_a2==2: a2primeIndex/a2primeProbs are per-dim factored. Fold the two per-dim lower
        % indexes into the four corners of the a2 lattice, and interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:),[N_d2,N_a2,N_e]);
        loIdx_2=reshape(a2primeIndex(2,:),[N_d2,N_a2,N_e]);
        prob_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_e]);
        prob_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_e]);
        a2prime_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        a2prime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        a2prime_lh=loIdx_1+n_a2_1*loIdx_2;
        a2prime_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
    end

    % Integrate out eprime first (e is i.i.d. start-of-period); EVpre is [N_a, N_bothz]
    EVpre=sum(reshape(vfoptions.V_Jplus1,[N_a,N_bothz,N_e]).*shiftdim(pi_e_J(:,N_j+1),-2),3);

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]); % (d2,a2,e_cur,bothzcur)
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz); % [N_d2,N_a2,N_e,N_bothz]
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper; % (d2,a2,e_cur,bothzcur)
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur) -- match return-fn dim order

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, n_e, d123_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_gridvals_J(:,:,N_j), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]);
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, special_n_e, d123_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,N_j), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]);
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,N_j);
                DiscountedEV_z=DiscountedEV(:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV_z(:,:,:,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,N_j),pi_semiz_J(:,:,d3_c,N_j));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]);
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,N_j);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,N_j);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, special_n_e, d123_gridvals_val, a2_gridvals, z_val, e_val, ReturnFnParamsVec);

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
    Policy3(3,:,:,:,N_j)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy3(1,:,:,:,N_j)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,N_j)=ceil(d12_ind/N_d1);    % d2
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
    if length(n_a2)==1
        aprimeplus1Index=a2primeIndex+1;
    else
        % l_a2==2: a2primeIndex/a2primeProbs are per-dim factored. Fold the two per-dim lower
        % indexes into the four corners of the a2 lattice, and interp inside the d3 loop.
        n_a2_1=n_a2(1);
        loIdx_1=reshape(a2primeIndex(1,:),[N_d2,N_a2,N_e]);
        loIdx_2=reshape(a2primeIndex(2,:),[N_d2,N_a2,N_e]);
        prob_1=reshape(a2primeProbs(1,:),[N_d2,N_a2,N_e]);
        prob_2=reshape(a2primeProbs(2,:),[N_d2,N_a2,N_e]);
        a2prime_ll=loIdx_1+n_a2_1*(loIdx_2-1);
        a2prime_hl=(loIdx_1+1)+n_a2_1*(loIdx_2-1);
        a2prime_lh=loIdx_1+n_a2_1*loIdx_2;
        a2prime_hh=(loIdx_1+1)+n_a2_1*loIdx_2;
    end

    EVpre=sum(V(:,:,:,jj+1).*shiftdim(pi_e_J(:,jj+1),-2),3); % [N_a, N_bothz]

    if vfoptions.lowmemory==0
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]); % (d2,a2,e_cur,bothzcur)
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)

            ReturnMatrix_d3=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, n_e, d123_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,jj), e_gridvals_J(:,:,jj), ReturnFnParamsVec);

            entireRHS=ReturnMatrix_d3+DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            [Vtemp,maxindex]=max(entireRHS,[],1);
            V_ford3_jj(:,:,:,d3_c)=shiftdim(Vtemp,1);
            Policy_ford3_jj(:,:,:,d3_c)=shiftdim(maxindex,1);
        end
    elseif vfoptions.lowmemory==1
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]);
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                ReturnMatrix_d3e=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, n_bothz, special_n_e, d123_gridvals_val, a2_gridvals, bothz_gridvals_J(:,:,jj), e_val, ReturnFnParamsVec);

                entireRHS_e=ReturnMatrix_d3e+DiscountedEV(:,:,:,e_c);

                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V_ford3_jj(:,:,e_c,d3_c)=shiftdim(Vtemp,1);
                Policy_ford3_jj(:,:,e_c,d3_c)=shiftdim(maxindex,1);
            end
        end
    elseif vfoptions.lowmemory==2
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]);
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            for z_c=1:N_z
                semizblock=(z_c-1)*N_semiz+(1:N_semiz);
                z_val=bothz_gridvals_J(semizblock,:,jj);
                DiscountedEV_z=DiscountedEV(:,:,semizblock,:);
                for e_c=1:N_e
                    e_val=e_gridvals_J(e_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_semiz, special_n_e, d123_gridvals_val, a2_gridvals, z_val, e_val, ReturnFnParamsVec);

                    entireRHS_ze=ReturnMatrix_d3ze+DiscountedEV_z(:,:,:,e_c);

                    [Vtemp,maxindex]=max(entireRHS_ze,[],1);
                    V_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(Vtemp,1);
                    Policy_ford3_jj(:,semizblock,e_c,d3_c)=shiftdim(maxindex,1);
                end
            end
        end
    elseif vfoptions.lowmemory==3
        for d3_c=1:N_d3
            d123_gridvals_val=[d12_gridvals,repelem(d3_grid(d3_c),N_d12,1)];
            pi_bothz=kron(pi_z_J(:,:,jj),pi_semiz_J(:,:,d3_c,jj));

            EV=EVpre.*shiftdim(pi_bothz',-1);
            EV(isnan(EV))=0;
            EV=sum(EV,2);

            if length(n_a2)==1
                Vlower=reshape(EV(a2primeIndex,:),[N_d2,N_a2,N_e,N_bothz]);
                Vupper=reshape(EV(aprimeplus1Index,:),[N_d2,N_a2,N_e,N_bothz]);
                skipinterp=(Vlower==Vupper);
                aprimeProbs_d3=repmat(a2primeProbs,1,1,1,N_bothz);
                aprimeProbs_d3(skipinterp)=0;
                entireEV=aprimeProbs_d3.*Vlower+(1-aprimeProbs_d3).*Vupper;
                entireEV(aprimeProbs_d3==0)=Vupper(aprimeProbs_d3==0); % includes the skipinterp positions; a zero weight against an infinite node gives 0*(-Inf)=NaN
                entireEV(aprimeProbs_d3==1)=Vlower(aprimeProbs_d3==1);
            else
                % l_a2==2: nested 2-corner interp over the four a2 corners, with skipinterp at each
                % level and per-contribution NaN cleanup so that a zero weight against an infinite
                % node (0*(-Inf)=NaN) does not poison the sum.
                V_ll=reshape(EV(a2prime_ll,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hl=reshape(EV(a2prime_hl,:),[N_d2,N_a2,N_e,N_bothz]);
                V_lh=reshape(EV(a2prime_lh,:),[N_d2,N_a2,N_e,N_bothz]);
                V_hh=reshape(EV(a2prime_hh,:),[N_d2,N_a2,N_e,N_bothz]);
                prob_1_d3=repmat(prob_1,1,1,1,N_bothz);
                prob_2_d3=repmat(prob_2,1,1,1,N_bothz);
                % inner level: interpolate over the a2_1 dimension, at each a2_2 corner
                p1_lo=prob_1_d3; p1_lo(V_ll==V_hl)=0;
                c_ll=p1_lo.*V_ll; c_ll(isnan(c_ll))=0;
                c_hl=(1-p1_lo).*V_hl; c_hl(isnan(c_hl))=0;
                EV_lo=c_ll+c_hl;
                p1_hi=prob_1_d3; p1_hi(V_lh==V_hh)=0;
                c_lh=p1_hi.*V_lh; c_lh(isnan(c_lh))=0;
                c_hh=(1-p1_hi).*V_hh; c_hh(isnan(c_hh))=0;
                EV_hi=c_lh+c_hh;
                % outer level: interpolate those two over the a2_2 dimension
                p2=prob_2_d3; p2(EV_lo==EV_hi)=0;
                c_lo=p2.*EV_lo; c_lo(isnan(c_lo))=0;
                c_hi=(1-p2).*EV_hi; c_hi(isnan(c_hi))=0;
                entireEV=c_lo+c_hi;
            end
            entireEV=permute(entireEV,[1,2,4,3]); % (d2,a2,bothzcur,e_cur)
            DiscountedEV=DiscountFactorParamsVec*repelem(entireEV,N_d1,1,1,1);

            for e_c=1:N_e
                e_val=e_gridvals_J(e_c,:,jj);
                for z_c=1:N_bothz
                    z_val=bothz_gridvals_J(z_c,:,jj);
                    ReturnMatrix_d3ze=CreateReturnFnMatrix_Case2_Disc_e(ReturnFn, [n_d1,n_d2,1], n_a2, special_n_bothz, special_n_e, d123_gridvals_val, a2_gridvals, z_val, e_val, ReturnFnParamsVec);

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
    Policy3(3,:,:,:,jj)=shiftdim(maxindex,-1); % d3
    maxindex=reshape(maxindex,[N_a*N_bothz*N_e,1]);
    d12_ind=reshape(Policy_ford3_jj((1:1:N_a*N_bothz*N_e)'+(N_a*N_bothz*N_e)*(maxindex-1)),[1,N_a,N_bothz,N_e]);
    Policy3(1,:,:,:,jj)=rem(d12_ind-1,N_d1)+1; % d1
    Policy3(2,:,:,:,jj)=ceil(d12_ind/N_d1);    % d2
end


end
