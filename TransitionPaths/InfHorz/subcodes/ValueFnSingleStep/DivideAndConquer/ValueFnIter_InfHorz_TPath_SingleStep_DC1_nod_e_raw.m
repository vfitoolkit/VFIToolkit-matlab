function [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_DC1_nod_e_raw(Vnext,n_a,n_z,n_e, a_grid, z_gridvals, e_gridvals, pi_z, pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Single Bellman step, divide-and-conquer (one endogenous state), no d, with z and iid e.
% Vnext is [N_a,N_z,N_e]; pi_z is [N_z,N_zprime]; pi_e is [N_e,1], the distribution of next period e.
% lowmemory: 0 is everything at once; 1 loops over e; 2 loops over z and e.

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

if vfoptions.lowmemory>=1
    special_n_e=ones(1,length(n_e));
end
if vfoptions.lowmemory>=2
    special_n_z=ones(1,length(n_z));
end

zind=shiftdim(gpuArray(0:1:N_z-1),-1); % [1,1,N_z], already includes -1

% n-Monotonicity
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

V=zeros(N_a,N_z,N_e,'gpuArray');
Policy=zeros(N_a,N_z,N_e,'gpuArray'); %first dim indexes the optimal choice for aprime rest of dimensions a,z,e

%%

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);

DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Calc the condl expectation term (except beta), computed once and used (or sliced) by every lowmemory branch
% First e', then z'
EV=Vnext.*shiftdim(pi_e,-2); % [N_a,N_z,N_e], aprime-by-zprime-by-eprime
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the probabilities)
EV=sum(EV,3); % sum over e', [N_a,N_z], aprime-by-zprime
EV=EV.*shiftdim(pi_z',-1); % [N_a,N_zprime,N_z]
EV(isnan(EV))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
EV=sum(EV,2); % sum over z', leaving a singular second dimension, [N_a,1,N_z], aprime-by-1-by-z
DiscountedEV=DiscountFactorParamsVec*EV; % [N_a,1,N_z]; linear index of (aprime,1,z) is aprime+N_a*(z-1)

if vfoptions.lowmemory==0

    % n-Monotonicity
    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, a_grid, a_grid(level1ii), z_gridvals, e_gridvals, ReturnFnParamsVec,1); % [N_a,level1n,N_z,N_e], aprime-by-a-by-z-by-e

    entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % DiscountedEV autoexpands over a and e

    %Calc the max and it's index
    [Vtempii,maxindex1]=max(entireRHS_ii,[],1); % [1,level1n,N_z,N_e]

    V(level1ii,:,:)=shiftdim(Vtempii,1);
    Policy(level1ii,:,:)=shiftdim(maxindex1,1);

    % Attempt for improved version
    maxgap=max(max(maxindex1(1,2:end,:,:)-maxindex1(1,1:end-1,:,:),[],4),[],3); % max over z,e
    for ii=1:(vfoptions.level1n-1)
        curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-n_z-by-n_e
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are maxgap(ii)+1-by-1-by-n_z-by-n_e
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, reshape(a_grid(aprimeindexes),[maxgap(ii)+1,1,N_z,N_e]), a_grid(curraindex), z_gridvals, e_gridvals, ReturnFnParamsVec,2); % [maxgap(ii)+1,npts,N_z,N_e]
            aprimez=aprimeindexes+N_a*zind; % [maxgap(ii)+1,1,N_z,N_e], index into DiscountedEV
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimez),[maxgap(ii)+1,1,N_z,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:,:)=shiftdim(Vtempii,1);
            Policy(curraindex,:,:)=shiftdim(maxindex+loweredge-1,1);
        else % maxgap(ii)==0
            loweredge=maxindex1(1,ii,:,:);
            % loweredge is 1-by-1-by-n_z-by-n_e
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, n_e, reshape(a_grid(loweredge),[1,1,N_z,N_e]), a_grid(curraindex), z_gridvals, e_gridvals, ReturnFnParamsVec,2); % [1,npts,N_z,N_e]
            aprimez=loweredge+N_a*zind; % [1,1,N_z,N_e], index into DiscountedEV
            entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimez),[1,1,N_z,N_e]); % autoexpand level1iidiff(ii) in 2nd-dim
            % can skip max() over as just a single point
            V(curraindex,:,:)=shiftdim(entireRHS_ii,1);
            Policy(curraindex,:,:)=repelem(reshape(loweredge,[1,N_z,N_e]),level1iidiff(ii),1,1);
        end
    end

elseif vfoptions.lowmemory==1
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, a_grid, a_grid(level1ii), z_gridvals, e_val, ReturnFnParamsVec,1); % [N_a,level1n,N_z], aprime-by-a-by-z

        entireRHS_ii=ReturnMatrix_ii+DiscountedEV; % DiscountedEV autoexpands over a

        %Calc the max and it's index
        [Vtempii,maxindex1]=max(entireRHS_ii,[],1); % [1,level1n,N_z]

        V(level1ii,:,e_c)=shiftdim(Vtempii,1);
        Policy(level1ii,:,e_c)=shiftdim(maxindex1,1);

        % Attempt for improved version
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3); % max over z
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_z
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, reshape(a_grid(aprimeindexes),[maxgap(ii)+1,1,N_z]), a_grid(curraindex), z_gridvals, e_val, ReturnFnParamsVec,2); % [maxgap(ii)+1,npts,N_z]
                aprimez=aprimeindexes+N_a*zind; % [maxgap(ii)+1,1,N_z], index into DiscountedEV
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimez),[maxgap(ii)+1,1,N_z]); % autoexpand level1iidiff(ii) in 2nd-dim
                [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                V(curraindex,:,e_c)=shiftdim(Vtempii,1);
                Policy(curraindex,:,e_c)=shiftdim(maxindex+loweredge-1,1);
            else % maxgap(ii)==0
                loweredge=maxindex1(1,ii,:);
                % loweredge is 1-by-1-by-n_z
                % Just use aprime(ii) for everything
                ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, n_z, special_n_e, reshape(a_grid(loweredge),[1,1,N_z]), a_grid(curraindex), z_gridvals, e_val, ReturnFnParamsVec,2); % [1,npts,N_z]
                aprimez=loweredge+N_a*zind; % [1,1,N_z], index into DiscountedEV
                entireRHS_ii=ReturnMatrix_ii+reshape(DiscountedEV(aprimez),[1,1,N_z]); % autoexpand level1iidiff(ii) in 2nd-dim
                % can skip max() over as just a single point
                V(curraindex,:,e_c)=shiftdim(entireRHS_ii,1);
                Policy(curraindex,:,e_c)=repelem(reshape(loweredge,[1,N_z]),level1iidiff(ii),1);
            end
        end
    end

elseif vfoptions.lowmemory==2
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        DiscountedEV_z=DiscountedEV(:,1,z_c); % [N_a,1]

        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid, a_grid(level1ii), z_val, e_val, ReturnFnParamsVec,1); % [N_a,level1n], aprime-by-a

            entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z;

            %Calc the max and it's index
            [Vtempii,maxindex1]=max(entireRHS_ii,[],1); % [1,level1n]

            V(level1ii,z_c,e_c)=Vtempii';
            Policy(level1ii,z_c,e_c)=maxindex1';

            % Attempt for improved version
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid(aprimeindexes), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2); % [maxgap(ii)+1,npts]
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(aprimeindexes); % autoexpand level1iidiff(ii) in 2nd-dim
                    [Vtempii,maxindex]=max(entireRHS_ii,[],1);
                    V(curraindex,z_c,e_c)=Vtempii';
                    Policy(curraindex,z_c,e_c)=maxindex'+loweredge-1;
                else % maxgap(ii)==0
                    loweredge=maxindex1(1,ii);
                    % Just use aprime(ii) for everything
                    % No need for reshape(a_grid(loweredge),...) as is just a single point
                    ReturnMatrix_ii=CreateReturnFnMatrix_Disc_DC1_nod_e(ReturnFn, special_n_z, special_n_e, a_grid(loweredge), a_grid(curraindex), z_val, e_val, ReturnFnParamsVec,2); % [1,npts]
                    entireRHS_ii=ReturnMatrix_ii+DiscountedEV_z(loweredge);
                    % can skip max() over as just a single point
                    V(curraindex,z_c,e_c)=entireRHS_ii';
                    Policy(curraindex,z_c,e_c)=loweredge;
                end
            end
        end
    end
end

%% Policy in transition paths
Policy=reshape(ind2sub_vec_homemade(n_a,Policy(:))',[length(n_a),N_a,N_z,N_e]);

end
