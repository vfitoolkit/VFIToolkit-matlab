function [V,Policy]=ValueFnIter_FHorz_DC1_GI_noz_e_raw(n_d,n_a,n_e,N_j, d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(3,N_a,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)

%%
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
d_gridvals=CreateGridvals(n_d,d_grid,1);

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(N_d,1,N_a,N_e,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    midpoints_jj=zeros(N_d,1,N_a,1,'gpuArray');
    special_n_e=ones(1,length(n_e));
end

aind=0:1:N_a-1; % already includes -1
eind=shiftdim((0:1:N_e-1),-1); % already includes -1

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension (normally -2, but no z so -1)

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_grid, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % First, we want aprime conditional on (d,1,a,e)
        [~,maxindex1]=max(ReturnMatrix_ii,[],2);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(:,1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                [~,maxindex]=max(ReturnMatrix_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=shiftdim(maxindex+(loweredge-1),1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_grid, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            % First, we want aprime conditional on (d,1,a,e)
            [~,maxindex1]=max(ReturnMatrix_ii,[],2);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,3);
                    [~,maxindex]=max(ReturnMatrix_ii,[],2);
                    midpoints_jj(:,1,curraindex)=shiftdim(maxindex+(loweredge-1),1);
                else
                    loweredge=maxindex1(:,1,ii,:);
                    midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,N_j)=d_ind; % d
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

        end
    end

else
    % Using V_Jplus1
    V_Jplus1=reshape(vfoptions.V_Jplus1,[N_a,N_e]);    % First, switch V_Jplus1 into Kron form

    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(V_Jplus1.*pi_e_J(1,:,N_j),2);
    entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1]

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);
    entireEVinterp=repmat(shiftdim(EVinterp,-1),N_d,1,1,1); % [d,aprime,1,z]

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_grid, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,z)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(:,1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,3);
                daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d,(maxgap(ii)+1),level1iidiff(ii),N_e]));
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,N_j)=d_ind; % d
        Policy(2,:,:,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy(3,:,:,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_grid, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

            % First, we want aprime conditional on (d,1,a,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,3);
                    daprime=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprime,[N_d,(maxgap(ii)+1),level1iidiff(ii)]));
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,N_j)=d_ind; % d
            Policy(2,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
            Policy(3,:,e_c,N_j)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind
        end
    end
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj),2);
    entireEV=repmat(shiftdim(EV,-1),N_d,1,1,1); % [d,aprime,1]
    
    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);
    entireEVinterp=repmat(shiftdim(EVinterp,-1),N_d,1,1,1); % [d,aprime,1,z]

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_grid, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

        % First, we want aprime conditional on (d,1,a,e)
        [~,maxindex1]=max(entireRHS_ii,[],2);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(:,1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=squeeze(max(max(maxindex1(:,1,2:end,:)-maxindex1(:,1,1:end-1,:),[],4),[],1)); % max over d,e
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(:,1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is n_d-by-1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii));
                % aprime possibilities are n_d-by-maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,3);
                daprime=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprime,[N_d,(maxgap(ii)+1),level1iidiff(ii),N_e]));
                [~,maxindex]=max(entireRHS_ii,[],2);
                midpoints_jj(:,1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(:,1,ii,:);
                midpoints_jj(:,1,curraindex,:)=repelem(loweredge,1,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is n_d-by-1-by-n_a-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        d_ind=rem(maxindexL2-1,N_d)+1;
        allind=d_ind+N_d*aind+N_d*N_a*eind; % midpoint is n_d-by-1-by-n_a-by-n_e
        Policy(1,:,:,jj)=d_ind; % d
        Policy(2,:,:,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
        Policy(3,:,:,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_grid, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV;

            % First, we want aprime conditional on (d,1,a,e)
            [~,maxindex1]=max(entireRHS_ii,[],2);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(:,1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=squeeze(max(maxindex1(:,1,2:end)-maxindex1(:,1,1:end-1),[],1)); % max over d,e
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(:,1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is n_d-by-1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii));
                    % aprime possibilities are n_d-by-maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn, n_d, special_n_e, d_gridvals, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,3);
                    daprimez=(1:1:N_d)'+N_d*repelem(aprimeindexes-1,1,1,level1iidiff(ii),1); % the current aprimeii(ii):aprimeii(ii+1)
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*entireEV(reshape(daprimez,[N_d,(maxgap(ii)+1),level1iidiff(ii)]));
                    [~,maxindex]=max(entireRHS_ii,[],2);
                    midpoints_jj(:,1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(:,1,ii);
                    midpoints_jj(:,1,curraindex)=repelem(loweredge,1,1,length(curraindex),1);
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is n_d-by-1-by-n_a
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short); % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_Par2(ReturnFn,n_d,special_n_e,d_gridvals,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            daprime=(1:1:N_d)'+N_d*(aprimeindexes-1);
            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(entireEVinterp(daprime(:)),[N_d*n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);
            d_ind=rem(maxindexL2-1,N_d)+1;
            allind=d_ind+N_d*aind; % midpoint is n_d-by-1-by-n_a
            Policy(1,:,e_c,jj)=d_ind; % d
            Policy(2,:,e_c,jj)=shiftdim(squeeze(midpoints_jj(allind)),-1); % midpoint
            Policy(3,:,e_c,jj)=shiftdim(ceil(maxindexL2/N_d),-1); % aprimeL2ind

        end
    end
end

% Currently Policy(2,:) is the midpoint, and Policy(3,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(2,:) to 'lower grid point' and then have Policy(3,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(3,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(2,:,:,:)=Policy(2,:,:,:)-adjust; % lower grid point
Policy(3,:,:,:)=adjust.*Policy(3,:,:,:)+(1-adjust).*(Policy(3,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy(1,:,:,:)+N_d*(Policy(2,:,:,:)-1)+N_d*N_a*(Policy(3,:,:,:)-1));

end
