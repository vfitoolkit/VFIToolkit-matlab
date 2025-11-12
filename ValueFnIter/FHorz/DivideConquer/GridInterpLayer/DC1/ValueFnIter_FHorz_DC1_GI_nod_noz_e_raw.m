function [V, Policy]=ValueFnIter_FHorz_DC1_GI_nod_noz_e_raw(n_a,n_e,N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a=prod(n_a);
N_e=prod(n_e);

V=zeros(N_a,N_e,N_j,'gpuArray');
Policy=zeros(2,N_a,N_e,N_j,'gpuArray'); % first dim indexes the optimal choice for aprime and aprime2 (in GI layer)

%%
% Preallocate
if vfoptions.lowmemory==0
    midpoints_jj=zeros(1,N_a,N_e,'gpuArray');
elseif vfoptions.lowmemory==1 % loops over e
    midpoints_jj=zeros(1,N_a,'gpuArray');
    special_n_e=ones(1,length(n_e));
end

% n-Monotonicity
% vfoptions.level1n=5;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
% level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

% Grid interpolation
% vfoptions.ngridinterp=9;
n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
aprime_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short));
% n2aprime=length(aprime_grid);

% For debugging, uncomment next two lines, with this 'aprime_grid' you
% should get exact same value fn as without interpolation (as it doesn't
% really interpolate, it just repeats points)
% aprime_grid=repelem(a_grid,1+n2short,1);
% aprime_grid=aprime_grid(1:(N_a+(N_a-1)*n2short));

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames, N_j);

pi_e_J=shiftdim(pi_e_J,-1); % Move to second dimension

if ~isfield(vfoptions,'V_Jplus1')
    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        % Calc the max and it's index
        [~,maxindex1]=max(ReturnMatrix_ii,[],1);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                [~,maxindex]=max(ReturnMatrix_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_z
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_z
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            % Calc the max and it's index
            [~,maxindex1]=max(ReturnMatrix_ii,[],1);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    [~,maxindex]=max(ReturnMatrix_ii,[],1);
                    midpoints_jj(1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii);
                    midpoints_jj(1,curraindex)=repelem(loweredge,1,length(curraindex));
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            [Vtempii,maxindexL2]=max(ReturnMatrix_ii,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
            Policy(2,:,e_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end

else
    % Using V_Jplus1
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,N_j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

    EV=sum(reshape(vfoptions.V_Jplus1,[N_a,N_e]).*pi_e_J(1,:,N_j),2); % Use V_Jplus1
    
    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0

        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

        %Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,N_j), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(reshape(aprimeindexes(:),[1,1,N_e])); % autofill level1iidiff(ii) in the 2nd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,N_j),ReturnFnParamsVec,2);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,N_j)=shiftdim(Vtempii,1);
        Policy(1,:,:,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
        Policy(2,:,:,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,N_j);

            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

            %Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1-by-n_e
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(aprimeindexes(:)); % autofill level1iidiff(ii) in the 2nd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex)=maxindex+(loweredge-1);
                else
                    loweredge=maxindex1(1,ii);
                    midpoints_jj(1,curraindex)=repelem(loweredge,1,length(curraindex));
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            % aprime=aprimeindexes;
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,e_c,N_j)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,N_j)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
            Policy(2,:,e_c,N_j)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end    
end



%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    EV=sum(V(:,:,jj+1).*pi_e_J(1,:,jj),2);

    % Interpolate EV over aprime_grid
    EVinterp=interp1(a_grid,EV,aprime_grid);

    if vfoptions.lowmemory==0
        % n-Monotonicity
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid, a_grid(level1ii), e_gridvals_J(:,:,jj), ReturnFnParamsVec,1);

        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

        % Calc the max and it's index
        [~,maxindex1]=max(entireRHS_ii,[],1);

        % Just keep the 'midpoint' vesion of maxindex1 [as GI]
        midpoints_jj(1,level1ii,:)=maxindex1;

        % Attempt for improved version
        maxgap=max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3);
        for ii=1:(vfoptions.level1n-1)
            curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
            if maxgap(ii)>0
                loweredge=min(maxindex1(1,ii,:),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                % loweredge is 1-by-1-by-n_e
                aprimeindexes=loweredge+(0:1:maxgap(ii))';
                % aprime possibilities are maxgap(ii)+1-by-1-by-n_e
                ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_gridvals_J(:,:,jj), ReturnFnParamsVec,2);
                entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(reshape(aprimeindexes(:),[maxgap(ii)+1,1,N_e])); % autofill level1iidiff(ii) in the 2nd-dim
                [~,maxindex]=max(entireRHS_ii,[],1);
                midpoints_jj(1,curraindex,:)=maxindex+(loweredge-1);
            else % maxgap(ii)==0
                loweredge=maxindex1(1,ii,:);
                midpoints_jj(1,curraindex,:)=repelem(loweredge,1,length(curraindex),1);
            end
        end

        % Turn this into the 'midpoint'
        midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
        % midpoint is 1-by-n_a-by-n_e
        aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
        % aprime possibilities are n_d-by-n2long-by-n_a-by-n_e
        ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,n_e,aprime_grid(aprimeindexes),a_grid,e_gridvals_J(:,:,jj),ReturnFnParamsVec,2);
        entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a,N_e]);
        [Vtempii,maxindexL2]=max(entireRHS_ii,[],1);
        V(:,:,jj)=shiftdim(Vtempii,1);
        Policy(1,:,:,jj)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
        Policy(2,:,:,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind

    elseif vfoptions.lowmemory==1
        for e_c=1:N_e
            e_val=e_gridvals_J(e_c,:,jj);
            % n-Monotonicity
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid, a_grid(level1ii), e_val, ReturnFnParamsVec,1);

            entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV;

            % Calc the max and it's index
            [~,maxindex1]=max(entireRHS_ii,[],1);

            % Just keep the 'midpoint' vesion of maxindex1 [as GI]
            midpoints_jj(1,level1ii)=maxindex1;

            % Attempt for improved version
            maxgap=maxindex1(1,2:end)-maxindex1(1,1:end-1);
            for ii=1:(vfoptions.level1n-1)
                curraindex=level1ii(ii)+1:1:level1ii(ii+1)-1;
                if maxgap(ii)>0
                    loweredge=min(maxindex1(1,ii),n_a-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
                    % loweredge is 1-by-1-by-n_e
                    aprimeindexes=loweredge+(0:1:maxgap(ii))';
                    % aprime possibilities are maxgap(ii)+1-by-1
                    ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, special_n_e, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), e_val, ReturnFnParamsVec,2);
                    entireRHS_ii=ReturnMatrix_ii+DiscountFactorParamsVec*EV(aprimeindexes(:));  % autofill level1iidiff(ii) in the 2nd-dim
                    [~,maxindex]=max(entireRHS_ii,[],1);
                    midpoints_jj(1,curraindex)=maxindex+(loweredge-1);
                else % maxgap(ii)==0
                    loweredge=maxindex1(1,ii);
                    midpoints_jj(1,curraindex)=repelem(loweredge,1,length(curraindex));
                end
            end

            % Turn this into the 'midpoint'
            midpoints_jj=max(min(midpoints_jj,n_a-1),2); % avoid the top end (inner), and avoid the bottom end (outer)
            % midpoint is 1-by-n_a
            aprimeindexes=(midpoints_jj+(midpoints_jj-1)*n2short)+(-n2short-1:1:1+n2short)'; % aprime points either side of midpoint
            % aprime possibilities are n_d-by-n2long-by-n_a
            ReturnMatrix_ii_e=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn,special_n_e,aprime_grid(aprimeindexes),a_grid,e_val,ReturnFnParamsVec,2);
            entireRHS_ii_e=ReturnMatrix_ii_e+DiscountFactorParamsVec*reshape(EVinterp(aprimeindexes(:)),[n2long,N_a]);
            [Vtempii,maxindexL2]=max(entireRHS_ii_e,[],1);
            V(:,e_c,jj)=shiftdim(Vtempii,1);
            Policy(1,:,e_c,jj)=shiftdim(squeeze(midpoints_jj),-1); % midpoint
            Policy(2,:,e_c,jj)=shiftdim(maxindexL2,-1); % aprimeL2ind
        end
    end    
end


% Currently Policy(1,:) is the midpoint, and Policy(2,:) the second layer
% (which ranges -n2short-1:1:1+n2short). It is much easier to use later if
% we switch Policy(1,:) to 'lower grid point' and then have Policy(2,:)
% counting 0:nshort+1 up from this.
adjust=(Policy(2,:,:,:)<1+n2short+1); % if second layer is choosing below midpoint
Policy(1,:,:,:)=Policy(1,:,:,:)-adjust; % lower grid point
Policy(2,:,:,:)=adjust.*Policy(2,:,:,:)+(1-adjust).*(Policy(2,:,:,:)-n2short-1); % from 1 (lower grid point) to 1+n2short+1 (upper grid point)

Policy=squeeze(Policy(1,:,:,:)+N_a*(Policy(2,:,:,:)-1));


end
