function  [V,Policy]=ValueFnIter_Case1_DC2B_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)
% DC2B: two endogenous states, divide-and-conquer only on the first endogenous state

N_a=prod(n_a);
N_z=prod(n_z);

N_a1=n_a(1);
N_a2=n_a(2);
a1_grid=a_grid(1:N_a1);
a2_grid=a_grid(N_a1+1:end);

% n-Monotonicity
% vfoptions.level1n=7;
level1ii=round(linspace(1,N_a1,vfoptions.level1n(1)));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;


%% Start by setting up ReturnFn for the first-level (as we can reuse this every iteration)
ReturnMatrixLvl1=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid, a2_grid, a1_grid(level1ii), a2_grid, z_gridvals, ReturnFnParamsVec,1);
% [N_a1prime,N_a2prime,N_a1,N_a2,N_z]
% if vfoptions.actualV0==0
%     %% Solve first-level ignoring the second level
%     % This will hopefully work to give a good initial guess for the second level
% 
%     % Now, do value function iteration on just this first level 
%     [V,~]=ValueFnIter_Case1_NoD_Par2_raw(zeros(vfoptions.level1n*N_a2,N_z,'gpuArray'), [vfoptions.level1n,N_a2], n_z, pi_z, DiscountFactorParamsVec, reshape(ReturnMatrixLvl1(level1ii,:,:,:,:),[vfoptions.level1n*N_a2,vfoptions.level1n*N_a2,N_z]), vfoptions.howards, vfoptions.maxhowards, 100*vfoptions.tolerance, vfoptions.maxiter);
%     % Note: uses 100*vfoptions.tolerance, as it is just an intial guess
% 
%     % Turn this V, which is currently only on the first level grid, into a full V by linear interpolation
% 
%     V=interp3(reshape(V,[vfoptions.level1n(1),N_a2,N_z]),(1:1:N_a2),linspace(1,vfoptions.level1n(1),N_a1)',(1:1:N_z));
%     % Note: For reasons known only to matlab, you use interp3(V,Xq,Yq,Zq) with X=1:n, Y=1:m, Z=1:p, where [m,n,p] = size(V).
%     %       So X and Y are 'reversed' from what you would expect them to be.
%     % Weird behaviour, presumably something to do with how Matlab views columns as the first dimension.
% 
%     V=reshape(V,[N_a,N_z]);
% 
%     if vfoptions.verbose==1
%         fprintf('Created the initial guess for V based on level 1 of divide-and-conquer \n')
%     end
% else
V=reshape(V0,[N_a,N_z]);
% end




%% Now that we are armed with a (hopefully decent) initial guess, we can get on with the full problem
% V=zeros(N_a,N_z,'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray'); % first dim indexes the optimal choice for aprime rest of dimensions a1,a2,z

Ftemp=zeros(N_a,N_z,'gpuArray');

% Precompute
Epi_z=shiftdim(pi_z',-1); % pi_z in the form we need it to compute the expections

% For Howards we want
bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

% I want the print that tells you distance to have number of decimal points corresponding to vfoptions.tolerance
distvstolstr=['ValueFnIter: after %i iterations the dist is %4.',num2str(-round(log10(vfoptions.tolerance))),'f \n'];

currdist=Inf;
tempcounter=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter

    Vold=V;

    EV=Vold.*Epi_z;
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension
    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a1,N_a2,1,1,N_z]);  % autoexpand (a,z)

    % n-Monotonicity
    entireRHS_ii=ReturnMatrixLvl1+DiscountedEV; % autoexpand (a,z)

    % Calc the max and it's index
    [~,maxindex1]=max(entireRHS_ii,[],1);

    % Now, get and store the full (d,aprime)
    [Vtempii,maxindex2]=max(reshape(entireRHS_ii,[N_a1*N_a2,vfoptions.level1n*N_a2,N_z]),[],1);
    % Store
    curraindex=repmat(level1ii',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',vfoptions.level1n,1);
    V(curraindex,:)=shiftdim(Vtempii,1);
    Policy(curraindex,:)=shiftdim(maxindex2,1);

    % Need to keep Ftemp for Howards policy iteration improvement
    Ftemp(curraindex,:)=ReturnMatrixLvl1(shiftdim(maxindex2,1)+N_a1*N_a2*(0:1:vfoptions.level1n*N_a2-1)'+N_a1*N_a2*vfoptions.level1n*N_a2*(0:1:N_z-1));

    % Attempt for improved version
    maxgap=squeeze(max(max(max(maxindex1(1,:,2:end,:,:)-maxindex1(1,:,1:end-1,:,:),[],5),[],4),[],2));
    for ii=1:(vfoptions.level1n-1)
        curraindex=repmat((level1ii(ii)+1:1:level1ii(ii+1)-1)',N_a2,1)+N_a1*repelem((0:1:N_a2-1)',level1iidiff(ii),1);
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,:,ii,:,:),N_a1-maxgap(ii)); % maxindex1(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii))';
            % aprime possibilities are (maxgap(ii)+1)-n_a2-by-1-by-n_a2-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid(aprimeindexes), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,2);
            aprimez=repelem(aprimeindexes,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1)+N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[(maxgap(ii)+1)*N_a2,level1iidiff(ii)*N_a2,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            a1primeind=rem(maxindex-1,maxgap(ii)+1)+1;
            a2primeind=ceil(maxindex/(maxgap(ii)+1));
            maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=a2primeind+N_a2*a2ind+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
            Policy(curraindex,:)=shiftdim(maxindexfix+loweredge(allind)-1,1);

            % Need to keep Ftemp for Howards policy iteration improvement
            Ftemp(curraindex,:)=ReturnMatrix_ii(shiftdim(maxindex,1)+(maxgap(ii)+1)*N_a2*(0:1:level1iidiff(ii)*N_a2-1)'+(maxgap(ii)+1)*N_a2*level1iidiff(ii)*N_a2*(0:1:N_z-1));

        else
            loweredge=maxindex1(1,:,ii,:,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC2B_nod_Par2(ReturnFn, n_z, a1_grid(loweredge), a2_grid, a1_grid(level1ii(ii)+1:level1ii(ii+1)-1), a2_grid, z_gridvals, ReturnFnParamsVec,2);
            aprimez=repelem(loweredge,1,1,level1iidiff(ii),1,1)+N_a1*(0:1:N_a2-1)+N_a*shiftdim((0:1:N_z-1),-3); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedEV(reshape(aprimez,[1*N_a2,level1iidiff(ii)*N_a2,N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:)=shiftdim(Vtempii,1);
            % maxindex needs to be reworked:
            %  the a2prime is only an 'after maxgap(ii)+1, but needs to be after N_a1'
            a1primeind=1;
            a2primeind=maxindex;
            maxindexfix=a1primeind+N_a1*(a2primeind-1); % put maxindex back together, using N_a1 to determine a2prime, rather than using (maxgap(ii)+1) which is what it originally was in maxindex
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            a2ind=repelem((0:1:N_a2-1),1,level1iidiff(ii)); % already includes -1
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            allind=a2primeind+N_a2*a2ind+N_a2*N_a2*zind; % loweredge is 1-by-n_a2-by-1-by-n_a2-by-n_z
            Policy(curraindex,:)=shiftdim(maxindexfix+loweredge(allind)-1,1);

            % Need to keep Ftemp for Howards policy iteration improvement
            Ftemp(curraindex,:)=ReturnMatrix_ii(shiftdim(maxindex,1)+N_a2*(0:1:level1iidiff(ii)*N_a2-1)'+N_a2*level1iidiff(ii)*N_a2*(0:1:N_z-1));
        end
    end


    %% Finish up
    % Update currdist
    Vdist=V(:)-Vold(:); 
    Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards % Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=V(Policy(:),:).*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=sum(EVKrontemp,2);
            V=Ftemp+DiscountFactorParamsVec*reshape(EVKrontemp,[N_a,N_z]);
        end
    end
    
    if vfoptions.verbose==1
        if rem(tempcounter,10)==0 % Every 10 iterations
            fprintf(distvstolstr, tempcounter,currdist) % use enough decimal points to be able to see countdown of currdist to 0
        end
    end

    tempcounter=tempcounter+1;

end



%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(V,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, 0, n_a, n_z,vfoptions);
else
    return
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(Policy,0,n_a,n_z,[], a_grid);
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end





end