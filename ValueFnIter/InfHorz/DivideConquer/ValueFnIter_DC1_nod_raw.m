function [VKron, Policy]=ValueFnIter_DC1_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

N_a=prod(n_a);
N_z=prod(n_z);

% n-Monotonicity
% vfoptions.level1n=11;
level1ii=round(linspace(1,n_a,vfoptions.level1n));
level1iidiff=level1ii(2:end)-level1ii(1:end-1)-1;

%% Start by setting up ReturnFn for the first-level (as we can reuse this every iteration)
ReturnMatrixLvl1=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid, a_grid(level1ii), z_gridvals, ReturnFnParamsVec, 1);

V=reshape(V0,[N_a,N_z]);
Policy=zeros(N_a,N_z,'gpuArray');

% for Howards, preallocate
Ftemp=zeros(N_a,N_z,'gpuArray');
% and we need 
bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

% precompute
Epi_z=shiftdim(pi_z',-1); % pi_z in the form we need it to compute the expectations

% I want the print that tells you distance to have number of decimal points corresponding to vfoptions.tolerance
distvstolstr=['ValueFnIter: after %i iterations the dist is %4.',num2str(-round(log10(vfoptions.tolerance))),'f \n'];

%%
currdist=Inf;
tempcounter=1;
while currdist>vfoptions.tolerance && tempcounter<=vfoptions.maxiter

    Vold=V;
    
    %Calc the condl expectation term (except beta), which depends on z but not on control variables
    EV=Vold.*Epi_z;
    EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
    EV=sum(EV,2); % sum over z', leaving a singular second dimension

    DiscountedEV=DiscountFactorParamsVec*reshape(EV,[N_a,1,N_z]);

    %% Level 1
    % We can just reuse ReturnMatrixLvl1
    entireRHS_ii=ReturnMatrixLvl1+DiscountedEV;

    % First, we want aprime conditional on (1,a,z)
    [Vtempii,maxindex1]=max(entireRHS_ii,[],2);

    % Store
    % curraindex=level1ii';
    V(level1ii,:)=shiftdim(Vtempii,1);
    Policy(level1ii,:)=shiftdim(maxindex1,1);

    % Need to keep Ftemp for Howards policy iteration improvement
    Ftemp(level1ii,:)=ReturnMatrixLvl1(shiftdim(maxindex1,1)+N_a*(0:1:vfoptions.level1n-1)'+N_a*vfoptions.level1n*(0:1:N_z-1));

    % Attempt for improved version
    maxgap=squeeze(max(maxindex1(1,2:end,:)-maxindex1(1,1:end-1,:),[],3));
    for ii=1:(vfoptions.level1n-1)
        curraindex=(level1ii(ii)+1:1:level1ii(ii+1)-1)';
        if maxgap(ii)>0
            loweredge=min(maxindex1(1,ii,:),N_a-maxgap(ii)); % maxindex(ii,:), but avoid going off top of grid when we add maxgap(ii) points
            % loweredge is 1-by-1-by-n_z
            aprimeindexes=loweredge+(0:1:maxgap(ii));
            % aprime possibilities are maxgap(ii)+1-by-1-by-n_z
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(aprimeindexes), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
            aprimez=repelem(aprimeindexes-1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_z-1),-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(aprimez,[(maxgap(ii)+1),level1iidiff(ii),N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:)=shiftdim(Vtempii,1);
            %  the a1prime is relative to loweredge(allind), need to 'add' the loweredge
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            Policy(curraindex,:)=shiftdim(maxindex+N_d*(loweredge(zind)-1),1);

            % Need to keep Ftemp for Howards policy iteration improvement
            Ftemp(curraindex,:)=ReturnMatrix_ii(shiftdim(maxindex,1)+(maxgap(ii)+1)*(0:1:level1iidiff(ii)-1)'+(maxgap(ii)+1)*level1iidiff(ii)*(0:1:N_z-1));

        else
            loweredge=maxindex1(:,1,ii,:);
            % Just use aprime(ii) for everything
            ReturnMatrix_ii=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, a_grid(loweredge), a_grid(level1ii(ii)+1:level1ii(ii+1)-1), z_gridvals, ReturnFnParamsVec,2);
            aprimez=repelem(loweredge-1,1,level1iidiff(ii),1)+N_a*shiftdim((0:1:N_z-1),-1); % the current aprimeii(ii):aprimeii(ii+1)
            entireRHS_ii=ReturnMatrix_ii+DiscountedentireEV(reshape(aprimez,[1,level1iidiff(ii),N_z]));
            [Vtempii,maxindex]=max(entireRHS_ii,[],1);
            V(curraindex,:)=shiftdim(Vtempii,1);
            %  the aprime is relative to loweredge(allind), need to 'add' the loweredge
            zind=shiftdim((0:1:N_z-1),-1); % already includes -1
            Policy(curraindex,:)=shiftdim(maxindex+N_d*(loweredge(zind)-1),1);

            % Need to keep Ftemp for Howards policy iteration improvement
            Ftemp(curraindex,:)=ReturnMatrix_ii(shiftdim(maxindex,1)+(0:1:level1iidiff(ii)-1)'+level1iidiff(ii)*(0:1:N_z-1));

        end
    end

    %% Finish up
    % Update currdist
    Vdist=V(:)-Vold(:);
    Vdist(isnan(Vdist))=0;
    currdist=max(abs(Vdist));

    if isfinite(currdist) && currdist/vfoptions.tolerance>10 && tempcounter<vfoptions.maxhowards % Use Howards Policy Fn Iteration Improvement
        Policy_Vind=Policy(:);
        for Howards_counter=1:vfoptions.howards
            EVKrontemp=V(Policy_Vind,:);
            EVKrontemp=EVKrontemp.*aaa;
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
    Policy=PolicyInd2Val_Case1(Policy,0,n_a,n_z,d_grid, a_grid);
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