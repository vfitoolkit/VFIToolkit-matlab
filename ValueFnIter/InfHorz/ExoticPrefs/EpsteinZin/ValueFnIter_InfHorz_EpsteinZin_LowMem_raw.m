function [VKron, Policy]=ValueFnIter_InfHorz_EpsteinZin_LowMem_raw(VKron, n_d,n_a,n_z, d_gridvals,a_grid,z_grid, pi_z,DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, Howards,Howards2,Tolerance, maxiter, Verbose, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)
% Does pretty much exactly the same as ValueFnIter_InfHorz_EpsteinZin_raw, but builds the return matrix one z at a time inside the loop

% Progress print for Verbose (same form as the standard raws)
distvstolstr=['ValueFnIter (Epstein-Zin): after %i iterations the dist is %4.',num2str(-round(log10(Tolerance))),'f \n'];

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexes=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

%%
l_z=length(n_z);
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
elseif all(size(z_grid)==[prod(n_z),l_z])
    z_gridvals=z_grid;
end

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    for z_c=1:N_z
        zvals=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Disc(ReturnFn,n_d, n_a, ones(l_z,1),d_gridvals, a_grid, zvals,ReturnFnParamsVec,0);
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=isfinite(ReturnMatrix_z); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
        temp2_z=ReturnMatrix_z;
        temp2_z(becareful)=ReturnMatrix_z(becareful).^ezc2; % Otherwise can get things like 0 to negative power equals infinity

        % Part of Epstein-Zin is before taking expectation
        temp=VKronold;
        temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
        temp(VKronold==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

        %Calc the expectation term (except beta)
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension

        entireEV_z=repelem(EV_z,N_d,1);
        temp4=entireEV_z;
        temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
        temp4(entireEV_z==0)=0;

        entireRHS_z=ezc1*temp2_z+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        entireRHS_z(isnan(entireRHS_z))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_z==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        VKron(:,z_c)=Vtemp;
        PolicyIndexes(:,z_c)=maxindex;

        tempmaxindex=maxindex+(0:1:N_a-1)*(N_d*N_a);
%         Ftemp(:,z_c)=ReturnMatrix_z(tempmaxindex);
        Ftemp(:,z_c)=temp2_z(tempmaxindex);
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist)); %IS THIS reshape() & max() FASTER THAN max(max()) WOULD BE?

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(ceil(PolicyIndexes/N_d),:);

            % Part of Epstein-Zin is before taking expectation
            temp=EVKrontemp;
            temp(isfinite(EVKrontemp))=(ezc4*EVKrontemp(isfinite(EVKrontemp))).^ezc5;
            temp(EVKrontemp==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

            EVKrontemp=temp.*aaa;
            EVKrontemp(isnan(EVKrontemp))=0;
            EVKrontemp=reshape(sum(EVKrontemp,2),[N_a,N_z]);

            temp4=EVKrontemp;
            temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
            temp4(EVKrontemp==0)=0;

            % Note that Ftemp already includes all the relevant Epstein-Zin modifications
            VKron=ezc1*Ftemp+ezc3*DiscountFactorParamsVec*temp4;

            VKron(isnan(VKron))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
            temp5=~(VKron==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
            VKron(temp5)=VKron(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity
        end
    end

    if Verbose==1
        if rem(tempcounter,10)==0 % Every 10 iterations
            fprintf(distvstolstr, tempcounter,currdist)
        end
    end

    tempcounter=tempcounter+1;

end

if currdist>Tolerance
    warning(['Value fn iteration (Epstein-Zin) has stopped due to reaching the maximum number of iterations ', ...
             '(not due to convergence); can be set by vfoptions.maxiter. ', ...
             'Last currdist = %.16g; tolerance = %.16g.'], ...
             currdist, Tolerance)
end

Policy=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:)=shiftdim(rem(PolicyIndexes-1,N_d)+1,-1);
Policy(2,:,:)=shiftdim(ceil(PolicyIndexes/N_d),-1);

end
