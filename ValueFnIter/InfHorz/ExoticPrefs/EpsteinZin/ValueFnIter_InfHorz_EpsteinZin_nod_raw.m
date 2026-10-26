function [VKron, Policy]=ValueFnIter_InfHorz_EpsteinZin_nod_raw(VKron, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, Howards,Howards2,Tolerance, maxiter, Verbose, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7)
% Does pretty much exactly the same as ValueFnIter_InfHorz_EpsteinZin_raw, only without any decision variable (n_d=0)

% Progress print for Verbose (same form as the standard raws)
distvstolstr=['ValueFnIter (Epstein-Zin): after %i iterations the dist is %4.',num2str(-round(log10(Tolerance))),'f \n'];

N_a=prod(n_a);
N_z=prod(n_z);

Policy=zeros(N_a,N_z,'gpuArray');

Ftemp=zeros(N_a,N_z,'gpuArray');

bbb=reshape(shiftdim(pi_z,-1),[1,N_z*N_z]);
ccc=kron(ones(N_a,1,'gpuArray'),bbb);
aaa=reshape(ccc,[N_a*N_z,N_z]);

DiscountFactorParamsVec=prod(DiscountFactorParamsVec);

% Modify the Return Function appropriately for Epstein-Zin Preferences
becareful=isfinite(ReturnMatrix); % a zero return is a value, not infeasible: 0^ezc2 is Inf when ezc2<0, and Inf^ezc7 is then the floor 0
temp2=ReturnMatrix;
temp2(becareful)=ReturnMatrix(becareful).^ezc2;

%%
tempcounter=1;
currdist=Inf;
while currdist>Tolerance && tempcounter<=maxiter
    VKronold=VKron;

    for z_c=1:N_z
%         ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        temp2_z=temp2(:,:,z_c);

        % Part of Epstein-Zin is before taking expectation
        temp=VKronold;
        temp(isfinite(VKronold))=(ezc4*VKronold(isfinite(VKronold))).^ezc5;
        temp(VKronold==-Inf)=-Inf*ezc4; % a -Inf (dead) continuation must stay infinitely bad: with ezc4=-1 the untouched -Inf would come out of ezc3*beta*(.) as +Inf

        %Calc the condl expectation term (except beta), which depends on z but not on control variables
        EV_z=temp.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multiplications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilities)
        EV_z=sum(EV_z,2); % sum over z', leaving a singular second dimension

        temp4=EV_z;
        temp4(~(temp4==-Inf))=temp4(~(temp4==-Inf)).^ezc6;
        temp4(EV_z==0)=0;

        entireRHS_z=ezc1*temp2_z+ezc3*DiscountFactorParamsVec*temp4.*ones(1,N_a,1);

        entireRHS_z(isnan(entireRHS_z))=-Inf; % Inf-Inf (a zero return today with a dead continuation) is infeasible
        temp5=~(entireRHS_z==-Inf); % apply the power to everything else, +Inf included (Inf^ezc7 is the floor 0 when ezc7<0)
        entireRHS_z(temp5)=entireRHS_z(temp5).^ezc7;  % matlab otherwise puts 0 to negative power to infinity

        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS_z,[],1);
        VKron(:,z_c)=Vtemp;
        Policy(:,z_c)=maxindex;

        tempmaxindex=maxindex+(0:1:N_a-1)*N_a;
        Ftemp(:,z_c)=temp2_z(tempmaxindex); % note that temp2_z is the EZ ReturnMatrix_z
    end

    VKrondist=reshape(VKron-VKronold,[N_a*N_z,1]); VKrondist(isnan(VKrondist))=0;
    currdist=max(abs(VKrondist));

    if isfinite(currdist) && currdist/Tolerance>10 && tempcounter<Howards2 %Use Howards Policy Fn Iteration Improvement
        for Howards_counter=1:Howards
            EVKrontemp=VKron(Policy,:);

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

end
