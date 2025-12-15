function [dPolicy_gridvals, aprimePolicy_gridvals]=CreateGridvals_Policy(Policy,n_d,n_aprime,n_a,n_z,d_grid,aprime_grid,simoptions,Case1orCase2, MatrixOrCell)
% Creates the 'gridvals' versions of the optimal policy. These allow for
% easier evaluation of functions on the grids via the EvalFnOnAgentDist
% commands.
% For Case1or2=1, aprime_gridvals is always returned, and d_gridvals is returned or equal to nan as appropriate based on n_d.
% For Case1or2=2, aprime_gridvals=nan, and d_gridvals is always returned.
% For MatrixOrCell=1, output takes form of matrices
% For MatrixOrCell=2, output takes form of cells.
%
% Gridvals contain N_a*N_z rows, and the columns for a given row contain
% all the values of all the 'a' variables. (ie. a_gridvals is N_a*N_z-by-l_a)
% These contain no more information than the standard grid format (e.g.,
% a_grid), but are substantially larger (use more memory), however for
% certain purposes they are much easier to use quickly or in parallel.
%
% If either of d or aprime is not relevant, then a value of nan will be returned for the corresponding gridvals output.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_aprime=length(n_aprime);

N_a=prod(n_a);
N_z=prod(n_z);

if N_z==0
    N_z=1;
end

Policy=reshape(Policy,[size(Policy,1),N_a*N_z]); % Note: Policy will be (l_d+l_a,n_a,n_z)

if simoptions.gridinterplayer==1
    % switch aprime_grid and Policy to be 'fine' index
    Policy(l_d+1,:)=(1+simoptions.ngridinterp)*(Policy(l_d+1,:)-1)+Policy(end,:);
    Policy=Policy(1:end-1,:);
    if l_aprime==1
        aprime_grid=interp1(1:1:n_aprime,aprime_grid,linspace(1,n_aprime,n_aprime+(n_aprime-1)*simoptions.ngridinterp));
    else
        N_a1=n_aprime(1);
        a1prime_grid=interp1(1:1:N_a1,aprime_grid(1:n_aprime(1))',linspace(1,N_a1,N_a1+(N_a1-1)*simoptions.ngridinterp))';
        aprime_grid=[a1prime_grid; aprime_grid(n_aprime(1)+1:end)];
        n_aprime(1)=n_aprime(1)+(n_aprime(1)-1)*simoptions.ngridinterp;
    end
end

if simoptions.experienceasset==1 || simoptions.experienceassetu==1
    l_aprime=l_aprime-1;
end


% Check if doing Case1 or Case2, and if Case1, then check if need d_gridvals
if Case1orCase2==1
    if l_d>0
        Policy_d=Policy(1:l_d,:)';
        Policy_aprime=Policy(l_d+1:end,:)';
        if l_d==1
            dPolicy_gridvals=d_grid(Policy_d);
        elseif l_d==2
            d1_grid=d_grid(1:n_d(1));
            d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
            dPolicy_gridvals=[d1_grid(Policy_d(:,1)),d2_grid(Policy_d(:,2))];
        elseif l_d==3
            d1_grid=d_grid(1:n_d(1));
            d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
            d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
            dPolicy_gridvals=[d1_grid(Policy_d(:,1)),d2_grid(Policy_d(:,2)),d3_grid(Policy_d(:,3))];
        elseif l_d==4
            d1_grid=d_grid(1:n_d(1));
            d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
            d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
            d4_grid=d_grid(n_d(1)+n_d(2)+n_d(3)+1:n_d(1)+n_d(2)+n_d(3)+n_d(4));
            dPolicy_gridvals=[d1_grid(Policy_d(:,1)),d2_grid(Policy_d(:,2)),d3_grid(Policy_d(:,4)),d4_grid(Policy_d(:,4))];
        end
    else
        dPolicy_gridvals=nan;
        Policy_aprime=Policy';
    end
    
    if l_aprime==1
        aprimePolicy_gridvals=aprime_grid(Policy_aprime);
        aprimePolicy_gridvals=reshape(aprimePolicy_gridvals,size(Policy_aprime)); % not sure why I even need this line, but I seem to need it
    elseif l_aprime==2
        a1prime_grid=aprime_grid(1:n_aprime(1));
        a2prime_grid=aprime_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2));
        aprimePolicy_gridvals=[a1prime_grid(Policy_aprime(:,1)),a2prime_grid(Policy_aprime(:,2))];
    elseif l_aprime==3
        a1prime_grid=aprime_grid(1:n_aprime(1));
        a2prime_grid=aprime_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2));
        a3prime_grid=aprime_grid(n_aprime(1)+n_aprime(2)+1:n_aprime(1)+n_aprime(2)+n_aprime(3));
        aprimePolicy_gridvals=[a1prime_grid(Policy_aprime(:,1)),a2prime_grid(Policy_aprime(:,2)),a3prime_grid(Policy_aprime(:,3))];
    elseif l_aprime==4
        a1prime_grid=aprime_grid(1:n_aprime(1));
        a2prime_grid=aprime_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2));
        a3prime_grid=aprime_grid(n_aprime(1)+n_aprime(2)+1:n_aprime(1)+n_aprime(2)+n_aprime(3));
        a4prime_grid=aprime_grid(n_aprime(1)+n_aprime(2)+n_aprime(3)+1:n_aprime(1)+n_aprime(2)+n_aprime(3)+n_aprime(4));
        aprimePolicy_gridvals=[a1prime_grid(Policy_aprime(:,1)),a2prime_grid(Policy_aprime(:,2)),a3prime_grid(Policy_aprime(:,3)),a4prime_grid(Policy_aprime(:,4))];
    end
else % Case1orCase2==2
    aprimePolicy_gridvals=nan; % there is no aprime when using Case2
    Policy_d=Policy';
    if l_d==1
        dPolicy_gridvals=d_grid(Policy_d);
    elseif l_d==2
        d1_grid=d_grid(1:n_d(1));
        d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
        dPolicy_gridvals=[d1_grid(Policy_d(:,1)),d2_grid(Policy_d(:,2))];
    elseif l_d==3
        d1_grid=d_grid(1:n_d(1));
        d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
        d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
        dPolicy_gridvals=[d1_grid(Policy_d(:,1)),d2_grid(Policy_d(:,2)),d3_grid(Policy_d(:,3))];
    elseif l_d==4
        d1_grid=d_grid(1:n_d(1));
        d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
        d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
        d4_grid=d_grid(n_d(1)+n_d(2)+n_d(3)+1:n_d(1)+n_d(2)+n_d(3)+n_d(4));
        dPolicy_gridvals=[d1_grid(Policy_d(:,1)),d2_grid(Policy_d(:,2)),d3_grid(Policy_d(:,4)),d4_grid(Policy_d(:,4))];
    end
end

if MatrixOrCell==2
    dPolicy_gridvals=num2cell(dPolicy_gridvals);
    aprimePolicy_gridvals=num2cell(aprimePolicy_gridvals);
end


end
