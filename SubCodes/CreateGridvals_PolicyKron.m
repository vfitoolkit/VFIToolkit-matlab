function [dPolicy_gridvals, aprimePolicy_gridvals]=CreateGridvals_PolicyKron(PolicyKron,n_d,n_aprime,n_a,n_z,d_grid,aprime_grid,Case1orCase2, MatrixOrCell)
% Identical to CreateGridvals_Policy, except that it works with PolicyKron instead of Policy
%
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

% NOTE TO SELF: THE CPU VERSION COULD CERTAINLY BE SPED UP BY BETTER USE OF VECTORIZATION.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_aprime=length(n_aprime);

N_a=prod(n_a);
N_z=prod(n_z);

% Check if doing Case1 or Case2, and if Case1, then check if need d_gridvals
if Case1orCase2==1
    if l_d>0
        PolicyKron=reshape(PolicyKron,[2,N_a*N_z]);
        Policy_d=PolicyKron(1,:)';
        Policy_aprime=PolicyKron(2,:)';
        if l_d==1
            dPolicy_gridvals=d_grid(Policy_d);
        elseif l_d==2
            d1_grid=d_grid(1:n_d(1));
            d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
            dPolicy_gridvals=[d1_grid(rem(Policy_d-1,n_d(1))+1),d2_grid(ceil(Policy_d/n_d(1)))];
        elseif l_d==3
            d1_grid=d_grid(1:n_d(1));
            d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
            d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
            dPolicy_gridvals=[d1_grid(rem(Policy_d-1,n_d(1))+1),d2_grid(rem(ceil(Policy_d/prod(n_d(1)))-1,n_d(2))+1),d3_grid(ceil(Policy_d/prod(n_d(1:2))))];
        elseif l_d==4
            d1_grid=d_grid(1:n_d(1));
            d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
            d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
            d4_grid=d_grid(n_d(1)+n_d(2)+n_d(3)+1:n_d(1)+n_d(2)+n_d(3)+n_d(4));
            dPolicy_gridvals=[d1_grid(rem(Policy_d-1,n_d(1))+1),d2_grid(rem(ceil(Policy_d/n_d(1))-1,n_d(2))+1),d3_grid(rem(ceil(Policy_d/prod(n_d(1:2)))-1,n_d(3))+1),d4_grid(ceil(Policy_d/prod(n_d(1:3))))];
        end
    else
        dPolicy_gridvals=nan;
        Policy_aprime=reshape(PolicyKron,[N_a*N_z,1]);
    end

    if l_aprime==1
        aprimePolicy_gridvals=aprime_grid(Policy_aprime);
    elseif l_aprime==2
        a1prime_grid=aprime_grid(1:n_aprime(1));
        a2prime_grid=aprime_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2));
        aprimePolicy_gridvals=[a1prime_grid(rem(Policy_aprime-1,n_aprime(1))+1),a2prime_grid(ceil(Policy_aprime/n_aprime(1)))];
    elseif l_aprime==3
        a1prime_grid=aprime_grid(1:n_aprime(1));
        a2prime_grid=aprime_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2));
        a3prime_grid=aprime_grid(n_aprime(1)+n_aprime(2)+1:n_aprime(1)+n_aprime(2)+n_aprime(3));
        aprimePolicy_gridvals=[a1prime_grid(rem(Policy_aprime-1,n_aprime(1))+1),a2prime_grid(rem(ceil(Policy_aprime/prod(n_aprime(1)))-1,n_aprime(2))+1),a3prime_grid(ceil(Policy_aprime/prod(n_aprime(1:2))))];
    elseif l_aprime==4
        a1prime_grid=aprime_grid(1:n_aprime(1));
        a2prime_grid=aprime_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2));
        a3prime_grid=aprime_grid(n_aprime(1)+n_aprime(2)+1:n_aprime(1)+n_aprime(2)+n_aprime(3));
        a4prime_grid=aprime_grid(n_aprime(1)+n_aprime(2)+n_aprime(3)+1:n_aprime(1)+n_aprime(2)+n_aprime(3)+n_aprime(4));
        aprimePolicy_gridvals=[a1prime_grid(rem(Policy_aprime-1,n_aprime(1))+1),a2prime_grid(rem(ceil(Policy_aprime/n_aprime(1))-1,n_aprime(2))+1),a3prime_grid(rem(ceil(Policy_aprime/prod(n_aprime(1:2)))-1,n_aprime(3))+1),a4prime_grid(ceil(Policy_aprime/prod(n_aprime(1:3))))];
    end
else % Case1orCase2==2
    aprimePolicy_gridvals=nan; % there is no aprime when using Case2
    Policy_d=reshape(PolicyKron,[N_a*N_z,1]);
    if l_d==1
        dPolicy_gridvals=d_grid(Policy_d);
    elseif l_d==2
        d1_grid=d_grid(1:n_d(1));
        d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
        dPolicy_gridvals=[d1_grid(rem(Policy_d-1,n_d(1))+1),d2_grid(ceil(Policy_d/n_d(1)))];
    elseif l_d==3
        d1_grid=d_grid(1:n_d(1));
        d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
        d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
            dPolicy_gridvals=[d1_grid(rem(Policy_d-1,n_d(1))+1),d2_grid(rem(ceil(Policy_d/prod(n_d(1)))-1,n_d(2))+1),d3_grid(ceil(Policy_d/prod(n_d(1:2))))];
    elseif l_d==4
        d1_grid=d_grid(1:n_d(1));
        d2_grid=d_grid(n_d(1)+1:n_d(1)+n_d(2));
        d3_grid=d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3));
        d4_grid=d_grid(n_d(1)+n_d(2)+n_d(3)+1:n_d(1)+n_d(2)+n_d(3)+n_d(4));
        dPolicy_gridvals=[d1_grid(rem(Policy_d-1,n_d(1))+1),d2_grid(rem(ceil(Policy_d/n_d(1))-1,n_d(2))+1),d3_grid(rem(ceil(Policy_d/prod(n_d(1:2)))-1,n_d(3))+1),d4_grid(ceil(Policy_d/prod(n_d(1:3))))];
    end
end

if MatrixOrCell==2
    dPolicy_gridvals=num2cell(dPolicy_gridvals);
    aprimePolicy_gridvals=num2cell(aprimePolicy_gridvals);
end



end
