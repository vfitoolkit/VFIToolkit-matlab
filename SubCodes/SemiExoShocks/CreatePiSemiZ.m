function pi_semiz=CreatePiSemiZ(n_d,n_semiz,d_grid,semiz_grid,SemiExoStateFn,SemiExoStateFnParams)

ParamCell=cell(length(SemiExoStateFnParams),1);
for ii=1:length(SemiExoStateFnParams)
    if ~all(size(SemiExoStateFnParams(ii))==[1,1])
        fprintf('ERROR: Using GPU for the return fn does not allow for any of SemiExoStateFnParams to be anything but a scalar, problem with %i-th parameter',ii)
    end
    ParamCell(ii,1)={SemiExoStateFnParams(ii)};
end

N_semiz=prod(n_semiz);

l_d=length(n_d);
l_semiz=length(n_semiz);
if l_semiz>5
    error('ERROR: Using GPU for the return fn does not allow for more than five semi-exogenous state variables (you have vfoptions.semiexostates>5 (or simoptions))')
end
if length(n_d)>2
    error('Do not currently allow more than two decision variables that influence semi-exo state (contact me if you need/want more; vfoptions.vfoptions.numd_semiz is greater than 3)')
end
% Note: l_d is hardcoded to one during setup of semi-exogenous state
% problems (only the relevant decision variable for the semi-exogenous
% states is passed to CreatePiSemiZ()).
N_d=prod(n_d);
% dvals=d_grid;

if all(size(semiz_grid)==[sum(n_semiz),1]) % kroneker product semiz_grid
    if l_semiz>=1
        z1vals=semiz_grid(1:n_semiz(1));
        z1primevals=shiftdim(z1vals,-l_semiz);
        if l_semiz>=2
            z2vals=shiftdim(semiz_grid(n_semiz(1)+1:sum(n_semiz(1:2))),-1);
            z2primevals=shiftdim(z2vals,-l_semiz);
            if l_semiz>=3
                z3vals=shiftdim(semiz_grid(sum(n_semiz(1:2))+1:sum(n_semiz(1:3))),-2);
                z3primevals=shiftdim(z3vals,-l_semiz);
                if l_semiz>=4
                    z4vals=shiftdim(semiz_grid(sum(n_semiz(1:3))+1:sum(n_semiz(1:4))),-3);
                    z4primevals=shiftdim(z4vals,-l_semiz);
                    if l_semiz>=5
                        z5vals=shiftdim(semiz_grid(sum(n_semiz(1:4))+1:sum(n_semiz(1:5))),-4);
                        z5primevals=shiftdim(z5vals,-l_semiz);
                    end
                end
            end
        end
    end
    if l_d==1
        d1vals=shiftdim(d_grid,-l_semiz-l_semiz);
    elseif l_d==2
        d1vals=shiftdim(kron(ones(n_d(2),1),d_grid(1:n_d(1))),-l_semiz-l_semiz); % Note, I am just going to create an N_d dimension
        d2vals=shiftdim(kron(d_grid(n_d(1)+1:end),ones(n_d(1),1)),-l_semiz-l_semiz);
    end
elseif all(size(semiz_grid)==[prod(n_semiz),l_semiz]) % joint semiz_grid
    if l_semiz>=1
        z1vals=semiz_grid(:,1);
        z1primevals=shiftdim(z1vals,-1);
        if l_semiz>=2
            z2vals=semiz_grid(:,2);
            z2primevals=shiftdim(z2vals,-1);
            if l_semiz>=3
                z3vals=semiz_grid(:,3);
                z3primevals=shiftdim(z3vals,-1);
                if l_semiz>=4
                    z4vals=semiz_grid(:,4);
                    z4primevals=shiftdim(z4vals,-1);
                    if l_semiz>=5
                        z5vals=semiz_grid(:,5);
                        z5primevals=shiftdim(z5vals,-1);
                    end
                end
            end
        end
    end
    if l_d==1
        d1vals=shiftdim(kron(ones(n_d(2),1),d_grid(1:n_d(1))),-2);
    elseif l_d==2
        d2vals=shiftdim(kron(d_grid(n_d(1)+1:end),ones(n_d(1),1)),-2);
    end
end


% SemiExoStateFn(z,zprime,d,paremeters)
if l_d==1
    if l_semiz==1
        pi_semiz=arrayfun(SemiExoStateFn, z1vals, z1primevals, d1vals, ParamCell{:}); % Note: z1primevals is just z1vals
    elseif l_semiz==2
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals, z1primevals,z2primevals, d1vals, ParamCell{:});
    elseif l_semiz==3
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals,z3vals, z1primevals,z2primevals,z3primevals, d1vals, ParamCell{:});
    elseif l_semiz==4
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals,z3vals,z4vals, z1primevals,z2primevals,z3primevals,z4primevals, d1vals, ParamCell{:});
    elseif l_semiz==5
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals,z3vals,z4vals,z5vals, z1primevals,z2primevals,z3primevals,z4primevals,z5primevals, d1vals, ParamCell{:});
    end
elseif l_d==2 % Note, I am just going to create an N_d dimension
    if l_semiz==1
        pi_semiz=arrayfun(SemiExoStateFn, z1vals, z1primevals, d1vals,d2vals, ParamCell{:}); % Note: z1primevals is just z1vals
    elseif l_semiz==2
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals, z1primevals,z2primevals, d1vals,d2vals, ParamCell{:});
    elseif l_semiz==3
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals,z3vals, z1primevals,z2primevals,z3primevals, d1vals,d2vals, ParamCell{:});
    elseif l_semiz==4
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals,z3vals,z4vals, z1primevals,z2primevals,z3primevals,z4primevals, d1vals,d2vals, ParamCell{:});
    elseif l_semiz==5
        pi_semiz=arrayfun(SemiExoStateFn, z1vals,z2vals,z3vals,z4vals,z5vals, z1primevals,z2primevals,z3primevals,z4primevals,z5primevals, d1vals,d2vals, ParamCell{:});
    end
end
pi_semiz=reshape(pi_semiz,[N_semiz,N_semiz,N_d]);







end