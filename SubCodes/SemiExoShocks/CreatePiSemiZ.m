function pi_semiz=CreatePiSemiZ(n_d,n_semiz,d_grid,semiz_grid,SemiExoStateFn,SemiExoStateFnParams)
% d is only the decision variables relevant to the transitions of semiz

ParamCell=cell(length(SemiExoStateFnParams),1);
for ii=1:length(SemiExoStateFnParams)
    if ~all(size(SemiExoStateFnParams(ii))==[1,1])
        error('Using SemiExoStateFn does not allow for any of SemiExoStateFnParams to be anything but a scalar, problem with %i-th parameter',ii)
    end
    ParamCell(ii,1)={SemiExoStateFnParams(ii)};
end

N_semiz=prod(n_semiz);

l_d=length(n_d);
l_semiz=length(n_semiz);
if l_semiz>5
    error('Using SemiExoStateFn does not allow for more than five semi-exogenous state variables (you have vfoptions.n_semiz>5 (or simoptions))')
end
if length(n_d)>4
    error('Using SemiExoStateFn does not allow for more than four decision variables influencing the semi-exo state')
end
N_d=prod(n_d);

% joint semiz_grid is hardcoded
if l_semiz>=1
    semiz1vals=semiz_grid(:,1);
    semiz1primevals=shiftdim(semiz1vals,-1);
    if l_semiz>=2
        semiz2vals=semiz_grid(:,2);
        semiz2primevals=shiftdim(semiz2vals,-1);
        if l_semiz>=3
            semiz3vals=semiz_grid(:,3);
            semiz3primevals=shiftdim(semiz3vals,-1);
            if l_semiz>=4
                semiz4vals=semiz_grid(:,4);
                semiz4primevals=shiftdim(semiz4vals,-1);
                if l_semiz>=5
                    semiz5vals=semiz_grid(:,5);
                    semiz5primevals=shiftdim(semiz5vals,-1);
                end
            end
        end
    end
end

if l_d==1
    d1vals=shiftdim(d_grid,-2);
elseif l_d==2
    d1vals=shiftdim(repmat(d_grid(1:n_d(1)),n_d(2),1),-2);
    d2vals=shiftdim(repelem(d_grid(n_d(1)+1:n_d(1)+n_d(2)),n_d(1),1),-2);
elseif l_d==3
    d1vals=shiftdim(repmat(d_grid(1:n_d(1)),n_d(2)*n_d(3),1),-2);
    d2vals=shiftdim(repmat(repelem(d_grid(n_d(1)+1:n_d(1)+n_d(2)),n_d(1),1),n_d(3)),-2);
    d3vals=shiftdim(repelem(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),n_d(1)*n_d(2),1),-2);
elseif l_d==4
    d1vals=shiftdim(repmat(d_grid(1:n_d(1)),n_d(2)*n_d(3)*n_d(4),1),-2);
    d2vals=shiftdim(repmat(repelem(d_grid(n_d(1)+1:n_d(1)+n_d(2)),n_d(1),1),n_d(3)*n_d(4)),-2);
    d3vals=shiftdim(repmat(repelem(d_grid(n_d(1)+n_d(2)+1:n_d(1)+n_d(2)+n_d(3)),n_d(1)*n_d(2),1),n_d(4)),-2);
    d4vals=shiftdim(replem(d_grid(n_d(1)+n_d(2)+n_d(3)+1:n_d(1)+n_d(2)+n_d(3)+n_d(4)),n_d(1)*n_d(2)*n_d(3),1),-2);
end


% SemiExoStateFn(z,zprime,d,paremeters)
if l_d==1
    if l_semiz==1
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals, semiz1primevals, d1vals, ParamCell{:}); % Note: z1primevals is just z1vals
    elseif l_semiz==2
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals, semiz1primevals,semiz2primevals, d1vals, ParamCell{:});
    elseif l_semiz==3
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals, semiz1primevals,semiz2primevals,semiz3primevals, d1vals, ParamCell{:});
    elseif l_semiz==4
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals, d1vals, ParamCell{:});
    elseif l_semiz==5
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals,semiz5vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals,semiz5primevals, d1vals, ParamCell{:});
    end
elseif l_d==2 % Note, I am just going to create an N_d dimension
    if l_semiz==1
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals, semiz1primevals, d1vals,d2vals, ParamCell{:}); % Note: z1primevals is just z1vals
    elseif l_semiz==2
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals, semiz1primevals,semiz2primevals, d1vals,d2vals, ParamCell{:});
    elseif l_semiz==3
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals, semiz1primevals,semiz2primevals,semiz3primevals, d1vals,d2vals, ParamCell{:});
    elseif l_semiz==4
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals, d1vals,d2vals, ParamCell{:});
    elseif l_semiz==5
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals,semiz5vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals,semiz5primevals, d1vals,d2vals, ParamCell{:});
    end
elseif l_d==3 % Note, I am just going to create an N_d dimension
    if l_semiz==1
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals, semiz1primevals, d1vals,d2vals,d3vals, ParamCell{:}); % Note: z1primevals is just z1vals
    elseif l_semiz==2
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals, semiz1primevals,semiz2primevals, d1vals,d2vals,d3vals, ParamCell{:});
    elseif l_semiz==3
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals, semiz1primevals,semiz2primevals,semiz3primevals, d1vals,d2vals,d3vals, ParamCell{:});
    elseif l_semiz==4
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals, d1vals,d2vals,d3vals, ParamCell{:});
    elseif l_semiz==5
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals,semiz5vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals,semiz5primevals, d1vals,d2vals,d3vals, ParamCell{:});
    end
elseif l_d==4 % Note, I am just going to create an N_d dimension
    if l_semiz==1
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals, semiz1primevals, d1vals,d2vals,d3vals,d4vals, ParamCell{:}); % Note: z1primevals is just z1vals
    elseif l_semiz==2
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals, semiz1primevals,semiz2primevals, d1vals,d2vals,d3vals,d4vals, ParamCell{:});
    elseif l_semiz==3
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals, semiz1primevals,semiz2primevals,semiz3primevals, d1vals,d2vals,d3vals,d4vals, ParamCell{:});
    elseif l_semiz==4
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals, d1vals,d2vals,d3vals,d4vals, ParamCell{:});
    elseif l_semiz==5
        pi_semiz=arrayfun(SemiExoStateFn, semiz1vals,semiz2vals,semiz3vals,semiz4vals,semiz5vals, semiz1primevals,semiz2primevals,semiz3primevals,semiz4primevals,semiz5primevals, d1vals,d2vals,d3vals,d4vals, ParamCell{:});
    end
end
pi_semiz=reshape(pi_semiz,[N_semiz,N_semiz,N_d]); % I think this reshape is actually superflous, so can probably comment it out







end