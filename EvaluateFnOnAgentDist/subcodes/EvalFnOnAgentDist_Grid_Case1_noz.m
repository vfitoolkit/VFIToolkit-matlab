function Values=EvalFnOnAgentDist_Grid_Case1_noz(FnToEvaluate,FnToEvaluateParams,PolicyValuesPermute,n_d,n_a,a_grid,Parallel)

if Parallel~=2
    fprintf('Need to use EvalFnOnAgentDist_Grid_Case1_cpu instead of EvalFnOnAgentDist_Grid_Case1 \n')
    dbstack
    return
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
N_a=prod(n_a);

ParamCell=cell(length(FnToEvaluateParams),1);
for ii=1:length(FnToEvaluateParams)
    ParamCell(ii,1)={FnToEvaluateParams(ii)};
end

% if l_d>4
%     error
% end
% if l_a>4
%     error
% end
% if l_z>4
%     error
% end

if l_a>=1
    a1vals=a_grid(1:n_a(1));
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        if l_a>=3
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
            if l_a>=4
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
            end
        end
    end
end

if l_a==1
    if l_d>=1
        d1vals=PolicyValuesPermute(:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,l_d+4);
                end
            end
        end
    end
end
if l_a==2
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,l_d+4);
                end
            end
        end
    end
end
if l_a==3
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,l_d+4);
                end
            end
        end
    end
end
if l_a==4
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,l_d+4);
                end
            end
        end
    end
end
if l_a==5
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,l_d+4);
                end
            end
        end
    end
end
if l_a==6
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,1);
        if l_d>=2
            d2vals=PolicyValuesPermute(:,:,:,:,:,:,2);
            if l_d>=3
                d3vals=PolicyValuesPermute(:,:,:,:,:,:,3);
                if l_d>=4
                    d4vals=PolicyValuesPermute(:,:,:,:,:,:,4);
                end
            end
        end
    end
    if l_a>=1
        a1primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+1);
        if l_a>=2
            a2primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+2);
            if l_a>=3
                a3primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+3);
                if l_a>=4
                    a4primevals=PolicyValuesPermute(:,:,:,:,:,:,l_d+4);
                end
            end
        end
    end
end

if l_d==0 && l_a==1
    Values=arrayfun(FnToEvaluate, a1primevals, a1vals, ParamCell{:});
elseif l_d==0 && l_a==2
    Values=arrayfun(FnToEvaluate, a1primevals,a2primevals, a1vals,a2vals, ParamCell{:}); 
elseif l_d==0 && l_a==3
    Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==0 && l_a==4
    Values=arrayfun(FnToEvaluate, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==1 && l_a==1
    Values=arrayfun(FnToEvaluate, d1vals, a1primevals, a1vals, ParamCell{:});
elseif l_d==1 && l_a==2
    Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals, a1vals,a2vals, ParamCell{:});
elseif l_d==1 && l_a==3
    Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==1 && l_a==4
    Values=arrayfun(FnToEvaluate, d1vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==2 && l_a==1
    Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals, a1vals, ParamCell{:});
elseif l_d==2 && l_a==2
    Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, ParamCell{:});
elseif l_d==2 && l_a==3
    Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==2 && l_a==4
    Values=arrayfun(FnToEvaluate, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==3 && l_a==2
    Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, ParamCell{:});
elseif l_d==3 && l_a==3
    Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==3 && l_a==4
    Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==4 && l_a==2
    Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, ParamCell{:});
elseif l_d==4 && l_a==3
    Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==4 && l_a==4
    Values=arrayfun(FnToEvaluate, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
end

Values=reshape(Values,[N_a,1]);


end