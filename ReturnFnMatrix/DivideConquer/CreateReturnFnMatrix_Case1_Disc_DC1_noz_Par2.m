function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_grid, aprime_grid, a_grid, ReturnFnParams,Level)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_a=length(a_grid); % Because l_a=1
N_aprime=length(aprime_grid); % Because l_a=1

l_d=length(n_d); % won't get here if l_d=0
l_a=1; % (or else won't get here)
if l_d>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end

if l_d>=1
    d1vals=d_grid(1:n_d(1)); 
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:sum(n_d(1:2))),-1);
        if l_d>=3
            d3vals=shiftdim(d_grid(sum(n_d(1:2))+1:sum(n_d(1:3))),-2);
            if l_d>=4
                d4vals=shiftdim(d_grid(sum(n_d(1:3))+1:sum(n_d(1:4))),-3);
            end
        end
    end
end

if l_d==1
%     d1vals(1,1,1)=d_grid(1); % Requires special treatment 
    Fmatrix=arrayfun(ReturnFn, d1vals, shiftdim(aprime_grid,-l_d), shiftdim(a_grid,-l_d-1), ParamCell{:});
elseif l_d==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, shiftdim(aprime_grid,-l_d), shiftdim(a_grid,-l_d-1), ParamCell{:});
elseif l_d==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, shiftdim(aprime_grid,-l_d), shiftdim(a_grid,-l_d-1), ParamCell{:});
elseif l_d==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, shiftdim(aprime_grid,-l_d), shiftdim(a_grid,-l_d-1), ParamCell{:});
end

if Level==1 % For level 1
    Fmatrix=reshape(Fmatrix,[N_d,N_aprime,N_a]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a]);
end

end


