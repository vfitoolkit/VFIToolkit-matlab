function WGmatrix=CreateWarmGlowFnMatrix_Case1_Disc_Par2(WarmGlowFn, n_a, a_grid, WarmGlowFnParams)

ParamCell=cell(length(WarmGlowFnParams),1);
for ii=1:length(WarmGlowFnParams)
    if ~all(size(WarmGlowFnParams(ii))==[1,1])
        fprintf('ERROR: Using GPU for the warm glow of bequests fn does not allow for any of WarmGlowFnParams to be anything but a scalar, problem with %i-th parameter',ii)
    end
    ParamCell(ii,1)={WarmGlowFnParams(ii)};
end

N_a=prod(n_a);

l_a=length(n_a); 
if l_a>4
    error('ERROR: Using GPU for the warm-glow of bequests fn does not allow for more than four of a variable (you have length(n_a)>4)')
end

if nargin(WarmGlowFn)~=l_a+length(WarmGlowFnParams)
    fprintf('Next line is numbers relevant to the error \n')
    [nargin(WarmGlowFn),l_d,l_a,l_z,length(WarmGlowFnParams)]
    error('ERROR: Number of inputs to vfoptions.WarmGlowFn does not fit with size of WarmGlowFnParams')
end

if l_a>=1
    aprime1vals=a_grid(1:n_a(1));
    if l_a>=2
        aprime2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        if l_a>=3
            aprime3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
            if l_a>=4
                aprime4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
            end
        end
    end
end

if l_a==1
    WGmatrix=arrayfun(WarmGlowFn, aprime1vals, ParamCell{:});
elseif l_a==2
    WGmatrix=arrayfun(WarmGlowFn, aprime1vals,aprime2vals, ParamCell{:});
elseif l_a==3
    WGmatrix=arrayfun(WarmGlowFn, aprime1vals,aprime2vals,aprime3vals, ParamCell{:});
elseif l_a==4
    WGmatrix=arrayfun(WarmGlowFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, ParamCell{:});
end

WGmatrix=reshape(WGmatrix,[N_a,1]);

end


