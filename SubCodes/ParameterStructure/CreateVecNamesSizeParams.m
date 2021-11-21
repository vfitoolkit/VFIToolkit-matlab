function Parameters=CreateVecNamesSizeParams(ParamsVec,ParamsNames,ParamsSize)
% Creates Parameters from ParamsVec,ParamsNames,ParamsSize
% You can do the reverse using CreateParamsVecNamesSize().
%
% ParamsVec is a column vector containing all the parameters in Params (if a parameter is a matrix, it gets reshaped)
% ParamsNames is a cell containing strings for the name of each parameter
% ParamsSize is a matrix, each row of which contains the size of the parameters (is only really needed so it is possible to reconstruct Parameters)

ii=1;
for jj=1:length(ParamsVec)
    currsize=ParamsSize(jj,:);
    currparam=ParamsVec(ii:(ii+prod(currsize)-1));
    Parameters.(ParamsNames{jj})=reshape(currparam,currsize);
    ii=ii+prod(currsize);
end


end

