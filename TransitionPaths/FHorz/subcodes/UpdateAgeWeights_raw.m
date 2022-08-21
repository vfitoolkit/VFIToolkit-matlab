function mewj=UpdateAgeWeights_raw(mewjlag,sj,n,immigrantmewj,immigrationrate,emmigrantmewj,emmigrationrate)
% Important note: n is NOT the population growth rate, it is the rate of growth of the population of age j=1
%
% immigrationrate is immigrants as a percentage of the population
% emmigrationrate is emmigrants as a percentage of the population
% immigrantmewj is (unit mass) of age weights of immigrant population
% emmigrantmewj is (unit mass) of age weights of emmigrant population

%% I commented out the following as it would require n to be growth rate of
% model population (ages j=1,...,J) and not the population as a whole.
% % n is the population growth rate. We need to turn this into the 
% % 'rate of growth of population of age j=1'
% % Note: n should be population growth rate of ages j=1,2,..J (which is
% % unlikely to calibrate to exact same as population growth rate of whole
% % population in empirical data)
% mewj1=(1+n)/sum(sj(1:end-1).*mewjlag(1:end-1));

%% n is the rate of growth of the population of age j=1
mewj1=(1+n)*mewjlag(1);

mewj=[mewj1;sj(1:end-1).*mewjlag(1:end-1)];
mewj=mewj./sum(mewj); % Normalize to one


%% Immigration and emmigration (if used)
if exist('immigrantmewj','var')
    if ~isnan(immigrantmewj)
        if sum(immigrantmewj)~=1
            fprintf('ERROR: Mass of immigrantmewj should be equal to one [sum(immigrantmewj)=1] \n')
            dbstack
            return
        end
        mewj=mewj+immigrationrate*immigrantmewj;
    end
end
if exist('emmigrantmewj','var')
    if ~isnan(emmigrantmewj)
        if sum(emmigrantmewj)~=1
            fprintf('ERROR: Mass of emmigrantmewj should be equal to one [sum(emmigrantmewj)=1] \n')
            dbstack
            return
        end
        mewj=mewj-emmigrationrate*emmigrantmewj;
    end
end

mewj=mewj./sum(mewj); % Normalize to one

end





