function InvertedParetoCoeff=EvalFnOnAgentDist_InvertedParetoCoeff_Case1(SteadyStateDist, PolicyIndexes, FnsToEvaluate, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s, p_val)
disp('THIS CODE (EvalFnOnAgentDist_InvertedParetoCoeff_Case1) IS INCOMPLETE. DO NOT USE.')
dbstack
return

% Calculates the Inverted Pareto Coefficient.
% Parameter alpha on wikipedia (Pareto Type I)
% ( http://en.wikipedia.org/wiki/Pareto_distribution )
%
% From appendix to Chapter 10 of Piketty - Capital in 21st Century
% ( http://piketty.pse.ens.fr/files/capital21c/en/Piketty2014TechnicalAppendix.pdf)
% (Piketty refers to alpha as b; I have edited it to match wikipedia.)
%
% Pareto distribution is a particular form of distribution that follows this kind of mathematic law:
%   1-F(y)=(c/y)^a
% In which 1-F(y) is the share of the population whose income or wealth is higher than
% y, c is a constant and a the coefficient of the Pareto law. The idiosyncrasy of the Pareto
% law is that if we calculate the average income or wealth y* of all the people whose income 
% or wealth is higher than y, then the y*/y ratio is equal to a constant alpha. This coefficient, 
% called “inverted Pareto coefficient”, is simply linked to the coefficient:
%   alpha = a/(a-1)
% Intuitively, the higher is alpha, the thicker is the top of the distribution, and thus the
% stronger is the concentration of wealth. As a consequence, the coefficient alpha
% measures the distribution inequality (whereas the coefficient a varies in the opposite
% direction, and thus measures the distribution equality). On the mathematical link
% between a and alpha, and the way they can be empirically measured, see for instance,
% Les hauts revenus en France au 20 siècle..., 2001, Annexe B (It is in French)
% 
% To estimate alpha, we can use
% the maximum likelihood of alpha, given by
%    \widehat \alpha = n/(\sum _i (ln(x_i) - ln(\widehat x_m)))
% where \widehat x_m is the maximum likelihood of x_m, given by 
%    \widehat x_m = \min_i x_i
% where the i index is understood to run across the sample x = (x1, x2, ..., xn).
% The standard error of the estimate of alpha (which Piketty calls b) is
% given by
%    \sigma = (\widehat \alpha)/(sqrt(n))
% (From: http://en.wikipedia.org/wiki/Pareto_distribution#Parameter_estimation )
x_m=zeros(length(FnsToEvaluate),1);
alpha=zeros(length(FnsToEvaluate),1);

l_d=length(n_d);
l_a=length(n_a);
l_s=length(n_s);
N_a=prod(n_a);
N_s=prod(n_s);

AggVars=zeros(length(FnsToEvaluate),1);
LorenzCurve=zeros(length(FnsToEvaluate),100);
d_val=zeros(l_d,1);
aprime_val=zeros(l_a,1);
a_val=zeros(l_a,1);
s_val=zeros(l_s,1);
PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_s]);
SteadyStateDistVec=reshape(SteadyStateDist,[N_a*N_s,1]);
for i=1:length(FnsToEvaluate)
    Values=zeros(N_a,N_s);
    for j1=1:N_a
        a_ind=ind2sub_homemade([n_a],j1);
        for jj1=1:l_a
            if jj1==1
                a_val(jj1)=a_grid(a_ind(jj1));
            else
                a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
            end
        end
        for j2=1:N_s
            s_ind=ind2sub_homemade([n_s],j2);
            for jj2=1:l_s
                if jj2==1
                    s_val(jj2)=s_grid(s_ind(jj2));
                else
                    s_val(jj2)=s_grid(s_ind(jj2)+sum(n_s(1:jj2-1)));
                end
            end
            d_ind=PolicyIndexesKron(1:l_d,j1,j2);
            aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
            for kk=1:l_d
                if kk==1
                    d_val(kk)=d_grid(d_ind(kk));
                else
                    d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                end
            end
            for kk=1:l_a
                if kk==1
                    aprime_val(kk)=a_grid(aprime_ind(kk));
                else
                    aprime_val(kk)=a_grid(aprime_ind(kk)+sum(n_a(1:kk-1)));
                end
            end
            Values(j1,j2)=FnsToEvaluate{i}(d_val,aprime_val,a_val,s_val,pi_s,p_val);
        end
    end
    
    Values=reshape(Values,[N_a*N_s,1]);
    
    % Calculate what wikipedia calls x_m
    temp=Values.*(SteadyStateDistVec>0); % Consider only non-zero values
    x_m(i)=min(min(temp(temp>0))); % Take the smallest one which is 
    
    % Calculate the maximum likelihood estimate for alpha
    alpha(i)=
    
    WeightedValues=Values.*SteadyStateDistVec;
    AggVars(i)=sum(WeightedValues);
    
    
    [trash1,SortedValues_index] = sort(Values);

    SortedSteadyStateDistVec=SteadyStateDistVec(SortedValues_index);
    SortedWeightedValues=WeightedValues(SortedValues_index);
    
    CumSumSortedSteadyStateDistVec=cumsum(SortedSteadyStateDistVec);
    
    %We now want to use interpolation, but this won't work unless all
    %values in are CumSumSortedSteadyStateDist distinct. So we now remove
    %any duplicates (ie. points of zero probability mass/density). We then
    %have to remove the corresponding points of SortedValues
    [trash2,UniqueIndex] = unique(CumSumSortedSteadyStateDistVec,'first');
    CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec(sort(UniqueIndex));
    SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
    
    CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);

    InverseCDF_xgrid=0.01:0.01:1;
    
    InverseCDF_SSvalues=interp1(CumSumSortedSteadyStateDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
    % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
    % have already sorted and removed duplicates this will just be the last
    % point so we can just grab it directly.
    InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
    % interp1 may have similar problems at the bottom of the cdf
    j=1; %use j to figure how many points with this problem
    while InverseCDF_xgrid(j)<CumSumSortedSteadyStateDistVec_NoDuplicates(1)
        j=j+1;
    end
    for jj=1:j-1 %divide evenly through these states (they are all identical)
        InverseCDF_SSvalues(jj)=(jj/j)*InverseCDF_SSvalues(j);
    end

    
    LorenzCurve(i,:)=InverseCDF_SSvalues./AggVars(i);
end



end