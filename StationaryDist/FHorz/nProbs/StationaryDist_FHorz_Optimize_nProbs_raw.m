function [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj, N_a1,N_a2,N_z_input,jj, epsilon,total_zeros_created,jj_at_max_a2)
 
epsilon_round=7;

age_zeros_created=total_zeros_created;

if N_z_input==0
    N_z=1;
else
    N_z=N_z_input;
end

for z_c=1:N_z
    if N_z_input==0
        StationaryDist_row_jj=reshape(StationaryDist_jj,[N_a1,N_a2]);
    else
        StationaryDist_row_jj=reshape(StationaryDist_jj(:,z_c),[N_a1,N_a2]);
    end
    row_prob_sum=full(sum(StationaryDist_row_jj,'all'));
    if row_prob_sum==0 || all(arrayfun(@(r) nnz(StationaryDist_row_jj(r,:)), 1:size(StationaryDist_row_jj,1))<3)
        % Sometimes nobody chooses the path less taken
        continue
    end
    if jj<jj_at_max_a2 && any(StationaryDist_row_jj(:,N_a2)~=0)
        jj_at_max_a2=jj;
    end

    [rows,~]=find(StationaryDist_row_jj~=0);
    for row=unique(rows')
        % Process agents' ExpAssets row by row (i.e., each N_a1 asset mixture)
        row_prob_sum=full(sum(StationaryDist_row_jj(row,:),2));

        % We have two strategies for dealing with gaps.  The first is
        % to see whether the gap disappears when we look at the whole
        % picture.  It can happen that lower/upper columns are
        % disjoint like this: [2 4] [3 5] which gives 4-in-a-row.
        % If we see we have [2 3 4 5] we have to reconstruct
        % lower/upper columns somehow.  In this case, it is possible
        % for a singleton upper to have a lower index than a lower,
        % so we take care to fix that.

        % The second strategy is to fill in a single hole if that
        % allows us to connect lower and upper columns.  In the above
        % case we get [2 3* 4] [3 4* 5] where * means inserted zero.
        % This is simpler because lower/upper are already split.
        [~,ea_all_idx,all_vals_jj]=find(StationaryDist_row_jj(row,:));
        p=find(diff(ea_all_idx)>1); % p columns are start and end of consecutive elements
        runs=[ea_all_idx(1),ea_all_idx(p+1);ea_all_idx(p),ea_all_idx(end)];

        if size(runs,2)>1
            single_gaps=ea_all_idx(p+1)-ea_all_idx(p)==2;
            if any(single_gaps)
                % A zero logically bubbled in...so make space for it for now
                s=p(single_gaps);
                z = zeros(1,length(ea_all_idx)+length(s));  %initialise a new vector of the appropriate size
                z(s+(1:length(s))) = ea_all_idx(s)+1; % set locations in 's' to s+1, which will have the value zero
                z(z==0) = ea_all_idx; %insert the original values in ea_all_idx into the new vector at their new positions.
                ea_all_idx=z;
                z = nan(1,length(all_vals_jj)+length(s));  %initialise a new vector of the appropriate size
                z(s+(1:length(s))) = 0; % set value locations in 'p' to zero
                z(isnan(z)) = all_vals_jj; %insert the original values in all_vals_jj into the new vector at their new positions.
                p=find(diff(ea_all_idx)>1); % p columns are start and end of consecutive elements
                runs=[ea_all_idx(1),ea_all_idx(p+1);ea_all_idx(p),ea_all_idx(end)];
                gap_idx=ea_all_idx(s)';
                single_gaps=sum(gap_idx>=runs(1,:) & gap_idx<=runs(2,:),1);
                all_vals_jj=z;
            else
                single_gaps=zeros(1,size(runs,2));
            end
            for ridx=1:size(runs,2)
                if runs(2,ridx)-runs(1,ridx)<2
                    % Remove traces of any short sequences
                    temp=ea_all_idx<runs(1,ridx) | ea_all_idx>runs(2,ridx);
                    ea_all_idx=ea_all_idx(temp);
                    all_vals_jj=all_vals_jj(temp);
                else
                    % Fill in zeros for everything we track
                    valid_ridx=ea_all_idx>=runs(1,ridx) & ea_all_idx<=runs(2,ridx);
                    if ~any(valid_ridx)
                        continue
                    end
                    ea_1=find(valid_ridx,1,'first');
                    ea_end=find(valid_ridx,1,'last');
                    if ~any(single_gaps(ridx))
                        % Check for single gaps intra lower/upper
                        if any(diff(ea_all_idx(ea_1:ea_end))==2)
                            single_gaps(ridx)=true;
                        end
                    end
                    if ea_end>ea_1 || any(single_gaps(ridx))
                        % Non-singleton, so maybe insert zeros; note ea_lower_idx is continuous, so ea_lower_idx(ea_lower_end)-ea_lower_idx(ea_lower_1)==ea_lower_end-ea_lower_1
                        new_vals=zeros(1,ea_end-ea_1+1); % pick up the new zeros
                        temp=ea_all_idx>=runs(1,ridx) & ea_all_idx<=runs(2,ridx);
                        new_vals(ea_all_idx(temp)-ea_all_idx(ea_1)+1)=all_vals_jj(temp);
                        all_vals_jj=[all_vals_jj(1:ea_1-1), new_vals, all_vals_jj(ea_end+1:end)];
                        ea_all_idx=[ea_all_idx(1:ea_1-1), ea_all_idx(ea_1):ea_all_idx(ea_end), ea_all_idx(ea_end+1:end)];
                    end
                end
            end
            runs=runs(:,runs(2,:)-runs(1,:)>1);
            if isempty(runs)
                % We have disqualified all merging opportunities
                continue
            else
                ea_gaps_isempty=size(runs,2)==1;
            end
        else
            if ea_all_idx(end)-ea_all_idx(1)>=length(ea_all_idx)
                % We have no gaps in the big picture, but lower gaps
                p=find(diff(ea_all_idx)>1); % p columns are start and end of consecutive elements
                runs=[ea_all_idx(1),ea_all_idx(p+1);ea_all_idx(p),ea_all_idx(end)];
                z = zeros(1,length(ea_all_idx)+length(p));  %initialise a new vector of the appropriate size
                z(p+(1:length(p))) = ea_all_idx(p)+1; % set locations in 'p' to p+1, which will have the value zero
                z(z==0) = ea_all_idx; %insert the original values in ea_lower_idx into the new vector at their new positions.
                ea_all_idx=z;
                z = nan(1,length(all_vals_jj)+length(p));  %initialise a new vector of the appropriate size
                z(p+(1:length(p))) = 0; % set value locations in 'p' to zero
                z(isnan(z)) = all_vals_jj; %insert the original values in ea_lower_vals into the new vector at their new positions.
                ea_lower_vals=z;
            end
            ea_gaps_isempty=true;
        end

        [ea_all_idx,sort_idx]=sort(ea_all_idx);
        all_vals_jj=all_vals_jj(sort_idx);
        if ea_gaps_isempty
            ea_gaps=[];
        else
            ea_gaps=find(diff(ea_all_idx)>1);
        end

        group_idx=[0,ea_gaps,length(ea_all_idx)];
        for ll=1:length(group_idx)-1
            all_idx=ea_all_idx(group_idx(ll)+1:group_idx(ll+1));
            vals=all_vals_jj(group_idx(ll)+1:group_idx(ll+1));

            multiplier=all_idx-all_idx(1)+1;
            assert(allunique(multiplier));

            % Attempt to consolidate min and max values to the middle
            starting_zeros=sum(vals==0);

            zero_created=false;
            if length(vals)>2
                cidx=length(vals);
                zero_candidate=zeros(1,cidx);
                zero_candidate(cidx)=1;
                while nnz(vals)>2
                    % Aggressively try to zero out largest indices
                    new_vals=linsolve([multiplier;ones(1,length(vals));zero_candidate],[sum(vals.*multiplier); row_prob_sum; 0])';
                    new_vals=round(new_vals,epsilon_round+abs(fix(log10(row_prob_sum))));
                    if all(new_vals==vals) || any(new_vals<0)
                        zero_candidate(cidx)=0;
                        break
                    end
                    vals=new_vals;
                    zero_created=true;
                    cidx=find(zero_candidate==0,1,'last');
                    zero_candidate(cidx)=1;
                end
                if zero_created
                    zero_candidate(cidx)=0;
                end
                cidx=1;
                zero_candidate(cidx)=1;
                while nnz(vals)>1
                    % Try to zero out least index
                    new_vals=linsolve([multiplier;ones(1,length(vals));zero_candidate],[sum(vals.*multiplier); row_prob_sum; 0])';
                    new_vals=round(new_vals,epsilon_round+abs(fix(log10(row_prob_sum))));
                    if all(new_vals==vals) || any(new_vals<0) || any(isnan(new_vals))
                        break
                    end
                    vals=new_vals;
                    zero_created=true;
                    cidx=find(zero_candidate==0,1,'first');
                    zero_candidate(cidx)=1;
                end
            end
            if ~zero_created
                % Just re-balance the indices (possibly creating a zero in the middle we cannot move to either end of vals
                new_vals=linsolve([multiplier;ones(1,length(vals))],[sum(vals.*multiplier); row_prob_sum])';
                new_vals=round(new_vals,epsilon_round);
                if any(new_vals<0)
                    break
                end
                vals=new_vals;
            end
            temp=sparse(row,all_idx,vals,N_a1,N_a2);
            temp_cols=all_idx(1):all_idx(end);
            if N_z_input==0
                StationaryDist_jj(sub2ind([N_a1,N_a2],row,temp_cols))=temp(row,temp_cols);
            else
                StationaryDist_jj(sub2ind([N_a1,N_a2,N_z],row,temp_cols,z_c))=temp(row,temp_cols);
            end
            total_zeros_created=total_zeros_created+sum(vals==0)-starting_zeros;
        end
    end
end

fprintf("Age %3d: zeros created = %d \n", jj, total_zeros_created-age_zeros_created);


end
