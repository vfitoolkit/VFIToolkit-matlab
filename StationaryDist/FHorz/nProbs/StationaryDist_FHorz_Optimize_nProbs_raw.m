function [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj, N_a1_input,N_a2_input,N_z_input,N_e,jj, epsilon,total_zeros_created,jj_at_max_a2, simoptions)

epsilon_round=7;

new_zeros_created=zeros(1,N_z_input);

if N_a2_input==0
    N_a2=N_a1_input;
    N_a1=1;
else
    N_a1=N_a1_input;
    N_a2=N_a2_input;
end

if N_z_input==0
    N_z=1;
else
    N_z=N_z_input;
end

assert(N_e==0); % haven't implemented N_e of any kind yet

% For large N_z, this loop can be changed to `parfor` for greater CPU parallelism
for z_c=1:N_z
    % When N_z=1, the index z_c is only every 1, which does nothing
    StationaryDist_row_jj=reshape(StationaryDist_jj(:,z_c),[N_a1,N_a2]);

    row_prob_sum=full(sum(StationaryDist_row_jj,'all'));
    if row_prob_sum==0 || all(arrayfun(@(r) nnz(StationaryDist_row_jj(r,:)), 1:size(StationaryDist_row_jj,1))<3)
        % Sometimes nobody chooses the path less taken
        continue
    end

    [rows,~]=find(StationaryDist_row_jj~=0);
    for row=unique(rows')
        % Process agents' ExpAssets row by row (i.e., each N_a1 asset mixture)

        % Find and join up runs that are reasonably close together
        [~,ea_all_idx,all_vals_jj]=find(StationaryDist_row_jj(row,:));
        p=find(diff(ea_all_idx)>2); % p columns are start and end of mostly consecutive elements
        runs=[ea_all_idx(1),ea_all_idx(p+1);ea_all_idx(p),ea_all_idx(end)];

        if size(runs,2)>1
            ea_1=1;
            for ridx=1:size(runs,2)
                if runs(2,ridx)-runs(1,ridx)<2
                    % Remove traces of any short sequences
                    temp=ea_all_idx<runs(1,ridx) | ea_all_idx>runs(2,ridx);
                    ea_all_idx=ea_all_idx(temp);
                    all_vals_jj=all_vals_jj(temp);
                    % ea_1 doesn't move
                    continue
                end
                % Fill in zeros for everything we track
                valid_ridx=ea_all_idx>=runs(1,ridx) & ea_all_idx<=runs(2,ridx);
                if sum(valid_ridx)==runs(2,ridx)-runs(1,ridx)+1
                    % We have a full run with no gaps to fill
                    ea_1=ea_1+runs(2,ridx)-runs(1,ridx)+1;
                    continue
                end
                ea_end_plus1=ea_1+length(all_vals_jj(valid_ridx));
                new_idx=runs(1,ridx):runs(2,ridx);
                new_ridx=ismember(new_idx, ea_all_idx(valid_ridx));
                new_vals=zeros(1,length(new_idx));
                new_vals(new_ridx)=all_vals_jj(valid_ridx);
                all_vals_jj=[all_vals_jj(1:ea_1-1), new_vals, all_vals_jj(ea_end_plus1:end)];
                ea_all_idx=[ea_all_idx(1:ea_1-1), new_idx, ea_all_idx(ea_end_plus1:end)];
                ea_1=ea_1+length(new_idx);
            end
            % Remove runs ignored above (but used to cut down idx and vals)
            runs=runs(:,runs(2,:)-runs(1,:)>1);
            if isempty(runs)
                % We have disqualified all merging opportunities
                continue
            end
        end

        [ea_all_idx,sort_idx]=sort(ea_all_idx);
        all_vals_jj=all_vals_jj(sort_idx);
        ea_gaps=find(diff(ea_all_idx)>1);

        group_idx=[0,ea_gaps,length(ea_all_idx)];
        for ll=1:length(group_idx)-1
            all_idx=ea_all_idx(group_idx(ll)+1:group_idx(ll+1));
            vals=all_vals_jj(group_idx(ll)+1:group_idx(ll+1));
            run_prob_sum=sum(vals);

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
                    new_vals=linsolve([multiplier;ones(1,length(vals));zero_candidate],[sum(vals.*multiplier); run_prob_sum; 0])';
                    new_vals=round(new_vals,epsilon_round+abs(fix(log10(run_prob_sum))));
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
                    new_vals=linsolve([multiplier;ones(1,length(vals));zero_candidate],[sum(vals.*multiplier); run_prob_sum; 0])';
                    new_vals=round(new_vals,epsilon_round+abs(fix(log10(run_prob_sum))));
                    if all(new_vals==vals) || any(new_vals<0) || any(isnan(new_vals))
                        break
                    end
                    vals=new_vals;
                    cidx=find(zero_candidate==0,1,'first');
                    zero_candidate(cidx)=1;
                end
            end
            temp=sparse(row,all_idx,vals,N_a1,N_a2);
            temp_cols=all_idx(1):all_idx(end);
            StationaryDist_row_jj(sub2ind([N_a1,N_a2],row,temp_cols))=temp(row,temp_cols);
            new_zeros_created(z_c)=new_zeros_created(z_c)+sum(vals==0)-starting_zeros;
        end
    end
    StationaryDist_jj(:,z_c)=reshape(StationaryDist_row_jj,[N_a1*N_a2,1]);
end

temp=reshape(full(StationaryDist_jj),[N_a1,N_a2,N_z]);
if jj<jj_at_max_a2 && any(temp(:,N_a2,:)~=0,'all')
    jj_at_max_a2=jj;
end

sum_new_zeros=sum(new_zeros_created);
total_zeros_created=total_zeros_created+sum_new_zeros;
if simoptions.verbose>=1
    if sum_new_zeros || simoptions.verbose==2
        fprintf("Age %3d: zeros created = %d \n", jj, sum_new_zeros);
    end
end


end
