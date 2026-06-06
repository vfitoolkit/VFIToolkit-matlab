function output = importCEX_interviewfile(filename)
% Get the current FMLI file (filename points to it)

% % If dataLines is not specified, define defaults
% dataLines = [2, Inf];

%% Set up some of the import options
% First, use row 1 to get the variable names

fid = fopen(filename,'r');
VariableNamesraw = fgetl(fid);

% First, figure out the names of the variables
VariableNames={};
currstart=1;
counter=0;
for tt=1:length(VariableNamesraw)
    if strcmp(VariableNamesraw(tt),',')
        counter=counter+1;
        VariableNames{counter}=VariableNamesraw(currstart:tt-1);
        currstart=tt+1;
    end
end
% And now the final one (which does not have a comma after it)
counter=counter+1;
VariableNames{counter}=VariableNamesraw(currstart:tt);

% Second, based on rows 2 to 11, figure out if they are double or categorical
VariableTypes={};
fullentries=cell(10,counter); % 10 will be way too few
rr=0; % row counter
while ~feof(fid)
    tline = fgetl(fid);
    rr=rr+1;

    % for rr=1:10
    currentrow=cell(1,counter); % Get first 10 values for each string
    tt=0; currstart=1;
    for vv=1:counter
        found=0;
        while found==0 && tt<length(tline) % Look for next comma (unless reach end of line)
            tt=tt+1;
            if strcmp(tline(tt),',')
                found=1;
            end
        end
        if tt-1<currstart % Two consequtive commas signifies missing value
            currentrow(vv)={'nan'};
        elseif vv==counter % Treat end of line specially
            currentrow(vv)={string(tline(currstart:end))};
        else
            currentrow(vv)={string(tline(currstart:tt-1))};
        end
        currstart=tt+1; % The one after the current comma
    end
    fullentries(rr,:)=currentrow;
end

% Some CEX years (2000+) wrap each header in double-quotes; strip them so
% downstream code can reference table columns by their bare names.
for vv=1:counter
    nm = VariableNames{vv};
    if numel(nm)>=2 && nm(1)=='"' && nm(end)=='"'
        VariableNames{vv} = nm(2:end-1);
    end
end

output=cell2table(fullentries); % Turn fullentries into a table
output.Properties.VariableNames = VariableNames; % Assign names to the columns of the table

for vv=1:counter
    try 
        temp = str2double(output.(vv)); % Try to convert the column to double
    catch

    end
    % Check if temp is just all nan, if so, leave it as string
    if all(isnan(temp))
        % Do nothing
    else
        output.(vv)=temp;
    end
end

% disp('Convert the numbers into numbers')
% % Where ever possible, convert to numbers
% for vv=1:counter
%     for rr2=1:rr
%         temp=output{rr,vv};
%         try 
%             temp=str2double(temp);
%         catch
%             % Just leave it as a string
%         end
%     end
%     output{rr,vv}=temp;
% end



% disp('fullentries \n')
% fullentries{1:10,1:3}
% disp('currentrow')
% currentrow
% disp('fullentries11 \n')
% fullentries{1,1}

% Now go through each variable, and either covert all to double or all to string

% 
% 
% % Now go through and based on the first ten values decide if it is double or string
% % WHY BOTHER. JUST GO OVER EVERY VALUE (ALREADY CREATED THEM ALL IN
% % CELL) AND TRY CONVERT NUMBER, WHEN FAIL, LEAVE AS CELL.
% for vv=1:counter
%     currfirst10=first10(:,vv);
%     appearstobedouble=0; 
%     appearstobecategorical=0;
%     numberofnan=0;
%     for rr=1:10
%         if strcmp(currfirst10{rr},'nan')
%             % Ignore it in terms of is categorical or double as there is a missing value
%             numberofnan=numberofnan+1;
%         else
%             try str2double(currfirst10(rr))
%                 appearstobedouble=appearstobedouble+1;
%             catch
%                 appearstobecategorical=appearstobecategorical+1;
%             end
%         end
%     end
%     if appearstobedouble>appearstobecategorical
%         if appearstobecategorical==0
%             % Is double
%             VariableTypes{vv}={"double"};
%         else
%             [appearstobedouble,appearstobecategorical,numberofnan]
%             warning(['Unable to categorize the %i th variable ',VariableNames{vv}],vv)
%         end
%     else
%         if appearstobedouble==0
%             % Is categorical
%             VariableTypes{vv}={"categorical"};
%         else
%             [appearstobedouble,appearstobecategorical,numberofnan]
%             warning(['Unable to categorize the %i th variable ',VariableNames{vv}],vv)
%         end
% 
%     end
% end
% 
% 
% fclose(fid);
% 
% VariableTypes(1:5)
% 
% %%
% opts = delimitedTextImportOptions("NumVariables", counter);
% 
% % Specify range and delimiter
% opts.DataLines = dataLines;
% opts.Delimiter = ",";
% 
% % Specify column names and types
% opts.VariableNames = VariableNames;
% % ["NEWID", "ACCESS", "ACCESS_", "AGE_REF", "AGE_REF_", "AGE2", "AGE2_", "AIR_TYPE", "AIR__YPE", "AIR_FUEL", "AIR__UEL", "AIRCOND", "AIRCOND_", "ALIMCSUP", "ALIM_SUP", "APARTMNT", "APAR_MNT", "APT_NOQ", "APT_NOQ_", "AS_COMP1", "AS_C_MP1", "AS_COMP2", "AS_C_MP2", "AS_COMP3", "AS_C_MP3", "AS_COMP4", "AS_C_MP4", "AS_COMP5", "AS_C_MP5", "BARN", "BARN_", "BASEMENT", "BASE_ENT", "BATHRMQ", "BATHRMQ_", "BEDROOMQ", "BEDR_OMQ", "BLS_URBN", "BOTTLED", "BOTTLED_", "BSINVSTX", "BSIN_STX", "BUILDING", "BUIL_ING", "BUILT", "BUILT_", "CBSGFTX", "CBSGFTX_", "CKBKACTX", "CKBK_CTX", "CLLGEQTR", "CLLG_QTR", "CNTEDORX", "CNTE_ORX", "CNTRCHRX", "CNTR_HRX", "CNTRELGX", "CNTR_LGX", "CNTRPOLX", "CNTR_OLX", "COALCOOK", "COAL_OOK", "COLLEXPX", "COLL_XPX", "COMKITCH", "COMK_TCH", "COMPBND", "COMPBND_", "COMPBNDX", "COMP_NDX", "COMPCKG", "COMPCKG_", "COMPCKGX", "COMP_KGX", "COMPLET1", "COMP_ET1", "COMPLET2", "COMP_ET2", "COMPENSX", "COMP_NSX", "COMPOWD", "COMPOWD_", "COMPOWDX", "COMP_WDX", "COMPSAV", "COMPSAV_", "COMPSAVX", "COMP_AVX", "COMPSEC", "COMPSEC_", "COMPSECX", "COMP_ECX", "CORBLOCK", "CORB_OCK", "CORBRICK", "CORB_ICK", "CORCONCR", "CORC_NCR", "CORDONT", "CORDONT_", "CORFRAME", "CORF_AME", "COROTHER", "CORO_HER", "CORSTONE", "CORS_ONE", "CSHCNTBX", "CSHC_TBX", "CUREMPL1", "CURE_PL1", "CUREMPL2", "CURE_PL2", "CUTENURE", "CUTE_URE", "DONTKNOW", "DONT_NOW", "EARNCOMP", "EARN_OMP", "EARNINCX", "EARN_NCX", "EDUC_REF", "EDUC0REF", "EDUCA2", "EDUCA2_", "ELECCOOK", "ELEC_OOK", "ELECTRIC", "ELEC_RIC", "ENCPORCH", "ENCP_RCH", "EXALUMIN", "EXAL_MIN", "EXASBEST", "EXAS_EST", "EXBLOCK", "EXBLOCK_", "EXBRICK", "EXBRICK_", "EXOTHER", "EXOTHER_", "EXSHINGL", "EXSH_NGL", "EXSIDING", "EXSI_ING", "EXSTONE", "EXSTONE_", "EXSTUCCO", "EXST_CCO", "FAM_SIZE", "FAM__IZE", "FAM_TYPE", "FAM__YPE", "FAMTFEDX", "FAMT_EDX", "FEDRFNDX", "FEDR_NDX", "FEDTAXX", "FEDTAXX_", "FFRMINCX", "FFRM_NCX", "FGOVRETX", "FGOV_ETX", "FINCATAX", "FINCAT_X", "FINCBTAX", "FINCBT_X", "FINDRETX", "FIND_ETX", "FININCX", "FININCX_", "FINLWT01", "FINLWT02", "FINLWT03", "FINLWT04", "FINLWT05", "FINLWT06", "FINLWT07", "FINLWT08", "FINLWT09", "FINLWT10", "FINLWT11", "FINLWT12", "FINLWT13", "FINLWT14", "FINLWT15", "FINLWT16", "FINLWT17", "FINLWT18", "FINLWT19", "FINLWT20", "FINLWT21", "FIREPLCQ", "FIRE_LCQ", "FJSSDEDX", "FJSS_EDX", "FNONFRMX", "FNON_RMX", "FORCEAIR", "FORC_AIR", "FOUNDATN", "FOUN_ATN", "FPRIPENX", "FPRI_ENX", "FRRDEDX", "FRRDEDX_", "FRRETIRX", "FRRE_IRX", "FSALARYX", "FSAL_RYX", "FSLTAXX", "FSLTAXX_", "FSSIX", "FSSIX_", "FUEL_OIL", "FUEL0OIL", "GAS", "GAS_", "GRAVTAIR", "GRAV_AIR", "GREENHSE", "GREE_HSE", "GUESTHSE", "GUES_HSE", "HALFSAMP", "HALF_AMP", "HEATFUEL", "HEAT_UEL", "HLFBATHQ", "HLFBA_THQ", "HH_CU_Q", "HH_CU_Q_", "HHID", "HHID_", "HTPUMPCT", "HTPU_PCT", "HTPUMPWL", "HTPU_PWL", "INC_HRS1", "INC__RS1", "INC_HRS2", "INC__RS2", "INC_RANK", "INC__ANK", "INC_RNKU", "INC__NKU", "INCCONTX", "INCC_NTX", "INCLASS", "INCLOSSA", "INCL_SSA", "INCLOSSB", "INCL_SSB", "INCNONW1", "INCN_NW1", "INCNONW2", "INCN_NW2", "INCOMEY1", "INCO_EY1", "INCOMEY2", "INCO_EY2", "INCSORC1", "INCS_RC1", "INCSORC2", "INCS_RC2", "INCSTAT1", "INCS_AT1", "INCSTAT2", "INCS_AT2", "INCWEEK1", "INCW_EK1", "INCWEEK2", "INCW_EK2", "INSRFNDX", "INSR_NDX", "INTEARNX", "INTE_RNX", "JFDSTMPA", "JFDS_MPA", "JOTAXNET", "JOTA_NET", "KEROSENE", "KERO_ENE", "LOT_SIZE", "LOT__IZE", "LUMPSUMX", "LUMP_UMX", "MARITAL1", "MARI_AL1", "MISCNTRX", "MISC_TRX", "MISCTAXX", "MISC_AXX", "MONYOWDX", "MONY_WDX", "NO_EARNR", "NO_E_RNR", "NO_EARNX", "NO_E_RNX", "NO_FUEL", "NO_FUEL_", "NO_HEAT", "NO_HEAT_", "NONINCMX", "NONI_CMX", "NUM_AUTO", "NUM__UTO", "OCCEXPNX", "OCCE_PNX", "OCCUPRE1", "OCCU_RE1", "OCCUPRE2", "OCCU_RE2", "ORIGIN1", "ORIGIN1_", "ORIGIN2", "ORIGIN2_", "OTH_COOK", "OTH__OOK", "OTHERHT", "OTHERHT_", "OTHRFNDX", "OTHR_NDX", "OTHRINCX", "OTHR_NCX", "OWNLIVE", "OWNLIVE_", "PARK_FAC", "PARK0FAC", "PARKINGQ", "PARK_NGQ", "PENSIONX", "PENS_ONX", "PERSLT18", "PERS_T18", "PERSOT64", "PERS_T64", "PLUM_FAC", "PLUM0FAC", "POPSIZE", "POCC_REF", "POCC0REF", "PRINEARN", "PRIN_ARN", "PROPVALX", "PROP_ALX", "PTAXRFDX", "PTAX_FDX", "PUBSEWER", "PUBS_WER", "PURSSECX", "PURS_ECX", "QINTRVMO", "QINTRVYR", "QCURRIN1", "QCUR_IN1", "QCURRIN2", "QCUR_IN2", "QCURROC2", "QCUR_OC2", "QPREVIN1", "QPRE_IN1", "QPREVIN2", "QPRE_IN2", "QPREVOC2", "QPRE_OC2", "RACE2", "RACE2_", "REF_RACE", "REF__ACE", "REGION", "RENTEQVX", "RENT_QVX", "RESPSTAT", "RESP_TAT", "RMWFLUE", "RMWFLUE_", "RMWOFLUE", "RMWO_LUE", "ROOMSQ", "ROOMSQ_", "SALEINCX", "SALE_NCX", "SAVACCTX", "SAVA_CTX", "SECESTX", "SECESTX_", "SELLSECX", "SELL_ECX", "SETLINSX", "SETL_NSX", "SEX_REF", "SEX_REF_", "SEX2", "SEX2_", "SLOCTAXX", "SLOC_AXX", "SLRFUNDX", "SLRF_NDX", "SMSASTAT", "SOLARCK", "SOLARCK_", "SOLARHT", "SOLARHT_", "SSOVERPX", "SSOV_RPX", "STATE", "ST_HOUS", "ST_HOUS_", "STEAMSYS", "STEA_SYS", "STORIES", "STORIES_", "SWIMPOOL", "SWIM_OOL", "TAXPROPX", "TAXP_OPX", "TENNISCT", "TENN_SCT", "TERRACE", "TERRACE_", "TOTTXPDX", "TOTT_PDX", "UNEMPLX", "UNEMPLX_", "USBNDX", "USBNDX_", "VEHQ", "VEHQ_", "WALLFURN", "WALL_URN", "WATER", "WATER_", "WATERHT", "WATERHT_", "WDBSASTX", "WDBS_STX", "WDBSGDSX", "WDBS_DSX", "WELFAREX", "WELF_REX", "WOODCOOK", "WOOD_OOK", "TOTEXPPQ", "TOTEXPCQ", "FOODPQ", "FOODCQ", "FDHOMEPQ", "FDHOMECQ", "FDAWAYPQ", "FDAWAYCQ", "ALCBEVPQ", "ALCBEVCQ", "HOUSPQ", "HOUSCQ", "SHELTPQ", "SHELTCQ", "OWNDWEPQ", "OWNDWECQ", "RENDWEPQ", "RENDWECQ", "OTHLODPQ", "OTHLODCQ", "UTILPQ", "UTILCQ", "HOUSOPPQ", "HOUSOPCQ", "HOUSEQPQ", "HOUSEQCQ", "APPARPQ", "APPARCQ", "TRANSPQ", "TRANSCQ", "VEHICLPQ", "VEHICLCQ", "GASMOPQ", "GASMOCQ", "OTHVEHPQ", "OTHVEHCQ", "PUBTRAPQ", "PUBTRACQ", "HEALTHPQ", "HEALTHCQ", "ENTERTPQ", "ENTERTCQ", "PERSCAPQ", "PERSCACQ", "READPQ", "READCQ", "EDUCAPQ", "EDUCACQ", "TOBACCPQ", "TOBACCCQ", "MISCPQ", "MISCCQ", "CASHCOPQ", "CASHCOCQ", "PERINSPQ", "PERINSCQ", "LIFINSPQ", "LIFINSCQ", "RETPENPQ", "RETPENCQ"];
% opts.VariableTypes = VariableTypes;
% % ["double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "string", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "string", "categorical", "string", "categorical", "string", "categorical", "double", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "double", "categorical", "string", "categorical", "string", "categorical", "double", "categorical", "string", "categorical", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
% 
% % Specify file level properties
% opts.ExtraColumnsRule = "ignore";
% opts.EmptyLineRule = "read";
% 
% % Specify variable properties
% opts = setvaropts(opts, ["COALCOOK", "DONTKNOW", "FUEL_OIL", "HTPUMPWL", "KEROSENE", "NO_HEAT", "OTH_COOK", "RENTEQVX", "RMWOFLUE", "SOLARCK", "SOLARHT", "SSOVERPX", "WDBSASTX", "WDBSGDSX", "WOODCOOK"], "WhitespaceRule", "preserve");
% opts = setvaropts(opts, ["ACCESS_", "AGE_REF_", "AGE2_", "AIR__YPE", "AIR__UEL", "AIRCOND_", "ALIM_SUP", "APAR_MNT", "APT_NOQ_", "AS_C_MP1", "AS_C_MP2", "AS_C_MP3", "AS_C_MP4", "AS_C_MP5", "BARN_", "BASE_ENT", "BATHRMQ_", "BEDR_OMQ", "BOTTLED_", "BSIN_STX", "BUIL_ING", "BUILT_", "CBSGFTX_", "CKBK_CTX", "CLLG_QTR", "CNTE_ORX", "CNTR_HRX", "CNTR_LGX", "CNTR_OLX", "COALCOOK", "COAL_OOK", "COLL_XPX", "COMK_TCH", "COMPBND_", "COMP_NDX", "COMPCKG_", "COMP_KGX", "COMP_ET1", "COMP_ET2", "COMP_NSX", "COMPOWD_", "COMP_WDX", "COMPSAV_", "COMP_AVX", "COMPSEC_", "COMP_ECX", "CORB_OCK", "CORB_ICK", "CORC_NCR", "CORDONT_", "CORF_AME", "CORO_HER", "CORS_ONE", "CSHC_TBX", "CURE_PL1", "CURE_PL2", "CUTE_URE", "DONTKNOW", "DONT_NOW", "EARN_OMP", "EARN_NCX", "EDUC0REF", "EDUCA2_", "ELEC_OOK", "ELEC_RIC", "ENCP_RCH", "EXAL_MIN", "EXAS_EST", "EXBLOCK_", "EXBRICK_", "EXOTHER_", "EXSH_NGL", "EXSI_ING", "EXSTONE_", "EXST_CCO", "FAM__IZE", "FAM__YPE", "FAMT_EDX", "FEDR_NDX", "FEDTAXX_", "FFRM_NCX", "FGOV_ETX", "FINCAT_X", "FINCBT_X", "FIND_ETX", "FININCX_", "FIRE_LCQ", "FJSS_EDX", "FNON_RMX", "FORC_AIR", "FOUN_ATN", "FPRI_ENX", "FRRDEDX_", "FRRE_IRX", "FSAL_RYX", "FSLTAXX_", "FSSIX_", "FUEL_OIL", "FUEL0OIL", "GAS_", "GRAV_AIR", "GREE_HSE", "GUES_HSE", "HALF_AMP", "HEAT_UEL", "HLFBA_THQ", "HH_CU_Q_", "HHID_", "HTPU_PCT", "HTPUMPWL", "HTPU_PWL", "INC__RS1", "INC__RS2", "INC__ANK", "INC__NKU", "INCC_NTX", "INCL_SSA", "INCL_SSB", "INCN_NW1", "INCN_NW2", "INCO_EY1", "INCO_EY2", "INCS_RC1", "INCS_RC2", "INCS_AT1", "INCS_AT2", "INCW_EK1", "INCW_EK2", "INSR_NDX", "INTE_RNX", "JFDS_MPA", "JOTA_NET", "KEROSENE", "KERO_ENE", "LOT__IZE", "LUMP_UMX", "MARI_AL1", "MISC_TRX", "MISC_AXX", "MONY_WDX", "NO_E_RNR", "NO_E_RNX", "NO_FUEL_", "NO_HEAT", "NO_HEAT_", "NONI_CMX", "NUM__UTO", "OCCE_PNX", "OCCU_RE1", "OCCU_RE2", "ORIGIN1_", "ORIGIN2_", "OTH_COOK", "OTH__OOK", "OTHERHT_", "OTHR_NDX", "OTHR_NCX", "OWNLIVE_", "PARK0FAC", "PARK_NGQ", "PENS_ONX", "PERS_T18", "PERS_T64", "PLUM0FAC", "POCC0REF", "PRIN_ARN", "PROP_ALX", "PTAX_FDX", "PUBS_WER", "PURS_ECX", "QCUR_IN1", "QCUR_IN2", "QCUR_OC2", "QPRE_IN1", "QPRE_IN2", "QPRE_OC2", "RACE2_", "REF__ACE", "RENTEQVX", "RENT_QVX", "RESP_TAT", "RMWFLUE_", "RMWOFLUE", "RMWO_LUE", "ROOMSQ_", "SALE_NCX", "SAVA_CTX", "SECESTX_", "SELL_ECX", "SETL_NSX", "SEX_REF_", "SEX2_", "SLOC_AXX", "SLRF_NDX", "SOLARCK", "SOLARCK_", "SOLARHT", "SOLARHT_", "SSOVERPX", "SSOV_RPX", "ST_HOUS_", "STEA_SYS", "STORIES_", "SWIM_OOL", "TAXP_OPX", "TENN_SCT", "TERRACE_", "TOTT_PDX", "UNEMPLX_", "USBNDX_", "VEHQ_", "WALL_URN", "WATER_", "WATERHT_", "WDBSASTX", "WDBS_STX", "WDBSGDSX", "WDBS_DSX", "WELF_REX", "WOODCOOK", "WOOD_OOK"], "EmptyFieldRule", "auto");
% 
% %% Import the data
% output = readtable(filename, opts);

end