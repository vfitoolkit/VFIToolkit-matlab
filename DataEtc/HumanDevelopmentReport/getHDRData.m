function [output]=getHDRData(series_id, countrycode3L, observation_start, observation_end, structure)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Human Development Report Office database needs three pieces of info:
%
% series_id: the code for the variable you want (annoyingly they make these
%            unintelligble 5 or 6 digit numbers; see 'Dictionary' below; HDR call it indicator_id)
%
% countrycode3L: the country as a three-letter code
%
% observation_start: year, as a four-digit number in form of string (e.g., '1999-01-01', or '1991')
% observation_end: year, as a four-digit number in form of string (e.g., '2019-01-01' or '2019')
%
% structure (optional): by default ciy (Country-Indicator-Year). Can be set
%            to any of: ciy, yic, yci, iyc, icy
%
% To get a description of all the individual codes run: getHDRdata('Dictionary')
%
% If you do not input observation_start and observation_end you will be
% given data for all available dates. Equivalently, set
% observation_start=[], and similarly for observation_end=[].
%
% Note: All data is annual, allowing '1991-01-01' is just to make use the same as getFredData().
% Note: HDR allows requesting multiple countries or indicators in a single
%       call, but this is not supported here. (use a for-loop)
%
% I have deliberately made it so that the actual output is similar to that
% from getFredData(). Especially that the dates and data are in output.Data
%
% Some examples of usage can be found on my website, or on my github.
%
% 2022
% Robert Kirkby
% robertdkirkby.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

baseurl='http://ec2-54-174-131-205.compute-1.amazonaws.com/API/HDRO_API.php/';
if ~isstr(series_id)
    warning('series_id (the number of the indicator/variable) should be a string, not a number (has been converted internally)')
    series_id=num2str(series_id);
end
indicatorurl=['/indicator_id=',series_id];

    
if exist('structure','var')
    structureurl=['/structure=',structure];
else
    structureurl='/structure=ciy'; % Country, Indicator, Year (default)
end

if exist('countrycode3L','var')
    countryurl=['/country_code=',countrycode3L];
end

if exist('observation_start','var')
    if exist('observation_end','var')
        nyears=str2num(observation_end(1:4))-str2num(observation_start(1:4))+1;
        yearurl=['/year=',observation_start(1:4)];
        for ii=1:nyears-1
            yearurl=[yearurl,',',num2str(str2num(observation_start(1:4))+ii)];
        end
    else
        yearurl=['/year=',observation_start(1:4)];
        nyears=1;
    end 
end


%%
if strcmp(series_id,'Dictionary')
    fprintf('The following info is the Dictionary for the HDR (Human Development Report) database. \n')
    fprintf('It is taken from: http://ec2-54-174-131-205.compute-1.amazonaws.com/API/Information.php \n')
    fprintf('\n')
    fprintf('Years starts from 1990 \n')
    fprintf('Resource Selection Index for Indicators, Countries and Years (Updated: 15 December 2020) \n')
    fprintf(' \n')
    fprintf('ID 	Indicator Name \n')
    fprintf('164406	Adjusted net savings (% of GNI) \n');
    fprintf('36806	Adolescent birth rate (births per 1,000 women ages 15-19) \n');
    fprintf('185106	Age-standardized mortality rate attributed to noncommunicable diseases, female \n');
    fprintf('185206	Age-standardized mortality rate attributed to noncommunicable diseases, male \n');
    fprintf('175206	Antenatal care coverage, at least one visit (%) \n');
    fprintf('186806	Average annual change in the share of bottom 40 percent (%) \n');
    fprintf('181106	Birth registration (% under age 5) \n');
    fprintf('186706	Carbon dioxide emissions, per unit of GDP (kg per 2010 US$ of GDP) \n');
    fprintf('195606	Carbon dioxide emissions, production emissions per capita (tonnes) \n');
    fprintf('181306	Child labour (% ages 5-17) \n');
    fprintf('98306	Child malnutrition, stunting (moderate or severe) (% under age 5) \n');
    fprintf('181406	Child marriage, women married by age 18 (% of women ages 20?24 who are married or in union) \n');
    fprintf('135006	Coefficient of human inequality \n');
    fprintf('170006	Concentration index (exports) (value) \n');
    fprintf('175506	Contraceptive prevalence, any method (% of married or in-union women of reproductive age, 15?49 years) \n');
    fprintf('117806	Contribution of deprivation in education to the Multidimensional Poverty Index \n');
    fprintf('117906	Contribution of deprivation in health to the Multidimensional Poverty Index \n');
    fprintf('118006	Contribution of deprivation in standard of living to the Multidimensional Poverty Index \n');
    fprintf('181806	Current health expenditure (% of GDP) \n');
    fprintf('185006	Degraded land (% of total land area) \n');
    fprintf('195906	Domestic material consumption per capita, (tonnes) \n');
    fprintf('103706	Education index \n');
    fprintf('150606	Employment in agriculture (% of total employment) \n');
    fprintf('150706	Employment in services (% of total employment) \n');
    fprintf('148306	Employment to population ratio (% ages 15 and older) \n');
    fprintf('123506	Estimated gross national income per capita, female (2017 PPP $) \n');
    fprintf('123606	Estimated gross national income per capita, male (2017 PPP $) \n');
    fprintf('69706	Expected years of schooling (years) \n');
    fprintf('123306	Expected years of schooling, female (years) \n');
    fprintf('123406	Expected years of schooling, male (years) \n');
    fprintf('133206	Exports and imports (% of GDP) \n');
    fprintf('175006	Female share of employment in senior and middle management (%) \n');
    fprintf('53506	Foreign direct investment, net inflows (% of GDP) \n');
    fprintf('100806	Forest area (% of total land area) \n');
    fprintf('164206	Forest area, change (%) \n');
    fprintf('174306	Fossil fuel energy consumption (% of total energy consumption) \n');
    fprintf('97106	Fresh water withdrawals (% of total renewable water resources) \n');
    fprintf('194906	GDP per capita (2017 PPP $) \n');
    fprintf('137906	Gender Development Index (GDI) \n');
    fprintf('68606	Gender Inequality Index (GII) \n');
    fprintf('149206	Government expenditure on education (% of GDP) \n');
    fprintf('178306	Gross capital formation (% of GDP) \n');
    fprintf('194506	Gross domestic product (GDP), total (2017 PPP $ billions) \n');
    fprintf('133006	Gross enrolment ratio, pre-primary (% of preschool-age children) \n');
    fprintf('63206	Gross enrolment ratio, primary (% of primary school-age population) \n');
    fprintf('63306	Gross enrolment ratio, secondary (% of secondary school-age population) \n');
    fprintf('63406	Gross enrolment ratio, tertiary (% of tertiary school-age population) \n');
    fprintf('65606	Gross fixed capital formation (% of GDP) \n');
    fprintf('195706	Gross national income (GNI) per capita (constant 2017 PPP$) \n');
    fprintf('146206	HDI rank \n');
    fprintf('58006	HIV prevalence, adult (% ages 15-49) \n');
    fprintf('149406	Homeless people due to natural disaster (average annual per million people) \n');
    fprintf('61006	Homicide rate (per 100,000 people) \n');
    fprintf('137506	Human Development Index (HDI) \n');
    fprintf('136906	Human Development Index (HDI), female \n');
    fprintf('137006	Human Development Index (HDI), male \n');
    fprintf('103606	Income index \n');
    fprintf('67106	Income inequality, Gini coefficient \n');
    fprintf('135106	Income inequality, quintile ratio \n');
    fprintf('186906	Income share held by poorest 40% \n');
    fprintf('186106	Income share held by richest 1% \n');
    fprintf('187006	Income share held by richest 10 % \n');
    fprintf('101606	Inequality in education (%) \n');
    fprintf('101706	Inequality in income (%) \n');
    fprintf('101806	Inequality in life expectancy (%) \n');
    fprintf('71406	Inequality-adjusted education index \n');
    fprintf('138806	Inequality-adjusted HDI (IHDI) \n');
    fprintf('71606	Inequality-adjusted income index \n');
    fprintf('71506	Inequality-adjusted life expectancy index \n');
    fprintf('64406	Infants lacking immunization, DTP (% of one-year-olds) \n');
    fprintf('64306	Infants lacking immunization, measles (% of one-year-olds) \n');
    fprintf('111106	International inbound tourists (thousands) \n');
    fprintf('147206	International student mobility (% of total tertiary enrolment) \n');
    fprintf('178106	Internet users, female (% of female population) \n');
    fprintf('43606	Internet users, total (% of population) \n');
    fprintf('148206	Labour force participation rate (% ages 15 and older) \n');
    fprintf('48706	Labour force participation rate (% ages 15 and older), female \n');
    fprintf('48806	Labour force participation rate (% ages 15 and older), male \n');
    fprintf('183906	Labour share of GDP, comprising wages and social protection transfers (%) \n');
    fprintf('69206	Life expectancy at birth (years) \n');
    fprintf('120606	Life expectancy at birth, female (years) \n');
    fprintf('121106	Life expectancy at birth, male (years) \n');
    fprintf('103206	Life expectancy index \n');
    fprintf('101406	Literacy rate, adult (% ages 15 and older) \n');
    fprintf('182106	Malaria incidence (per 1,000 people at risk) \n');
    fprintf('128106	Mandatory paid maternity leave (days) \n');
    fprintf('89006	Maternal mortality ratio (deaths per 100,000 live births) \n');
    fprintf('103006	Mean years of schooling (years) \n');
    fprintf('24106	Mean years of schooling, female (years)    \n');
    fprintf('24206	Mean years of schooling, male (years) \n');
    fprintf('47906	Median age (years) \n');
    fprintf('46006	Mobile phone subscriptions (per 100 people) \n');
    fprintf('174506	Mortality rate attributed to household and ambient air pollution (per 100,000 population, age-standardized) \n');
    fprintf('174606	Mortality rate attributed to unsafe water, sanitation and hygiene services (per 100,000 population) \n');
    fprintf('57806	Mortality rate, female adult (per 1,000 people) \n');
    fprintf('57206	Mortality rate, infant (per 1,000 live births) \n');
    fprintf('57906	Mortality rate, male adult (per 1,000 people) \n');
    fprintf('57506	Mortality rate, under-five (per 1,000 live births) \n');
    fprintf('38406	Multidimensional poverty index (MPI) \n');
    fprintf('97306	Natural resource depletion (% of GNI) \n');
    fprintf('110806	Net migration rate (per 1,000 people) \n');
    fprintf('99106	Net official development assistance received (% of GNI) \n');
    fprintf('184306	Number of deaths and missing persons attributed to disasters (per 100,000 population) \n');
    fprintf('122006	Old-age (65 and older) dependency ratio (per 100 people ages 15-64) \n');
    fprintf('123806	Old-age pension recipients (% of statutory pension age population) \n');
    fprintf('73506	Overall loss in HDI due to inequality (%) \n');
    fprintf('183206	Overall loss in HDI value due to inequality, average annual change (%) \n');
    fprintf('185306	Percentage of primary schools with access to the internet \n');
    fprintf('185406	Percentage of secondary schools with access to the internet \n');
    fprintf('63106	Population ages 15-64 (millions) \n');
    fprintf('132706	Population ages 65 and older (millions) \n');
    fprintf('38606	Population in multidimensional poverty, headcount (%) \n');
    fprintf('102006	Population in multidimensional poverty, headcount (thousands) (for the year of the survey) \n');
    fprintf('183406	Population in multidimensional poverty, headcount (thousands) (projection for 2018) \n');
    fprintf('38506	Population in multidimensional poverty, intensity of deprivation (%) \n');
    fprintf('101006	Population in severe multidimensional poverty (%) \n');
    fprintf('39006	Population living below income poverty line, national poverty line (%) \n');
    fprintf('167106	Population living below income poverty line, PPP $1.90 a day (%) \n');
    fprintf('132806	Population under age 5 (millions) \n');
    fprintf('182806	Population using safely managed drinking-water services (%) \n');
    fprintf('195206	Population using safely managed sanitation services (%) \n');
    fprintf('142506	Population vulnerable to multidimensional poverty (%) \n');
    fprintf('23806	Population with at least some secondary education (% ages 25 and older) \n');
    fprintf('23906	Population with at least some secondary education, female (% ages 25 and older) \n');
    fprintf('24006	Population with at least some secondary education, male (% ages 25 and older) \n');
    fprintf('181506	Prevalence of female genital mutilation/cutting among girls and women (% of girls and women ages 15-49) \n');
    fprintf('46106	Primary school dropout rate (% of primary school cohort) \n');
    fprintf('45806	Primary school teachers trained to teach (%) \n');
    fprintf('128306	Prison population (per 100,000 people) \n');
    fprintf('111306	Private capital flows (% of GDP) \n');
    fprintf('177006	Programme for International Student Assessment (PISA) score in mathematics \n');
    fprintf('176906	Programme for International Student Assessment (PISA) score in reading \n');
    fprintf('176806	Programme for International Student Assessment (PISA) score in science \n');
    fprintf('181206	Proportion of births attended by skilled health personnel (%) \n');
    fprintf('184006	Proportion of informal employment in nonagricultural employment (% of total employment in nonagriculture) \n');
    fprintf('193006	Proportion of informal employment in nonagricultural employment, female (% of total employment in nonagriculture) \n');
    fprintf('46206	Pupil-teacher ratio, primary school (pupils per teacher) \n');
    fprintf('179706	Ratio of education and health expenditure to military expenditure \n');
    fprintf('181606	Red List Index (value) \n');
    fprintf('21806	Refugees by country of origin (thousands) \n');
    fprintf('52606	Remittances, inflows (% of GDP) \n');
    fprintf('52306	Research and development expenditure (% of GDP) \n');
    fprintf('181706	Rural population with access to electricity (%) \n');
    fprintf('49006	Sex ratio at birth (male to female births) \n');
    fprintf('175106	Share of employment in nonagriculture, female (% of total employment in nonagriculture) \n');
    fprintf('183506	Share of graduates from science, technology, engineering and mathematics programmes in tertiary education who are female (%) \n');
    fprintf('183706	Share of graduates from science, technology, engineering and mathematics programmes in tertiary education who are male (%) \n');
    fprintf('175906	Share of graduates in science, technology, engineering and mathematics programmes at tertiary level, female (%) \n');
    fprintf('180706	Share of graduates in science, technology, engineering and mathematics programmes at tertiary level, male (%) \n');
    fprintf('194306	Share of seats held by women in local government (%)  \n');
    fprintf('31706	Share of seats in parliament (% held by women) \n');
    fprintf('179406	Skilled labour force (% of labour force) \n');
    fprintf('112606	Suicide rate, female (per 100,000 people, age-standardized) \n');
    fprintf('112506	Suicide rate, male (per 100,000 people, age-standardized) \n');
    fprintf('177706	Survival rate to the last grade of lower secondary general education (%) \n');
    fprintf('174206	Total debt service (% of exports of goods, services and primary income) \n');
    fprintf('44206	Total population (millions) \n');
    fprintf('169706	Total unemployment rate (female to male ratio) \n');
    fprintf('182206	Tuberculosis incidence (per 100,000 people) \n');
    fprintf('140606	Unemployment, total (% of labour force) \n');
    fprintf('110906	Unemployment, youth (% ages 15-24) \n');
    fprintf('175606	Unmet need for family planning (% of married or in-union women of reproductive age, 15?49 years) \n');
    fprintf('45106	Urban population (%) \n');
    fprintf('196006	Use of fertilizer nutrient nitrogen (N), per area of cropland (kg per hectare) \n');
    fprintf('196106	Use of fertilizer nutrient phosphorus (expressed as P2O5), per area of cropland (kg per hectare) \n');
    fprintf('167406	Violence against women ever experienced, intimate partner (% of female population ages 15 and older) \n');
    fprintf('167506	Violence against women ever experienced, nonintimate partner (% of female population ages 15 and older) \n');
    fprintf('43006	Vulnerable employment (% of total employment) \n');
    fprintf('175706	Women with account at financial institution or with mobile money-service provider (% of female population ages 15 and older) \n');
    fprintf('153706	Working poor at PPP$3.20 a day (% of total employment) \n');
    fprintf('121206	Young age (0-14) dependency ratio (per 100 people ages 15-64) \n');
    fprintf('147906	Youth not in school or employment (% ages 15-24) \n');
    fprintf('169806	Youth unemployment rate (female to male ratio) \n');
    fprintf(' \n')
    fprintf('Countries \n')
    fprintf('Country Code 	Country Name \n')
    fprintf('AFG	Afghanistan \n');
    fprintf('ALB	Albania \n');
    fprintf('DZA	Algeria \n');
    fprintf('AND	Andorra \n');
    fprintf('AGO	Angola \n');
    fprintf('ATG	Antigua and Barbuda \n');
    fprintf('ARG	Argentina \n');
    fprintf('ARM	Armenia \n');
    fprintf('AUS	Australia \n');
    fprintf('AUT	Austria \n');
    fprintf('AZE	Azerbaijan \n');
    fprintf('BHS	Bahamas \n');
    fprintf('BHR	Bahrain \n');
    fprintf('BGD	Bangladesh \n');
    fprintf('BRB	Barbados \n');
    fprintf('BLR	Belarus \n');
    fprintf('BEL	Belgium \n');
    fprintf('BLZ	Belize \n');
    fprintf('BEN	Benin \n');
    fprintf('BTN	Bhutan \n');
    fprintf('BOL	Bolivia (Plurinational State of) \n');
    fprintf('BIH	Bosnia and Herzegovina \n');
    fprintf('BWA	Botswana \n');
    fprintf('BRA	Brazil \n');
    fprintf('BRN	Brunei Darussalam \n');
    fprintf('BGR	Bulgaria \n');
    fprintf('BFA	Burkina Faso \n');
    fprintf('BDI	Burundi \n');
    fprintf('CPV	Cabo Verde \n');
    fprintf('KHM	Cambodia \n');
    fprintf('CMR	Cameroon \n');
    fprintf('CAN	Canada \n');
    fprintf('CAF	Central African Republic \n');
    fprintf('TCD	Chad \n');
    fprintf('CHL	Chile \n');
    fprintf('CHN	China \n');
    fprintf('COL	Colombia \n');
    fprintf('COM	Comoros \n');
    fprintf('COG	Congo \n');
    fprintf('COD	Congo (Democratic Republic of the) \n');
    fprintf('CRI	Costa Rica \n');
    fprintf('CIV	Cote dIvoire \n');
    fprintf('HRV	Croatia \n');
    fprintf('CUB	Cuba \n');
    fprintf('CYP	Cyprus \n');
    fprintf('CZE	Czechia \n');
    fprintf('DNK	Denmark \n');
    fprintf('DJI	Djibouti \n');
    fprintf('DMA	Dominica \n');
    fprintf('DOM	Dominican Republic \n');
    fprintf('ECU	Ecuador \n');
    fprintf('EGY	Egypt \n');
    fprintf('SLV	El Salvador \n');
    fprintf('GNQ	Equatorial Guinea \n');
    fprintf('ERI	Eritrea \n');
    fprintf('EST	Estonia \n');
    fprintf('SWZ	Eswatini (Kingdom of) \n');
    fprintf('ETH	Ethiopia \n');
    fprintf('FJI	Fiji \n');
    fprintf('FIN	Finland \n');
    fprintf('FRA	France \n');
    fprintf('GAB	Gabon \n');
    fprintf('GMB	Gambia \n');
    fprintf('GEO	Georgia \n');
    fprintf('DEU	Germany \n');
    fprintf('GHA	Ghana \n');
    fprintf('GRC	Greece \n');
    fprintf('GRD	Grenada \n');
    fprintf('GTM	Guatemala \n');
    fprintf('GIN	Guinea \n');
    fprintf('GNB	Guinea-Bissau \n');
    fprintf('GUY	Guyana \n');
    fprintf('HTI	Haiti \n');
    fprintf('HND	Honduras \n');
    fprintf('HKG	Hong Kong, China (SAR) \n');
    fprintf('HUN	Hungary \n');
    fprintf('ISL	Iceland \n');
    fprintf('IND	India \n');
    fprintf('IDN	Indonesia \n');
    fprintf('IRN	Iran (Islamic Republic of) \n');
    fprintf('IRQ	Iraq \n');
    fprintf('IRL	Ireland \n');
    fprintf('ISR	Israel \n');
    fprintf('ITA	Italy \n');
    fprintf('JAM	Jamaica \n');
    fprintf('JPN	Japan \n');
    fprintf('JOR	Jordan \n');
    fprintf('KAZ	Kazakhstan \n');
    fprintf('KEN	Kenya \n');
    fprintf('KIR	Kiribati \n');
    fprintf('PRK	Korea (Democratic Peoples Rep. of) \n');
    fprintf('KOR	Korea (Republic of) \n');
    fprintf('KWT	Kuwait \n');
    fprintf('KGZ	Kyrgyzstan \n');
    fprintf('LAO	Lao Peoples Democratic Republic \n');
    fprintf('LVA	Latvia \n');
    fprintf('LBN	Lebanon \n');
    fprintf('LSO	Lesotho \n');
    fprintf('LBR	Liberia \n');
    fprintf('LBY	Libya \n');
    fprintf('LIE	Liechtenstein \n');
    fprintf('LTU	Lithuania \n');
    fprintf('LUX	Luxembourg \n');
    fprintf('MDG	Madagascar \n');
    fprintf('MWI	Malawi \n');
    fprintf('MYS	Malaysia \n');
    fprintf('MDV	Maldives \n');
    fprintf('MLI	Mali \n');
    fprintf('MLT	Malta \n');
    fprintf('MHL	Marshall Islands \n');
    fprintf('MRT	Mauritania \n');
    fprintf('MUS	Mauritius \n');
    fprintf('MEX	Mexico \n');
    fprintf('FSM	Micronesia (Federated States of) \n');
    fprintf('MDA	Moldova (Republic of) \n');
    fprintf('MCO	Monaco \n');
    fprintf('MNG	Mongolia \n');
    fprintf('MNE	Montenegro \n');
    fprintf('MAR	Morocco \n');
    fprintf('MOZ	Mozambique \n');
    fprintf('MMR	Myanmar \n');
    fprintf('NAM	Namibia \n');
    fprintf('NRU	Nauru \n');
    fprintf('NPL	Nepal \n');
    fprintf('NLD	Netherlands \n');
    fprintf('NZL	New Zealand \n');
    fprintf('NIC	Nicaragua \n');
    fprintf('NER	Niger \n');
    fprintf('NGA	Nigeria \n');
    fprintf('MKD	North Macedonia \n');
    fprintf('NOR	Norway \n');
    fprintf('OMN	Oman \n');
    fprintf('PAK	Pakistan \n');
    fprintf('PLW	Palau \n');
    fprintf('PSE	Palestine, State of \n');
    fprintf('PAN	Panama \n');
    fprintf('PNG	Papua New Guinea \n');
    fprintf('PRY	Paraguay \n');
    fprintf('PER	Peru \n');
    fprintf('PHL	Philippines \n');
    fprintf('POL	Poland \n');
    fprintf('PRT	Portugal \n');
    fprintf('QAT	Qatar \n');
    fprintf('ROU	Romania \n');
    fprintf('RUS	Russian Federation \n');
    fprintf('RWA	Rwanda \n');
    fprintf('KNA	Saint Kitts and Nevis \n');
    fprintf('LCA	Saint Lucia \n');
    fprintf('VCT	Saint Vincent and the Grenadines \n');
    fprintf('WSM	Samoa \n');
    fprintf('SMR	San Marino \n');
    fprintf('STP	Sao Tome and Principe \n');
    fprintf('SAU	Saudi Arabia \n');
    fprintf('SEN	Senegal \n');
    fprintf('SRB	Serbia \n');
    fprintf('SYC	Seychelles \n');
    fprintf('SLE	Sierra Leone \n');
    fprintf('SGP	Singapore \n');
    fprintf('SVK	Slovakia \n');
    fprintf('SVN	Slovenia \n');
    fprintf('SLB	Solomon Islands \n');
    fprintf('SOM	Somalia \n');
    fprintf('ZAF	South Africa \n');
    fprintf('SSD	South Sudan \n');
    fprintf('ESP	Spain \n');
    fprintf('LKA	Sri Lanka \n');
    fprintf('SDN	Sudan \n');
    fprintf('SUR	Suriname \n');
    fprintf('SWE	Sweden \n');
    fprintf('CHE	Switzerland \n');
    fprintf('SYR	Syrian Arab Republic \n');
    fprintf('TJK	Tajikistan \n');
    fprintf('TZA	Tanzania (United Republic of) \n');
    fprintf('THA	Thailand \n');
    fprintf('TLS	Timor-Leste \n');
    fprintf('TGO	Togo \n');
    fprintf('TON	Tonga \n');
    fprintf('TTO	Trinidad and Tobago \n');
    fprintf('TUN	Tunisia \n');
    fprintf('TUR	Turkey \n');
    fprintf('TKM	Turkmenistan \n');
    fprintf('TUV	Tuvalu \n');
    fprintf('UGA	Uganda \n');
    fprintf('UKR	Ukraine \n');
    fprintf('ARE	United Arab Emirates \n');
    fprintf('GBR	United Kingdom \n');
    fprintf('USA	United States \n');
    fprintf('URY	Uruguay \n');
    fprintf('UZB	Uzbekistan \n');
    fprintf('VUT	Vanuatu \n');
    fprintf('VEN	Venezuela (Bolivarian Republic of) \n');
    fprintf('VNM	Viet Nam \n');
    fprintf('YEM	Yemen \n');
    fprintf('ZMB	Zambia \n');
    fprintf('ZWE	Zimbabwe \n')
    output=[];
    return
end

%% Download the data

if nargin==1 % Note, already dealt with dictionary
    fullurl=[baseurl,indicatorurl,structureurl];
elseif nargin==2 
    fullurl=[baseurl,countryurl,indicatorurl,structureurl];
else
    fullurl=[baseurl,countryurl,indicatorurl,yearurl,structureurl];
end

JSONdata = webread(fullurl);

%% Get it ready for output
output.Data=nan(nyears,2);
for ii=1:nyears
    curryear=str2num(observation_start(1:4))+ii-1;
    % If you request a year with no data then the API just blanks you
    try
        output.Data(ii,2)=JSONdata.indicator_value.(countrycode3L).(['x',series_id]).(['x',num2str(curryear)]);
    catch
        output.Data(ii,2)=nan; 
    end
    % Dates, in matlab datetime format
    output.Data(ii,1)=datenum(curryear,1,1); % 1st January of that year
end
output.CountryCode3L=countrycode3L;
output.Country=JSONdata.country_name.(countrycode3L);
output.series_id=series_id;
output.description=JSONdata.indicator_name.(['x',series_id]);




end
