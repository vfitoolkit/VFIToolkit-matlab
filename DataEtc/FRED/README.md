getfreddata-matlab
==================

Matlab functions for directly importing data from FRED (Federal Reserve Economic Database)

Matlab has an inbuilt function for this, but this one allows for more control, 
letting you specify the frequency and 'transformation' (eg. level, log, percentage change on year ago).

Dates should be formatted as 'YYYY-MM-DD'. The functions returns dates in the standard matlab format 
(counting number of days since 0001-01-01, and matlab fns 'datevec', 'datenum', etc. can be used)

It also provides support for ALFRED data (Historical releases for US data, rather than just the final revised data)

The getOECDFredData.m provides an implementation to allow you to easily call the getFredData.m for all the 
OECD countries for a specific variable (eg. to get Real GDP for all the OECD countries for 1990:Q1-2000:Q4).
