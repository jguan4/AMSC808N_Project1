function [A,str,label,num,idem,igop] = read_data(toggle)
%% read data
A2012 = readmatrix('A2012.csv');
A2016 = readmatrix('A2016.csv');
% Format for A2012 and A2016:
% FIPS, County, #DEM, #GOP, then <str> up to Unemployment Rate
str = ["Median Income", "Migration Rate", "Birth Rate",...
    "Death Rate", "Bachelor Rate", "Unemployment Rate","log(#Votes)"];
%
% remove column county that is read by matlab as NaN
A2012(:,2) = [];
A2016(:,2) = [];
%% Remove rows with missing data
A = A2016;
% remove all rows with missing data
ind = find(~isfinite(A(:,2)) |  ~isfinite(A(:,3)) | ~isfinite(A(:,4)) ...
    | ~isfinite(A(:,5)) | ~isfinite(A(:,6)) | ~isfinite(A(:,7)) ...
    | ~isfinite(A(:,8)) | ~isfinite(A(:,9)));
A(ind,:) = [];

%% select CA, OR, WA, NJ, NY counties
switch toggle
    case 'CA'
        ind = find((A(:,1)>=6000 & A(:,1)<=6999)); % ...  %CA
        A = A(ind,:);
    case 'all_counties'
        ind = find((A(:,1)>=6000 & A(:,1)<=6999) ...  %CA
            | (A(:,1)>=53000 & A(:,1)<=53999) ...        %WA
            | (A(:,1)>=34000 & A(:,1)<=34999) ...        %NJ
            | (A(:,1)>=36000 & A(:,1)<=36999) ...        %NY
            | (A(:,1)>=41000 & A(:,1)<=41999));          %OR
        A = A(ind,:);
end
[n,dim] = size(A);

%% assign labels: -1 = dem, 1 = GOP
idem = find(A(:,2) >= A(:,3));
igop = find(A(:,2) < A(:,3));
num = A(:,2)+A(:,3);
label = zeros(n,1);
label(idem) = -1;
label(igop) = 1;

if strcmp(toggle,'CA') || strcmp(toggle,'all_counties')
    return
else
    %% select max subset of data with equal numbers of dem and gop counties
    ngop = length(igop);
    ndem = length(idem);
    if ngop > ndem
        rgop = randperm(ngop,ndem);
        Adem = A(idem,:);
        Agop = A(igop(rgop),:);
        A = [Adem;Agop];
    else
        rdem = randperm(ndem,ngop);
        Agop = A(igop,:);
        Adem = A(idem(rdem),:);
        A = [Adem;Agop];
    end
    [n,dim] = size(A)
    idem = find(A(:,2) >= A(:,3));
    igop = find(A(:,2) < A(:,3));
    num = A(:,2)+A(:,3);
    label = zeros(n,1);
    label(idem) = -1;
    label(igop) = 1;
    return
end
end