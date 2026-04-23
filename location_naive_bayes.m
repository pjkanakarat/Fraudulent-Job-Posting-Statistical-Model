clc; clear;

data = readtable("fake_job_postings.csv");
isFraud = data.fraudulent;  
raw_location = string(data.location);
n = height(data);

country = strings(n,1);

for i = 1:n
    if raw_location(i) == "" || ismissing(raw_location(i))
        country(i) = "Unknown";
    else
        parts = split(raw_location(i), ",");
        country(i) = strtrim(parts(1));
    end
end

% locations
group = strings(n,1);

for i = 1:n
    if country(i) == "US"
        group(i) = "US";
    elseif country(i) == "GB"
        group(i) = "GB";
    elseif country(i) == "Unknown"
        group(i) = "Unknown";
    else
        group(i) = "Other";
    end
end

group = categorical(group, ["US","GB","Other","Unknown"]);

locCats = categories(group);
heat = zeros(2, numel(locCats));

for i = 1:2  
    for j = 1:numel(locCats)
        heat(i,j) = sum(isFraud == (i-1) & group == locCats{j});
    end
end

% convert to probabilities
heat = heat ./ sum(heat(:));

% plot
figure
heatmap(locCats, ["Not Fraud","Fraud"], heat)

title("P(Fraud vs Location Group)")
xlabel("Location Group")
ylabel("Class")