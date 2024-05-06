function dprime = ES_from_ttest(d1,d2) 

M2 = mean(d2);
M1 = mean(d1);

SD1 = std(d1);
SD2 = std(d2);

SDpooled = sqrt((SD1.^2+SD2.^2)/2);

dprime = (M2-M1)/SDpooled;
