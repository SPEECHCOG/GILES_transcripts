


set(0,'defaultaxesfontsize',18);
set(0,'defaulttextfontsize',18);

% Finals
input_path = '/Users/rasaneno/rundata/GILES_CogSci/analysis_out_1word_100/reports/';
modelname = 'GILES_CHILDES_agecond_final_1word_5layers_512emb_512ff_8head_8000voc_5do_500k_100t_';

refname = 'childes_age_0';
drawlinefits = 1;

separateplots = 0;

valid_ages = [6,9,12,15,18,21,24,36,48];

a = dir([input_path '/' modelname '*.features']);
b = dir([input_path '/' refname '*.features']);

% Parse ages of references data
b_age = zeros(length(b),1);
for j = 1:length(b)
    b_age(j) = str2num(b(j).name(end-10:end-9));
end

fnam = [input_path '/' a(1).name];
D = importdata(fnam);

DATA = zeros(size(D.data,1),size(D.data,2),length(a),2);
age = zeros(length(a),1);

for k = 1:length(a)
    fnam = [input_path '/' a(k).name];

    D = importdata(fnam);

    DATA(:,:,k,1) = D.data;
    % Age of current model data
    age(k) = str2num(a(k).name(end-10:end-9));

    % Find the corresponding reference data based on age
    i = find(b_age == age(k));
    fnam2 = [input_path '/' b(i).name];
    Dref = importdata(fnam2);

    % Find a mapping between feature columns of target and reference data
    target_column = zeros(length(D.colheaders),1);
    for j = 1:length(D.colheaders)
        target_feat = D.colheaders{j};
        tmp_i = find(strcmp(Dref.colheaders,target_feat));
        target_column(j) = tmp_i;
    end

    sprintf([fnam(end-20:end) '\t' fnam2(end-20:end)])
    
    DATA(:,:,k,2) = Dref.data(1:size(D.data,1),target_column);

end


% Remove age bins that are not of interest
toremove = [];
for k = 1:length(age)
if(~sum(valid_ages == age(k)))
    toremove = [toremove;k];
end
end

DATA(:,:,toremove,:) = [];
age(toremove) = [];
N_bins = length(age);



headers = D.colheaders;
featnames = cell(size(headers));

for k = 1:length(headers)
    s = headers{k};
    s = strrep(s,'_','\_');
    featnames{k} = s;
    featnames{k} = headers{k};
end

% Plot metrics

if(separateplots)

    r = zeros(length(featnames),3);
    rp = zeros(length(featnames),3);
    pfit = zeros(length(featnames),2,2);

    %for feat = 3
    for feat = 1:length(featnames)

        d1 = squeeze(DATA(:,feat,:,1));
        d2 = squeeze(DATA(:,feat,:,2));

        sig = zeros(size(d1,2),1);
        p = zeros(size(d1,2),1);
        tstat = cell(size(d1,2),1);
        dprime = zeros(size(d1,2),1);

        d_to_plot = zeros(size(d1,1),size(d1,2)*2);
        x = 1;
        for k = 1:size(d1,2)
            d_to_plot(:,x) = d1(:,k);
            d_to_plot(:,x+1) = d2(:,k);

            [sig(k),p(k),~,tstat{k}] = ttest(d1(:,k),d2(:,k));
            dprime(k) = ES_from_ttest(d1(:,k),d2(:,k));

            x = x+2;
        end

        fh = figure(2);clf;
        fh.Position = [2651 518 1223 506];
        h = violinplot(d_to_plot);

        for k = 1:2:size(DATA,3)*2-1
            h(k).ViolinColor = {[0 0.4470 0.7410]};
            h(k+1).ViolinColor = {[0.8500 0.3250 0.0980]};
            h(k).ScatterPlot.Visible = 0;
            h(k+1).ScatterPlot.Visible = 0;
            h(k).ViolinAlpha = {[0.45]};
            h(k+1).ViolinAlpha = {[0.45]};


        end
        set(gca,'XTick',[1.5:2:2*N_bins]);
        xt = xticks;
        set(gca,'XTickLabel',age);
        xlim([0 max(xt)+1.5]);
        xlabel('age (months)');
        ylabel('value');
        title(sprintf(featnames{feat}),'Interpreter','none');




        m1 = mean(d1);
        %a = polyfit(xt,m1,1);
        %pfit(feat,1,:) = polyfit(age,m1,1);
        %plot(xt,xt*a(1)+a(2),'--','Color','black','LineWidth',2);
        %a = polyfit(xt,m1,2);
        %pfit(feat,1,:) = polyfit(age,m1,2);
        %plot(xt,xt.^2*a(1)+xt.*a(2)+a(3),'--','Color','black','LineWidth',2);

        %m2 = mean(d2);
        %a2 = polyfit(xt,m2,1);
        %pfit(feat,2,:) = polyfit(age,m2,1);
        %plot(xt,xt*a2(1)+a2(2),'--','Color','black','LineWidth',2);

        corrtype = 'Pearson';

        [r(feat,1),rp(feat,1)] = corr(age,mean(d1)','type',corrtype);
        [r(feat,2),rp(feat,2)] = corr(age,mean(d2)','type',corrtype);
        [r(feat,3),rp(feat,3)] = corr(mean(d1)',mean(d2)','type',corrtype);

        lh = legend({'','model','','','','','','','','CHILDES'});
        drawnow;

        teekuvajpg(sprintf('GILES_CogSci_comparison_%s',featnames{feat}));

    end

end

% Joku tiivistelmätaulukko?

% Features of interest

%f_interest = {'sent_length','ttr','lm_perplexity','voc_rank_diff','pos_rate_NOUN','pos_rate_VERB','pos_rate_PRON','intj_sent_rate'};
f_interest = {'sent_length','ttr','lm_perplexity','voc_rank_diff','phrase_per_sent','pos_rate_NOUN','pos_rate_VERB','pos_rate_PRON','pos_rate_ADJ','pos_rate_INTJ'};

new_featnames = {'utterance length','type-to-token ratio','LM perplexity','lexical divergence','dep. per root','rate NOUN','rate VERB','rate PRON','rate ADJ','rate INTJ'};

fgh = figure(5);clf;
fgh.Position = [887 859 1313 478];



%for feat = 3
for iter = 1:length(f_interest)
    subplot(2,5,iter);

    feat = cellfind(headers,f_interest{iter});

    d1 = squeeze(DATA(:,feat,:,1));
    d2 = squeeze(DATA(:,feat,:,2));

    sig = zeros(size(d1,2),1);
    p = zeros(size(d1,2),1);
    tstat = cell(size(d1,2),1);
    dprime = zeros(size(d1,2),1);

    d_to_plot = zeros(size(d1,1),size(d1,2)*2);
    x = 1;
    for k = 1:size(d1,2)
        d_to_plot(:,x) = d1(:,k);
        d_to_plot(:,x+1) = d2(:,k);

        [sig(k),p(k),~,tstat{k}] = ttest(d1(:,k),d2(:,k));
        dprime(k) = ES_from_ttest(d1(:,k),d2(:,k));

        x = x+2;
    end

    %fh = figure(2);clf;
    %fh.Position = [2651 518 1223 506];
    h = violinplot(d_to_plot);

    for k = 1:2:size(DATA,3)*2-1
        h(k).ViolinColor = {[0 0.4470 0.7410]};
        h(k+1).ViolinColor = {[0.8500 0.3250 0.0980]};
        h(k).ScatterPlot.Visible = 0;
        h(k+1).ScatterPlot.Visible = 0;
        h(k).ViolinAlpha = {[0.75]};
        h(k+1).ViolinAlpha = {[0.75]};


    end
    set(gca,'XTick',[1.5:2:2*N_bins]);
        xt = xticks;
        set(gca,'XTickLabel',age);
        xlim([0 max(xt)+1.5]);

    if(iter > 5)
        xlabel('age (months)');
    end

    if(iter == 1 || iter == 6)
        ylabel('value');
    end
    %title(sprintf(featnames{feat}),'Interpreter','none');
    title(sprintf(new_featnames{iter}),'Interpreter','none');




    m1 = mean(d1);
    %a = polyfit(xt,m1,1);
    %pfit(feat,1,:) = polyfit(age,m1,1);
    %plot(xt,xt*a(1)+a(2),'--','Color','black','LineWidth',2);
    a = polyfit(xt,m1,2);
    %pfit(feat,1,:) = polyfit(age,m1,2);
    if(drawlinefits)
        plot(xt,xt.^2*a(1)+xt.*a(2)+a(3),'--','Color','blue','LineWidth',2);
    end

    m2 = mean(d2);
    %a2 = polyfit(xt,m2,1);
    %pfit(feat,2,:) = polyfit(age,m2,1);
    %plot(xt,xt*a2(1)+a2(2),'--','Color','black','LineWidth',2);
    a2 = polyfit(xt,m2,2);
    %pfit(feat,1,:) = polyfit(age,m1,2);
    if(drawlinefits)
        plot(xt,xt.^2*a2(1)+xt.*a2(2)+a2(3),'--','Color','red','LineWidth',2);
    end

    corrtype = 'Pearson';

    [r(feat,1),rp(feat,1)] = corr(age,mean(d1)','type',corrtype);
    [r(feat,2),rp(feat,2)] = corr(age,mean(d2)','type',corrtype);
    [r(feat,3),rp(feat,3)] = corr(mean(d1)',mean(d2)','type',corrtype);

    drawnow;
    %teekuvajpg(sprintf('GILES_CogSci_comparison_%s',featnames{feat}));
    
    if(iter == 1)
        lh = legend({'','model','','','','','','','','CHILDES'},'Location','NorthWest');
    end

end

% Create overlap figs

olap_path = '/Users/rasaneno/rundata/GILES_CogSci/overlaps_500k_1word.csv';

D = importdata(olap_path);

ages = D.data(:,1);
uttlen = D.data(:,2);
propuq = D.data(:,3);
count = D.data(:,4);

uq_ages = unique(ages);

P = zeros(length(uq_ages),max(uttlen));
C = zeros(length(uq_ages),max(uttlen));

for k = 1:length(ages)
    i = find(uq_ages == ages(k)); 

    P(i,uttlen(k)) = propuq(k);
    C(i,uttlen(k)) = count(k);
end

maxlen = 14;

P = P(:,1:maxlen);
C = C(:,1:maxlen);

% 

olap_path = '/Users/rasaneno/rundata/GILES_CogSci/overlaps_CHILDES_1word_fixed.csv';

D2 = importdata(olap_path);

ages2 = D2.data(:,1);
uttlen2 = D2.data(:,2);
propuq2 = D2.data(:,3);
count2 = D2.data(:,4);

uq_ages2 = unique(ages2);

P2 = zeros(length(uq_ages2),max(uttlen2));
C2 = zeros(length(uq_ages2),max(uttlen2));

for k = 1:length(ages2)
    i = find(uq_ages2 == ages2(k)); 

    P2(i,uttlen2(k)) = propuq2(k);
    C2(i,uttlen2(k)) = count2(k);
end

maxlen = 14;

P2 = P2(:,1:maxlen);
C2 = C2(:,1:maxlen);

%plot(uq_ages,mean(P),'LineWidth',2);

h = figure(7);clf;hold on;
h.Position = [2801 561 556 286];

plot(mean(P),'LineWidth',2);
xlabel('utterance length (words)');
ylabel('proportion new (%)')
grid;
xlim([0.5 maxlen+0.5])

ylim([0 105])
%line([0.5,maxlen+0.5],[100,100],'LineStyle','--','LineWidth',2,'Color','red');
hold on;
plot(nanmean(P2),'LineWidth',2);
drawstds(h,(1:maxlen)-0.1,nanmean(P),nanstd(P),0.25,1.5,'black');
drawstds(h,(1:maxlen)+0.1,nanmean(P2),nanstd(P2),0.25,1.5,'black');
legend({'model','CHILDES'},'Location','SouthEast')

teekuva('GILES_generatednew_wordforms');
teekuvajpg('GILES_generatednew_wordforms');







