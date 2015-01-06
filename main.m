%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MMID Projekt
% VUT, FEKT, KAM
% Autoøi: DAVÍDEK D.
% Zadání:
% % Proveïte  identifikaci  zadaného  systému  pomocí  jedné  z  metod  pomocných  promìnných, 
% % která  zajistí  nevychýlený  odhad  neznámých  parametrù  (volbu  metody  zdùvodnìte).  Zvolte 
% % vhodnou periodu vzorkování a vstupní signál k identifikaci. Výsledky ovìøte pomocí Matlab 
% % Identification  Toolbox  (ident).  Výstup  z  neznámého  systému  získáte  z  pøiložené  funkce 
% % odezva.p. 
% Popis: 
% % Výsledkem  identifikace  by  mìl  být  diskrétní  model  deterministické  èásti  systému.  Pøi 
% % zpracování dejte pozor na to, že 
% % ?  systém je nelineární (maximální vstupní signál musí být omezen) 
% % ?  mùže být porouchané èidlo mìøení výstupního signálu (napø. segmentace dat) 
% % ?  systém mùže obsahovat dopravní zpoždìní a stejnosmìrnou složku 
% % ?  mìøení  je  drahé,  proto  v  hotovém  projektu  mùžete  volat  funkci  odezva.p  jen 
% % jednou,  množství  jednorázovì  získaných  dat  není  omezeno,  každé  další  volání  je 
% % penalizováno jedním bodem (pro vaše ladìní si ji puste kolikrát chcete). 
% Hlavní skript - volá všechny ostatní funkce
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;clc;clear all

% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
% vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% odkomentovat nebo zakomentovat pro zobrazení postupu výpoètu parametrù
% ident_param = 0; %parametry použije (již vypoètené - pro debugging)
ident_param = 1; %identifikuje parametry

% dále výbìr metody zmìnou promìnné [rms_type] viz níže (kekonci)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
% ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% init
id = 136510;
getResponse = @(u,t) (odezva(id, u, t)); 
% stairs = @(x) (stairs(x))

% PLOTTING
global screen_full screen_third 
scrsz = get(0,'ScreenSize');
wid = scrsz(3)/3;
% [left,bottom,width,height]
screen_third = [2*wid scrsz(4)/2 wid scrsz(4)/2];
screen_full = [1 1 scrsz(3) scrsz(4)];
global SI SX SY
str_tim = 'time [s]';
str_val = 'value [1]';
str_ind = 'index [1]';

global z

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vlastní identifikace

%% parametry
if ident_param == 1

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% vstupní signál

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: jednotkový skok - pro urèení èasové konstanty

% budu volit èasovou konstantu jako 1/8 - 1/16 èasu nábìhu
% kdy jako èas nábìhu beru (vzhledem k zašumìlému charakteru) èas kdy
% dosáhne 0.7 násobku ustálené hodnoty 
% (dalo by se spoèítat i exaktnì dle vzorce - prùnik inflexe s 0 a
% ustálenou hodnotou)


% rozsah volím uvážlivì
min_tstep = 0.1;
max_tstep = 10;
count_tstep = 20;

% dìlám že nevím kolik by mìlo být zesílení, které jsem pozdìji vypoèítal
% obecnì musím vidìt že signál je mimo šum
utry = 10; 

% pøedpokládám že èasová konstanta systému nebude zas až tak extrémní, a
% øeknìme, že bude maximálnì 100×delší než nejdelší odhadovaný vzorkovací
% èas (respektive 50× -> viz druhá pùlka pøi výpoètu dále)
tend = 100*max_tstep;


figure('Position',screen_full); SX=1 ;SY=2 ;SI=0;
SI=SI+1;subplot(SY,SX,SI)

tsteps = linspace(min_tstep,max_tstep,count_tstep);
len_ts = length(tsteps);
cols = hsv(count_tstep);
for q = 1:len_ts
    TSTEP = tsteps(q);
    [t,nsteps] = GET_t(tend,TSTEP);
    u = utry*ones(nsteps,1);
    y = getResponse(u,t);
    stairs(t,y,'Color',cols(q,:));
    hold on
    
    ystep{q} = y;
    ystep_tt{q} = t;
end
title(sprintf('[%.2f]×jednotkový skok, pro rùzné hodnoty èasové konstanty',utry));
xlabel(str_ind)
ylabel(str_val)

%% vypoèteme hodonotu ustáleného signálu, jakožto prùmìr ze druhé poloviny 
% èasového intervalu, signálù jenž nevypadli

% minimum nenulových krokù v druhé pùlce intervalu odezvy na jednotkový skok
min_steps = 2;
imean = 1;
for q = 1:len_ts
    y = ystep{q};
    half = round(length(y)/2);
    y_half = y( half:end );
    % výbìr druhé pùlky signálu (pøi výpadku èidla už je zkrácen)
    [yG, indG] = GET_intGood(y_half);
    if indG < min_steps
        % málo nenulových krokù
        continue
    end
    yG_mean(imean) = mean(yG);
    imean = imean+1;
end
% magnifyOnFigure.m

y_stable = mean(yG_mean(:));
coef = 0.63;
y_up = coef*y_stable;
% plot those two
tt = [0,tend];
ons = ones(2,1);
hold on
plot(tt,ons*y_stable,'k', 'LineWidth',4, 'LineStyle', '--');
plot(tt,ons*y_up,'g','LineWidth',4, 'LineStyle', '--');
xlabel(str_ind)
ylabel(str_val)

%% zjistíme èas ve kterém mìl výstup hodnotu menší než [y_up]
% ze signálu s nejmenší èasovou konstantou (pokud nemá u zaèátku výpadek)
for e = 1:10
    y = ystep{e};
    tt = ystep_tt{e};
    % hodnoty krokù které splòují podmínku inverzní - lépe se hledá

    f = find(y>=y_up);
    % první vìtší
    fG = f(1);

    [~, indG] = GET_intGood(y);
    if indG > fG
        % pøi tomto vzorkování vypadlo èidlo døíve než dosáhlo dané hodnoty
        % -> zkusíme to znovu 
    else 
        break
    end
    if e==10
       error('%s\n%s%i%s',...
           'Èidlo opakovanì (10×) vypadáva pøi nejmenším zvoleném èasu vzorkování [min_tstep],',...
           'prosím zvolte jinou hodnotu než ',min_tstep, '(pravdìpodobnì vìtší).')
    end
end

%% nabezny cas
T_UP = tt(fG);
SI=SI+1;subplot(SY,SX,SI)


tend = 4*T_UP;
TSTEP = T_UP / 10;
% TSTEP = 1;
[t,nsteps] = GET_t(tend,TSTEP);
u = utry*ones(nsteps,1);

%% plotting
y = getResponse(u,t);    
tt = [0,tend];
ons = ones(2,1);

stairs(t,y,'r')
hold on
plot(t,y,'o--')
plot(tt,ons*y_stable,'k', 'LineWidth',4, 'LineStyle', '--');
plot(tt,ons*y_up,'g','LineWidth',4, 'LineStyle', '--');

tt_up = [T_UP, T_UP+0.01*TSTEP];
yy_up = [0,y_stable*1.5];
plot(tt_up,yy_up,'b','LineWidth',4, 'LineStyle', '--');

legend('y-step output','y-stable','y-rising','t-up');
tit = sprintf('priblizna doba nabehu Tup[%.2f] -> vzorkovaci frekvence[%.2f]',T_UP,TSTEP);
title(tit);
grid on
axis tight



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% následující 4 vstupy vykreslit pospolu
figure('Position',screen_full); SX=1 ;SY=4 ;SI=0;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: bez vstupu
tit = 'bez vstupu - nulový vstupní signál';

tend = T_UP*1000;
[t,nsteps] = GET_t(tend,TSTEP);

u = zeros(nsteps,1);
y = getResponse(u,t);

% odstranit nulove
yG = y;
f = find(y==0);
ind0 = f(1);
yG(ind0) = [];

%% vypocet
% bezpeènostní koeficient
coef = 1.1;
NOISE_MAX = max(abs(yG)) * coef;
mu = mean(yG);

% preplot
tt = [0,tend];
ons = [1,1];
o = mu * ons;
N = NOISE_MAX * ons;

%% ploting
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'b')
hold on
stairs(t,y,'r')
stairs(tt,o,'o--','LineWidth',3)
stairs(tt,N,'g--','LineWidth',3)
stairs(tt,-N,'g--','LineWidth',3)
grid on
title(tit)

txmu = strcat('mean=[',num2str(mu),']');
txno = strcat('noise=[',num2str(NOISE_MAX),']');
legend('u(k)','y(k)',txmu,txno)
xlabel(str_tim)
ylabel(str_val)
axis tight

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% z odezvy na nulový signál vidíme že systém vykazuje:
% * šum s hodnotou spadající do intervalu [+-1]
% * oproti šumu zanedbatelný stejnosmìrnou složku 
%   - prùmìr výstupu je menší než 0.1
% -->
% volíme vstupní signál s dostateèným odstupem signál šum 
% --> umax = 10
% * ale pozor na nelinearity
% 
% ____________________________________________________
% :: jednotkový impulz
% tit = 'jednotkový impulz';
% u = zeros(nsteps,1); 
% u(1) = umax;
% y = getResponse(u,t);
% 
% % ploting
% SI=SI+1;subplot(SY,SX,SI)
% stairs(t,u,'g')
% hold on
% stairs(t,y,'r')
% title(tit)
% legend('u(k)','y(k)')
% xlabel(str_tim)
% ylabel(str_val)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: rostoucí signál
tit = 'lineárnì rostoucí signál';

tend = T_UP*1000;
[t,nsteps] = GET_t(tend,TSTEP);

% do kolika rùst 
% - viz nìjaký násobek noise_max, aby už se projevila nelinearita
utry = 5*NOISE_MAX; 

% poèet intervalù
qint_count = 20; 
% pøibližná délka jednoho intervalu
qint_len = floor(nsteps/qint_count); 

% minimální délka signálu pro výpoèet smìrnice
min_length_coef = 0.3;
min_length = round(qint_len * min_length_coef); 

% vstupní lineárnì rostoucí signál
u = linspace(NOISE_MAX,utry,nsteps)'; 
y = getResponse(u,t);


%% výpoèet maximální velikosti vstupu pøi kterém se systém chová lineárnì

umax = max(u(:));
ulen = numel(u);
unorm1 = u / umax * ulen; % ma smìrnici 1 protože je lineární

ymax = max(y(:));
ylen = numel(y);
ynorm1 = y / ymax * ylen ; % ma konstantní smìrnici dokud je lineární

qmax = qint_count+1; % poèet okrajù všech intervalù

% qlen = unorm1 / ;
qind = floor(linspace(1,ylen+1,qmax));

% init zeros
slope = zeros(1,qint_count);
% detekce výpadku v prvním kroku
q0_bad = 0;

for q=1:qint_count
    % interval ze vstupu unorm1
    tt = qind(q): (qind(q+1)-1);
    qy = ynorm1( tt );
    qu = unorm1( tt );
%     stairs(length(qu)); %- celkem stejnì dlouhý
    
    %% store it for later plotting
    tt_qy_plot{q} = tt;
    qy_plot{q} = qy;
    

    % odštípne pøípadnou chybu
    [qy_good, indG] = GET_intGood(qy);
    %% pro tento signál se bude poèítat smìrnice
    qlen = length(qy_good);
    if qlen < min_length
        % nemá cenu poèítat smìrnici pro málo prvkù
% % nezjednodušší -> necháme nulu        
        % pro zjednodušení dalších výpoètù použijeme pøedchozí
        q0_bad = 1;
        if q~=1
            % pozor kdyby byl první nulový..
            slope(q) = slope(q-1);
        end
        continue;
    end
    
    % spoèítáme smìrnici pomocí MNÈ
    phi = qy_good;
    
    qu = qu(1:indG);
    psi = cat(2, ones(qlen,1), qu);
    
    omega = psi \ phi;
%     offset = omega(1); % = noise_max
    slope(q) = omega(2);
    
end

    
% obèas se stane že smìrnice zachytí i strmý nástup nelinearity..
% každopádnì je dùležité zjistit pøedposlední ještì lineární hodnotu vstupu
% --> z kroku ve kterem se smìrnice pøíliš nelišila od ostatních

% najdu ten krok ve kterém se lišila
% % byla o dost procent !vìtší! než pøedchozí
% % byla o dost procent !vìtší! než prùmìr dosavadních
% % byla o dost procent !vìtší! než další
% % plus musím ignorovat ty co byli na výstupu nulové
% % --> to zní jako derivace

%% POZOR! pokud výpadek èidla ohrozí první krok
% mohla by v [derivaci smìrnice] vyskoèit v druhém kroku vysoká hodnota
if q0_bad == 1
% ošetøíme to pøiøazením prùmìrné smìrnice øeknìme z 5ti následujících
% krokù - tato nemá sice vypovídající hodnotu ale neohrozí tolik prùbìh
% derivace
    slope_mean = mean(slope(1:5));
    slope(1) = slope_mean ;
    % pokud jsou nìjaké další nulové nastavit na tuto hodnotu
    % pøedpokládáme nenulový 5tý interval
    for q=1:4
        if slope(q) == 0
            slope(q) = slope_mean;
        end
    end
end
    
%% derivace smìrnice
tstep2 = 1; % dame že krok je jedna protože jedeme pøes indexy ve slope
z = tf('z',tstep2);

% diskretni derivator
DFormula = tstep2/2*(z + 1)/(z-1); 
Td = 1;
N = 1;
tf_deriv = Td / (Td/N + DFormula);

% simulace derivace smìrnice
Fz = tf_deriv;
tslope = 1:length(slope);
slope_dif = lsim(Fz,slope,tslope);

% no a tady už jasnì vidím ten nejvìtší zlom se projeví jakožto vysoký peak
[~, q] = max(slope_dif);

% index posledního lineárnì se tváøícího intervalu 
% ještì jeden ubereme abysme si byli opravdu jistí
q_G = q - 2;
% hodnota vstupu pro kterou se systém ještì urèitì choval lineárnì
U_LINMAX = u(qind(q_G));
% u_lin_max = u(round(1900/tstep)) % odeèteno z grafu

%% ploting
sxx = 4;
sii = sxx+0;
SI = SI+1;

% rampa
sii=sii+1;
subplot(SY,sxx,sii)
stairs(t,u,'g')
hold on
stairs(t,y,'r')
title(tit)
legend('u(k)','y(k)')
xlabel(str_tim)
ylabel(str_val)
grid on
axis tight

 sii=sii+1;
subplot(SY,sxx,sii)
title('jednotlivé intervaly vstupního signálu pro výpoèet smìrnice');

cols = prism(qint_count);
for q=1:qint_count
    tt = tt_qy_plot{q};
    qy = qy_plot{q};
    stairs(tt,qy,'Color',cols(q,:))
    hold on
end
xlabel(str_ind)
ylabel(str_val)
grid on
title(tit)
axis tight

% smìrnice
sii=sii+1;
subplot(SY,sxx,sii)
for q=1:qint_count
    tt = tslope(q);
    yy = slope(q);
    stem(tt, yy, 'Color',cols(q,:))
    hold on
end
title(strcat('smìrnice - ',tit));
xlabel(str_ind)
ylabel(str_val)
grid on
axis tight

% derivace smìrnice
sii=sii+1;
subplot(SY,sxx,sii)
for q=1:qint_count
    tt = tslope(q);
    yy = slope_dif(q);
    stem(tt, yy, 'Color',cols(q,:))
    hold on
end
title(strcat('derivace smìrnice - ',tit));
xlabel(str_ind)
ylabel(str_val)
grid on

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: jednotkový skok - znovu tentokrát urèitì lineární
tit = sprintf('[%.2f]×jednotkový skok',U_LINMAX);

tend = T_UP*10000;
[t,nsteps] = GET_t(tend,TSTEP);

% pokud vynechám signál s poruchou èidla mùžu zjistit statické zesílení K
u = U_LINMAX * ones(nsteps,1);
y = getResponse(u,t);

% poèet intervalù
qint_count = 20; 
% poèet okrajù všech intervalù
qmax = qint_count+1; 
% minimální délka pro výpoèet
min_length = 5;

% pro odeznìní dynamiky
t_start = T_UP * 42;

% intervaly
qind = floor(linspace(t_start,nsteps+1,qmax));
% init zeros
slope = zeros(1,qint_count);

imean = 1;
for q=1:qint_count
    % interval ze vstupu unorm1
    tt = qind(q): (qind(q+1)-1);
    qy = y( tt );
    
    % odštípne pøípadnou chybu
    [qy_good, indG] = GET_intGood(qy);
    %% pro tento signál se bude poèítat smìrnice
    qlen = length(qy_good);
    if qlen < min_length
        % nemá cenu poèítat pro málo vzorkù
        continue;
    end
    y_stable_mean(imean) = mean(qy_good);
    imean=imean+1;
end
y_stable = mean(y_stable_mean);
K = y_stable / U_LINMAX ;
tit = sprintf('%s K[%.2f]',tit,K);

%% zpoždìní
coef = 1.2;
f = find( y > NOISE_MAX*coef);
indD = f(1);
TDELAY = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% z odezvy na jednotkový skok vidíme že systém vykazuje
% * zesílení rovné pomìru výstupní ku vstupní ustálené hodnotì 

%% ploting
SI=SI+1;subplot(SY,SX,SI)
% u
stairs(t,u,'g')
hold on
% y
stairs(t,y,'r')
% stable
tt= [0, tend];
ons = [1,1];
plot(tt,ons*y_stable,'k','LineWidth',3,'LineStyle','--');

title(tit)
legend('u(k)','y(k)')
xlabel(str_tim)
ylabel(str_val)
grid on
axis tight
title('odezva na ''jednotkový'' skok')
drawnow

SI=SI+1;subplot(SY,SX,SI)
u = U_LINMAX*100 * ones(nsteps,1);
y = getResponse(u,t);

coef = 0.001;
ind = floor(length(t)*coef);
t = t(1:ind);
u = u(1:ind);
y = y(1:ind);
tt= [0, ind*TSTEP];

hold on
tt= [TDELAY,TDELAY+0.0001];
plot(tt,[0,y_stable*1.2],'b','LineWidth',3,'LineStyle','--');

% u
stairs(t,u,'g')
hold on
% y
stairs(t,y,'r')
% stable
ons = [1,1];
plot(tt,ons*y_stable,'k','LineWidth',3,'LineStyle','--');

title(tit)
legend('u(k)','y(k)')
xlabel(str_tim)
ylabel(str_val)
grid on
axis tight
title('odezva na ''jednotkový'' skok- výøez')

drawnow
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% žadne dopravní zpoždìní

%% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% error('I don''t want to play anymore')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: sinus
% u = sin(2*pi*0.1*t)+sin(2*pi+0.3*t)+sin(2*pi+0.01*t);


% ____________________________________________________


%% odstranìní stejnosmìrné složky
% není potøeba - není tam 
% provedlo by se odeètením 

%% odstranìní dopravního zpoždìní

else
    %% v pøípadì že nechci poèítat
%     K = 1.999;
%     U_LINMAX = 2.6123;
%     TSTEP = 0.42;
%     NOISE_MAX = 0.8707;
    K = 1.00;
    U_LINMAX = 2.32;
    TSTEP = 0.35;
    NOISE_MAX = 1.29;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % žadne dopravní zpoždìní

    T_UP = TSTEP * 10;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%% zobraz použité parametry
txt = sprintf('Paramerty:\nK = %.2f;\nU_LINMAX = %.2f;\nTSTEP = %.2f;\nNOISE_MAX = %.2f;',...
    K, U_LINMAX, TSTEP, NOISE_MAX);
disp(txt)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: PRBS

% tend = T_UP*100;
full_prbs_nsteps = 40950;
tend = ceil(full_prbs_nsteps * TSTEP);
% [t,nsteps] = GET_t(tend,TSTEP);
t = ( 0:TSTEP:tend )';
nsteps = full_prbs_nsteps;
% size(t)
t = t(1:nsteps);
% size(t)
% nsteps

% where B is such that the signal is constant over intervals of length 1/B
% (the clock period)
% const_interval = ceil(tstep * 3);
band = [0, 0.01];
% = nejmenì 10 kmitu je stejných..
% prbs 0 1
% = 00000000 1111111
% band = [0, ceil(1/const_interval)];

% ____________________________________________________
% rozsah vstupniho signalu
coef = 1 / 2  ;
umax = U_LINMAX * coef;
umin = -umax; 

% umax = U_LINMAX;
% umin = 0;

levels = [umin, umax];
% levels = [-1 1];

uall = idinput(nsteps, 'prbs', band, levels); % full prbs sequence
tall = t;

%% prumìrování
% y_ = zeros(nsteps,1);
% % qmax = 42;
% qmax = 10;
% for q=1:qmax
%     y_ = y_ + getResponse(u,t);
% end
% y = y_ / qmax;

%% odezva
yall = getResponse(uall,tall);

coef = 4/5;
half = floor(nsteps * coef);
u = uall(1:half);
y = yall(1:half);
t = tall(1:half);

%% lineární regrese


% orders
M = 2;
N = 2;

% není pøevýšení
% impulzová charakteristika
% staèí 2 øád

o_off = max([M, N]);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% typ MNÈ
% odkomentovat jeden typ pro výbìr
rms_types = {'normal RMS', 'delayed_observation','added_model'};
rms_type = 'normal RMS'; % nejlepší výsledky
% rms_type = 'delayed_observation'; % výsledky
% rms_type = 'added_model'; % - nedodìláno

fprintf('\nPoužitá metoda: %s\n\n',rms_type);
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% normání metoda nejmenších ètvercù - bez pomocných promìnných
if strcmp(rms_type, 'normal RMS')
    lenu = length(u);
    k = (o_off+1):lenu;
    %% phi
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y(k-n) );
    end
    phi = cat( 2, uks, -yks );
    %% psi
    psi = y(k);
    
    %% odstranìní výpadku èidla
    [psi,phi] = GET_psiphiGood(y,psi,phi,o_off,1);
    
    %% lineární regrese
    theta = phi \ psi;

    %% pøenos
    Fz = GET_tf(TSTEP, theta, M, N);

    %% odezva modelu identifikovaného systému
    yi = lsim(Fz,u,t);
    % phi(u(k-1), u(k-2), .., u(k-m), -y(k-1), -y(k-2), .., -y(k-n) ) = [uks,yks]
    % theta(b0, b1, b2,.., bm, a0, a1, a2,..,an)
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Zpoždìné pozorování - MNÈ s pomocnými promìnnými
% zpozdíme výstupy - nejménì o 1 krok 
% --> výstupy už potom nebudou korelované se šumem v aktuálním kroku
if strcmp(rms_type, 'delayed_observation')
    d_off = 1;
    lenu = length(u);
    k = (o_off+1+d_off) : lenu ;
    %% phi
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y(k-n) );
    end
    phi = cat( 2, uks, -yks );
    %% zeta
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y(k-n-d_off) );
    end
    zeta = cat( 2, uks, -yks );
    
    %% psi
    psi = y(k);
    %% odstranìní výpadku èidla
    psi_bad = psi;
    [psi,phi] = GET_psiphiGood(y,psi_bad, phi,o_off,1);
    [~,  zeta ] = GET_psiphiGood(y,psi_bad, zeta,o_off,0);
    
    %% regrese
    theta = (zeta'*phi) \ (zeta'*psi);

    %% pøenos
    Fz = GET_tf(TSTEP, theta, M, N);

    %% odezva modelu identifikovaného systému
    yi = lsim(Fz,u,t);
end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pomocný model - MNÈ s pomocnými promìnnými
if strcmp(rms_type, 'added_model')
    lenu = length(u);
    k = lenu:-1:(o_off+1);
    %% phi
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y(k-n) );
    end
    phi = cat( 2, uks, -yks );
    %% zeta
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y(k-n) );
    end
    zeta = cat( 2, uks, -yks );
    
    %% psi
    psi = y(k);
    %% odstranìní výpadku èidla
    psi_bad = psi;
    [psi,phi] = GET_psiphiGood(y,psi_bad, phi,o_off,0);
    [~,  zeta ] = GET_psiphiGood(y,psi_bad, zeta,o_off,0);
    
    %% regrese
    theta_addon = phi \ psi;

    %% pøenos
    Fz_addon = GET_tf(TSTEP, theta_addon, M, N);

    %% odezva modelu identifikovaného systému
    y_model = lsim(Fz_addon,u,t);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % :: vlastni odezva modelu
    lenu = length(u);
    k = lenu:-1:(o_off+1);
    %% phi
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y(k-n) );
    end
    phi = cat( 2, uks, -yks );
    %% zeta
    uks = [];
    yks = [];
    for m = 0 : (M-1)
        uks = cat(2, uks, u(k-m) );
    end
    for n = 1 : (N)
        yks = cat(2, yks, y_model(k-n) );
    end
    zeta = cat( 2, uks, -yks );
    
    %% psi
    psi = y(k);
    %% odstranìní výpadku èidla
    psi_bad = psi;
    [psi,phi] = GET_psiphiGood(y,psi_bad, phi,o_off,1);
    [~,  zeta ] = GET_psiphiGood(y,psi_bad, zeta,o_off,0);
    
    %% regrese
    theta = (zeta'*phi) \ (zeta'*psi);

    %% pøenos
    Fz = GET_tf(TSTEP, theta, M, N);

    %% odezva modelu identifikovaného systému
    yi = lsim(Fz,u,t);
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% pøenosovka
fprintf('\n\nPøenosová funkce identifikovaného systému:\n%s');
Fz

uall_len = length(uall);
str_from = @(from,to) (sprintf('/[%i] = %.2f%%', ...
    uall_len,(to-from)/uall_len*100));

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTY
figure('Position',screen_full);  SX=1 ;SY=3 ;SI=0;
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'g','LineWidth',2)
hold on
stairs(t,y,'r')
stairs(t,yi,'b','LineWidth',2)

grid on
legend('u(k)','y(k)','y_{ident}(k)')
xlabel(str_tim)
ylabel(str_val)
axis tight
title(sprintf('nauèená data - trénovací množina - zkouška identifikace [1:%i]%s',...
    half,str_from(1,half) ))



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ovìøení identifikace
half2 = half;
lenu = length(u);
% lenu = length(uall);
u = uall(half2:end);
y = yall(half2:end);
t = tall(half2:end) - t(half2);
yi_half = lsim(Fz,u,t);
yi = yi_half;
%% nìjakou porovnávací funkci..


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ploty
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'g','LineWidth',2)
hold on
stairs(t,y,'r')
stairs(t,yi,'b','LineWidth',2)

grid on
legend('u(k)','y(k)','y_{ident}(k)')
xlabel(str_tim)
ylabel(str_val)
axis tight
title(sprintf('nenauèená data - ovìøení identifikace [%i:%i]%s', ...
    half,uall_len,str_from(half,uall_len) ))

%% výøez

coef_zoom = 1/100;
zoom = floor(nsteps * coef_zoom);
zoom_and_half = zoom + half;
u = uall(half:zoom_and_half);
y = yall(half:zoom_and_half);
t = tall(half:zoom_and_half);
yi = yi_half(1:(zoom+1));

%% ploty
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'g')
hold on
stairs(t,y,'r')
stairs(t,yi,'b')

grid on
legend('u(k)','y(k)','y_{ident}(k)')
xlabel(str_tim)
ylabel(str_val)
axis tight


title(sprintf('nenauèená data - ovìøení identifikace - výøez[%i:%i]%s',...
    half,zoom_and_half, str_from(half,zoom_and_half) ));

%% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% error('I don''t want to play anymore')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% verry good
%  - complementar sy
%   -6.981e-17 z + 1.097e-15
%   ------------------------
%   4.673e-15 z - 4.219e-15


% phi(u(k+1), u(k+2), .., u(k+m-1), y(k+1), y(k+2), .., y(k+n-1) ) 
% theta(b0, b1, b2,.., bm, a0, a1, a2,..,an)