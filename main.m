%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MMID Projekt
% VUT, FEKT, KAM
% Auto�i: DAV�DEK D.
% Zad�n�:
% % Prove�te  identifikaci  zadan�ho  syst�mu  pomoc�  jedn�  z  metod  pomocn�ch  prom�nn�ch, 
% % kter�  zajist�  nevych�len�  odhad  nezn�m�ch  parametr�  (volbu  metody  zd�vodn�te).  Zvolte 
% % vhodnou periodu vzorkov�n� a vstupn� sign�l k identifikaci. V�sledky ov��te pomoc� Matlab 
% % Identification  Toolbox  (ident).  V�stup  z  nezn�m�ho  syst�mu  z�sk�te  z  p�ilo�en�  funkce 
% % odezva.p. 
% Popis: 
% % V�sledkem  identifikace  by  m�l  b�t  diskr�tn�  model  deterministick�  ��sti  syst�mu.  P�i 
% % zpracov�n� dejte pozor na to, �e 
% % ?  syst�m je neline�rn� (maxim�ln� vstupn� sign�l mus� b�t omezen) 
% % ?  m��e b�t porouchan� �idlo m��en� v�stupn�ho sign�lu (nap�. segmentace dat) 
% % ?  syst�m m��e obsahovat dopravn� zpo�d�n� a stejnosm�rnou slo�ku 
% % ?  m��en�  je  drah�,  proto  v  hotov�m  projektu  m��ete  volat  funkci  odezva.p  jen 
% % jednou,  mno�stv�  jednor�zov�  z�skan�ch  dat  nen�  omezeno,  ka�d�  dal��  vol�n�  je 
% % penalizov�no jedn�m bodem (pro va�e lad�n� si ji pus�te kolikr�t chcete). 
% Hlavn� skript - vol� v�echny ostatn� funkce
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% init
close all;clc;clear all
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

global z

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Vlastn� identifikace

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% casovy vektor
tend = 3400;
tstep = 1.5;
z = tf('z',tstep);


% [K,Td,T] = c02_1(tend, tstep, nmean, stepRespFcn)
% t = ( 0:tstep:(tend-tstep) )';

[t,nsteps] = GET_t(tend,tstep);
% nsteps = length(t);
% N = nsteps;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% vstupn� sign�l



%% rozd�lit nam��en� data na identifika�n� a verifika�n�
% -aby se volal jenom jednou odezva.p

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: jednotkov� skok - pro ur�en� �asov� konstanty

% budu volit �asovou konstantu jako 1/8 - 1/16 �asu n�b�hu
% kdy jako �as n�b�hu beru (vzhledem k za�um�l�mu charakteru) �as kdy
% dos�hne 0.7 n�sobku ust�len� hodnoty 
% (dalo by se spo��tat i exaktn� dle vzorce - pr�nik inflexe s 0 a
% ust�lenou hodnotou)


% rozsah vol�m uv�liv�
min_tstep = 0.1;
max_tstep = 10;
count_tstep = 20;

utry = 10; 
% d�l�m �e nev�m kolik by m�lo b�t zes�len�, kter� jsem pozd�ji vypo��tal
% obecn� mus�m vid�t �e sign�l je mimo �um
tend = 100*max_tstep;
% p�edpokl�d�m �e �asov� konstanta syst�mu nebude zas a� tak extr�mn�, a
% �ekn�me, �e bude maxim�ln� 100�del�� ne� nejdel�� odhadovan� vzorkovac�
% �as (respektive 50� -> viz druh� p�lka p�i v�po�tu d�le)

figure('Position',screen_full); SX=1 ;SY=2 ;SI=0;
SI=SI+1;subplot(SY,SX,SI)

tsteps = linspace(min_tstep,max_tstep,count_tstep);
len_ts = length(tsteps);
cols = hsv(count_tstep);
for q = 1:len_ts
    tstep = tsteps(q);
    [t,nsteps] = GET_t(tend,tstep);
    u = utry*ones(nsteps,1);
    y = getResponse(u,t);
    stairs(t,y,'Color',cols(q,:));
    hold on
    
    ystep{q} = y;
end

% vypo�teme hodonotu ust�len�ho sign�lu, jako�to pr�m�r ze druh� poloviny 
% �asov�ho intervalu, sign�l� jen� nevypadli

% minimum nenulov�ch krok� v druh� p�lce intervalu odezvy na jednotkov� skok
min_steps = 2;
imean = 1;
for q = 1:len_ts
    y = ystep{q};
    half = round(length(y)/2);
    y_half = y( half:end );
    % v�b�r druh� p�lky sign�lu (p�i v�padku �idla u� je zkr�cen)
    [yG, indG] = GET_intGood(y_half);
    if indG < min_steps
        % m�lo nenulov�ch krok�
        continue
    end
    yG_mean(imean) = mean(yG);
    imean = imean+1;
end

y_stable = mean(yG_mean(:));
y_up = 0.7*y_stable;
% plot those two
tt = [0,tend];
ons = ones(2,1);
hold on
plot(tt,ons*y_stable,'k', 'LineWidth',4, 'LineStyle', '--');
plot(tt,ons*y_up,'g','LineWidth',4, 'LineStyle', '--');

% zjist�me �as ve kter�m m�l v�stup hodnotu men�� ne� [y_up]
% ze sign�lu s nejmen�� �asovou konstantou (pokud nem� u za��tku v�padek)
y = ystep{end};
% hodnoty krok� kter� spl�uj� podm�nku inverzn� - l�pe se hled�

figure
subplot(121)
f = find(y>=y_up);
plot( f )
subplot(122)
plot( f,y(f) )
hold on
val = y( f(1) );
tt2=[0,length(f)];
plot(tt2,ons*val,'LineWidth',3);
axis normal

SI=SI+1;subplot(SY,SX,SI)


% [yG, indG] = GET_intGood(y);
% if indG


%% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
error('I don''t want to play anymore')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n�sleduj�c� 3 vstupy vykreslit pospolu
figure('Position',screen_full); SX=1 ;SY=3 ;SI=0;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: bez vstupu
tit = 'bez vstupu';
u = zeros(nsteps,1);
y = getResponse(u,t);

mu = mean(y);
o = mu * ones(1,nsteps);
noise_max = max(abs(y));
n = noise_max * ones(1,nsteps);

% ploting
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'g')
hold on
stairs(t,y,'r')
stairs(t,o,'c')
stairs(t,n,'g')
stairs(t,-n,'g')
grid on
title(tit)

txmu = strcat('mean=[',num2str(mu),']');
txno = strcat('noise=[',num2str(noise_max),']');
legend('u(k)','y(k)',txmu,txno)
xlabel('time [s]')
ylabel('value [1]')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% z odezvy na nulov� sign�l vid�me �e syst�m vykazuje:
% * �um s hodnotou spadaj�c� do intervalu [+-1]
% * oproti �umu zanedbateln� stejnosm�rnou slo�ku 
%   - pr�m�r v�stupu je men�� ne� 0.1
% -->
% vol�me vstupn� sign�l s dostate�n�m odstupem sign�l �um 
% --> umax = 10
% * ale pozor na nelinearity
% 
% ____________________________________________________
% :: jednotkov� impulz
% tit = 'jednotkov� impulz';
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
% xlabel('time [s]')
% ylabel('value [1]')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: rostouc� sign�l
tit = 'line�rn� rostouc� sign�l';


tend = 3000;
[t,nsteps] = GET_t(tend,tstep);

% do kolika r�st 
% - viz n�jak� n�sobek noise_max, aby u� se projevila nelinearita
utry = 5*noise_max; 

% po�et interval�
qint_count = 20; 
% p�ibli�n� d�lka jednoho intervalu
qint_len = floor(nsteps/qint_count); 

% minim�ln� d�lka sign�lu pro v�po�et sm�rnice
min_length_coef = 0.3;
min_length = round(qint_len * min_length_coef); 

% vstupn� line�rn� rostouc� sign�l
u = linspace(noise_max,utry,nsteps)'; 
y = getResponse(u,t);

y(1:50) = 0;

%% v�po�et maxim�ln� velikosti vstupu p�i kter�m se syst�m chov� line�rn�

umax = max(u(:));
ulen = numel(u);
unorm1 = u / umax * ulen; % ma sm�rnici 1 proto�e je line�rn�

ymax = max(y(:));
ylen = numel(y);
ynorm1 = y / ymax * ylen ; % ma konstantn� sm�rnici dokud je line�rn�

qmax = qint_count+1; % po�et okraj� v�ech interval�

% qlen = unorm1 / ;
qind = floor(linspace(1,ylen+1,qmax));

% init zeros
slope = zeros(1,qint_count);
% detekce v�padku v prvn�m kroku
q0_bad = 0;

for q=1:qint_count
    % interval ze vstupu unorm1
    tt = qind(q): (qind(q+1)-1);
    qy = ynorm1( tt );
    qu = unorm1( tt );
%     stairs(length(qu)); %- celkem stejn� dlouh�
    
    %% store it for later plotting
    tt_qy_plot{q} = tt;
    qy_plot{q} = qy;
    

    % od�t�pne p��padnou chybu
    [qy_good, indG] = GET_intGood(qy);
    %% pro tento sign�l se bude po��tat sm�rnice
    qlen = length(qy_good);
    if qlen < min_length
        % nem� cenu po��tat sm�rnici pro m�lo prvk�
% % nezjednodu��� -> nech�me nulu        
        % pro zjednodu�en� dal��ch v�po�t� pou�ijeme p�edchoz�
        q0_bad = 1;
        if q~=1
            % pozor kdyby byl prvn� nulov�..
            slope(q) = slope(q-1);
        end
        continue;
    end
    
    % spo��t�me sm�rnici pomoc� MN�
    phi = qy_good;
    
    qu = qu(1:indG);
    psi = cat(2, ones(qlen,1), qu);
    
    omega = psi \ phi;
%     offset = omega(1); % = noise_max
    slope(q) = omega(2);
    
end

    
% ob�as se stane �e sm�rnice zachyt� i strm� n�stup nelinearity..
% ka�dop�dn� je d�le�it� zjistit p�edposledn� je�t� line�rn� hodnotu vstupu
% --> z kroku ve kterem se sm�rnice p��li� neli�ila od ostatn�ch

% najdu ten krok ve kter�m se li�ila
% % byla o dost procent !v�t��! ne� p�edchoz�
% % byla o dost procent !v�t��! ne� pr�m�r dosavadn�ch
% % byla o dost procent !v�t��! ne� dal��
% % plus mus�m ignorovat ty co byli na v�stupu nulov�
% % --> to zn� jako derivace

%% POZOR! pokud v�padek �idla ohroz� prvn� krok
% mohla by v [derivaci sm�rnice] vysko�it v druh�m kroku vysok� hodnota
if q0_bad == 1
% o�et��me to p�i�azen�m pr�m�rn� sm�rnice �ekn�me z 5ti n�sleduj�c�ch
% krok� - tato nem� sice vypov�daj�c� hodnotu ale neohroz� tolik pr�b�h
% derivace
    slope_mean = mean(slope(1:5));
    slope(1) = slope_mean ;
    % pokud jsou n�jak� dal�� nulov� nastavit na tuto hodnotu
    % p�edpokl�d�me nenulov� 5t� interval
    for q=1:4
        if slope(q) == 0
            slope(q) = slope_mean;
        end
    end
end
    
%% derivace sm�rnice
tstep2 = 1; % dame �e krok je jedna proto�e jedeme p�es indexy ve slope
zz = tf('z',tstep2);

% diskretni derivator
DFormula = tstep2/2*(zz + 1)/(zz-1); 
Td = 1;
N = 1;
tf_deriv = Td / (Td/N + DFormula);

% simulace derivace sm�rnice
Fz = tf_deriv;
tslope = 1:length(slope);
slope_dif = lsim(Fz,slope,tslope);

% no a tady u� jasn� vid�m ten nejv�t�� zlom se projev� jako�to vysok� peak
[~, q] = max(slope_dif);
% index posledn�ho line�rn� se tv���c�ho intervalu 
% je�t� jeden ubereme abysme si byli opravdu jist�
q_G = q - 2;
% hodnota vstupu pro kterou se syst�m je�t� ur�it� choval line�rn�
u_lin_max = u(qind(q_G));
% u_lin_max = u(round(1900/tstep)) % ode�teno z grafu

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
xlabel('time [s]')
ylabel('value [1]')
grid on

 sii=sii+1;
subplot(SY,sxx,sii)
title('jednotliv� intervaly vstupn�ho sign�lu pro v�po�et sm�rnice');

cols = prism(qint_count);
for q=1:qint_count
    tt = tt_qy_plot{q};
    qy = qy_plot{q};
    stairs(tt,qy,'Color',cols(q,:))
    hold on
end
xlabel('time [s]')
ylabel('value [1]')
grid on
title(tit)

% sm�rnice
sii=sii+1;
subplot(SY,sxx,sii)
for q=1:qint_count
    tt = tslope(q);
    yy = slope(q);
    stem(tt, yy, 'Color',cols(q,:))
    hold on
end
title(strcat('sm�rnice - ',tit));
grid on

% derivace sm�rnice
sii=sii+1;
subplot(SY,sxx,sii)
for q=1:qint_count
    tt = tslope(q);
    yy = slope_dif(q);
    stem(tt, yy, 'Color',cols(q,:))
    hold on
end
title(strcat('derivace sm�rnice - ',tit));
grid on

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: jednotkov� skok - znovu tentokr�t ur�it� line�rn�
tit = sprintf('[%.2f]�jednotkov� skok',u_lin_max);

tend = 3000;
[t,nsteps] = GET_t(tend,tstep);

u = u_lin_max * ones(nsteps,1);
y = getResponse(u,t);
off = round(25/tstep);
K = mean(y(off:end)) / u_lin_max
tit = sprintf('%s K[%.2f]',tit,K);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% z odezvy na jednotkov� skok vid�me �e syst�m vykazuje
% * zes�len� rovn� pom�ru v�stupn� ku vstupn� ust�len� hodnot� 
%   - K =  (pr�m�r po 20 kroku)
% ploting
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'g')
hold on
stairs(t,y,'r')
title(tit)
legend('u(k)','y(k)')
xlabel('time [s]')
ylabel('value [1]')
grid on
% ____________________________________________________
% :: sinus
% u = sin(2*pi*0.1*t)+sin(2*pi+0.3*t)+sin(2*pi+0.01*t);


% ____________________________________________________

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% :: PRBS

tend = 1600;
[t,nsteps] = GET_t(tend,tstep);

% where B is such that the signal is constant over intervals of length 1/B
% (the clock period)
% const_interval = ceil(tstep * 3);
band = [0, 0.1];
% = nejmen� 10 kmitu je stejn�ch..
% prbs 0 1
% = 00000000 1111111
% band = [0, ceil(1/const_interval)];

% ____________________________________________________
% rozsah vstupniho signalu
u_lin_max = 5;
umin = -u_lin_max; 
% umin = 0;
levels = [umin, u_lin_max];
% levels = [-1 1];

u = idinput(nsteps, 'prbs', band, levels);

%% odezva
y_ = zeros(nsteps,1);
% qmax = 42;
qmax = 10;
for q=1:qmax
    y_ = y_ + getResponse(u,t);
end
y = y_ / qmax;

%% odstran�n� stejnosm�rn� slo�ky
% nen� pot�eba - nen� tam

%% odstran�n� dopravn�ho zpo�d�n�

%% odstran�n� v�padku �idla

%% line�rn� regrese
% u(k)
% u(k+1) = u( (1+1):(end-off+1) ) = u(2:end-1)

% phi(u(k+1), u(k+2), .., u(k+m-1), y(k+1), y(k+2), .., y(k+n-1) ) 
% theta(b0, b1, b2,.., bm, a0, a1, a2,..,an)

m = 2;
n = 3;


off = max([m,n]);
k = 1:(length(u)-off);

% vytvo�en� matice phi
uks = [];
yks = [];
for q = 1 : (m+1)
    uks = cat(2, uks, u(k+m) );
end
for q = 1 : (n+1)
    yks = cat(2, yks, -y(k+m) );
end

phi = cat( 2, uks, yks );

%% reseni
% k = 2:length(u)
% phi1 = [u(k-1), - y(k-1);
% Y = y(k)
% TH1 = PHI1 \ Y
% Fz1 = (TH1(1)) / (z+TH1(2))

%%
rad = 2;
theta = phi \ y(k);
% theta = phi(1:end-2,:) \ y(3:end);


% phi(u(k-1), u(k-2), .., u(k-m-1), y(k-1), y(k-2), .., y(k-n-1) ) 
% theta(b0, b1, b2,.., bm, a0, a1, a2,..,an)

numerator = 0;
denominator = 1;
for q = 1 : (m+1)
    numerator = numerator + theta(q) * z^-q;
end

for q = 1 : (n+1)
    denominator = denominator + theta(m+1+q) * z^-q;
end
% for q = 0 : (n)
%     denominator = denominator + theta(m+1+q) * z^-q;
% end

% b0 = theta(q); q=q+1;
% b1 = theta(q); q=q+1;
% a0 = theta(q); q=q+1;
% a1 = theta(q); q=q+1;

% Fz = (b0*z^-1) / (1 + a0*z^-1)
% Fz = (b0*z^-1) / (1 + a0*z^-1 + a1*z^-2)
% Fz = (b0*z^-1 + b1*z^-2) / (1 + a0*z^-1 + a1*z^-2)

Fz = K * numerator / denominator 
% * z^-1

% count the response of identified system
yi = lsim(Fz,u,t);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% metoda pomocn�ch prom�nn�ch

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOTY
figure('Position',screen_full);  SX=1 ;SY=1 ;SI=0;
SI=SI+1;subplot(SY,SX,SI)
stairs(t,u,'g')
hold on
stairs(t,y,'r')
stairs(t,yi,'b')

grid on
legend('u(k)','y(k)','y_{ident}(k)')

