function [psi,phi] = GET_psiphiGood(y,psi,phi,o_off,display)
%GET_PSIPHIGOOD odstranìní výpadku èidla
%   Detailed explanation goes here
% abychom neodstranili první vzorek jenž je nulový
% a zároveò zkrátíme abese nemazali indexi které nejsou do psi a phi vkladany
% display - wheter to write text..

y_not0 = y(2:(end-o_off));

ind0 = find(y_not0==0);
% ind0 = ind0+1;

if ind0
    psi(ind0) = [];
    qmax = length(ind0);
    for q = qmax:-1:1
        phi(ind0(q),:) = [];
    end
    nrow = size(phi,1);
    if display == 1
        fprintf('Chyba èidla detekována!\n  Celkem [%i z %i] vzorkù muselo být vyøazeno!\n  [%.2f%%] z celkového množství øádkù v maticích psi,phi (a zeta)',...
            qmax,nrow,qmax/nrow*100);
    end
end

end

