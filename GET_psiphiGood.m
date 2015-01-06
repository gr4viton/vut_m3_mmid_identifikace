function [psi,phi] = GET_psiphiGood(y,psi,phi,o_off,display)
%GET_PSIPHIGOOD odstran�n� v�padku �idla
%   Detailed explanation goes here
% abychom neodstranili prvn� vzorek jen� je nulov�
% a z�rove� zkr�t�me abese nemazali indexi kter� nejsou do psi a phi vkladany
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
        fprintf('Chyba �idla detekov�na!\n  Celkem [%i z %i] vzork� muselo b�t vy�azeno!\n  [%.2f%%] z celkov�ho mno�stv� ��dk� v matic�ch psi,phi (a zeta)',...
            qmax,nrow,qmax/nrow*100);
    end
end

end

