function [ out, indG ] = GET_intGood( u )
%GET_INTGOOD najdi chybu èidla v datech (nulove hodnoty)
% pokud nenajde tak nastaví [indG] na index posledního prvku [u]
% indG - vdy obsahuje index prvku kterı není v posloupnosti pøedcházen nulou
%      - pokud je 0 první v posloupnosti --> indG=0
ind0 = find(u==0,1);
if ind0
    % nalezen alespoò jeden nulovı prvek
    % vra první èást intervalu ne pøišla chyba
    indG = ind0 - 1;
    out = u(1:indG);
else
    % vra celı interval
    indG = length(u);
    out = u;
end

end

