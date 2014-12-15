function [ out, indG ] = GET_intGood( u )
%GET_INTGOOD najdi chybu �idla v datech (nulove hodnoty)
% pokud nenajde tak nastav� [indG] na index posledn�ho prvku [u]
% indG - v�dy obsahuje index prvku kter� nen� v posloupnosti p�edch�zen nulou
%      - pokud je 0 prvn� v posloupnosti --> indG=0
ind0 = find(u==0,1);
if ind0
    % nalezen alespo� jeden nulov� prvek
    % vra� prvn� ��st intervalu ne� p�i�la chyba
    indG = ind0 - 1;
    out = u(1:indG);
else
    % vra� cel� interval
    indG = length(u);
    out = u;
end

end

