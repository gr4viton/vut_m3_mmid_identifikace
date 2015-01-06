function [Fz] = GET_tf(TSTEP, theta, M, N)
%GET_TF returns transfer function from created input
global z

z = tf('z',TSTEP);
numerator = 0;
denominator = 1;

% theta(b0, b1, b2,.., bm, a0, a1, a2,..,an)

for m = 1 : (M)
    numerator = numerator + theta(m) * z^-(m);
end

o_off = M;
for n = 1 : (N)
    denominator = denominator + theta(o_off+n) * z^-(n);
end
% denominator
% 1 / denominator
% denominator = denominator + 1;
% denominator
% 1 / denominator

Fz = minreal(numerator / denominator);

end

