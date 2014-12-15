function [ t, nsteps ] = GET_t( tend, tstep )
%GET_T Summary of this function goes here
%   Detailed explanation goes here
%% casovy vektor
% tend = 340;
% tstep = 1.5;

% [K,Td,T] = c02_1(tend, tstep, nmean, stepRespFcn)

t = ( 0:tstep:(tend-tstep) )';
nsteps = length(t);

end

