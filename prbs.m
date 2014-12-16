
TSTEP = 0.42;
tend = 10000;
[t,nsteps] = GET_t(tend,TSTEP);


band = [0, 0.1];
umax = 3;
umin = -umax; 
levels = [umin, umax];

u = idinput(nsteps, 'prbs', band, levels);

% Warning: The PRBS signal delivered is the 238 first values of a full sequence of length 310. 