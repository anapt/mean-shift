%
% SCRIPT: DEMO_MEANSHIFT
%
%   Sample script on usage of mean-shift function.
%
% DEPENDENCIES
%
%   meanshift
%
%


%% CLEAN-UP

clear;
close all;


%% PARAMETERS

% dataset options
basepath = './code/';
filename = 'r15';
varX     = 'X';
varL     = 'L';

% mean shift options
h = 1;
optMeanShift.epsilon = 1e-4*h;
optMeanShift.verbose = true;
optMeanShift.display = true;


%% (BEGIN)

fprintf('\n *** begin %s ***\n\n',mfilename);


%% READ DATA

fprintf('...reading data...\n')

matFile = [basepath filesep filename '.mat'];

fprintf('   - file: %s...\n', matFile)

ioData = matfile( matFile );

x = ioData.(varX);
l = ioData.(varL);

figure('name', 'original_data')
scatter(x(:,1),x(:,2), 8, l);


%% PERFORM MEAN SHIFT

fprintf('...computing mean shift...')

tic;
y = meanshift( x, h, optMeanShift );
tElapsed = toc;

fprintf('DONE in %.2f sec\n', tElapsed);


%% SHOW FINAL POSITIONS

figure('name', 'final_local_maxima_points')
scatter(y(:,1),y(:,2), 8, l);


%% (END)

fprintf('\n *** end %s ***\n\n',mfilename);


%%------------------------------------------------------------
%
% AUTHORS
%
%   Dimitris Floros                         fcdimitr@auth.gr
%
% VERSION
%
%   0.1 - December 29, 2017
%
% CHANGELOG
%
%   0.1 (Dec 29, 2017) - Dimitris
%       * initial implementation
%
% ------------------------------------------------------------
