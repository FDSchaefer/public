clc; clear; close all;

%Add subfolder to paths
folder = fileparts(which(mfilename)); 
addpath(genpath(folder));


%To ask for user input for next frame
manual = false;
inp = 'n';
inp = input('Ask for input for next frame? ','s');
if inp == 'y' || inp == 'Y'
    disp('Press any key to continue simulation...')
    manual = true;
end

%Generaly Best between 20 -> 500
RunSimulation(400,manual)