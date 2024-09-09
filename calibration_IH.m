clear all 

%%
% Extrae la información en los contornos de la malla principal en la cual
% se utilizan los inputs de espectros, ademas se obtienen los parametros de
% calibración de Hs, extra la serie temporal y la calibra con los
% parametros estimados y los guarda. Estos pueden ser utilizados para la
% calibracion de los espectros y para un posible paso de espectros
% trihorarios a espectros horarios.

dir_out = 'data/cal_IH/';
lang = 'eng';

%% Se calibran
name_boya = {'Bilbao-Vizcaya','Cabo_de_Pennas','Estaca_de_Bares','Villano-Sisargas','Cabo_Silleiro','Golfo_de_Cadiz','Cabo_de_Gata','Cabo_de_Palos','Valencia','Tarragona','Cabo_Begur','Dragonera','Mahon','Gran_Canaria','Tenerife'};
inds_time = [43035, 35037, 34689, 33585, 25362, 24903, 19885, 17524, 18607, 22196, 23458, 15311, 33813, 21521, 24065];

for i=1:length(name_boya)
    name_boya_c = name_boya{i};
    
    boya = load(['data/processed/boya_',name_boya_c,'.mat']);
    copernicus = load(['data/processed/copernicus_',name_boya_c,'.mat']);
    gow = load(['data/processed/gow_',name_boya_c,'.mat']);

    % Train y test
    ind_time = inds_time(i);

    hs_boya_train = double(boya.hs(1:ind_time)'); 
    
    hs_gow_train = double(gow.hs(1:ind_time)'); 
    dir_gow_train = double(boya.dir(1:ind_time)');
    hs_gow_test = double(gow.hs(ind_time+1:end)'); 
    dir_gow_test = double(boya.dir(ind_time+1:end)');
    
    hs_cop_train = double(copernicus.hs(1:ind_time)'); 
    dir_cop_train = double(copernicus.dir(1:ind_time)');
    hs_cop_test = double(copernicus.hs(ind_time+1:end)'); 
    dir_cop_test = double(copernicus.dir(ind_time+1:end)');

    %% Calibracion GOW
    [dataCal_gow,dataGraph_gow] = utils.getCalibrationParameters(hs_boya_train,hs_gow_train,dir_gow_train);
    
    utils.getCalibrationParametersReport(dataGraph_gow,-3,5,'folder',dir_out,'nameFile','Test','svopt',[1 1 0]);

%     dataGraph_gow.Hscal_test = utils.applyCalibrationParameters(hs_gow_test,dir_gow_test,dataCal_gow);
%     save(fullfile(dir_out,[name_boya_c,'_gow.mat']),'dataGraph_gow')

    %% Calibracion GOW
    [dataCal_cop,dataGraph_cop] = utils.getCalibrationParameters(hs_boya_train,hs_cop_train,dir_cop_train);
    
%     dataGraph_cop.Hscal_test = utils.applyCalibrationParameters(hs_cop_test,dir_cop_test,dataCal_cop);
%     save(fullfile(dir_out,[name_boya_c,'_cop.mat']),'dataGraph_cop')
end