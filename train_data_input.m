%% Load Data
FS = 12000; 
NUM_SAMPLES = FS*2;     
NUM_CHANNELS = 32;

bandfilt = designfilt('bandpassfir','FilterOrder',500,'CutoffFrequency1',40,'CutoffFrequency2',1280,'SampleRate',12000);

% Set Path to DATA
prefix = '/Volumes/icex6/ICEX_UNCLASS/ICEX16/macrura/2016-03-13/DURIP/DURIP_20160313T055853/';
%prefix = '/Volumes/icex6/ICEX_UNCLASS/ICEX16/macrura/2016-03-14_andbefore/DURIP/DURIP_20160314T002324/';
 
directory = dir([prefix 'ACO0000*.DAT']);

first_file = 2000+0*(1800/4);
last_file = first_file + 10;
 
% Read DATA
aco_in = zeros(NUM_SAMPLES * (last_file-first_file), 32);
  
% Start looping over ACO*.DAT files
counter=0;
for i = first_file:last_file-1
 
    counter=counter+1;
    filename = [prefix directory(i).name];
    fid = fopen (filename, 'r', 'ieee-le');
 
    if (fid <= 0)
        continue;
    end
 
    % Read the single precision float acoustic data samples (in uPa)
    for j = 1:NUM_CHANNELS
        aco_in(((counter-1)*NUM_SAMPLES+1):(counter*NUM_SAMPLES),j) = fread (fid, NUM_SAMPLES, 'float32');
    end
     
    fclose (fid);
end

timestamp = 1457848722.58 + first_file*2;
data_name = datestr ((timestamp / 86400) + datenum (1970,1,1), 31);

data_filt = filtfilt(bandfilt,aco_in);
time = (1/(FS))*(0:length(data_filt)-1/FS);
figure
plot(time,data_filt(:,16));

%%
timesteps = 32;
chns = 32;
samples = 8192;
overlap = 0.5;

win_len = samples;
window_start = (timesteps+1)*overlap*round(win_len-win_len*overlap);
step_start = round(win_len-win_len*overlap);
num_window = floor(size(data_filt,1)/window_start)-1;
t_end = size(data_filt,1)/FS;

train_dataset = zeros(num_window,timesteps,chns,samples,1);

for l = 1:num_window
    disp([num2str(l), '/', num2str(num_window)])

    t(l) = ((l+1)*window_start-window_start+1)/FS;
    data_seg = data_filt(l*window_start-window_start+1:l*window_start-window_start+(timesteps+1)*overlap*win_len,:);
    
    for i = 1:timesteps
        train_dataset(l,i,:,:,1) = data_seg(i*step_start-step_start+1:i*step_start-step_start+win_len,:).';
    end

end

t = t - t(1);
t_end = t_end - t(1);

train_dataset_save = reshape(train_dataset,[num_window*timesteps,chns*samples]);

%%
FID = fopen('quarter1.dat', 'w');
fwrite(FID, data_filt, 'double');
fclose(FID);
