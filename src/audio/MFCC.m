fold_dir = '/Users/mac/Downloads/avec2017/';

fold_list = dir(fold_dir);

for i = 1:length(fold_list)
    if any(regexp(fold_list(i).name,'_P$'))  %% Will return true
        
        % Load a speech waveform
        [d,sr] = audioread(strcat(fold_dir, fold_list(i).name, '/', fold_list(i).name(1:3), '_AUDIO.wav'));
        % % Look at its regular spectrogram
        % subplot(411)
        % specgram(d, 256, sr);

        % Calculate 12th order PLP features without RASTA
        [cep2, spec2] = rastaplp(d, sr, 0, 12);
        % .. and plot them
        % subplot(414)
        % imagesc(10*log10(spec2));
        % axis xy
        % Notice the greater level of temporal detail compared to the 
        % RASTA-filtered version.  There is also greater spectral detail 
        % because our PLP model order is larger than the default of 8

        % Append deltas and double-deltas onto the cepstral vectors
        del = deltas(cep2);
        % Double deltas are deltas applied twice with a shorter window
        ddel = deltas(deltas(cep2,5),5);
        % Composite, 39-element feature vector, just like we use for speech recognition
        cepDpDD = [cep2;del;ddel];

%         % run PCA
%         cepDpDD = cepDpDD';
%         [loading, score, latent, explained] = pca(cepDpDD);
% 
%         c3 = loading(:,1:5);
%         coefforth = loading / diag(std(cepDpDD));
        csvwrite(strcat(fold_dir, fold_list(i).name, '/', fold_list(i).name(1:3), '_mfcc.csv'), cepDpDD')
    end
end


