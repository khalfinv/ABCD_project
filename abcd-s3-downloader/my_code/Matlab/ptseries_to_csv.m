%This function find all .ptseries.nii (time series in CIFTI format) 
%files in a root directory and convert to csv files. 
%Input:
    % dir: root directory
    % pattern: files patterns
    % new_file_name: output file name
%Output:
    % time series in csv format in each sub directory
    % list of subject keys in csv format in root folder
    % list of parcels in csv formt in root folder
%Run example: 
%ptseries_to_csv('C:\Users\skhalfin.TD-ST\Desktop\ABCD_project\abcd-s3-downloader\Gordon_time_series\derivatives\abcd-hcp-pipeline','filtered_timeseries.ptseries','time_series.csv')
function ptseries_to_csv(dir, pattern, new_file_name)
    disp(dir)
    fn = getfn(dir, pattern);
    subject_keys = strings(1, numel(fn));
    disp("number of subjects: " + numel(fn));
    for i = 1:numel(fn)
        %disp(fn{i});
        [folder,filename,~] = fileparts(fn{i});
        pat = regexpPattern("NDARINV[A-Za-z0-9]{8}");
        subject_key = extract(filename,pat);
        subject_key = insertAfter(subject_key,"NDAR","_");
        subject_keys(i) = subject_key;
        mycifti = ft_read_cifti(fn{i});
        writematrix(mycifti.ptseries,fullfile(folder,new_file_name));
    end
    T_subjects = array2table(subject_keys(:),'VariableNames',{'SUBJECTKEY'});
    writetable(T_subjects,fullfile(dir,'subjects.csv'));
    T_parcels = array2table(mycifti.label,'VariableNames',{'parcels'});
    writetable(T_parcels,fullfile(dir,'parcels.csv'));
    
end


    