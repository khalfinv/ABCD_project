%This function find all .pconn.nii (correlation matrix in CIFTI format) 
%files in a root directory and convert to csv files. 
%Input:
    % dir: root directory
    % pattern: files patterns
    % new_file_name: output file name
%Output:
    % correlation matrix in csv format in each sub directory
    % list of subject keys in csv format in root folder
%Run example: 
%pconn_to_csv('C:\Users\skhalfin.TD-ST\Desktop\ABCD_project\abcd-s3-downloader\Connectivity_Matrix\derivatives\abcd-hcp-pipeline','5min_conndata-network_connectivity.pconn','mat.csv')
function pconn_to_csv(dir, pattern, new_file_name)
    disp(dir)
    fn = getfn(dir, pattern);
    subject_keys = strings(1, numel(fn));
    disp("number of subjects: " + numel(fn));
    for i = 1:numel(fn)
        disp(fn{i});
        [folder,filename,~] = fileparts(fn{i});
        pat = regexpPattern("NDARINV[A-Za-z0-9]{8}");
        subject_key = extract(filename,pat);
        subject_key = insertAfter(subject_key,"NDAR","_");
        subject_keys(i) = subject_key;
        mycifti = ft_read_cifti(fn{i});
        T_corr_mat = array2table(mycifti.pconn,'VariableNames',mycifti.label);
        writetable(T_corr_mat,fullfile(folder,new_file_name));
    end
    T = array2table(subject_keys(:),'VariableNames',{'SUBJECTKEY'});
    writetable(T,fullfile(dir,'subjects.csv'));
end


    
    