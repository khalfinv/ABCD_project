function pconn_to_csv(dir, pattern, new_file_name)
    fn = getfn(dir, pattern);
    subject_keys = strings(1, numel(fn));
    disp("number of subjects: " + numel(fn));
    for i = 1:4%numel(fn)
        disp(fn{i});
        [folder,filename,ext] = fileparts(fn{i});
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


    
    