% 1. We want to create masks of the Gordon atlas- running with for on each
% voxel number- turning it to 1, and all the other numbers to 0
% 2. we will multiply the mask by the anatomical atlas
% 3. We will use unique to find the unique areas that are present in our
% ROI mask
% 4. Using a txt file containing the names of the regions in your
% anatomical atlas- we will convert the numbers to ROI names
% 5. creating an excel
func_atlas_info=load_nii('gordon_Parcels_MNI_111.nii')
func_atlas=func_atlas_info.img;
anat_atlas_info=load_nii('AAL_space-MNI152NLin6_res-1x1x1.nii')
anat_atlas=anat_atlas_info.img;
slice=squeeze(anat_atlas(:,100,:));
slice2=squeeze(func_atlas(:,100,:));

anat_labels = readtable('aal_labels.csv');
display(anat_labels(ismember(anat_labels.Var1,1),:).Var2)

% figure
% subplot(2,1,1)
% imagesc(slice); axis image
% 
% subplot(2,1,2)
% imagesc(slice2); axis image
% which=unique(anat_atlas)
func_regions=unique(func_atlas)
func_regions(1)=[]
number_of_regions = length(func_regions)
anat_regions = strings(number_of_regions,1)
out_table = table(func_regions,anat_regions)
for i=1:number_of_regions
    mask = zeros(size(func_atlas)); % creating a zero matrix in the dimensions of the atlas
    mask(func_atlas == i) = 1;
    regions_mask=mask.*anat_atlas;
    numbers=unique(regions_mask)
    numbers(1)=[]
    array_labels = []
    for j=1:length(numbers)
        label = anat_labels(ismember(anat_labels.Var1,numbers(j)),:).Var2
        %if (contains(label,"Gray Matter"))
        array_labels = [array_labels,label]
        %end
    end
    if(isempty(array_labels) == 0)
        str_labels = strjoin(array_labels,', ')
        out_table{i,2}={str_labels}
    end
end

writetable(out_table,'func_anat_overlay.csv')


