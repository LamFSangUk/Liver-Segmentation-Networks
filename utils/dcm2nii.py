import nibabel
import pydicom
import glob
import numpy as np
import dicom2nifti
import os

def dcm2nii(path_dcm,
            path_dst):

    dicom2nifti.convert_directory(path_dcm, path_dst, compression=True, reorient=True)


if __name__ == "__main__":
    base_path_dcm = "E:/Data/INFINITT/CT/Pancreas-CT/*"
    base_path_dst = "E:/Data/INFINITT/Integrated"

    dcm_list = glob.glob(base_path_dcm)
    for i, dcm_dir in enumerate(dcm_list):
        path_dcm = glob.glob(dcm_dir+"/*/*")

        path_dst = os.path.join(base_path_dst, str(i+1))
        os.mkdir(path_dst)
        dcm2nii(path_dcm[0], os.path.join(path_dst))
