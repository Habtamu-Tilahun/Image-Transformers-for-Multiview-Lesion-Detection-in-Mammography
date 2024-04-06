# Imports
import SimpleITK as sitk
import pandas as pd
import os

# Registration
df_lesion = pd.read_csv('/home/habtamu/Analyze_Mammograms_for_Registration/lesion_case.csv')
df_contralateral = pd.read_csv('/home/habtamu/Analyze_Mammograms_for_Registration/cropped_contralateral_case.csv')
parameterMap = sitk.ReadParameterFile('Parameters_Affine.txt')

for idx_lesion, row_lesion in df_lesion.iterrows():
    for idx_contralateral, row_contralateral in df_contralateral.iterrows():
        if row_lesion['client'] == row_contralateral['client_id'] and row_lesion['side'] != row_contralateral['image_laterality'] and row_lesion['view_position'] == row_contralateral['view_position']:
            lesion_image_path = os.path.join("/home/robert/data/mammo/iceberg_selection/HOLOGIC/ffdm/st"+"{0:03}".format(row_lesion["subtype"]), row_lesion['filename'])
            contralateral_image_path = os.path.join('/home/habtamu/Analyze_Mammograms_for_Registration/cropped_contralateral_images', row_contralateral['filename'])
            splitted_lesion_image_path = row_lesion['filename'].split('.png')
            splitted_contralateral_image_path = row_contralateral['filename'].split('.png')
            registered_image_path = os.path.join('registered_images', splitted_lesion_image_path[0] + '_' + row_contralateral['filename'])
            transform_parameter_map_path = os.path.join('transform_parameters', splitted_lesion_image_path[0] +'_' + splitted_contralateral_image_path[0] + '.txt')

            if row_lesion['side'] == 'R':
                fixedImage = sitk.ReadImage(lesion_image_path)
                movingImage = sitk.ReadImage(contralateral_image_path)
            else:
                fixedImage = sitk.ReadImage(contralateral_image_path)
                movingImage = sitk.ReadImage(lesion_image_path)
            
            elastixImageFilter = sitk.ElastixImageFilter()
            elastixImageFilter.SetFixedImage(fixedImage)
            elastixImageFilter.SetMovingImage(movingImage)
            elastixImageFilter.SetParameterMap(parameterMap)
            elastixImageFilter.SetOutputDirectory('registration_output')

            elastixImageFilter.Execute()
            resultImage = elastixImageFilter.GetResultImage()
            transformParameterMap = elastixImageFilter.GetTransformParameterMap()[0]

            # Cast the pixel type to unsigned char
            resultImage = sitk.Cast(resultImage, sitk.sitkUInt8)

            # Write the result image
            sitk.WriteImage(resultImage, registered_image_path)

            # Write the transform parameter map
            sitk.WriteParameterFile(transformParameterMap, transform_parameter_map_path)