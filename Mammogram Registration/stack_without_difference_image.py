import pandas as pd
import os
import numpy as np
import cv2

df_lesion = pd.read_csv('/home/habtamu/Analyze_Mammograms_for_Registration/lesion_case.csv')
df_contralateral = pd.read_csv('/home/habtamu/Analyze_Mammograms_for_Registration/cropped_contralateral_case.csv')

for idx_lesion, row_lesion in df_lesion.iterrows():
    for idx_contralateral, row_contralateral in df_contralateral.iterrows():
        if row_lesion['client'] == row_contralateral['client_id'] and row_lesion['side'] != row_contralateral['image_laterality'] and row_lesion['view_position'] == row_contralateral['view_position']:
            lesion_image_path = os.path.join("/home/robert/data/mammo/iceberg_selection/HOLOGIC/ffdm/st"+"{0:03}".format(row_lesion["subtype"]), row_lesion['filename'])
            contralateral_image_path = os.path.join('/home/habtamu/Analyze_Mammograms_for_Registration/cropped_contralateral_images', row_contralateral['filename'])
            splitted_lesion_image_path = row_lesion['filename'].split('.png')
            registered_image_path = os.path.join('registered_images', splitted_lesion_image_path[0] + '_' + row_contralateral['filename'])
            stacked_image_path = os.path.join('stacked_without_difference_image', row_lesion['filename'])

            # Load the grayscale images
            if row_lesion['side'] == 'R':
                lesion_image = cv2.imread(lesion_image_path, cv2.IMREAD_GRAYSCALE)
                normal_image = cv2.imread(registered_image_path, cv2.IMREAD_GRAYSCALE)
            else:
                lesion_image = cv2.imread(registered_image_path, cv2.IMREAD_GRAYSCALE)
                normal_image = cv2.imread(contralateral_image_path, cv2.IMREAD_GRAYSCALE)
            
            # Stack the images
            merged_image = cv2.merge((lesion_image, lesion_image, normal_image)) 
            
            # Save the resulting image
            cv2.imwrite(stacked_image_path, merged_image)