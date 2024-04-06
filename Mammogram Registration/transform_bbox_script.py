# Imports
import pandas as pd
import os
import numpy as np
import SimpleITK as sitk

df_contralateral = pd.read_csv('/home/habtamu/Analyze_Mammograms_for_Registration/cropped_contralateral_case.csv')

# A function to transform bbox_roi coordinates
def transform_bbox(input_csv, output_csv):
    df_lesion = pd.read_csv(input_csv)
    df_lesion['transformed_bbox_roi'] = None

    for idx_lesion, row_lesion in df_lesion.iterrows():
        if row_lesion['side'] == 'L':
            for idx_contralateral, row_contralateral in df_contralateral.iterrows():
                if row_lesion['client'] == row_contralateral['client_id'] and row_lesion['side'] != row_contralateral['image_laterality'] and row_lesion['view_position'] == row_contralateral['view_position']:
                    splitted_lesion_image_path = row_lesion['filename'].split('.png')
                    splitted_contralateral_image_path = row_contralateral['filename'].split('.png')
                    transform_parameter_map_path = os.path.join('transform_parameters', splitted_lesion_image_path[0] +'_' + splitted_contralateral_image_path[0] + '.txt')

                    # Read the transformation parameter map from a file
                    transformParameterMap = sitk.ReadParameterFile(transform_parameter_map_path)

                    # Filter the transform parameters
                    params = transformParameterMap['TransformParameters']
                    parameters = [float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4]), float(params[5])]

                    # Filter the center of rotation
                    center = transformParameterMap['CenterOfRotationPoint']
                    center_of_rotation = np.array([float(center[0]), float(center[1])])
                    
                    # Construct the matrix
                    A = np.array([[parameters[0], parameters[1], parameters[4]],
                                [parameters[2], parameters[3], parameters[5]]])  
                    A_3x3 = np.vstack([A, [0, 0, 1]])

                    # Invert the matrix
                    A_inv_3x3 = np.linalg.inv(A_3x3)
                    A_inv = A_inv_3x3[:2, :3]   
                    
                    # The final parameters
                    parameters = [A_inv[0][0], A_inv[0][1], A_inv[1][0], A_inv[1][1], A_inv[0][2], A_inv[1][2]]    

                    # Build the transform object
                    transform = sitk.AffineTransform(2)
                    transform.SetMatrix([parameters[0], parameters[1],
                                        parameters[2], parameters[3]])
                    transform.SetTranslation([parameters[4], parameters[5]])
                    transform.SetCenter(center_of_rotation.tolist())

                    # Bounding box breast area         
                    bbox = row_lesion["bbox"][12:-1]
                    coords = bbox.split(',')
                    r= np.array([0,0,0,0])
                    r_indx = 0
                    for c in coords:
                        aux = c.split('=')
                        r[r_indx]=(int(aux[1]))
                        r_indx +=1
                    
                    # Bounding box roi  
                    bbox_roi = row_lesion["bbox_roi"][12:-1]
                    coords = bbox_roi.split(',')
                    s= np.array([0,0,0,0])
                    s_indx = 0
                    for c in coords:
                        aux = c.split('=')
                        s[s_indx]=(int(aux[1]))
                        s_indx +=1
                    
                    # Make bbox roi coordinates relative to the cropped image
                    s[0] = s[0]-r[0]
                    s[1] = s[1]-r[1]
                    s[2] = s[2]-r[0]
                    s[3] = s[3]-r[1]
                    
                    # The two points of bbox_roi
                    point1 = np.array([s[0], s[1]])
                    point2 = np.array([s[2], s[3]])

                    # Create a nested list of the two points
                    points = np.array([point1, point2]).tolist()

                    # Transform the points
                    transformed_points = [transform.TransformPoint(p) for p in points]
                    tp1 = transformed_points[0]
                    tp2 = transformed_points[1]

                    # Add transformed bbox to the csv
                    t_bbox_roi = 'BoundingBox(x1='+ str(tp1[0]) + ', y1='+ str(tp1[1]) + ', x2=' + str(tp2[0]) + ', y2=' + str(tp2[1]) + ')'
                    df_lesion.at[idx_lesion, 'transformed_bbox_roi'] = t_bbox_roi

    df_lesion.to_csv(output_csv, index=False)

# Transform bbox_roi coordinates of the training set
transform_bbox('/home/habtamu/Analyze_Mammograms_for_Registration/lesion_training_set.csv', 'transformed_lesion_training_set.csv')
# Transform bbox_roi coordinates of the validation set
transform_bbox('/home/habtamu/Analyze_Mammograms_for_Registration/lesion_validation_set.csv', 'transformed_lesion_validation_set.csv')