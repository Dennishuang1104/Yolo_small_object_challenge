import json
from pathlib import Path
from utils import DataPreProcessing




# for folder in ['valid']:  # convert order: train -> val, you can also add 'test' if you want to use it
#     convert_annotations(
#         types = folder,
#         annotation_path = f'datasets/SkyFusion/annotations/{folder}.json',  # annotation file
#         target_path = f'datasets/SkyFusion/{folder}/labels',  # labels folder
#     )



if __name__ == '__main__':
    service = DataPreProcessing(data_type='valid')
    service.process_traindata()