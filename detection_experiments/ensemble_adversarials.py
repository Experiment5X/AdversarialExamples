import os
from pathlib import Path

directory = 'C:/Users/Adam/Developer/AdversarialExamples/test_images/test_set20'
file_names = os.listdir(directory)

print('Found ' + str(len(file_names)) + ' files')

for filename in file_names:
    if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
        image_path = os.path.join(directory, filename)

        original_file_name = os.path.basename(image_path)
        base_file_name = Path(original_file_name).stem
        adversarial_image_path = f'./ensemble_adversarial_{base_file_name}.png'

        if os.path.exists(adversarial_image_path):
            print('Skipping: ', adversarial_image_path)
            continue

        print('Creating adversarial image from: ', filename)

        os.system(f'python ensemble_adversarial.py {image_path}')
    