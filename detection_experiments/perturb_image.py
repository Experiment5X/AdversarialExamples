import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
from predict_yolo import predict as predict_yolo
from predict_rcnn import predict as predict_rcnn


def add_noise(noise_amount):
    def modify(image_array):
        noised_image = image_array + np.random.rand(*image_array.shape) * noise_amount
        return noised_image

    return modify


def adjust_contrast(enhance_amount):
    def modify(image_array):
        image = Image.fromarray(image_array)
        adjusted_image = ImageEnhance.Contrast(image).enhance(enhance_amount)

        return np.array(adjusted_image)

    return modify


def grayscale(enhance_amount):
    def modify(image_array):
        image = Image.fromarray(image_array)
        adjusted_image = ImageEnhance.Color(image).enhance(enhance_amount)

        return np.array(adjusted_image)

    return modify


def perturb_image(image_path, out_path, perturb_func):
    image = Image.open(image_path)
    image_array = np.array(image)

    modified_image_array = perturb_func(image_array)
    normalized_image_array = (
        (modified_image_array - modified_image_array.min())
        / (modified_image_array.max() - modified_image_array.min())
        * 255
    )

    modified_image = Image.fromarray(np.uint8(normalized_image_array))
    modified_image.save(out_path)

    return modified_image


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(
            'Usage: python3 perturb_image.py IMAGE_PATH PERTURBATION PERTURB_PARAMETER'
        )

    perturb_functions = {
        'noise': add_noise,
        'contrast': adjust_contrast,
        'grayscale': grayscale,
    }

    image_path = sys.argv[1]
    perturb_function_name = sys.argv[2]
    perturb_parameter = float(sys.argv[3])

    if perturb_function_name not in perturb_functions:
        print(f'Unknown perturb function "{perturb_function_name}"')
        exit(-1)

    base_file_name = Path(image_path).stem
    perturb_param_str = (
        str(int(perturb_parameter))
        if perturb_parameter.is_integer()
        else f'{perturb_parameter:.3f}'.replace('.', '_')
    )
    out_image_path = f'../test_images/perturbed/{base_file_name}_{perturb_function_name}_{perturb_param_str}.png'

    perturb_func = perturb_functions[perturb_function_name](perturb_parameter)
    perturb_image(image_path, out_image_path, perturb_func)

    print(
        f'Perturbed the image with "{perturb_function_name}" and wrote it to '
        + out_image_path
    )
    predict_yolo(out_image_path)
    predict_rcnn(out_image_path)
