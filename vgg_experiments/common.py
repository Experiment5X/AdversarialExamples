import torch
import torchvision.models as models
import torchvision.transforms as transforms

from labels import label_lookup


model = models.vgg16(pretrained=True).eval()
CrossEntropyLoss = torch.nn.CrossEntropyLoss()

loader = transforms.Compose([transforms.ToTensor()])

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
normalize = transforms.Normalize(mean.tolist(), std.tolist())
inv_normalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())


def load_image(image):
    image_tensor = loader(image).float()
    image_normalized = normalize(image_tensor)

    return image_normalized.unsqueeze(0)


def predict(image):
    image_tensor = load_image(image)
    prediction_tensor = model.forward(image_tensor)
    label_index = int(prediction_tensor.argmax().numpy())

    return f'{label_lookup[label_index]} - {prediction_tensor[0][label_index]}'
