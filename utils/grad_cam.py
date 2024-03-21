import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.feature_gradients = None
        self.feature_activations = None

        # Register hook to the target layer
        def backward_hook(module, grad_input, grad_output):
            self.feature_gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.feature_activations = output

        # Find the target layer and register hooks
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    def preprocess_image(img_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel to create a 3-channel image
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(img_path).convert('RGB')  # Convert grayscale to RGB if needed
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor.requires_grad_(True)


    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        model_output = self.model(input_tensor)
        if target_class is None:
            target_class = model_output.argmax().item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot_output = torch.zeros(model_output.shape, dtype=torch.float)
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)

        # Generate CAM
        gradients = self.feature_gradients.data.numpy()[0]
        activations = self.feature_activations.data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

    def overlay_cam_on_image(self, img_path, cam):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.4 + img
        return superimposed_img

# Example usage:
    

if __name__ == "__main__":
    # Load model
    model = models.resnet18(pretrained=True)

    # Initialize GradCAM with the model and target layer
    grad_cam = GradCAM(model, 'layer4')

    # Preprocess the image and generate CAM
    # Correctly define the path to your image file
    img_path = "F:/UNIVERCITY/sharifian/t1/datasets/tumor_dataset/all/img/5.png"

    # Call preprocess_image with the correct image path
    img_tensor = grad_cam.preprocess_image(img_path)


    cam = grad_cam.generate_cam(img_tensor)

    # Overlay CAM on image
    superimposed_img = grad_cam.overlay_cam_on_image('F:/UNIVERCITY/sharifian/t1/datasets/tumor_dataset/5.png', cam)
    cv2.imwrite('grad_cam.jpg', superimposed_img)
