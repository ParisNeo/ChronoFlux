import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from .model import Generator

class AgeTransformer:
    def __init__(self, repo_id="ParisNeo/ChronoFlux", model_file="age_transformer.pth"):
        """
        Initialize the Age Transformer model.
        
        Args:
            repo_id (str): Hugging Face repository ID
            model_file (str): Name of the model file in the repository
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(repo_id, model_file)
        self.transform = self._get_transforms()
        
    def _load_model(self, repo_id, model_file):
        """
        Load model from Hugging Face Hub.
        
        Args:
            repo_id (str): Hugging Face repository ID
            model_file (str): Name of the model file
            
        Returns:
            Generator: Loaded and configured generator model
        """
        model_path = hf_hub_download(repo_id, model_file)
        model = Generator().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def _get_transforms(self):
        """
        Get image transformations for preprocessing.
        
        Returns:
            transforms.Compose: Composition of image transformations
        """
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def _postprocess_image(self, tensor):
        """
        Convert model output tensor to PIL Image.
        
        Args:
            tensor (torch.Tensor): Output tensor from model
            
        Returns:
            PIL.Image: Processed output image
        """
        tensor = tensor.squeeze(0).cpu()
        tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
        return transforms.ToPILImage()(tensor)
    
    @torch.no_grad()
    def transform(self, image, target_age, output_size=256):
        """
        Transform an image to the target age.
        
        Args:
            image (PIL.Image): Input image
            target_age (int): Target age (0-100)
            output_size (int): Desired output size
            
        Returns:
            PIL.Image: Transformed image
        """
        # Validate inputs
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if not (0 <= target_age <= 100):
            raise ValueError("Target age must be between 0 and 100")
        
        # Preprocess
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        age_tensor = torch.tensor([target_age/100], device=self.device)
        
        # Inference
        output_tensor = self.model(image_tensor, age_tensor)
        
        # Postprocess
        output_image = self._postprocess_image(output_tensor)
        
        # Resize if needed
        if output_size != 128:
            output_image = output_image.resize((output_size, output_size))
            
        return output_image
    
    def batch_transform(self, images, target_ages, output_size=256):
        """
        Transform multiple images to their target ages.
        
        Args:
            images (list): List of PIL Images
            target_ages (list): List of target ages (0-100)
            output_size (int): Desired output size
            
        Returns:
            list: List of transformed PIL Images
        """
        if len(images) != len(target_ages):
            raise ValueError("Number of images must match number of target ages")
            
        return [self.transform(img, age, output_size) 
                for img, age in zip(images, target_ages)]
    
    def create_age_progression(self, image, ages=None, output_size=256):
        """
        Create age progression/regression sequence.
        
        Args:
            image (PIL.Image): Input image
            ages (list): List of target ages (default: [0, 10, 20, ..., 100])
            output_size (int): Desired output size
            
        Returns:
            list: List of transformed PIL Images
        """
        if ages is None:
            ages = list(range(0, 101, 10))
            
        return self.batch_transform([image]*len(ages), ages, output_size)

def load_model_from_hub(repo_id="ParisNeo/ChronoFlux", model_file="age_transformer.pth"):
    """
    Convenience function to load model from Hugging Face Hub.
    
    Args:
        repo_id (str): Hugging Face repository ID
        model_file (str): Name of the model file
        
    Returns:
        AgeTransformer: Initialized AgeTransformer instance
    """
    return AgeTransformer(repo_id, model_file)