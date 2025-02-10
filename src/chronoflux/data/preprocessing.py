from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader

def get_transforms(resolution=128):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def create_dataloader(dataset_name="utk-face", batch_size=32, resolution=128):
    dataset = load_dataset(dataset_name)
    
    def preprocess(examples):
        examples["pixel_values"] = [get_transforms(resolution)(img.convert("RGB")) for img in examples["image"]]
        examples["age"] = [age / 100 for age in examples["age"]]
        return examples
    
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type="torch", columns=["pixel_values", "age"])
    
    return DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )