import argparse
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms # تمت الإضافة لتوليد البيانات

# ملاحظة: تم إيقاف استيراد train_model القديمة لأننا سنستخدم الدالة الجديدة المدمجة أدناه
# from src.core.train import train_model 
from src.data.dataio import get_mnist_loaders, get_sudoku_loaders
from src.evaluate.evaluate import evaluate_model
from src.model.model import ConvNet, ResNet152
from src.preprocess.build_features import process_sudoku_image

# --- كلاس مساعد لتطبيق توليد البيانات (Data Augmentation) ---
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# --- دالة التدريب الاحترافية (تتضمن Early Stopping و LR Scheduler) ---
def train_model_with_early_stopping(model, train_loader, criterion, optimizer, device, num_epochs=10, patience=5):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        
        # تحديث سرعة التعلم
        scheduler.step(epoch_loss)
        
        # التوقف المبكر (Early Stopping)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n🛑 التوقف المبكر! النموذج لم يتحسن منذ {patience} جولات.")
                break

    print("\n✅ انتهى التدريب. جاري استرجاع أفضل نسخة من النموذج...")
    model.load_state_dict(best_model_wts)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train digit recognition models")
    parser.add_argument("--model", type=str, choices=["convnet", "resnet"], required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "sudoku"], required=True)
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs") # تم رفع الافتراضي إلى 50
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--finetune", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--eval-both", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    is_resnet = args.model == "resnet"
    model = ResNet152().to(device) if is_resnet else ConvNet().to(device)
    print(f"Model: {'ResNet152' if is_resnet else 'ConvNet'}")

    if args.finetune:
        try:
            model.load_state_dict(torch.load(args.finetune, map_location=device))
            print(f"Loaded pretrained model: {args.finetune}")
        except FileNotFoundError:
            print(f"Error: Pretrained model not found at {args.finetune}")
            return

    # --- إعدادات توليد البيانات (Data Augmentation) ---
    # هذه التحويلات تعمل على الـ Tensors مباشرة لإمالتها وإزاحتها
    augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(15), # إمالة حتى 15 درجة
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # إزاحة 10%
    ])

    # Load dataset
    if args.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders("data/raw/MNIST", batch_size=args.batch_size, for_resnet=is_resnet)
        dataset_name = "MNIST"
    else:
        sudoku_train_dirs = [
            "data/raw/sudoku/v1_training/v1_training",
            "data/raw/sudoku/v2_train/v2_train",
        ]
        sudoku_test_dir = "data/raw/sudoku/v1_test/v1_test"
        train_loader, test_loader = get_sudoku_loaders(
            sudoku_train_dirs,
            cell_processor=lambda img: process_sudoku_image(img, invert_for_mnist_compatibility=True),
            test_dir=sudoku_test_dir,
            batch_size=args.batch_size,
            for_resnet=is_resnet,
        )
        dataset_name = "Sudoku"

        if train_loader is None or test_loader is None:
            print("Failed to load Sudoku data. Exiting.")
            return

    # تطبيق الـ Augmentation على بيانات التدريب فقط
    augmented_train_dataset = AugmentedDataset(train_loader.dataset, transform=augmentation_transforms)
    train_loader = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Dataset: {dataset_name} (مع Data Augmentation)")
    print(f"Training up to {args.epochs} epochs with lr={args.lr}")

    if args.finetune:
        print(f"\nPerformance before fine-tuning:")
        evaluate_model(model, test_loader)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # استخدام الدالة الجديدة المدمجة للتدريب
    model = train_model_with_early_stopping(
        model, 
        train_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=args.epochs,
        patience=7 # سيتوقف التدريب إذا لم يتحسن الأداء لـ 7 جولات متتالية
    )

    # Evaluate after training
    print(f"\nPerformance on {dataset_name} after training:")
    evaluate_model(model, test_loader)

    # Cross-evaluation
    if args.eval_both:
        if args.dataset == "mnist":
            print("\nEvaluating on Sudoku test set:")
            _, sudoku_loader = get_sudoku_loaders(
                "data/raw/sudoku/v1_test/v1_test",
                cell_processor=lambda img: process_sudoku_image(img, invert_for_mnist_compatibility=True),
                for_resnet=is_resnet,
            )
            evaluate_model(model, sudoku_loader)
        else:
            print("\nEvaluating on MNIST test set:")
            _, mnist_loader = get_mnist_loaders("data/raw/MNIST", for_resnet=is_resnet)
            evaluate_model(model, mnist_loader)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = args.output if args.output else f"models/best_{'resnet' if is_resnet else 'convnet'}_{dataset_name.lower()}.pkl"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

if __name__ == "__main__":
    main()
