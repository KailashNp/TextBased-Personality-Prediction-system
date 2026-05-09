import sys
sys.path.append("src")

from train_models import train
from predict import predict_personality

print("1. Train Models")
print("2. Predict Personality")

choice = input("Enter choice: ")

if choice == "1":
    train()

elif choice == "2":
    text = input("Enter text: ")
    result = predict_personality(text)
    print("\nPredicted Personality Traits:")
    for k,v in result.items():
        print(f"{k}: {v}")