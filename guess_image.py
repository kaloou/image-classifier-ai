import os, math, utils
from utils import *

network = utils.load_network()

#--------
# PATHS
#--------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
unsorted_dir = os.path.join(BASE_DIR, "images", "unsorted")
sorted_dir = os.path.join(BASE_DIR, "images", "sorted")
labels_path = os.path.join(BASE_DIR, "labels.txt")

#------------
# FUNCTIONS
#------------

def generate_sorted_data(sorted_dir, network):
    sorted_data = {}

    for type_folder in os.listdir(sorted_dir):
        type_path = os.path.join(sorted_dir, type_folder)

        if os.path.isdir(type_path):
            type_features = []
            
            for filename in os.listdir(type_path):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(type_path, filename)
                    img_features = transform_image(img_path, network)
                    type_features.append(img_features)

            if type_features:
                average_features = [
                    sum(feature[i] for feature in type_features) / len(type_features)
                    for i in range(len(type_features[0]))
                ]
                sorted_data[type_folder] = average_features

    return sorted_data

def read_labels(file_path):
    labels = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_path, img_label = parts[0], parts[1]
                    labels[img_path] = img_label
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
    return labels

def calculate_distance(features1, features2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(features1, features2)))

def guess_image_class(img_path, network, sorted_data):
    img_features = transform_image(img_path, network)
    
    min_distance = float('inf')
    determined_type = None
    
    for sorted_type, sorted_features in sorted_data.items():
        distance = calculate_distance(img_features, sorted_features)
        if distance < min_distance:
            min_distance = distance
            determined_type = sorted_type
            
    return determined_type

def update_labels(filename, correct_label):
    with open(labels_path, 'a') as file:
        file.write(f"{filename} {correct_label}\n")

def move_image_to_class(filename, correct_label):
    source_path = os.path.join(unsorted_dir, filename)
    target_path = os.path.join(sorted_dir, correct_label, filename)
    
    # Create class directory if it doesn't exist
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Move the image
    os.rename(source_path, target_path)

def get_available_classes(sorted_dir):
    classes = []
    for item in os.listdir(sorted_dir):
        if os.path.isdir(os.path.join(sorted_dir, item)):
            classes.append(item)
    return sorted(classes)

#---------------
# MAIN FUNCTION
# --------------

def main():
    print("Loading existing data...")
    # Load sorted data only once at the beginning
    sorted_data = generate_sorted_data(sorted_dir, network)
    read_labels(labels_path)
    
    # Display available classes
    available_classes = get_available_classes(sorted_dir)
    print("\nAvailable classes in the network:")
    print("-" * 30)
    for i, classe in enumerate(available_classes, 1):
        print(f"{i}. {classe}")
    print("-" * 30)
    
    print("\nWelcome to the image guessing system!")
    print("The program will analyze unsorted images and ask you to validate its predictions.")
    
    while True:
        # Check if there are images to process
        images = [f for f in os.listdir(unsorted_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        
        if not images:
            print("\nNo images to process in the unsorted folder.")
            break
            
        # Take the first image
        current_image = images[0]
        img_path = os.path.join(unsorted_dir, current_image)
        
        # Guess the class
        predicted_class = guess_image_class(img_path, network, sorted_data)
        
        print(f"\nImage: {current_image}")
        print(f"Prediction: {predicted_class}")
        
        # Ask for user validation
        while True:
            response = input("Is the prediction correct? (y/n): ").lower()
            if response in ['y', 'n']:
                break
            print("Please answer with 'y' or 'n'")
        
        if response == 'y':
            # Update labels and move the image
            update_labels(current_image, predicted_class)
            move_image_to_class(current_image, predicted_class)
            print(f"Image moved to folder {predicted_class}")
        else:
            # Ask for the correct class
            print("\nAvailable classes:")
            for i, classe in enumerate(available_classes, 1):
                print(f"{i}. {classe}")
            print("New class")
            
            while True:
                choice = input("\nChoose an existing class (number) or type 'n' for a new class: ")
                if choice.lower() == 'n':
                    correct_class = input("Enter the name of the new class: ")
                    available_classes.append(correct_class)
                    break
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(available_classes):
                        correct_class = available_classes[index]
                        break
                    else:
                        print("Invalid class number.")
                except ValueError:
                    print("Please enter a valid number or 'n'.")
            
            update_labels(current_image, correct_class)
            move_image_to_class(current_image, correct_class)
            print(f"Image moved to folder {correct_class}")
        
        if input("\nDo you want to continue? (y/n): ").lower() != 'y':
            break

if __name__ == "__main__":
    main() 