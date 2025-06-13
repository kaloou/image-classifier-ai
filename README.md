# Image Classification System

This project uses a neural network to classify images into different categories.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kaloou/image-classifier-ai.git
cd image-classifier-ai
```

2. Create a virtual environment:

```bash
python3.10 -m venv venv
```

3. Activate the virtual environment:

- On macOS/Linux:

```bash
source venv/bin/activate
```

- On Windows:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
.
├── images/
│   ├── unsorted/    # Images to classify
│   └── sorted/      # Classified images by category
├── labels.txt       # Labels file
├── utils.py         # Utility functions
└── guess_image.py   # Main program
```

## Usage

1. Place images to classify in the `images/unsorted/` folder
2. Run the program:

```bash
python guess_image.py
```

3. Follow the on-screen instructions to validate or correct predictions

## Notes

- Model improvements are taken into account on the next program launch
- New categories can be created during use
