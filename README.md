# COMP472-Project: Artificial Intelligence

Project done for COMP 472: Artificial Intelligence in the Summer 2024 semester

# Link to the Report



# Team Members:

| Name               | Student ID |
| ------------------ | ---------- |
|   Daniel Secelean  |  40227919  |
|   Bhavya Mehta     |  40251474  |
|||

# Instructions

## Dataset

- The dataset has two folders: Training and Testing
- Each folder has 4 classes (Neutral, Focused, Angry and Happy). The training folder has minimum of 400 images in each one of its classes, while the testing folder has minimum of 100 images in each one of its classes

- Link to the dataset : 

## Running instructions

- The repository contains a data_preprocessing.py file, which is responsible for resizing and adjusting our training data.
- Clone the repository
- Install the libraries by doing 'pip install -r Requirements.txt'
- Create a virual environment by running
  `$ python -m venv "environment name"`
- Run the train.py script to train the model by using "python train.py". You will be prompted to select the model to train:

Enter 1 for the main model (DeeperCNN).
Enter 2 for Variant 1 (Variant1CNN).
Enter 3 for Variant 2 (Variant2CNN).

- After training, run the evaluate.py "python evaluate.py" script to evaluate the model on the test dataset:

You will be prompted to select the model to evaluate. Follow the prompt:

Enter 1 for the main model (DeeperCNN).
Enter 2 for Variant 1 (Variant1CNN).
Enter 3 for Variant 2 (Variant2CNN).


