import pickle
import os
from src.logger import Instance as Logger
import argparse

class DatasetValidator:
    def __init__(self, annotations, questions, img_dir):
        self.annotations = annotations
        self.questions = questions
        self.img_dir = img_dir
        self.module_name = "DatasetValidator"
    def validate_dataset(self):
        try:
            with open(self.annotations, 'rb') as f:
                annotaions = pickle.load(f)
            with open(self.questions,'rb') as f:
                questions =  pickle.load(f)
            for item in annotaions:
                q = questions[item['question_id']]
                img_path = os.path.join(self.img_dir, f"{item['img_id']}.pickle")
                with open(img_path, 'rb') as f:
                    a = f
            Logger.log(self.module_name, "Dataset validation completed successfully.")
        except Exception as e:
            Logger.log(self.module_name, f"Invalid dataset with error {str(e)}")
        return
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose instances of fsvqa dataset")
    parser.add_argument('--annotations', help='annotations path')
    parser.add_argument('--questions', help='questions path')
    parser.add_argument('--img_dir', help='number of instances')
    args = parser.parse_args()
    dataset_validator = DatasetValidator(args.annotations, args.questions, args.img_dir)
    dataset_validator.validate_dataset()