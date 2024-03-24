import gdown
import os

class FilesDownloader:
    def __init__(self):
        self.qa_model_name = "multitask-qg-ag.ckpt"
        self.qa_model_id = "1ZhLe_nO_1VxIoVj5zRB61Ql-5_l0ZTJi"
        self.qa_model_path = "app/ml_models/question_generation/models/multitask-qg-ag.ckpt"
        self.sense2vec_model_path = "app/ml_models/sense2vec_distractor_generation/data/"
        self.distractor_model_path = "app/ml_models/distractor_generation/models/race-distractors.ckpt"
        self.trueorfalse_model_path = "app/ml_models/true_or_false/models/"
        self.sense2vec_model_id = "1QVlAI-JxMvE01oLGp0SV7GguKgEJBD15"
        self.distractor_model_id = "1e8VjG-6ccGp5QEbNl-eZRisP9BGKILNk"
        self.trueorfalse_model_id = "1kvDsmuPtNXjVHfs6MzQcR0IAr66Gf94s"
        self.download_qa_model()
        self.download_sense2vec_model()
        self.download_distractor_model()
        self.download_trueorfalse_model()

    def download_qa_model(self):
        if not os.path.exists(self.qa_model_path):
            gdown.download(id=self.qa_model_id, output=self.qa_model_path)
        print("QA model downloaded successfully!")

    def download_sense2vec_model(self):
        if not os.path.exists(self.sense2vec_model_path + "s2v_old"):
            gdown.download_folder(id=self.sense2vec_model_id, output=self.sense2vec_model_path)
        print("Sense2Vec model downloaded successfully!")

    def download_distractor_model(self):
        if not os.path.exists(self.distractor_model_path):
            gdown.download(id=self.distractor_model_id, output=self.distractor_model_path)
        print("Distractor model downloaded successfully!")
        
    def download_trueorfalse_model(self):
        if not os.path.exists(self.distractor_model_path + "BoolQ"):
            gdown.download_folder(id=self.trueorfalse_model_id, output=self.distractor_model_path)
        print("True or False model downloaded successfully!")
