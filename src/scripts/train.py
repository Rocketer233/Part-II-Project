import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import GLUEDataModule
from model import OPTClassifier
import src.models.opt.modeling_opt as modeling_opt
import src.models.llama.modeling_llama as modeling_llama

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
            'OPT': {
                    'model_type': 'OPT', 
                    'model_source': 'facebook/opt-125m',
                    'pretrained_model_class': modeling_opt.OPTForSequenceClassification,
                }, 
            # 'LLaMA': {
            #         'model_type': 'LLaMA', 
            #         'model_source': 'JackFram/llama-160m',
            #         'pretrained_model_class': modeling_llama.LlamaForSequenceClassification,
            #     }
        }

# tasks = [['sst2', 2], ['mnli', 3], ['qnli', 2]]
tasks = [['qnli', 2]]

for _, model in models.items():
    model_name = model['model_source']
    for task_name, num_labels in tasks:
        pretrained_model = model['pretrained_model_class'].from_pretrained(model['model_source'], num_labels=num_labels)


        data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
        data_module.setup("fit")

        classifier = OPTClassifier(pretrained_model)

        logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model['model_type'])
        trainer = pl.Trainer(max_epochs=3, logger=logger)
        trainer.fit(classifier, data_module)