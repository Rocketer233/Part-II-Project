import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.datamodule import GLUEDataModule
from model import OPTClassifier
import src.models.opt.convert_checkpoint_opt as convert_checkpoint_opt
import src.models.opt.modeling_opt as modeling_opt
import src.models.opt.modeling_opt_gqa as modeling_opt_gqa
import src.models.llama.convert_checkpoint_llama as convert_checkpoint_llama
import src.models.llama.modeling_llama as modeling_llama
import src.models.llama.modeling_llama_gqa as modeling_llama_gqa

# tensorboard --logdir lightning_logs/ --port 6006

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models = {
            'OPT': {
                    'model_type': 'OPT-GQA', 
                    'model_source': 'facebook/opt-125m',
                    'pretrained_model_class': modeling_opt.OPTForSequenceClassification,
                    'model_class': modeling_opt_gqa.OPTForSequenceClassification,
                    'convert_ckpt': convert_checkpoint_opt
                }, 
            'LLaMA': {
                    'model_type': 'LLaMA-GQA', 
                    'model_source': 'JackFram/llama-160m',
                    'pretrained_model_class': modeling_llama.LlamaForSequenceClassification,
                    'model_class': modeling_llama_gqa.LlamaForSequenceClassification,
                    'convert_ckpt': convert_checkpoint_llama
                }
        }

tasks = ['sst2', 'mnli', 'qnli']
# tasks = ['sst2']

groups_idx = [ 
                [[0, 5], [2], [1, 7], [6], [8], [4, 10, 11], [9], [3]],
                [[4], [3], [9], [2], [0], [10, 11], [5], [1, 6], [7, 8]],
                [[5], [1, 6], [3, 4], [8, 10], [2], [0, 7], [9, 11]],
                [[6], [0], [1, 2, 7], [10], [5, 9], [4], [8, 11], [3]],
                [[8], [0, 10, 11], [2], [4, 6], [1], [3, 7], [5, 9]],
                [[11], [10], [5], [3, 4], [6, 8], [0], [2, 7], [1, 9]],
                [[1, 6, 10], [5, 7], [4], [0, 9, 11], [8], [3], [2]],
                [[0, 6], [2, 3, 5, 8], [4], [1], [11], [9], [10], [7]],
                [[8], [0, 3, 4, 5], [6, 7], [9], [2], [10], [11], [1]],
                [[0], [11], [8], [1, 10], [2, 5], [3], [9], [6, 7], [4]],
                [[0], [2], [5, 8], [6], [3], [1], [7], [4], [9], [10, 11]],
                [[10], [0], [11], [6, 7], [2, 3, 5], [1, 8], [4], [9]]
            ]

for _, model in models.items():
    model_name = model['model_source']
    for task_name in tasks:
        pretrained_model = model['pretrained_model_class'].from_pretrained(model['model_source'])
        pretrained_model.config.groups_idx = groups_idx
        
        gqa_model = model['model_class'](pretrained_model.config)
        state = pretrained_model.state_dict()
        gqa_model.load_state_dict(model['convert_ckpt'].mha2gqa(state, groups_idx, num_heads=12, transpose_layer=True))


        data_module = GLUEDataModule(model_name=model_name, task_name=task_name)
        data_module.setup("fit")

        classifier = OPTClassifier(gqa_model)

        logger = TensorBoardLogger("lightning_logs", name=task_name + "-" + model['model_type'])
        trainer = pl.Trainer(max_epochs=3, logger=logger)
        trainer.fit(classifier, data_module)