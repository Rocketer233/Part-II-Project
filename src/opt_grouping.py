import torch
import src.models.opt.modeling_opt as modeling_opt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def groups(size, tot=12):
    return [list(range(i, min(i + size, tot))) for i in range(0, tot, size)]

def neighbour_grouping(group_size):
    return [groups(group_size) for _ in range(12)]

def kv_grouping(model, avg):
    state = model.state_dict()
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    nums_groups = [6, 4, 3, 2]
    groupings = [[] for _ in range(len(nums_groups))]
    for layer_id in range(num_layers):
        weights = [[] for _ in range(num_heads)]
        bias = [[] for _ in range(num_heads)]
        
        for t in ('k', 'v'):
            mat_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.weight'
            layer = state[mat_name].transpose(0, 1)
            for i, x in enumerate(torch.tensor_split(layer, num_heads, dim=1)):
                weights[i].append(x)
            
            mat_name = f'model.decoder.layers.{layer_id}.self_attn.{t}_proj.bias'
            layer = state[mat_name]
            for i, x in enumerate(torch.tensor_split(layer, num_heads)):
                bias[i].append(x)
        
        vector = []
        for i in range(num_heads):
            if avg:
                weights[i] = torch.mean(torch.stack(weights[i]), dim=0)
                bias[i] = torch.mean(torch.stack(bias[i]), dim=0)
                print(weights[i].shape, bias[i].shape)
                vector.append(torch.concatenate(
                            [weights[i].flatten(), bias[i].flatten()]))
            else:
                vector.append(torch.concatenate(
                            [weights[i][0].flatten(), 
                            bias[i][0].flatten(),
                            weights[i][1].flatten(), 
                            bias[i][1].flatten()]))
        
        similarity = cosine_similarity(vector, vector)
        distance = 1 - similarity
        Z = linkage(distance, 'ward')
        
        j = 0
        tree = set(range(num_heads))
        nodes = [[i] for i in range(num_heads)]
        for i, z in enumerate(Z):
            x, y = int(z[0]), int(z[1])
            nodes.append(nodes[x] + nodes[y])
            tree.remove(x)
            tree.remove(y)
            tree.add(num_heads + i)
            if j < len(nums_groups) and len(tree) == nums_groups[j]:
                groups = [nodes[i] for i in tree]
                groupings[j].append(groups)
                j += 1
    
    for grouping in groupings:
        print(grouping)
    
    
    # plt.figure(f_avgigsize=(25, 25))
    # dendrogram(Z)
    # print(Z)
    # plt.savefig("dendrogram.png")
    # # print(similarity)
    # # for mat in key_bias[0]:
    # #     print(mat.shape)
    
    return groupings

# def random_grouping():
    

if __name__ == "__main__":
    model_name = 'facebook/opt-125m'
    num_labels = 2
    model = modeling_opt.OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    kv_grouping(model, avg=True)
    