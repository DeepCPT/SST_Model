import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad, softplus
import torch.nn.utils.rnn as rnn_utils
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
#from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader, Dataset
#from torchtext.vocab import build_vocab_from_iterator
#import torchtext.datasets as datasets
#import spacy
import pickle
import GPUtil
import warnings
from collections import Counter
#from torch.utils.data.distributed import DistributedSampler
#import torch.distributed as dist
#import torch.multiprocessing as mp
#from torch.nn.parallel import DistributedDataParallel as DDP




warnings.filterwarnings("ignore")
RUN_EXAMPLES = False

# Universial setting of the padding value
pad_idx = -1e2
torch.manual_seed(0) # Seed



def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


def Genral_Evaluate(G,P):
    relative_difference = torch.abs((P - G))
    MAE=relative_difference.mean()
    MAE_VAR = relative_difference.var()
    mask = torch.abs(G)>0.01 # exclude the denominators that are extremely small
    MAPE = (relative_difference[mask]/torch.abs(G[mask])).mean()

    return MAE,MAE_VAR,MAPE


def Perference_Evaluate(G,P):
    mask = G != 0
    relative_difference = torch.abs((P - G))
    filtered_relative_difference = relative_difference[mask]
    return str(filtered_relative_difference.mean().item())

def Shock_Evaluate(G,P):
    mask = torch.abs(G)!=0
    relative_difference = torch.abs((P - G))
    filtered_relative_difference = relative_difference[mask]
    return str(filtered_relative_difference.mean().item())

def Coeff_Evaluate_Consumser(G,P,Consumer,type):
    consumer_index, indices = torch.unique(Consumer, sorted=False, return_inverse=True)

    if type!="Shock" or type!="Post_Click_Shock":
        if type=="Updated_Pre":
            for i, unique_val in enumerate(consumer_index):
                index = Consumer == unique_val
                c_g = G[index]
                c_p = P[index]
                print("Consumer " + str(unique_val.item()) + " " + type + "   MAE: " + Shock_Evaluate(c_g, c_p))
        else:
            for i, unique_val in enumerate(consumer_index):
                index=torch.nonzero(Consumer == unique_val, as_tuple=True)[0][0]
                c_g = G[index]
                c_p = P[index]
                print("Consumer "+ str(unique_val.item())+" "+type+"   MAE: "+Perference_Evaluate(c_g,c_p))
    else:
        for i, unique_val in enumerate(consumer_index):
            index=Consumer==unique_val
            c_g = G[index]
            c_p = P[index]
            print("Consumer " + str(unique_val.item()) + " " + type + "   MAE: " + Shock_Evaluate(c_g, c_p))


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder,preference_embedding,cost_embedding,preference_shock,ru_learner,cu_learner,src_embed, tgt_embed,loss_generator,task):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.preference_embedding = preference_embedding
        self.cost_embedding = cost_embedding
        self.preference_shock=preference_shock
        self.ru_learner = ru_learner
        self.cu_learner = cu_learner
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.loss_generator = loss_generator
        self.task = task

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self,src,src_len,src_mask,post_click,tgt_mask,expo_list,expo_mask,cost_feature,click_index,click_seq,purchase_mask,tgt_len,cate_preference,updated_preferecne,shock_data,cost_coeff,post_click_coeff,preference_shock_flag,consumer_list,hyper_sigma):

        memory = self.encode(src, src_mask)

        # Learn consumer original preference and updated preference
        w_p_original = self.preference_embedding(consumer_list)
        shock = self.preference_shock(memory,src, src_mask,src_len)
        shock=shock*(1-preference_shock_flag).view(-1, 1)
        w_p = w_p_original + shock

        # Learn consumer cost coefficients
        w_c = self.cost_embedding(consumer_list)

        # Learn reserved utility of items in the list
        reserved_utility = self.ru_learner(w_p, w_c, expo_list, cost_feature)

        # Learn pre_click and post_click utility of clicked items
        pre_click_utility = torch.matmul(w_p.unsqueeze(1), click_seq.transpose(1, 2)).squeeze(1)
        post_click_utility,w_post_click = self.decode(memory, src_mask, post_click, tgt_mask, consumer_list)
        full_click_utility = pre_click_utility+post_click_utility

        # Get the loss of the estimation
        loss = self.loss_generator(full_click_utility,reserved_utility,tgt_mask,expo_mask,purchase_mask,click_index,tgt_len,shock,post_click_utility,hyper_sigma,self.task)


        post_click_ground = torch.matmul(post_click_coeff.unsqueeze(1), post_click.transpose(1, 2)).squeeze(1)

        return loss,w_p_original,w_p,w_c,w_post_click,post_click_utility,post_click_ground,preference_shock_flag

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask,consumer_list):
        if self.task=='training':
            return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        else:
            return self.decoder(tgt,consumer_list)

class RNNShock(nn.Module):
    def __init__(self, input_size, hidden_size,preference_shock_var):
        super(RNNShock, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.preference_shock_var = preference_shock_var
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, m,x, mask,lengths):
        h_0 = torch.zeros((1,x.shape[0], self.hidden_size),device=x.device)
        # Pack the padded sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_x,h_0)
        preference_shock = hidden[-1] # Last hidden state

        # # Compute mean and standard deviation for each row
        row_mean = preference_shock.mean(dim=1, keepdim=True)
        row_std = preference_shock.std(dim=1, keepdim=True)
        # #
        # # Normalize each row
        # preference_shock= (preference_shock - row_mean) / row_std
        # preference_shock=preference_shock*(self.preference_shock_var ** 0.5)

        # Create a column of zeros with the same number of rows as the matrix (price sensitivity is not in preference_shock)
        zero_column = torch.zeros(preference_shock.size(0), 1,device=preference_shock.device)
        # Concatenate the zero column to the matrix
        preference_shock = torch.cat((preference_shock, zero_column), dim=1)

        return preference_shock

class PreferenceShock(nn.Module):
    "learn user dynamic preference shock based on memory from encoder via attention"

    def __init__(self, size, pc_attn, feed_forward, preference_shock_var,dropout,len_pre):
        super(PreferenceShock, self).__init__()
        self.pc_attn = pc_attn
        self.feed_forward = feed_forward # make sure this is deep copy
        self.preference_shock_var = preference_shock_var
        self.sublayer = clones(SublayerConnection(size, dropout), 1)
        self.size = size
        self.len_pre = len_pre-1 # the price coefficient will not change
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

        self.preference_layer = nn.Linear(size, self.len_pre)

    def forward(self, m,x, mask,length):
        nbatches = m.size(0)
        query= torch.zeros(nbatches, self.size,device=m.device)
        query[:,::2] = 1 # this is the special query, aiming for learn preference and cost coefficients from memory
        # two steps below aims to do the same process in self.sublayer
        result = self.pc_attn(self.norm(query), m, m, mask)
        result = self.dropout(result)
        result = self.sublayer[0](result, self.feed_forward)


        preference_shock=self.preference_layer(result)

        # # Compute mean and standard deviation for each row
        # row_mean = preference_shock.mean(dim=1, keepdim=True)
        # row_std = preference_shock.std(dim=1, keepdim=True)
        #
        # # Normalize each row
        # preference_shock= (preference_shock - row_mean) / row_std
        # preference_shock = preference_shock * (self.preference_shock_var ** 0.5)

        # Create a column of zeros with the same number of rows as the matrix
        zero_column = torch.zeros(preference_shock.size(0), 1,device=preference_shock.device)
        # Concatenate the zero column to the matrix
        preference_shock = torch.cat((preference_shock, zero_column), dim=1)

        return preference_shock

class Embedding_Learner(nn.Module):
    "Learn utility of clicked items."
    def __init__(self,num_consumer,output_dimention):
        super(Embedding_Learner, self).__init__()

        self.output_dimention = output_dimention
        self.relu = nn.ReLU(inplace=True)
        # Define the layers with fully connected layers
        self.fc1 = nn.Linear(output_dimention*10, output_dimention*10)
        self.fc2 = nn.Linear(output_dimention*10, output_dimention*10)
        self.fc3 = nn.Linear(output_dimention*10, output_dimention*5)
        self.fc4 = nn.Linear(output_dimention*5, output_dimention)
        self.bn1 = nn.BatchNorm1d(output_dimention*10)
        self.bn2 = nn.BatchNorm1d(output_dimention*10)

        self.embedding_layer = nn.Embedding(num_consumer, output_dimention*10)



    def forward(self, input):
        embedding = self.embedding_layer(input)

        # First layer
        out = self.fc1(embedding)
        out = self.bn1(out)
        out = self.relu(out)
        # Second layer
        out = self.fc2(out)
        out = self.bn2(out)
        # Add the residual connection
        out += embedding
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)

        return out


class CU_Learner(nn.Module):
    "Learn utility of clicked items."
    def __init__(self):
        super(CU_Learner, self).__init__()
    def forward(self, w_pre,pre_x,post_click_utility):
        pre_click_utility=torch.matmul(w_pre.unsqueeze(1), pre_x.transpose(1, 2)).squeeze(1)
        CU=pre_click_utility+post_click_utility
        return CU


# Define the pre-trained module RU (the same architecture as when saved)
class PreTrainedModuleRU(nn.Module):
    "Mapping function from V,C to reserved utilities Z."
    def __init__(self):
        super(PreTrainedModuleRU, self).__init__()
        self.fc1 = nn.Linear(2, 16)  # Input layer to hidden layer
        self.fc2 = nn.Linear(16, 10)   # Hidden layer
        self.fc3 = nn.Linear(10, 8)  # Hidden layer
        self.fc4 = nn.Linear(8, 1)     # Hidden layer to output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation function
        x = torch.relu(self.fc2(x))  # Activation function
        x = torch.relu(self.fc3(x))  # Activation function
        x = self.fc4(x)               # Output layer
        return x

class RU_Learner(nn.Module):
    "Learn reserved utilities of unclicked items."
    def __init__(self, module_ru,len_pre,len_cost):
        super(RU_Learner, self).__init__()
        self.len_pre=len_pre
        self.len_cost=len_cost
        self.ru_inference = module_ru  # Use the pre-trained module_ru

    def forward(self, w_p,w_c,exp_list,cost_part):
        preference_part = exp_list[:, :, :, :self.len_pre]

        v=torch.matmul(w_p.unsqueeze(1).unsqueeze(2), preference_part.transpose(-1, -2)).squeeze()

        c=torch.matmul(w_c.unsqueeze(1).unsqueeze(2), cost_part.transpose(-1, -2)).squeeze()
        c=torch.clamp(c, min=None, max=50)
        c =torch.exp(c)

        return self.ru_inference(torch.stack((v, c), dim=-1)).squeeze()


class PC_Utility(nn.Module):
    "Learn reserved utilities of unclicked items."

    def __init__(self, pc_embedding):
        super(PC_Utility, self).__init__()
        self.pc_preference = pc_embedding  # Use the pre-trained module_ru

    def forward(self, tgt, consumer_list):
        w_post_click = self.pc_preference(consumer_list)
        post_click_utility = torch.matmul(w_post_click.unsqueeze(1), tgt.transpose(1, 2)).squeeze(1)
        return post_click_utility,w_post_click

class Loss_Generator(nn.Module):
    "Generting the loss for the model"

    def __init__(self):
        super(Loss_Generator, self).__init__()
        self.lamda_1 = 0.1
        self.lamda_2 = 0.01
        self.lamda_3 = 0.01
        self.lamda_4 = 0.01
        self.lamda_5 = 0.01


    def forward(self, cu, ru, tgt_mask,expo_mask, purchase_mask, click_index,tgt_len,shock,post_click_utility,hyper_sigma,task):
        nbatches=cu.size(0)
        len_interaction = cu.size(1)
        LR=nn.ReLU()

        index_matrix=torch.arange(cu.shape[1],device=cu.device).expand(cu.shape[0], cu.shape[1])
        cu_mask=(index_matrix <= tgt_len.unsqueeze(1))
        click_mask = (index_matrix < tgt_len.unsqueeze(1))

        ru = ru.masked_fill(expo_mask == 0, pad_idx)  # replace ru of the masked exposured items with pad_idx
        cu = cu.masked_fill(cu_mask == 0, pad_idx)  # replace masked items with pad_idx

        # Part 1: Purchase Decision

        # Part 1.1: purchase_loss_1
        # Compare the Utility of the purchased Item with all clicked but unpurchased items.
        purchase_u = cu[purchase_mask].unsqueeze(1)
        diff_1 = cu - purchase_u  # the diff should incline to negative
        unpurchase_mask = ~purchase_mask  # unpurchase items
        unpurchase_mask = unpurchase_mask & cu_mask  # exclude the purchased item and padding items in the ineraction sequence
        diff_1 = diff_1[unpurchase_mask]  # outside option is included here
        purchase_loss_1 = torch.sum(LR(diff_1))  # the purchased item should with the highest utility

        # Part 1.2: purchase_loss_2
        # Collect the RV of items in the last interaction of each session
        last_true_indices = tgt_len

        # Create a new mask that only keeps the last True in each row
        candidate_mask_last_true = torch.zeros_like(expo_mask[:,:,0])
        candidate_mask_last_true.scatter_(1, last_true_indices.unsqueeze(1), True)
        candidate_mask_last_interactions = candidate_mask_last_true.unsqueeze(-1) & expo_mask

        diff_2 = ru - purchase_u.unsqueeze(1)  # the diff should incline to negative
        diff_2 = diff_2[candidate_mask_last_interactions]
        purchase_loss_2 = torch.sum(LR(diff_2))  # the purchased item has the highest utility

        # Part 2: Click Decision
        # The clicked item is selected based on click_index within the ranking list
        clicked_ru = torch.sum(ru*click_index,dim=2)

        # Part 2.1: click_loss_1: compare the ru of the clicked item with unclicked items in the list
        diff_3 = (ru * (1 - click_index) - clicked_ru.unsqueeze(2))
        max_diff_3, _ = torch.max(diff_3, dim=2)  # compare the clicked item with the one has max ru in the ist
        max_diff_3 = max_diff_3[click_mask]
        click_loss_1 = torch.sum(LR(max_diff_3))

        # Part 2.2: click_loss_2: compare the ru of the clicked item with cv of previously clicked items
        cu_sofar = cu.unsqueeze(-2) * tgt_mask.float()
        max_cu_sofar, _ = torch.max(cu_sofar, dim=2)
        diff_4 = max_cu_sofar - clicked_ru
        diff_4 = diff_4[click_mask]
        click_loss_2 = torch.sum(LR(diff_4))  # the clicked item has the higher RU than CU of the previous clicked items



        total_loss = purchase_loss_1 + purchase_loss_2 + click_loss_1 + click_loss_2
        #p_l1_loss=self.lamda_1 * (torch.sum(torch.abs(w_p)))
        #p_l2_loss = self.lamda_1 * (torch.sum(torch.abs(w_c)))
        if task=='training': # add L2 regularization of shock and post_click_utility
            total_loss=total_loss+(shock ** 2).sum() / (2 * hyper_sigma**2)+(post_click_utility[click_mask] ** 2).sum()/2
        else: # no need L2 regularization in simulation data
            total_loss=total_loss

        return total_loss

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)


    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

        self.e_mean = nn.Linear(layer.size, 1)
        self.e_gamma= nn.Linear(layer.size, 1)
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        x=self.norm(x)
        post_click_shock = torch.squeeze(self.e_mean(x))
        return post_click_shock,None

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, pad_idx)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


def attention_pc(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query.unsqueeze(-2), key.transpose(-2, -1))/ math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, pad_idx)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention_pc(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_pc, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        key, value = [
           lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
           for lin, x in zip(self.linears, (key, value))
       ]
        query=self.linears[-2](query).view(nbatches, self.h, self.d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention_pc(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k).squeeze()
        )
        del query
        del key
        del value
        return self.linears[-1](x)



class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, interaction_feature_size):
        super(Embeddings, self).__init__()
        self.layer1 = nn.Linear(interaction_feature_size, d_model)
        self.layer2 = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.layer2(self.layer1(x).relu())


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def make_model(src_interaction_feature, tgt_interaction_feature, N=6, d_model=512, d_ff=1024, h=8, preference_shock_var=1,len_pre=20, len_cost=2, num_postclick_feature=4,num_consumer=10,
               dropout=0.1,task='training'):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    pc_attn = MultiHeadedAttention_pc(h, d_model)

    # emebdding function for the src and tgt should be different
    src_embedding = c(Embeddings(d_model, src_interaction_feature))
    tgt_embedding = c(Embeddings(d_model, tgt_interaction_feature))


    if task=='training':
        Preference_Shock=PreferenceShock(d_model, pc_attn, c(ff), preference_shock_var,dropout, len_pre)
        Decoder_module=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
    else: # for simulated data parameter estimation
        Preference_Shock=RNNShock(src_interaction_feature, src_interaction_feature - 2,preference_shock_var)
        Decoder_module=PC_Utility(Embedding_Learner(num_consumer, num_postclick_feature))


    # PreTrained RU module
    module_ru = PreTrainedModuleRU()


    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder_module,
        Embedding_Learner(num_consumer, len_pre),
        Embedding_Learner(num_consumer, len_cost),
        Preference_Shock,
        RU_Learner(module_ru, len_pre, len_cost),
        CU_Learner(),
        nn.Sequential(src_embedding, c(position)),
        nn.Sequential(tgt_embedding, c(position)),
        Loss_Generator(),
        task
    )


    # Initialize parameters with Glorot / fan_avg.
    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_normal_(p)


    # Load pre-trained RU moudule from the .pt file
    module_ru.load_state_dict(torch.load("model_ru_parameter.pt"))  # Load the saved weights
    module_ru.eval()  # Set to evaluation mode
    # Freeze the parameters of pre-trained module RU
    for param in module_ru.parameters():
        param.requires_grad = False  # Freezing all parameters of the module

    return model


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src,src_len,tgt,post_click,click_seq, expo,expo_cost,click_index, purchase,tgt_len,full_preference,cate_preference,updated_preferecne,shock,cost_coeff,post_click_coeff,preference_shock_flag,consumer_list, pad=pad_idx):  # 2 = <blank>

        self.src=src
        self.src_len = src_len
        self.src_mask = (src != pad)[:,:,0]
        #self.src_mask = self.src_mask[:,:,0]
        self.src_mask =self.src_mask.unsqueeze(-2)

        if tgt is not None:
            self.tgt_mask = self.make_std_mask(tgt, pad)
        self.post_click = post_click

        self.click_seq = click_seq
        self.expo = expo
        self.expo_mask = (expo != pad)[:,:,:,0]
        self.expo_cost =expo_cost
        self.click_index=click_index

        self.purchase_mask = (purchase == 1)

        self.tgt_len=tgt_len

        self.full_preference=full_preference
        self.cate_preference = cate_preference
        self.updated_preferecne=updated_preferecne
        self.shock=shock
        self.cost_coeff = cost_coeff
        self.post_click_coeff=post_click_coeff
        #self.category = category
        self.preference_shock_flag=preference_shock_flag
        self.consumer_list=consumer_list


    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad)
        tgt_mask=tgt_mask[:,:,0].unsqueeze(2)
        candidate_mask=subsequent_mask(tgt.size(-2)).type_as(tgt_mask.data)
        tgt_mask = tgt_mask & candidate_mask
        return tgt_mask


class Session_Date_Loader(Dataset):
    def __init__(self, device='cuda'):
        self.device = device
        self.sessions = self.DataLoader()
        self.X, self.Y, self.Post_Click,self.Click_Seq, self.expo_list, self.expo_cost, self.Click_Index,self.purchase, self.tgt_len, self.full_preference, self.cate_preference,self.update_preference, self.shock, self.cost_coeff,self.post_click_coeff, self.preference_shock_flag, self.consumer_list = self.Make_XY(
            self.sessions)

    def __len__(self):
        return len(self.Click_Seq)

    def __getitem__(self, idx):
        return self.X[idx],self.Y[idx],self.Post_Click[idx], self.Click_Seq[idx], self.expo_list[idx],self.expo_cost[idx],self.Click_Index[idx],self.purchase[idx],self.tgt_len[idx], self.full_preference[idx],self.cate_preference[idx],self.update_preference[idx],self.shock[idx],self.cost_coeff[idx], self.post_click_coeff[idx],self.preference_shock_flag[idx],self.consumer_list[idx]

    def DataLoader(self):
        with open('simulated_sessions.pkl', 'rb') as f:
            sessions = pickle.load(f)
        return sessions
    def Make_XY(self, sessions):
        X=[]
        Y=[]
        Post_Click=[]
        Click_Seq=[]
        Expo = []
        Expo_Cost=[]
        Click_Index=[]
        purchase = []
        tgt_len = []
        full_preference = []
        update_preference=[]
        shock=[]
        cate_preference = []
        cost_coeff = []
        post_click_coeff=[]
        category=[]
        preference_shock_flag=[]
        consumer_list=[]

        consumer_session_counts = Counter(sessions['consumer_id'])

        for consumer, count in consumer_session_counts.items():
            if count/len(sessions['consumer_id']) >= 0.01:
                consumer_mask = [candiate == consumer for candiate in sessions['consumer_id']]
                select_x = [value for value, m in zip(sessions['X'], consumer_mask) if m]
                select_y = [value for value, m in zip(sessions['Y'], consumer_mask) if m]
                select_post_click = [value for value, m in zip(sessions['Post_Click'], consumer_mask) if m]
                select_expo = [value for value, m in zip(sessions['Expo'], consumer_mask) if m]
                select_expo_cost = [value for value, m in zip(sessions['Expo_Cost'], consumer_mask) if m]
                select_click_index = [value for value, m in zip(sessions['Click_Index'], consumer_mask) if m]
                select_purchase = [value for value, m in zip(sessions['purchase'], consumer_mask) if m]
                select_tgtlen = [value for value, m in zip(sessions['tgt_len'], consumer_mask) if m]
                select_full_preference = [value for value, m in zip(sessions['original_preference_list'], consumer_mask) if m]
                select_update_preference = [value for value, m in zip(sessions['updated_preference_list'], consumer_mask) if m]
                select_cate_preference = [value for value, m in zip(sessions['original_preference_list'], consumer_mask) if m] # double-check the source
                select_shock = [value for value, m in zip(sessions['shock_list'], consumer_mask) if m]
                select_cost_coeff = [value for value, m in zip(sessions['cost_coeff'], consumer_mask) if m]
                select_post_click_coeff = [value for value, m in zip(sessions['post_click_coeff_list'], consumer_mask) if m]
                select_preference_shock_flag= [value for value, m in zip(sessions['preference_shock_flag'], consumer_mask) if m]
                #select_category = [value for value, m in zip(sessions['category'], consumer_mask) if m]

                avgerage_length=sum(select_tgtlen) / len(select_tgtlen)
                if avgerage_length>1.05:
                    for i in range(count):
                        X.append(select_x[i].t().to(self.device))
                        Y.append(select_y[i].t().to(self.device))
                        Post_Click.append(select_post_click[i].t().to(self.device))
                        Expo.append(select_expo[i].to(self.device))
                        Expo_Cost.append(select_expo_cost[i].to(self.device))
                        Click_Index.append(select_click_index[i].to(self.device))
                        purchase.append(select_purchase[i].to(self.device))
                        tgt_len.append(torch.tensor([select_tgtlen[i]], device=self.device))
                        full_preference.append(select_full_preference[i].to(self.device))
                        cate_preference.append(select_cate_preference[i].to(self.device))
                        update_preference.append(select_update_preference[i].to(self.device))
                        shock.append(select_shock[i].to(self.device))
                        cost_coeff.append(select_cost_coeff[i].to(self.device))
                        post_click_coeff.append(select_post_click_coeff[i].to(self.device))
                        preference_shock_flag.append(torch.tensor([select_preference_shock_flag[i]], device=self.device))
                        #category.append(torch.tensor([select_category[i]], device=self.device))
                        consumer_list.append(torch.tensor(consumer, device=self.device))


                        Click_Seq.append(select_y[i].t().to(self.device))


        return X,Y,Post_Click,Click_Seq, Expo, Expo_Cost,Click_Index, purchase, tgt_len, full_preference, cate_preference,update_preference,shock, cost_coeff,post_click_coeff,preference_shock_flag,consumer_list

# Custom collate function to pad sequences and create padding masks
def collate_fn(batch):


    X_batch,Y_batch,Post_Click_batch,Click_Seq_batch,Expo_batch,Expo_Cost_batch,Click_Index_batch, Purchase_bath,tgt_len, full_preference, cate_preference, update_preference,shock, cost_coeff,post_click_coeff,preference_shock_flag,consumer_list = zip(*batch)

    X_padded = rnn_utils.pad_sequence(X_batch, batch_first=True, padding_value=pad_idx)
    X_lengths = torch.tensor([len(seq) for seq in X_batch])
    Y_padded = rnn_utils.pad_sequence(Y_batch, batch_first=True, padding_value=pad_idx)
    Post_Click_padded = rnn_utils.pad_sequence(Post_Click_batch, batch_first=True, padding_value=pad_idx)
    Click_Seq_padded = rnn_utils.pad_sequence(Click_Seq_batch, batch_first=True, padding_value=pad_idx)
    Expo_padded = rnn_utils.pad_sequence(Expo_batch, batch_first=True, padding_value=pad_idx)
    Expo_Cost_padded = rnn_utils.pad_sequence(Expo_Cost_batch, batch_first=True, padding_value=pad_idx)
    Click_Index_padded = rnn_utils.pad_sequence(Click_Index_batch, batch_first=True, padding_value=0)
    Purchase_padded = rnn_utils.pad_sequence(Purchase_bath, batch_first=True, padding_value=pad_idx)

    tgt_len = torch.stack(tgt_len).squeeze()
    full_preference = torch.stack(full_preference).squeeze()
    cate_preference = torch.stack(cate_preference).squeeze()
    update_preference = torch.stack(update_preference).squeeze()
    shock = torch.stack(shock).squeeze()

    cost_coeff = torch.stack(cost_coeff).squeeze()
    post_click_coeff = torch.stack(post_click_coeff).squeeze()

    preference_shock_flag = torch.stack(preference_shock_flag).squeeze()
    consumer_list=torch.stack(consumer_list).squeeze()

    return X_padded,X_lengths, Y_padded,Post_Click_padded,Click_Seq_padded, Expo_padded, Expo_Cost_padded,Click_Index_padded,Purchase_padded,tgt_len, full_preference, cate_preference,update_preference,shock, cost_coeff,post_click_coeff,preference_shock_flag,consumer_list

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed
    w_p: [] # preference_coeff
    w_c: [] # cost_coeff
    reserved_utility: []
    clicked_utility: []
    update_dis: []

class TestState:
    def __init__(self):
        self.original_preference_prediction=[]
        self.original_preference_groundtruth=[]
        self.updated_preference_prediction=[]
        self.updated_preference_groundtruth=[]
        self.preference_shock_prediction=[]
        self.preference_shock_groundtruth=[]
        self.post_click_shock_prediction=[]
        self.post_click_shock_groundtruth=[]
        self.cost_coeff_prediction=[]
        self.ocost_coeff_groundtruth= []



def run_epoch(
    config,
    data_iter,
    model,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        loss, w_p_original, w_p, w_c, w_post_click, post_click_utility, post_click_ground,preference_shock_flag = model.forward(batch.src,
                                                                                                          batch.src_len,
                                                                                                          batch.src_mask,
                                                                                                          batch.post_click,
                                                                                                          batch.tgt_mask,
                                                                                                          batch.expo,
                                                                                                          batch.expo_mask,
                                                                                                          batch.expo_cost,
                                                                                                          batch.click_index,
                                                                                                          batch.click_seq,
                                                                                                          batch.purchase_mask,
                                                                                                          batch.tgt_len,
                                                                                                          batch.cate_preference,
                                                                                                          batch.updated_preferecne,
                                                                                                          batch.shock,
                                                                                                          batch.cost_coeff,
                                                                                                          batch.post_click_coeff,
                                                                                                          batch.preference_shock_flag,
                                                                                                          batch.consumer_list,config["hyper_sigma"]
                                                                                                          )
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss.backward()
            train_state.step += 1
            train_state.samples += batch.tgt_mask.shape[0]
            train_state.w_p=w_p
            train_state.w_c=w_c
            #train_state.update_dis=update_dis


            if i % accum_iter == 0:
                optimizer.step()
                #the gradients accumulate in the grad attribute of the model's parameters. Therefore, call optimizer.zero_grad() before the backpropagation step to reset the gradients to zero, ensuring that gradients from the previous step don't interfere with the current one.
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Learning Rate: %6.1e"
                )
                % (i, n_accum, loss , lr)
            )
            start = time.time()
            tokens = 0
        del loss
    return total_loss, train_state




def run_evluation(
    data_iter,
    model,
    config,
    mode="test",
    test_results=TestState(),

):

    model.eval()
    for i, batch in enumerate(data_iter):
        loss, w_p_original, w_p, w_c, w_post_click, post_click_utility, post_click_ground,preference_shock_flag = model.forward(batch.src,
                                                                                                          batch.src_len,
                                                                                                          batch.src_mask,
                                                                                                          batch.post_click,
                                                                                                          batch.tgt_mask,
                                                                                                          batch.expo,
                                                                                                          batch.expo_mask,
                                                                                                          batch.expo_cost,
                                                                                                          batch.click_index,
                                                                                                          batch.click_seq,
                                                                                                          batch.purchase_mask,
                                                                                                          batch.tgt_len,
                                                                                                          batch.cate_preference,
                                                                                                          batch.updated_preferecne,
                                                                                                          batch.shock,
                                                                                                          batch.cost_coeff,
                                                                                                          batch.post_click_coeff,
                                                                                                          batch.preference_shock_flag,
                                                                                                          batch.consumer_list,config["hyper_sigma"]
                                                                                                          )

        test_results.cost_coeff_prediction.append(w_c)
        test_results.ocost_coeff_groundtruth.append(batch.cost_coeff)
        test_results.original_preference_prediction.append(w_p_original)
        test_results.original_preference_groundtruth.append(batch.cate_preference)
        test_results.updated_preference_prediction.append(w_p)
        test_results.updated_preference_groundtruth.append(batch.updated_preferecne)
        test_results.preference_shock_prediction.append(w_p-w_p_original)
        test_results.preference_shock_groundtruth.append(batch.updated_preferecne-batch.cate_preference)
        test_results.post_click_shock_prediction.append(post_click_utility[batch.tgt_mask[:,:,0]])
        test_results.post_click_shock_groundtruth.append(post_click_ground[batch.tgt_mask[:,:,0]])


    # evaluation
    cost_coeff_prediction=torch.cat(test_results.cost_coeff_prediction, dim=0)
    ocost_coeff_groundtruth = torch.cat(test_results.ocost_coeff_groundtruth, dim=0)
    MAE,MAE_VAR,MAPE=Genral_Evaluate(ocost_coeff_groundtruth, cost_coeff_prediction)
    print(f" Cost_Coeff MAE: {MAE}, MAE_VAR: {MAE_VAR}, MAPE: {MAPE}")

    original_preference_prediction=torch.cat(test_results.original_preference_prediction, dim=0)
    original_preference_groundtruth = torch.cat(test_results.original_preference_groundtruth, dim=0)
    MAE,MAE_VAR,MAPE=Genral_Evaluate(original_preference_groundtruth, original_preference_prediction)
    print(f" Org_Pre MAE: {MAE}, MAE_VAR: {MAE_VAR}, MAPE: {MAPE}")

    updated_preference_prediction=torch.cat(test_results.updated_preference_prediction, dim=0)
    updated_preference_groundtruth = torch.cat(test_results.updated_preference_groundtruth, dim=0)
    MAE,MAE_VAR,MAPE=Genral_Evaluate(updated_preference_groundtruth, updated_preference_prediction)
    print(f" Updated_Pre MAE: {MAE}, MAE_VAR: {MAE_VAR}, MAPE: {MAPE}")

    preference_shock_prediction=torch.cat(test_results.preference_shock_prediction, dim=0)
    preference_shock_groundtruth = torch.cat(test_results.preference_shock_groundtruth, dim=0)
    MAE,MAE_VAR,MAPE=Genral_Evaluate(preference_shock_groundtruth, preference_shock_prediction)
    print(f" Preference_Shock MAE: {MAE}, MAE_VAR: {MAE_VAR}, MAPE: {MAPE}")

    post_click_shock_prediction=torch.cat(test_results.post_click_shock_prediction, dim=0)
    post_click_shock_groundtruth = torch.cat(test_results.post_click_shock_groundtruth, dim=0)
    MAE,MAE_VAR,MAPE=Genral_Evaluate(post_click_shock_groundtruth, post_click_shock_prediction)
    print(f" Post_Click_Shock MAE: {MAE}, MAE_VAR: {MAE_VAR}, MAPE: {MAPE}")

    return test_results


def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def train_worker(
    gpu,
    ngpus_per_node,
    config,
    is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)


    dataset=Session_Date_Loader(device=gpu)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=True)

    model = make_model(src_interaction_feature=dataset.sessions["num_features"] + 1,
                       tgt_interaction_feature=dataset.sessions["num_postclick_feature"], N=config["en_de_layers"],
                       d_model=config["d_model"],d_ff=config["d_ff"],preference_shock_var=dataset.sessions["preference_shock_var"],
                       len_pre=dataset.sessions["tgt_interaction_feature"],
                       len_cost=dataset.sessions["len_cost_coeff"], num_postclick_feature=dataset.sessions["num_postclick_feature"],num_consumer=dataset.sessions["num_consumer"],task=config["task"])
    model.cuda(gpu)
    module = model
    is_main_process = True



    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, config["d_model"], factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        model.train()
        model.ru_learner.ru_inference.training = False # the pre-trained module should not be updated

        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            config,
            (Batch(b[0], b[1], b[2], b[3],b[4], b[5], b[6], b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15],b[16],b[17], pad_idx) for b in dataloader),
            model,
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )
        GPUtil.showUtilization()
        # if is_main_process and epoch%100==0:
        #     file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
        #     #torch.save(module.state_dict(), file_path)
        # torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path) # save the parameters of the trained model
    return train_state


def test_worker(
    gpu,
    ngpus_per_node,
    config,
    is_distributed=False,
):
    torch.cuda.set_device(gpu)
    dataset=Session_Date_Loader(device=gpu)
    model = make_model(src_interaction_feature=dataset.sessions["num_features"] + 1,
                       tgt_interaction_feature=dataset.sessions["num_postclick_feature"], N=config["en_de_layers"],
                       d_model=config["d_model"],d_ff=config["d_ff"],preference_shock_var=dataset.sessions["preference_shock_var"],
                       len_pre=dataset.sessions["tgt_interaction_feature"],
                       len_cost=dataset.sessions["len_cost_coeff"], num_postclick_feature=dataset.sessions["num_postclick_feature"],num_consumer=dataset.sessions["num_consumer"],task=config["task"])
    model.load_state_dict(torch.load("%sfinal.pt" % config["file_prefix"])) # load trained model parameters
    model.cuda(gpu)

    # evaluate the model
    test_state = TestState()
    dataloader_test = DataLoader(dataset, batch_size=config["batch_size"], collate_fn=collate_fn, shuffle=False)
    test_state = run_evluation(
        (Batch(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15],
               b[16],b[17], pad_idx) for b in dataloader_test),
        model,
        config,
        mode="test",
        test_results=test_state,
    )
    return test_state

config = {
    "batch_size": 512,
    "num_epochs": 5000,
    "accum_iter": 10,
    "base_lr": 1.0,
    "warmup": 3000,
    "d_model": 128,
    "en_de_layers": 4,
    "d_ff":256,
    "file_prefix": "DeepStructural_model_",
    "hyper_sigma": 1,
    #"full_feature": 20,
    #"exposure_feature": 10,
    #"len_cost_coeff": 2,
    "task": "simulation"
}
torch.cuda.empty_cache()
if not exists("%sfinal.pt" % config["file_prefix"]): # if model is not pre-trained and parameter file is not available, train the model first
    train_result=train_worker(0, 1, config,False)

test_result=test_worker(0, 1, config,False)

