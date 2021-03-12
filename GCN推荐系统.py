# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:36:13 2020

@author: peter
"""
#importing packages
import csv
import os
import operator
import numpy as np
import torch
import collections
import torch.nn as nn
import math
import random
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv, SAGEConv, global_mean_pool
from datetime import datetime
from sklearn.preprocessing import RobustScaler



def seed_torch(seed=0):
    random.seed(seed)#
    os.environ['PYTHONHASHSEED'] = str(seed)#
    np.random.seed(seed)#
    torch.manual_seed(seed)#
    torch.cuda.manual_seed(seed)#
    torch.cuda.manual_seed_all(seed)# # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False#
    torch.backends.cudnn.deterministic = True#

#setting working directory
os.chdir("C:/Users/peter\'s lap/Master thesis/")

#setting data directory (code from Dr.Jacopo Tagliabue)
LOCAL_FOLDER = 'd31bca15-ef59-4794-9992-6d6117c11065/'
  # set here the local SECURE folder in which you have stored the dataset
DATASET_NAME = 'small.csv'  # for illustration purposes, I'm running a small version of the dataset
DATASET_FILE = os.path.join(LOCAL_FOLDER, DATASET_NAME)


#read out the sessions from dataset (code from Dr.Jacopo Tagliabue)
def map_site_action(event_type, product_action):
    """
    This function maps events to integers to make for easy downstream processing
    """
    if event_type in ['pageview', 'event'] and not product_action:
        return 1
    elif product_action == 'detail':
        return 2
    elif product_action == 'add':
        return 3
    elif product_action == 'remove':
        return 4
    elif product_action == 'purchase':
        return 5
    elif product_action == 'click':
        return 6
    else:
        raise Exception("User action not mappable {}".format(product_action))

    return None





def cut_and_pad_session(session, max_length, intervals, session_skus, current_timestamps):
    """
    This function pads sessions and assigns as label 1/0 depending on session containing a target event.
    If it contains a target event, it's cut just before that. 
    All session are padded to MAX_LENGTH, we also build a time interval list for future use.
    """
    session_object = dict()
    session_class = 0

    # fianally add properties to session object
    session_object['label'] = session_class
    session_object['values'] = [session[idx] if idx < len(session) else 0 for idx in range(max_length)]
    # time intervals vector
    session_object['times'] = [intervals[idx] if idx < len(intervals) else 0 for idx in range(max_length)]
    # skus
    session_object['skus'] = [session_skus[idx] if idx < len(session_skus) else [] for idx in range(max_length)]
    # timestamps
    session_object['timestamps'] = [current_timestamps[idx] if idx < len(current_timestamps) else 0 for idx in range(max_length)]
    
    return  session_object


MIN_SEQ_LENGTH = 5
MAX_SEQ_LENGTH = 200


last_session_id = None
last_timestamp = None
current_session = []
current_intervals = []
current_timestamps = []
current_skus = []
sessions = []
with open(DATASET_FILE) as csvfile:
    reader = csv.DictReader(csvfile)
    # loop over all events
    for idx, row in enumerate(reader):
        # print all columns for first row
        if idx == 0:
            print(row.keys())
        # read the current values for the event
        current_session_id = row['session_id_hash']
        current_timestamp = int(row['server_timestamp_epoch_ms'])
        current_action = map_site_action(row['event_type'], row['product_action'])
        
        # this is an array of the form [SKU,SKU] but as a string in the csv!
        products = row['product_skus_hash'][1:-1] 
        if products:
            current_products = [p.strip() for p in products.split(',')]
        else:
            current_products = []
        # if the SKU was not recorded we use a placeholder label anyway
        if not current_products and current_action > 1:
            current_products = ['MISSING']
        if current_session_id != last_session_id and last_session_id:
            # check if we keep the session
            if len(current_session) >= MIN_SEQ_LENGTH and len(current_session) < MAX_SEQ_LENGTH:
                # check data is in order
                assert len(current_intervals) + 1 == len(current_session)
                padded_session = cut_and_pad_session(current_session, 
                                                     MAX_SEQ_LENGTH, 
                                                     current_intervals,
                                                     current_skus,
                                                     current_timestamps)
                sessions.append(padded_session)
            # reset session
            current_session = [current_action]
            current_intervals = []
            current_skus = [current_products]
            current_timestamps = [current_timestamp]
        else:
            current_session.append(current_action)
            current_skus.append(current_products)
            current_timestamps.append(current_timestamp)
            if last_timestamp:
                current_intervals.append(current_timestamp - last_timestamp)
        # update session id and timestamp
        last_session_id = current_session_id
        last_timestamp = current_timestamp

# check how many session we have
print("Total of {} sessions, first session is {}".format(len(sessions), (sessions[0])))

#Get the sessions with the lable "detail" and their dwell time. These are the sessions that we care.
sess_clicks = []
sess_times = []
for session in sessions:
    items = []
    time = []
    action_indexes = []
    for num, __ in enumerate(session["values"]):
        if session["values"][num] == 2:
            action_indexes.append(num)
    if action_indexes != []:
        for index in action_indexes:
            if session["skus"][index] != []:
                items.append(session["skus"][index])
                time.append(session["times"][index])
    sess_clicks.append(items)
    sess_times.append(time)

print("length of original sessions & original dwelltime pairs")
print(len(sess_clicks), len(sess_times))

#Get rid of those empty sessions
sess_clicks_noempty = [session for session in sess_clicks if session != []]
sess_times_noempty = [session for session in sess_times if session != []]

print("length of sessions & dwelltime pairs without empty sessions")
print(len(sess_clicks_noempty), len(sess_times_noempty))

#Get rid of those items which appear less than 5
iid_counts = {}
for session in sess_clicks_noempty:
    for iid in session:
        if str(iid) in iid_counts.keys():
            iid_counts[str(iid)] += 1
        else:
            iid_counts[str(iid)] = 1

iid_items = []
for key in iid_counts.keys():
    if iid_counts[key] < 5:
        iid_items.append(key)

itr = 0
while itr <3:
    for session, time in zip(sess_clicks_noempty, sess_times_noempty):
        for num, item in enumerate(session):
            if str(item) in iid_items:
                session.pop(num)
                time.pop(num)
    itr += 1
#Check again whether the data is cleaned
iid_counts = {}
for session in sess_clicks_noempty:
    for iid in session:
        if str(iid) in iid_counts.keys():
            iid_counts[str(iid)] += 1
        else:
            iid_counts[str(iid)] = 1
sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))
sorted_counts[:10]

print("sorted item dictionary in form (['itemharsh'],frequency):") 
print(sorted_counts[:10])

#Get rid of those sessions with only one item
itr = 0
while itr <3:
    for num, session, time in zip(range(len(sess_clicks_noempty)), sess_clicks_noempty, sess_times_noempty):
        if len(session) < 2:
            sess_clicks_noempty.pop(num)
            sess_times_noempty.pop(num)
    itr += 1


sess_len = [len(session) for session in sess_clicks_noempty]
sess_len.count(1) # there is no session with a length of 1

print("length of session & dwelltime pairs with more than one item:")
print(len(sess_clicks_noempty), len(sess_times_noempty))

#Make an item dictionary and convert each item into numeric label.
item_dict = {}
item_ctr = 1
sess_seq = []
for session in sess_clicks_noempty:
    outseq = []
    for item in session:
        if str(item) in item_dict:
            outseq += [item_dict[str(item)]]
        else:
            outseq += [item_ctr]
            item_dict[str(item)] = item_ctr
            item_ctr += 1
    sess_seq += [outseq]

print("here is an example of session and dwelltime pair:")
print(sess_seq[1], sess_times_noempty[1])

#Data augmentation (E.g., S = [1,2,3,4] will become [1,2,3], [1,2], [1] and their labels (y) will be 4, 3, 2, 1)
labs = []
out_seqs = []
out_times = []
for seq, time in zip(sess_seq, sess_times_noempty):
    for num in range(1, len(seq)):
        tar = seq[-num]
        labs += [tar]
        out_seqs += [seq[:-num]]
        out_times += [time[:-num]]
print("length of sessions, dwelltime after data augmentation:")
print(len(out_seqs), len(out_times))

#Get rid of those sessions which only have 1 item, as these sessions can't be made into session graphs.
def unique(list1):
    newlist = []
    for i in list1:
        if i not in newlist:
            newlist.append(i)
    return len(newlist)

itr = 0
while itr <7:
    for num, session, time in zip(range(len(out_seqs)),out_seqs, out_times):
        if unique(session) <= 2:
            out_seqs.pop(num)
            out_times.pop(num)
            labs.pop(num)
    itr += 1

print("length of sessions, targets and dwelltime with more than one item:")
print(len(out_seqs), len(labs), len(out_times)) #959830

#shuffle the data:
before_shuffle = []
for i in range(len(out_seqs)):
    before_shuffle.append([out_seqs[i], labs[i], out_times[i]])
random.shuffle(before_shuffle)
after_shuffle = before_shuffle

out_seqs = [i[0] for i in after_shuffle]
labs = [i[1] for i in after_shuffle]
out_times = [i[2] for i in after_shuffle]


#time standarization and normalization, the variance of time is too big.
def normalization(time):
    new_time = []
    for list in time:
        time_list = []
        maxnum = max(list)
        minnum = min(list)+0.01
        for x in list:
            content = round((x - minnum)/(maxnum-minnum), 2)
            if content>=0:
                time_list.append(content)
            else:
                time_list.append(0)
        new_time.append(time_list)
    return new_time



def weight(time):
    new_time = []
    for list in time:
        time_list = []
        maxnum = max(list)
        minnum = min(list)+0.01
        for x in list:
            content = round((maxnum-x)/(maxnum-minnum), 2)
            time_list.append(content)
        new_time.append(time_list)
    return new_time

norm_times = normalization(out_times)
weight_times = weight(out_times)

        

#Split out the demo dataset which only contains 1000 pairs (i.e., sess_clicks_noempty & sess_times_noempty) of data,, training : testing = 7 : 3
sess_clicks_demo = out_seqs[:1000]
sess_labs_demo = labs[:1000]
sess_times_demo = weight_times[:1000]

sess_clicks_train = out_seqs[:885500]
sess_labs_train = labs[:885500]
sess_times_train = weight_times[:885500]

sess_clicks_val = out_seqs[885500:934000]
sess_labs_val = labs[885500:934000]
sess_times_val = weight_times[885500:934000]

sess_clicks_valtrain = out_seqs[:934000]
sess_labs_valtrain = labs[:934000]
sess_times_valtrain = weight_times[:934000]

sess_clicks_test = out_seqs[934000:]
sess_labs_test = labs[934000:]
sess_times_test = weight_times[934000:]

#Making session graphs. (inspired by Qiu et al.(2020))
def making_graphs(sess_clicks, sess_labs, sess_times, graph_types, shuffle):
    assert graph_types in ["no_time", "with_time", "time_only"]
    if graph_types == "no_time":
        data_list = []
        for session, y in zip(sess_clicks, sess_labs):
            i = 0
            nodes = {}
            senders = []
            x = []
            for node in session:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            
            if len(senders) != 1:
                del senders[-1]
                del receivers[0]
        
            pair = {}
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            i = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender)+'-'+str(receiver) in pair:
                    pair[str(sender)+'-'+str(receiver)] += 1
                    del senders[i]
                    del receivers[i]
                else:
                    pair[str(sender)+'-'+str(receiver)] = 1
                    i += 1
        
        
            count_out = collections.Counter(senders)
            count_in = collections.Counter(receivers)
            out_degree_inv = torch.tensor([1/count_out[i] for i in senders], dtype=torch.float)
            in_degree_inv = torch.tensor([1 / count_in[i] for i in receivers], dtype=torch.float)
            edge_attr = torch.tensor([pair[str(senders[i])+'-'+str(receivers[i])] for i in range(len(senders))],
                                             dtype=torch.float)
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            sequence = torch.tensor(session, dtype=torch.long)
            sequence_len = torch.tensor([len(session)], dtype=torch.long)
            session_graph = Data(x=x, y=y,
                                         edge_index=edge_index, edge_attr=edge_attr,
                                         sequence=sequence, sequence_len=sequence_len,
                                         out_degree_inv=out_degree_inv, in_degree_inv=in_degree_inv)
            data_list.append(session_graph)
            
    elif graph_types == "with_time":
        data_list = []
        for session, y, time in zip(sess_clicks, sess_labs, sess_times):
            i = 0
            nodes = {}
            senders = []
            x = []
            for node in session:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            time = time[:]
            
            if len(senders) != 1:
                del senders[-1]
                del receivers[0]
                del time[-1]

            pair = {}
            pair_time = {}
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            sur_time = time[:]
            i = 0
            for sender, receiver, time in zip(sur_senders, sur_receivers, sur_time):
                if str(sender)+'-'+str(receiver) in pair:
                    pair[str(sender)+'-'+str(receiver)] += 1
                    pair_time[str(sender)+'-'+str(receiver)] += time
                    del senders[i]
                    del receivers[i]
                else:
                    pair[str(sender)+'-'+str(receiver)] = 1
                    pair_time[str(sender)+'-'+str(receiver)] = time
                    i += 1
            pair_time_fi = [pair_time[str(senders[i])+'-'+str(receivers[i])] for i in range(len(senders))]
            pair_fi = [pair[str(senders[i])+'-'+str(receivers[i])] for i in range(len(senders))]

            time_incor = []
            for fre, time in zip(pair_fi, pair_time_fi):
                time_incor.append(fre*time)

            count_out = collections.Counter(senders)
            count_in = collections.Counter(receivers)
            out_degree_inv = torch.tensor([1/count_out[i] for i in senders], dtype=torch.float)
            in_degree_inv = torch.tensor([1 / count_in[i] for i in receivers], dtype=torch.float)
            edge_attr = torch.tensor(time_incor,
                                             dtype=torch.float)
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            sequence = torch.tensor(session, dtype=torch.long)
            sequence_len = torch.tensor([len(session)], dtype=torch.long)
            session_graph = Data(x=x, y=y,
                                         edge_index=edge_index, edge_attr=edge_attr,
                                         sequence=sequence, sequence_len=sequence_len,
                                         out_degree_inv=out_degree_inv, in_degree_inv=in_degree_inv)
            data_list.append(session_graph)

    elif graph_types == "time_only":
        data_list = []
        for session, y, time in zip(sess_clicks, sess_labs, sess_times):
            i = 0
            nodes = {}
            senders = []
            x = []
            for node in session:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            time = time[:]
            
            if len(senders) != 1:
                del senders[-1]
                del receivers[0]
                del time[-1]

            pair = {}
            pair_time = {}
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            sur_time = time[:]
            i = 0
            for sender, receiver, time in zip(sur_senders, sur_receivers, sur_time):
                if str(sender)+'-'+str(receiver) in pair:
                    pair[str(sender)+'-'+str(receiver)] += 1
                    pair_time[str(sender)+'-'+str(receiver)] += time
                    del senders[i]
                    del receivers[i]
                else:
                    pair[str(sender)+'-'+str(receiver)] = 1
                    pair_time[str(sender)+'-'+str(receiver)] = time
                    i += 1
            pair_time_fi = [pair_time[str(senders[i])+'-'+str(receivers[i])] for i in range(len(senders))]
        
            count_out = collections.Counter(senders)
            count_in = collections.Counter(receivers)
            out_degree_inv = torch.tensor([1/count_out[i] for i in senders], dtype=torch.float)
            in_degree_inv = torch.tensor([1 / count_in[i] for i in receivers], dtype=torch.float)
            edge_attr = torch.tensor(pair_time_fi,
                                             dtype=torch.float)
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            sequence = torch.tensor(session, dtype=torch.long)
            sequence_len = torch.tensor([len(session)], dtype=torch.long)
            session_graph = Data(x=x, y=y,
                                         edge_index=edge_index, edge_attr=edge_attr,
                                         sequence=sequence, sequence_len=sequence_len,
                                         out_degree_inv=out_degree_inv, in_degree_inv=in_degree_inv)
            data_list.append(session_graph)
    return DataLoader(data_list, batch_size = 32, shuffle=shuffle)

#aking session graphs for each set
#seed_torch(2) #graphs without time
loader_demo_notime = making_graphs(sess_clicks_demo, sess_labs_demo, sess_times_demo, "no_time", True)
loader_train_notime = making_graphs(sess_clicks_train, sess_labs_train, sess_times_train, "no_time", True)
loader_val_notime = making_graphs(sess_clicks_val, sess_labs_val, sess_times_val, "no_time", True)
loader_valtrain_notime = making_graphs(sess_clicks_valtrain, sess_labs_valtrain, sess_times_valtrain, "no_time", True)
loader_test_notime = making_graphs(sess_clicks_test, sess_labs_test, sess_times_test, "no_time", True)

#seed_torch(2) #graphs with time
loader_demo_withtime = making_graphs(sess_clicks_demo, sess_labs_demo, sess_times_demo, "with_time", True)
loader_train_withtime = making_graphs(sess_clicks_train, sess_labs_train, sess_times_train, "with_time", True)
loader_val_withtime = making_graphs(sess_clicks_val, sess_labs_val, sess_times_val, "with_time", True)
loader_valtrain_withtime = making_graphs(sess_clicks_valtrain, sess_labs_valtrain, sess_times_valtrain, "with_time", True)
loader_test_withtime = making_graphs(sess_clicks_test, sess_labs_test, sess_times_test, "with_time", True)

seed_torch(2) #graphs with time only
loader_demo_timeonly = making_graphs(sess_clicks_demo, sess_labs_demo, sess_times_demo, "time_only", True)
loader_train_timeonly = making_graphs(sess_clicks_train, sess_labs_train, sess_times_train, "time_only", True)
loader_val_timeonly = making_graphs(sess_clicks_val, sess_labs_val, sess_times_val, "time_only", True)
loader_valtrain_timeonly = making_graphs(sess_clicks_valtrain, sess_labs_valtrain, sess_times_valtrain, "time_only", True)
loader_test_timeonly = making_graphs(sess_clicks_test, sess_labs_test, sess_times_test, "time_only", True)

#building model
#readout function from SR-GNN, i = qT (W1vn +W2vi + c); (inspired by Wu et al.(2019))
class Embedding2Score(nn.Module): 
    def __init__(self, hidden_size):
        super(Embedding2Score, self).__init__()
        self.hidden_size = hidden_size
        self.W_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.q = nn.Linear(self.hidden_size, 1)
        self.W_3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def forward(self, session_embedding, all_item_embedding, batch):
        sections = torch.bincount(batch) #record the number of items of the same individual session graph [11, 10,  9,  8,  7,  6,  5, ...
        v_i = torch.split(session_embedding, tuple(sections.cpu().numpy()))    # split whole batched graph back into individual graphs
        v_n_repeat = tuple(graph[-1].view(1, -1).repeat(graph.shape[0], 1) for graph in v_i)    # repeat |V|_i times for the last node embedding
        #
        # Eq(6)
        alpha = self.q(torch.sigmoid(self.W_1(torch.cat(v_n_repeat, dim=0)) + self.W_2(session_embedding)))    # |V|_i * 1
        s_g_whole = alpha * session_embedding    # |V|_i * hidden_size
        s_g_split = torch.split(s_g_whole, tuple(sections.cpu().numpy()))    # split whole s_g into graphs G_i
        s_g = tuple(torch.sum(embeddings, dim=0).view(1, -1) for embeddings in s_g_split)
        
        # Eq(7)
        v_n = tuple(nodes[-1].view(1, -1) for nodes in v_i)
        s_h = self.W_3(torch.cat((torch.cat(v_n, dim=0), torch.cat(s_g, dim=0)), dim=1))
        
        # Eq(8)
        z_i_hat = torch.matmul(s_h, all_item_embedding.weight[1:].transpose(1, 0))
        # change mm into matmul, add [1:] behind weight
        
        return z_i_hat

#nn layers
class testGNN(nn.Module):
    """
    Args:
        hidden_size: the number of units in a hidden layer.
        n_node: the number of items in the whole item set for embedding layer.
    """
    def __init__(self, hidden_size, n_node):
        super(testGNN, self).__init__()
        self.hidden_size, self.n_node = hidden_size, n_node
        self.embedding = torch.nn.Embedding(self.n_node, self.hidden_size)
        self.gcn = GCNConv(self.hidden_size, self.hidden_size)
        self.gcn2 = GCNConv(self.hidden_size, self.hidden_size)
        self.ggcn = GatedGraphConv(self.hidden_size, self.hidden_size)
        self.gat1 = GATConv(self.hidden_size, self.hidden_size, heads=8, negative_slope=0.2)
        self.gat2 = GATConv(8 * self.hidden_size, self.hidden_size, heads=1, negative_slope=0.2)
        self.e2s = Embedding2Score(self.hidden_size)
        
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        embedding = self.embedding(x).squeeze()
        x = self.gcn(embedding, edge_index, edge_attr.float())
        #x = self.gcn2(x, edge_index, edge_attr.float())
        #x = self.gcn(x, edge_index, edge_attr.float())
        #x = F.relu(x)
        #x = self.gat1(x, edge_index)
        #x = F.relu(x)
        #x = self.gat2(x, edge_index)
        x = self.e2s(x, self.embedding, batch)
        return x


#forward function to make the model runing (inspired by Qiu et al.(2020))
def forward(model, loader, device, top_k=20, optimizer=None, train_flag=True):
    running_loss = []
    train_loss = []
    if train_flag:
        model.train()
    else:
        model.eval()
        hit, mrr = [], []

    for batch in loader:
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device))
        targets = batch.y
        loss = criterion(scores, targets)

        if train_flag:
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 0.5)
            optimizer.step()
            running_loss.append(loss.item())
        else:
            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr.append(0)
                else:
                    mrr.append(1 / (np.where(score == target)[0][0] + 1))

    if train_flag:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("this loop finish at", current_time)
        train_loss = np.average(running_loss)
        print("the train loss is", train_loss)
        return current_time, train_loss
        

    else:
        hit = np.mean(hit) * 100
        mrr = np.mean(mrr) * 100
        return  hit, mrr


# demo batch: the fourth batch in the demo batch list, could be used to understand how the representation looks like
batchlist = []
for batch in loader_demo_notime:
    batchlist.append(batch)
testbatch = batchlist[0]

# pre-running settings
seed_torch(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #the runing time per loop for cuda is 8 mins, for cpu is 22 mins
model = testGNN(125, 11000).to(device) #the num_nodes depends on the number of las nodes
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss() #this is a multiclass graph classification task
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1) 



#reseting model
#seed_torch(2)
#for layer in model.children():z
#   if hasattr(layer, 'reset_parameters'):
#       layer.reset_parameters()

#training and validation

seed_torch(2)
turn_result = []
train_loss = []
for epoch in range(13):
    time, loss = forward(model, loader_train_notime, device, top_k=20, optimizer=optimizer, train_flag=True)
    train_loss.append([time, loss])
    scheduler.step()
    hit_val, mrr_val = forward(model, loader_val_notime, device, top_k=20, optimizer=optimizer, train_flag=False)
    turn_result.append([hit_val, mrr_val])





#testing
seed_torch(2)
for epoch in range(12):
    forward(model, loader_valtrain_timeonly, device, top_k=20, optimizer=optimizer, train_flag=True)
    scheduler.step()
hit, mrr = forward(model, loader_test_timeonly, device, top_k=20, optimizer=optimizer, train_flag=False)


#save results
import pandas as pd
df = pd.DataFrame(turn_result, columns = ["hit", 'mrr'])
df.to_csv("direct_notime_scores.csv", index = False)
df = pd.DataFrame(train_loss, columns = ["time", 'loss'])
df.to_csv("direct_notime_loss.csv", index = False)




#making graphs
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

os.chdir("C:/Users/peter\'s lap/Master thesis/")


def fig(loss, name, title):
    f = plt.figure()
    x_major_locator=MultipleLocator(1)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.plot(loss)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    f.savefig(name, bbox_inches='tight')

notime_loss = [65.84438273,
93.63968265,
97.05681963,
91.06143775,
95.16821282,
12.62434361,
9.575152116,
8.683380355,
8.045302126,
7.688036423,
6.546929793,
6.412597543,
6.35093828
    ]

fig(notime_loss, "withtime_loss_inverse_0.007.pdf", "\"occurrence inverse-normalized dwell time\" validation loss")
import numpy as np
np.var([1,1,1,1,1,1,1,1,2,1,1,1,1,2,1,1,1,1,1,1,1,1])
