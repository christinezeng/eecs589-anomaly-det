import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn
from imblearn.over_sampling import SMOTE

import torch
torch.manual_seed(0)

df = pd.read_csv("UNSW-NB15/88695f0f620eb568_MOHANAD_A4706/data/NF-UNSW-NB15.csv")
df = df.drop(columns=['L4_SRC_PORT', 'L4_DST_PORT', 'Label'])
# df['Date first seen'] = pd.to_datetime(df['Date first seen'])

count_labels = df['Attack'].value_counts() / len(df) * 100
print(count_labels)
plt.pie(count_labels[:3], labels=df['Attack'].unique()[:3], autopct='%.0f%%')
# plt.show()
plt.savefig('./countUNSW-NB15.png')
# df['weekday'] = df['Date first seen'].dt.weekday
# df = pd.get_dummies(df, columns=['weekday']).rename(columns = {'weekday_0': 'Monday',
#                                                               'weekday_1': 'Tuesday',
#                                                               'weekday_2': 'Wednesday',
#                                                               'weekday_3': 'Thursday',
#                                                               'weekday_4': 'Friday',
#                                                               'weekday_5': 'Saturday',
#                                                               'weekday_6': 'Sunday',
#                                                              })

# df['daytime'] = (df['Date first seen'].dt.second +df['Date first seen'].dt.minute*60 + df['Date first seen'].dt.hour*60*60)/(24*60*60)

# def one_hot_flags(input):
#     return [1 if char1 == char2 else 0 for char1, char2 in zip('APRSF', input[1:])]

# df = df.reset_index(drop=True)
# # ohe_flags = one_hot_flags(df['Flags'].to_numpy())
# ohe_flags = df['Flags'].apply(one_hot_flags).to_list()
# df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohe_flags, columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])
# df = df.drop(columns=['Date first seen', 'Flags'])
# df
ip_to_index = {}
index = 0

for src_ip in df['IPV4_SRC_ADDR']:
    if src_ip not in ip_to_index:
        ip_to_index[src_ip] = index
        index += 1

for dest_ip in df['IPV4_DST_ADDR']:
    if dest_ip not in ip_to_index:
        ip_to_index[dest_ip] = index
        index += 1

temp = pd.DataFrame()
temp['SrcIP'] = df['IPV4_SRC_ADDR'].map(ip_to_index)
# temp = temp['SrcIP'].str.split('.', expand=True).rename(columns = {2: 'ipsrc3', 3: 'ipsrc4'}).astype(int)[['ipsrc3', 'ipsrc4']]
temp['SrcIP'] = temp['SrcIP'].apply(lambda x: format(x, "b").zfill(6))
df = df.join(temp['SrcIP'].str.split('', expand=True)
            .drop(columns=[0, 7])
            .rename(columns=dict(enumerate([f'ipsrc_{i}' for i in range(7)])))
            .astype('int32'))
print("----------------------------------------------")
print("Source IPs in binary form")
print(df.head(5))

temp = pd.DataFrame()
temp['DstIP'] = df['IPV4_DST_ADDR'].map(ip_to_index)
# temp = temp['DstIP'].str.split('.', expand=True).rename(columns = {2: 'ipdst3', 3: 'ipdst4'}).astype(int)[['ipdst3', 'ipdst4']]
temp['DstIP'] = temp['DstIP'].apply(lambda x: format(x, "b").zfill(6))
df = df.join(temp['DstIP'].str.split('', expand=True)
            .drop(columns=[0, 7])
            .rename(columns=dict(enumerate([f'ipdst_{i}' for i in range(7)])))
            .astype('int32'))
print("----------------------------------------------")
print("Destination IPs in binary form")
print(df.head(5))

def string_to_int(col_name):
    m_index = df[pd.to_numeric(df[col_name], errors='coerce').isnull()].index
    df[col_name].loc[m_index] = df[col_name].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce', downcast='integer')


def string_to_int_flow(col_name):
    m_index = df[pd.to_numeric(df[col_name], errors='coerce').isnull()].index
    df[col_name].loc[m_index] = df[col_name].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
    df[col_name] = pd.to_numeric(df[col_name] * 0.001, errors='coerce', downcast='integer')


string_to_int('IN_BYTES')
string_to_int('OUT_BYTES')
string_to_int('IN_PKTS')
string_to_int('OUT_PKTS')
string_to_int_flow('FLOW_DURATION_MILLISECONDS')


df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Attack'])
print("----------------------------------------------")
print("final transformation")
print(df.head(5))


labels = ['Benign', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS', 'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
df_train, df_test = train_test_split(df, random_state=0, test_size=0.2, stratify=df[labels])
df_val, df_test = train_test_split(df_test, random_state=0, test_size=0.5, stratify=df_test[labels])

# Separate features and labels for SMOTE
# X_train = df_train.drop(labels, axis=1)
# y_train = df_train[labels]
# smote = SMOTE(random_state=0)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
# df_train_smote = pd.concat([X_train_smote, y_train_smote], axis=1)

scaler = PowerTransformer()
# df_train_smote[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']] = scaler.fit_transform(df_train_smote[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']])
df_train[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']] = scaler.fit_transform(df_train[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']])
df_val[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']] = scaler.transform(df_val[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']])
df_test[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']] = scaler.transform(df_test[['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS']])
df_train[df_train['Benign'] == 1]
# df_train_smote[df_train_smote['Benign'] == 1]

BATCH_SIZE = 16
features_host = [f'ipsrc_{i}' for i in range(1, 7)] + [f'ipdst_{i}' for i in range(1, 7)]
features_flow = ['FLOW_DURATION_MILLISECONDS', 'IN_PKTS', 'OUT_PKTS', 'IN_BYTES', 'OUT_BYTES', 'PROTOCOL', 'L7_PROTO', 'TCP_FLAGS']
# features_flow = ['Duration', 'Packets', 'Bytes', 'ACK', 'PSH', 'RST', 'SYN', 'FIN', 'ICMP ', 'IGMP ', 'TCP  ', 'UDP  ']

def get_connections(ip_map, src_ip, dst_ip):
    src1 = [ip_map[ip] for ip in src_ip]
    src2 = [ip_map[ip] for ip in dst_ip]
    src = np.column_stack((src1, src2)).flatten()
    dst = list(range(len(src_ip)))
    dst = np.column_stack((dst, dst)).flatten()
    
    return torch.Tensor([src, dst]).int(), torch.Tensor([dst, src]).int()

def create_dataloader(df, subgraph_size=1024):
    data = []
    n_subgraphs = len(df) // subgraph_size
    for i in range(1, n_subgraphs+1):
        subgraph = df[(i-1)*subgraph_size:i*subgraph_size]
        src_ip = subgraph['IPV4_SRC_ADDR'].to_numpy()
        dst_ip = subgraph['IPV4_DST_ADDR'].to_numpy()
        
        ip_map = {ip:index for index, ip in enumerate(np.unique(np.append(src_ip, dst_ip)))}
        host_to_flow, flow_to_host = get_connections(ip_map, src_ip, dst_ip)

        batch = HeteroData()
        batch['host'].x = torch.Tensor(subgraph[features_host].to_numpy().astype(float))
        batch['flow'].x = torch.Tensor(subgraph[features_flow].to_numpy().astype(float))
        batch['flow'].y = torch.Tensor(subgraph[labels].to_numpy().astype(float))
        batch['host','flow'].edge_index = host_to_flow
        batch['flow','host'].edge_index = flow_to_host
        data.append(batch)

    return DataLoader(data, batch_size=BATCH_SIZE)

# train_loader = create_dataloader(df_train_smote)
train_loader = create_dataloader(df_train)
val_loader = create_dataloader(df_val)
test_loader = create_dataloader(df_test)


from torch_geometric.nn import Linear, HeteroConv, SAGEConv, GATConv

class HeteroGNN(torch.nn.Module):
    def __init__(self, dim_h, dim_out, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('host', 'to', 'flow'): SAGEConv((-1,-1), dim_h),
                ('flow', 'to', 'host'): SAGEConv((-1,-1), dim_h),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = Linear(dim_h, dim_out)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['flow'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGNN(dim_h=64, dim_out=10, num_layers=3).to(device)
optimizer = Adam(model.parameters(), lr=0.001)

@torch.no_grad()
def test(loader):
    model.eval()
    y_pred = []
    y_true = []
    n_subgraphs = 0
    total_loss = 0

    for batch in loader:
        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        # print("out shape:", out.shape)
        # targets = torch.flatten(batch['flow'].y.float())
        targets = batch['flow'].y.float()
        # print("targets shape:", targets.shape)
        loss = F.cross_entropy(out, targets)
        y_pred.append(out.argmax(dim=1))
        y_true.append(batch['flow'].y.argmax(dim=1))
        n_subgraphs += BATCH_SIZE
        total_loss += float(loss) * BATCH_SIZE
        
    y_pred = torch.cat(y_pred).cpu()
    y_true = torch.cat(y_true).cpu()
    f1score = f1_score(y_true, y_pred, average='macro')

    return total_loss/n_subgraphs, f1score, y_pred, y_true


model.train()
for epoch in range(101):
    n_subgraphs = 0
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        batch.to(device)
        out = model(batch.x_dict, batch.edge_index_dict)
        # print("out shape:", out.shape)
        targets = batch['flow'].y.float()
        # print("targets shape:", targets.shape)
        loss = F.cross_entropy(out, targets)
        loss.backward()
        optimizer.step()

        n_subgraphs += BATCH_SIZE
        total_loss += float(loss) * BATCH_SIZE

    if epoch % 10 == 0:
        val_loss, f1score, _, _ = test(val_loader)
        print(f'Epoch {epoch} | Loss: {total_loss/n_subgraphs:.4f} | Val loss: {val_loss:.4f} | Val F1-score: {f1score:.4f}')       

_, _, y_pred, y_true = test(test_loader)

print(classification_report(y_true, y_pred, target_names=labels, digits=4))

plt.rcParams.update({'font.size':8})
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, normalize='true')
disp.plot(xticks_rotation=45, values_format='.2f', cmap='Blues')
plt.savefig('./confusionMatrixUNSW-NB15.png')
