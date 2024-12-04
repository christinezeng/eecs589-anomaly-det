import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE  # Import SMOTE

from torch_geometric.loader import DataLoader
from torch_geometric.data import HeteroData
from torch.nn import functional as F
from torch.optim import Adam
from torch import nn

import torch
torch.manual_seed(0)

df = pd.read_csv("CIDDS-001-internal-week1.csv")
print("Read CSV")
df = df.drop(columns=['Src Pt', 'Dst Pt', 'Flows', 'Tos', 'class', 'attackID', 'attackDescription'])
df['attackType'] = df['attackType'].replace('---', 'benign')
df['Date first seen'] = pd.to_datetime(df['Date first seen'])
df

count_labels = df['attackType'].value_counts() / len(df) * 100
print(count_labels)
plt.pie(count_labels[:3], labels=df['attackType'].unique()[:3], autopct='%.0f%%')
#plt.show()
plt.savefig('./count.png')
df['weekday'] = df['Date first seen'].dt.weekday
df = pd.get_dummies(df, columns=['weekday']).rename(columns = {'weekday_0': 'Monday',
                                                              'weekday_1': 'Tuesday',
                                                              'weekday_2': 'Wednesday',
                                                              'weekday_3': 'Thursday',
                                                              'weekday_4': 'Friday',
                                                              'weekday_5': 'Saturday',
                                                              'weekday_6': 'Sunday',
                                                             })

df['daytime'] = (df['Date first seen'].dt.second +df['Date first seen'].dt.minute*60 + df['Date first seen'].dt.hour*60*60)/(24*60*60)

def one_hot_flags(input):
    return [1 if char1 == char2 else 0 for char1, char2 in zip('APRSF', input[1:])]

df = df.reset_index(drop=True)
ohe_flags = one_hot_flags(df['Flags'].to_numpy())
ohe_flags = df['Flags'].apply(one_hot_flags).to_list()
df[['ACK', 'PSH', 'RST', 'SYN', 'FIN']] = pd.DataFrame(ohe_flags, columns=['ACK', 'PSH', 'RST', 'SYN', 'FIN'])
df = df.drop(columns=['Date first seen', 'Flags'])
df

temp = pd.DataFrame()
temp['SrcIP'] = df['Src IP Addr'].astype(str)
temp['SrcIP'][~temp['SrcIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
temp = temp['SrcIP'].str.split('.', expand=True).rename(columns = {2: 'ipsrc3', 3: 'ipsrc4'}).astype(int)[['ipsrc3', 'ipsrc4']]
temp['ipsrc'] = temp['ipsrc3'].apply(lambda x: format(x, "b").zfill(8)) + temp['ipsrc4'].apply(lambda x: format(x, "b").zfill(8))
df = df.join(temp['ipsrc'].str.split('', expand=True)
            .drop(columns=[0, 17])
            .rename(columns=dict(enumerate([f'ipsrc_{i}' for i in range(1, 17)])))
            .astype('int32'))
df.head(5)

temp = pd.DataFrame()
temp['DstIP'] = df['Dst IP Addr'].astype(str)
temp['DstIP'][~temp['DstIP'].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
temp = temp['DstIP'].str.split('.', expand=True).rename(columns = {2: 'ipdst3', 3: 'ipdst4'}).astype(int)[['ipdst3', 'ipdst4']]
temp['ipdst'] = temp['ipdst3'].apply(lambda x: format(x, "b").zfill(8)) \
                + temp['ipdst4'].apply(lambda x: format(x, "b").zfill(8))
df = df.join(temp['ipdst'].str.split('', expand=True)
            .drop(columns=[0, 17])
            .rename(columns=dict(enumerate([f'ipdst_{i}' for i in range(1, 17)])))
            .astype('int32'))
df.head(5)


m_index = df[pd.to_numeric(df['Bytes'], errors='coerce').isnull() == True].index
df['Bytes'].loc[m_index] = df['Bytes'].loc[m_index].apply(lambda x: 10e6 * float(x.strip().split()[0]))
df['Bytes'] = pd.to_numeric(df['Bytes'], errors='coerce', downcast='integer')

df = pd.get_dummies(df, prefix='', prefix_sep='', columns=['Proto', 'attackType'])
df.head(5)


labels = ['benign', 'bruteForce', 'dos', 'pingScan', 'portScan']

# Add 'label' column
df['label'] = df[labels].idxmax(axis=1)

# Few-shot learning setup
K = 10  # Increased number of examples per class to allow SMOTE
df_train = df.groupby('label').apply(lambda x: x.sample(n=K, random_state=0)).reset_index(drop=True)
df_remaining = df.drop(df_train.index)

df_val, df_test = train_test_split(df_remaining, random_state=0, test_size=0.5, stratify=df_remaining['label'])

# Apply SMOTE to the training data
features_flow = ['daytime', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Duration', 'Packets', 'Bytes',
                 'ACK', 'PSH', 'RST', 'SYN', 'FIN', 'ICMP ', 'IGMP ', 'TCP  ', 'UDP  ']

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X = df_train[features_flow]
y = df_train['label']
y_encoded = le.fit_transform(y)

smote = SMOTE(random_state=0, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Reconstruct df_train with resampled data
df_resampled = pd.DataFrame(X_resampled, columns=features_flow)
df_resampled['label'] = le.inverse_transform(y_resampled)

# Assign 'Src IP Addr' and 'Dst IP Addr' by sampling from existing df_train
df_resampled['Src IP Addr'] = np.random.choice(df_train['Src IP Addr'], size=len(df_resampled))
df_resampled['Dst IP Addr'] = np.random.choice(df_train['Dst IP Addr'], size=len(df_resampled))

# Recompute 'ipsrc' and 'ipdst' features based on 'Src IP Addr' and 'Dst IP Addr'
def compute_ip_features(df, ip_column, prefix):
    temp = pd.DataFrame()
    temp[ip_column] = df[ip_column].astype(str)
    temp[ip_column][~temp[ip_column].str.contains('\d{1,3}\.', regex=True)] = '0.0.0.0'
    temp = temp[ip_column].str.split('.', expand=True).rename(columns = {2: f'{prefix}3', 3: f'{prefix}4'}).astype(int)[[f'{prefix}3', f'{prefix}4']]
    temp[f'{prefix}'] = temp[f'{prefix}3'].apply(lambda x: format(x, "b").zfill(8)) + temp[f'{prefix}4'].apply(lambda x: format(x, "b").zfill(8))
    ip_features = temp[f'{prefix}'].str.split('', expand=True).drop(columns=[0, 17]).rename(columns=dict(enumerate([f'{prefix}_{i}' for i in range(1,17)]))).astype('int32')
    return ip_features

df_resampled = df_resampled.reset_index(drop=True)
ipsrc_features = compute_ip_features(df_resampled, 'Src IP Addr', 'ipsrc')
df_resampled = pd.concat([df_resampled, ipsrc_features], axis=1)

ipdst_features = compute_ip_features(df_resampled, 'Dst IP Addr', 'ipdst')
df_resampled = pd.concat([df_resampled, ipdst_features], axis=1)

# Ensure that all necessary columns are included
df_train = df_resampled

# Apply scaler to numerical columns
scaler = PowerTransformer()
df_train[['Duration', 'Packets', 'Bytes']] = scaler.fit_transform(df_train[['Duration', 'Packets', 'Bytes']])
df_val[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_val[['Duration', 'Packets', 'Bytes']])
df_test[['Duration', 'Packets', 'Bytes']] = scaler.transform(df_test[['Duration', 'Packets', 'Bytes']])
df_train[df_train['benign'] == 1]

# Adjust BATCH_SIZE for few-shot learning
BATCH_SIZE = len(df_train)  # Set batch size to number of training examples

features_host = [f'ipsrc_{i}' for i in range(1, 17)] + [f'ipdst_{i}' for i in range(1, 17)]
# features_flow is already defined above

def get_connections(ip_map, src_ip, dst_ip):
    src1 = [ip_map[ip] for ip in src_ip]
    src2 = [ip_map[ip] for ip in dst_ip]
    src = np.column_stack((src1, src2)).flatten()
    dst = list(range(len(src_ip)))
    dst = np.column_stack((dst, dst)).flatten()

    return torch.Tensor([src, dst]).int(), torch.Tensor([dst, src]).int()

def create_dataloader(df, subgraph_size=None):
    data = []
    if subgraph_size is None:
        subgraph_size = len(df)
    n_subgraphs = len(df) // subgraph_size
    for i in range(1, n_subgraphs+1):
        subgraph = df[(i-1)*subgraph_size:i*subgraph_size]
        src_ip = subgraph['Src IP Addr'].to_numpy()
        dst_ip = subgraph['Dst IP Addr'].to_numpy()

        ip_map = {ip:index for index, ip in enumerate(np.unique(np.append(src_ip, dst_ip)))}
        host_to_flow, flow_to_host = get_connections(ip_map, src_ip, dst_ip)

        batch = HeteroData()
        batch['host'].x = torch.Tensor(subgraph[features_host].to_numpy().astype(float))
        batch['flow'].x = torch.Tensor(subgraph[features_flow].to_numpy().astype(float))
        batch['flow'].y = torch.Tensor(pd.get_dummies(subgraph['label'])[labels].to_numpy().astype(float))
        batch['host','flow'].edge_index = host_to_flow
        batch['flow','host'].edge_index = flow_to_host
        data.append(batch)

    return DataLoader(data, batch_size=BATCH_SIZE)

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
model = HeteroGNN(dim_h=64, dim_out=5, num_layers=3).to(device)
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
        loss = F.cross_entropy(out, batch['flow'].y.float())
        y_pred.append(out.argmax(dim=1))
        y_true.append(batch['flow'].y.argmax(dim=1))
        n_subgraphs += 1  # Adjusted for batch size
        total_loss += float(loss)

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
        loss = F.cross_entropy(out, batch['flow'].y.float())
        loss.backward()
        optimizer.step()

        n_subgraphs += 1  # Adjusted for batch size
        total_loss += float(loss)

    if epoch % 10 == 0:
        val_loss, f1score, _, _ = test(val_loader)
        print(f'Epoch {epoch} | Loss: {total_loss/n_subgraphs:.4f} | Val loss: {val_loss:.4f} | Val F1-score: {f1score:.4f}')

_, _, y_pred, y_true = test(test_loader)

print(classification_report(y_true, y_pred, target_names=labels, digits=4))
