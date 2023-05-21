import pandas as pd

df_path = './df.csv'
df = pd.read_csv(df_path)
class CFG:
    model_name = 'convnext_large_d_320' 
    model_data = 'laion2b_s29b_b131k_ft_soup'
    seed = 5
    workers = 8
    train_batch_size = 32
    valid_batch_size = 4
    emb_size = 512
    vit_bb_lr = {'8': 1.25e-6, '16': 2.5e-6, '20': 5e-6, '24': 10e-6} 
    vit_bb_wd = 1e-3
    hd_lr = 3e-4
    hd_wd = 1e-5
    autocast = True
    n_warmup_steps = 1000
    n_epochs = 3
    device = 'cuda'
    s=30.
    m=.45
    m_min=.05
    acc_steps = 4
    global_step = 0
    df = pd.read_csv(df_path).reset_index(drop=True)
    n_classes = len(df['label_idx'].unique())