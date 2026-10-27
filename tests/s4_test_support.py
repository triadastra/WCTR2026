"""Small sparse trade panel shared by isolated vendored-S4 CPU checks."""
import pandas as pd
import src.dataloaders.trade_unified as unified

def synthetic(data_dir, input_len, lag_count, aggregate=False):
    cfg=unified._GH.DataConfig(start='2009-01-01', end='2025-05-01', input_len=input_len, lag_count=lag_count,
        train_end='2023-12-01', val_start='2024-01-01', val_end='2024-12-01',
        test_start='2025-01-01', test_end='2025-05-01')
    pipe=unified._GH.TradeDataPipeline(cfg)
    rows=[]
    for si,s in enumerate(['A','B']):
      for ci,c in enumerate(['01','02','03']):
       for flow in ['Export','Import']:
        if (s,c,flow)==('B','03','Export'): continue
        for i,t in enumerate(pd.date_range(cfg.start,cfg.end,freq='MS')):
         rows.append(dict(State=s,Commodity=c,Country='X',Time=t,Value=(i+1)*(si+ci+1),Weight=(i+3)*(si+ci+2),**{'Import/Export':flow}))
    pipe.df_full=pd.DataFrame(rows)
    pipe.source_fingerprint='a'*64
    pipe.source_domain={'synthetic':True}
    pipe.create_splits()
    return pipe
