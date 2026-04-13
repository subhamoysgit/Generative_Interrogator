import pandas as pd
import numpy as np 
import h5py

def main():
    df = pd.read_csv('./data/index_sharps.csv')
    dict = {'file':[], 'dmin':[],
            'dmax':[], 'farea':[]}
    for i in range(len(df)):
        file = df['file'].iloc[i]
        dict['file'].append(file)
        file = h5py.File(file)
        key = list(file.keys())[0]
        data = np.array(file[key])
        data0 = data[0,:,:]
        mask = (data0 >=-1000)*(data0 <= -100) | (data0>=100)*(data0 <= 1000)
        mask_ = (data0 <= -100) | (data0 >= 100)
        dict['farea'].append(mask.sum()/mask_.sum())
        mn = np.nanmin(data0)
        mx = np.nanmax(data0)
        dict['dmin'].append(mn)
        dict['dmax'].append(mx)
        print(i, f'--of {len(df)}')
        dff = pd.DataFrame(dict)
        dff.to_csv('./data/sharp_Br_props.csv')

if __name__=='__main__':
    main()