import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import streamlit as st
import os
import openpyxl

st.text('ASAP Version 1.0 (2021, November')

uploaded_file = st.file_uploader('Upload data ASAP file.CSV', type =['csv', 'xls', 'xlsx'])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df.columns = ['mass', 'intensity']
    st.write(uploaded_file)

    intensity_cutoff = st.number_input('Intensity cutoff: ', value=1)
    ru = st.number_input('Repeating unit: ', value=108.0028, format="%.5f")
    er = st.number_input('Delta mass error: ', value=0.001, step=0.0005, format="%.5f")
    mz_delta_cutoff = st.number_input('Lowest delta mass: ', value=14.01, format="%.5f")

    # filter
    df_filter = df[df['intensity'] >= intensity_cutoff]
    # st.write(len(df_filter))

    # Calculate delta frequency

    # Calculate delta matrix
    mz = np.array(df_filter['mass'])
    mz_reshape = mz.reshape(len(mz),1)
    deltas = mz-mz_reshape
    deltas = deltas[deltas >= mz_delta_cutoff]

    # Generate sorted list
    mz_range = int (deltas.max() - deltas.min() )
    # deltas_hist = np.histogram(deltas, bins=mz_range*10000)
    deltas_hist = np.histogram(deltas, bins=int (mz_range/(er)))
    a = np.append(deltas_hist[0],0)
    b = np.round(deltas_hist[1], decimals=4)
    data = {'delta mz':b, 'frequency':a}
    delta_hist = pd.DataFrame(data=data)
    delta_hist = delta_hist.sort_values('frequency', ascending=False)
    delta_hist = delta_hist.reset_index()
    main_delta = delta_hist['delta mz'][0]
    st.write('Table main deltas', delta_hist.head(5))
    st.write('Main delta =', main_delta)

# Check for delta and calculate end-groups
# st.write('df_filter =', len(df_filter))
result_end_group = np.zeros(len(df_filter))
result_mz1 = np.zeros(len(df_filter))
result_mz2 = np.zeros(len(df_filter))
result_delta = np.zeros(len(df_filter))
result_detect = np.zeros(len(df_filter))

for i in range(len(df_filter)):
    mz1 = df_filter.iloc[i][0]
    it1 = df_filter.iloc[i][1]
    for j in range(len(df_filter)):
        mz2 = df_filter.iloc[j][0]
        it2 = df_filter.iloc[j][1]
        d = mz2 - mz1
        if (d > (ru - er)) and (d < (ru + er)):
            result_end_group[j] = (mz2 / ru - math.floor(mz2 / ru)) * ru
            result_mz1[j] = mz1
            result_mz2[j] = mz2
            result_delta[j] = d

#st.write(result_end_group)
df_filter['end group'] = result_end_group
df_filter['mz1'] = result_mz1
df_filter['mz2'] = result_mz2
df_filter['delta'] = result_delta
st.write(df_filter)

# plot mass versus end-groups
fig = plt.figure(figsize = (10, 5))
plt.scatter(x=df_filter['mz1']/ru, y=df_filter['end group'], s=df_filter['intensity']*10, alpha=0.5)
plt.xlabel("mass / mass repeating unit")
plt.ylabel("mass end group")
plt.title("Mass versus Mass End group")
st.pyplot(fig)

path_file = st.text_input('path+file: ', 'D:/processed.xlsx')
if st.button('Save processed data to Excel'):
    df_filter.to_excel(path_file)

