from tqdm import tqdm
from Derivatives import *

dir  = "DATA/"
file = 'BHBH_Delay_Time.csv'

BHBH = pd.read_csv(dir+file, nrows=5e5).drop(['Delay_Time', 'Eccentricity_Delay',
                                              'Unnamed: 0.1','Unnamed: 0'], axis=1)

BHBH = BHBH.apply(lambda x: x.sample(frac=1).values)

# Sequential computation of the delay time for each entry

# initial time step
h = 1e-5
t = 0

# number of rows to compute
n = 10000
tqdm.pandas()

delay = BHBH.progress_apply(func=delay_time, axis='columns', args=(ODE_RK, h, t))
delay.rename(columns={0:'Delay_Time', 1:'Eccentricity_Delay'}, inplace=True)

shuffled = pd.concat([BHBH, delay], axis=1)
shuffled.to_csv(dir+'BHBH_Delay_Time_Shuffled.csv')