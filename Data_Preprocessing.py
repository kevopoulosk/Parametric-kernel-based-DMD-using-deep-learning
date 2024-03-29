### File for the data processing FreeFem --> Python format.
### We read the vertices from the Finite element mesh.
### Then, these data are transformed to proper snapshot matrices in python format


import numpy as np
from tqdm import tqdm


def extract_vertices(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    l_num = int(np.ceil(v_num/5))
    TargetMSH.close()
    print('TOTAL VERTICES:', v_num)

    ### Create a list and copy out all the boundary vertices
    VerticeList = []

    for i in range(1,l_num+1):
        with open(directory_to_txt,'r') as txt:

            text = txt.readlines()
            currentline = text[i]
            coordinates = currentline.strip().split()

            for j in range(len(coordinates)):
                VerticeList.append(np.float64(coordinates[j]))
    return np.expand_dims(np.asarray(VerticeList),axis=1)


def vertices_num(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    TargetMSH.close()
    return v_num


### Collecting data generated from FreeFem
DirectoryData = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData"
DirectoryProcessed = "/Users/konstantinoskevopoulos/Documents/Heat_Eq_Thesis/SnapshotData_Processed"
SampleSize = 500
T_end = 5
dt = 0.01
t_span = np.linspace(dt, T_end, int(T_end/dt))

v_num = vertices_num(DirectoryData+'/mu0/t=0.01.txt')
snapshots = []

pbar = tqdm(position=0, leave=True, total=SampleSize, desc="Preprocessing of samples")
for i in range(SampleSize):
    c_list = np.empty((v_num, 0))
    num = 0
    for t in t_span:
        num += 1
        if num % 100 == 0:
            c = extract_vertices(DirectoryData+f"/mu{i}/t={'{0:.0f}'.format(t)}.txt")
        elif num % 10 == 0:
            c = extract_vertices(DirectoryData+f"/mu{i}/t={'{0:.1f}'.format(t)}.txt")
        else:
            c = extract_vertices(DirectoryData + f"/mu{i}/t={'{0:.2f}'.format(t)}.txt")
        c_list = np.hstack((c_list, c))

    snapshots.append(c_list)
    np.save(DirectoryProcessed+f"/sample{i}", c_list)

    pbar.update()
pbar.close()

print(f" All the snapshots are in the data structure {snapshots} with length {len(snapshots)}\n "
      f"and the first snapshot matrix is {snapshots[0]} with shape {snapshots[0].shape}")
