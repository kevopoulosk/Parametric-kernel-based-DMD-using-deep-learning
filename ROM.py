import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import TruncatedSVD


def vertices_num(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    ### Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    TargetMSH.close()
    return v_num


def extract_vertices(directory_to_txt):
    TargetMSH = open(directory_to_txt, "r+")

    # Read the total number of vertices(points) from .msh file
    line = TargetMSH.readline()
    words = line.strip().split()
    v_num = int(words[0])
    l_num = int(np.ceil(v_num / 5))
    TargetMSH.close()
    # print('TOTAL VERTICES:',v_num)

    # Create a list and copy out all the boundary vertices
    VerticeList = []

    for i in range(1, l_num + 1):
        with open(directory_to_txt, 'r') as txt:

            text = txt.readlines()
            currentline = text[i]
            coordinates = currentline.strip().split()

            for j in range(len(coordinates)):
                VerticeList.append(np.float64(coordinates[j]))
    return np.expand_dims(np.asarray(VerticeList), axis=1)


def SVD(directory, attribute):
    snapshot_matrix = np.load(directory + "POD" + attribute + ".npy")
    sampled_velocities = snapshot_matrix.shape[1]
    print(f"the snapshot matrix is{snapshot_matrix}")
    training_ratio = 0.75
    training_samples = int(training_ratio * sampled_velocities)

    training_X = snapshot_matrix[:, :training_samples]

    desired_components = 10
    svd = TruncatedSVD(n_components=desired_components, n_iter=7, random_state=42)
    svd.fit(training_X)

    system_energy = np.sum(svd.explained_variance_ratio_)
    while system_energy < 0.999:
        svd = TruncatedSVD(n_components=desired_components, n_iter=7, random_state=42)
        svd.fit(training_X)

        system_energy = np.sum(svd.explained_variance_ratio_)
        if system_energy < 0.999:
            desired_components += 1
        else:
            pass
    print(f"SVD completed with {system_energy} % of the system energy explained")
    print(f"The most informative singular values are {svd.singular_values_}")
    print(f"The number of singular values used (truncation threshold) is {desired_components}")

    plt.plot(svd.explained_variance_ratio_, "-o", label="Singular Values")
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Singular values")
    plt.legend()
    plt.title("Singular values")
    plt.show()

    return svd.fit_transform(training_X), snapshot_matrix


generated_data_directory = "/Users/konstantinoskevopoulos/Documents/SnapshotData/lid_driven_cavity/"
# number of sampled velocities
num_samples = 240

# Read number of vertices and create empty matrix for data storage
v_num = vertices_num(generated_data_directory + 'u/sample_0_u.txt')
# print(v_num)
ulist = np.empty((v_num, 0))
vlist = np.empty((v_num, 0))
plist = np.empty((v_num, 0))

# Collect all the snapshots and concatenate them in the "A" matrix
for samplenum in range(num_samples):
    print('Processing sample:', samplenum)
    u = extract_vertices(generated_data_directory + 'u/sample_' + str(samplenum) + '_u.txt')
    v = extract_vertices(generated_data_directory + 'v/sample_' + str(samplenum) + '_v.txt')
    p = extract_vertices(generated_data_directory + 'p/sample_' + str(samplenum) + '_p.txt')

    ulist = np.hstack((ulist, u))
    vlist = np.hstack((vlist, v))
    plist = np.hstack((plist, p))

# Save snapshot matrix (v_num,num_sample)
np.save(generated_data_directory + 'ROM/PODu.npy', ulist)
np.save(generated_data_directory + 'ROM/PODv.npy', vlist)
np.save(generated_data_directory + 'ROM/PODp.npy', plist)
print('Snapshot matrix shape: ', ulist.shape)


res, snapshot = SVD(directory="/Users/konstantinoskevopoulos/Documents/SnapshotData/lid_driven_cavity/ROM/", attribute="u")
plt.figure(figsize=(10,6))
plt.contourf(snapshot)
plt.colorbar()
plt.xlabel("Number of snapshots-samples")
plt.ylabel("Degrees of freedom of the mesh")
plt.title("Collection of snapshots of u")
plt.show()