from unidataset_IRMv1_3 import UniDataset1
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


def initialize_center(trajs,k):
    indices = np.random.choice(trajs.shape[0],k,replace = False)
    return trajs[indices]

def kmeans(trajs,k,max_iter = 100,tolenrance = 1e-4):
    centroids = initialize_center(trajs,k)
    prev_ventroids = centroids.copy()

    for i in range(max_iter):
        distance = cdist(trajs,centroids,'euclidean')
        
        cluster_labels = np.argmin(distance, axis = 1)

        for idx in range(k):
            if len(trajs[cluster_labels == idx]) > 0:
                centroids[idx] = np.mean(trajs[cluster_labels == idx],axis = 0)

        shift = np.linalg.norm(centroids-prev_ventroids)
        if shift < tolenrance:
            break

        prev_ventroids = centroids.copy()
    
    return cluster_labels , centroids

def plotpic(labels,trajs):
    fig, axes = plt.subplots(1, 1)
    color = ["#415f94","#f98d3a","#5ccac5"]
    for index,(label,traj) in enumerate(zip(labels,trajs)):
        traj = traj.reshape(-1,2)
        plt.plot(traj[:,0],traj[:,1],color[label],zorder=30,linewidth=0.95)
        if index>350:
            break
    plt.xlim(-17,17)
    plt.ylim(-20,45)
    plt.grid()
    plt.show()

    path='./picture/kmeans.png'
    fig.savefig(path,dpi=300)


if __name__=="__main__":
    trSet1=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/MA3s/train/',city="MA")
    trSet2=UniDataset1('/home/lb/Documents/unidataset/lastversion/processed/GL3s/val/',city="GL")

    trajs = []
    deltaxs = []
    deltavs = []
    cnt = 0
    for hist_c, fut_c, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask, motiva_list in trSet1:
        if cnt>300:
            break
        for i in range(hist_c.shape[0]):
            if i not in dest_mask or np.isnan(fut_c[i,-1,0]) or hist_c[i,0,1]==0.0:
                continue
            traj = np.concatenate((hist_c[i],fut_c[i]),axis =0)
            delta_traj = traj[1:,:] - traj[:-1,:]
            deltax = delta_traj[:,0]
            deltav = delta_traj[-1,0]*delta_traj[-1,0]+delta_traj[-1,1]*delta_traj[-1,1]

            trajs.append(traj)
            deltaxs.append(deltax)
            deltavs.append(deltav)
            cnt = cnt+1
    
    cnt =0
    for hist_c, fut_c, num, nbrs, vehtype, op_mask, mapfeats_list, dest_mask, motiva_list in trSet2:
        if cnt>300:
            break
        for i in range(hist_c.shape[0]):
            if i not in dest_mask or np.isnan(fut_c[i,-1,0]) or hist_c[i,0,1]==0.0:
                continue
            traj = np.concatenate((hist_c[i],fut_c[i]),axis =0)
            delta_traj = traj[1:,:] - traj[:-1,:]
            deltax = delta_traj[:,0]
            deltav = delta_traj[-1,0]*delta_traj[-1,0]+delta_traj[-1,1]*delta_traj[-1,1]
            trajs.append(traj)
            deltaxs.append(deltax)
            deltavs.append(deltav)
            cnt =cnt+1

    trajs = np.array(trajs,dtype=float).reshape(-1,50)
    deltaxs = np.array(deltaxs,dtype=float).reshape(-1,24)
    deltavs = np.array(deltavs,dtype=float).reshape(-1,1)
    print(deltavs.shape)
    print(trajs.shape)
    labels,centers = kmeans(deltavs,3)

    plotpic(labels,trajs)
