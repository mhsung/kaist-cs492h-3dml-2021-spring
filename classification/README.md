## CS492(H): Machine Learning for 3D Data — PointNet Classification

Minhyuk Sung (mhsung@kaist.ac.kr)<br>
Last Updated: Mar 14, 2021.

### 0. Setup JupyterLab.

Check out the KCloud setup instruction and see whether you can run JupyterLab.


### 1. Download ModelNet40 data.

Open a terminal and login to the KCloud.

Run:
```
cd ~/
mkdir classification && cd classification
mkdir data && cd data
mkdir ModelNet && cd ModelNet
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1zE1d_eYD_QEnmS01LlZlEOMSZTIXRwIA" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1zE1d_eYD_QEnmS01LlZlEOMSZTIXRwIA" -o modelnet_classification.h5
md5sum modelnet_classification.h5
```

The output should be exactly the same as this:
```
87e763a66819066da670053a360889ed  modelnet_classification.h5
```

### 2. Download PoineNet classification code and run.

Run:
```
cd ~/classification
wget https://raw.githubusercontent.com/mhsung/kaist-cs492h-3dml-2021-spring/main/classification/classification.ipynb?token=ABNDQXIHJZ27EDMTZAIJNMDAK335E -O classification.ipynb
wget https://raw.githubusercontent.com/mhsung/kaist-cs492h-3dml-2021-spring/main/classification/classification.py?token=ABNDQXOWO3E4IVRDGZHBU7LAK337Y -O classification.py
wget https://raw.githubusercontent.com/mhsung/kaist-cs492h-3dml-2021-spring/main/classification/pointnet.py?token=ABNDQXLWIEEI3TKDNSILIUDAK34CA -O pointnet.py
```

In the `~/classification` directory, run JupyterLab:
```
jupyter lab --port={PORT_ID} --no-browser
```

In JupyterLab, open 'classification.ipynb', and find the following two lines:
```
    'train': False,
```
```
    'model': 'outputs/model_50.pth',
```

Change the two lines like these:
```
    'train': True,
```
```
    'model': '',
```

Press the re-run button (<i class="fa fa-coffee"></i>) at the top. This will train the PointNet classification code with the ModelNet40 data.

Revert the above two lines, and re-run. This will test the trained PointNet classification model. Check whether the test accuracy is the same as the training result.
