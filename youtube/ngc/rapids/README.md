# Commands to Setup (see YouTube Video)

```
sudo lsblk
sudo file -s /dev/xvdf
sudo mkfs -t xfs /dev/xvdf
sudo mount /dev/xvdf /mnt/data
sudo chown ubuntu /mnt/data

aws s3 cp s3://jheaton-load-data /mnt/data/kaggle --recursive --exclude "*" --include "*.csv"

docker run -it --rm --gpus all -p 8888:8888 -v /mnt/data:/rapids/notebooks/host 709825985650.dkr.ecr.us-east-1.amazonaws.com/nvidia/containers/nvidia/rapidsai/rapidsai:0.17-cuda11.0-runtime-ubuntu18.04

import cudf df_local = cudf.read_csv('/data/sample.csv') df_remote = cudf.read_csv( 's3://<bucket>/sample.csv' , storage_options = {'anon': True})
```