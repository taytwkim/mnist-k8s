# MNIST on Docker + K8s

Train an MNIST model in Docker and serve predictions with a Flask app deployed on Kubernetes (GKE).

## Directory
* `train/`: training code and `Dockerfile` for the MNIST training job.
* `app/`: inference/Flask code and `Dockerfile` for serving predictions.
* `k8s/`: Kubernetes YAML manifests (PVC, training Job, app Deployment, Service).
* `model.py`: shared model architecture used by both training and inference.
* `index.html`: simple frontend UI that sends requests to the Flask `/predict/<id>` endpoint.

## Workflow

### 1. Build images and push to Dockerhub

Use docker `buildx` to build `linux/amd64` images (since I’m on an ARM Mac) and push them.

```bash!
docker buildx build \
  --platform linux/amd64 \
  -f train/Dockerfile \
  -t taytwkim/mnist-train:v2 \
  --push .
```

```bash!
docker buildx build \
  --platform linux/amd64 \
  -f app/Dockerfile \
  -t taytwkim/mnist-app:v2 \
  --push .
```

### 2. GKE cluster

* GCP Set Up
```bash!
gcloud auth login
gcloud projects list
gcloud config set project YOUR_PROJECT_ID
gcloud services enable container.googleapis.com
```

* Create a GKE Cluster
```bash!
gcloud container clusters create-auto mnist-cluster --region us-central1
```

* Hook up `kubectl`

`kubectl` is the command-line client for Kubernetes that runs locally. Fetch connection/auth info from GKE and update `kubeconfig`, so `kubectl` can talk to the cluster.
```bash!
gcloud container clusters get-credentials mnist-cluster --region us-central1
kubectl get nodes
```

### 3. PVC

A PVC (PersistentVolumeClaim) is a request for persistent storage in the cluster. It gets bound to a PersistentVolume, and any pods that mount that PVC share the same underlying storage.

```bash!
kubectl apply -f k8s/pvc.yaml
kubectl get pvc
```

### 4. Training

Training Job runs a container to train the model and save its weights to the shared PVC. When training  finishes, the container exits and the pod goes to completed, but the saved weights stay in the shared volume.

```bash!
kubectl apply -f k8s/train-job.yaml
kubectl get pods
kubectl logs -f job/mnist-train-job
```

### 5. Run app

Runs the Flask app pods, but not exposed externally.

```bash!
kubectl apply -f k8s/app-deploy.yaml
kubectl get pods
```

### 6. Expose app

Exposes the Flask app outside the cluster by giving it a stable IP and port.

```bash!
kubectl apply -f k8s/app-service.yaml
kubectl get svc
```

Once `EXTERNAL-IP` is sent, send requests.
```bash!
curl http://EXTERNAL-IP/predict/0
```

### 7. Clean up Deployment and Service

* Stop the running app/LB
```bash!
kubectl delete service mnist-app-svc
kubectl delete deployment mnist-app-deploy
```

* Optionally, clean traning job and PVC
```bash!
kubectl delete job mnist-train-job --ignore-not-found
kubectl delete pvc mnist-artifacts-pvc --ignore-not-found
```

* Optionally, delete the whole cluster.
```bash!
gcloud container clusters delete mnist-cluster --region us-central1
```
