# -----------------------------
# 1) SHELL COMMANDS (REFERENCE)
# -----------------------------
#
# Build the image (from course repo, in 05-deployment/homework folder):
#   docker build -f Dockerfile_full -t zoomcamp-model:3.13.10-hw10 .
# -----------------------------
# 2) deployment.yaml (Q6)
# -----------------------------
#
# Save everything between the '--- deployment.yaml' markers into a file
# called deployment.yaml, then apply:
#   kubectl apply -f deployment.yaml
#
# --- deployment.yaml ---
apiVersion: apps/v1
kind: Deployment
metadata:
    name: subscription
spec:
    selector:
        matchLabels:
            app: subscription
    replicas: 1
    template:
        metadata:
            labels:
                app: subscription
        spec:
            containers:
                - name: subscription
                  image: zoomcamp-model: 3.13.10-hw10
                   resources:
                        requests:
                            memory: "64Mi"
                            cpu: "100m"
                        limits:
                            memory: "128Mi"
                            cpu: "200m"
                    ports:
                        - containerPort: 9696
# --- end deployment.yaml ---
#
# Notes:
#   - containerPort = 9696  (this is the answer for Q6)
#
# -----------------------------
# 3) service.yaml (Q7)
# -----------------------------
#
# Save everything between the '--- service.yaml' markers into a file
# called service.yaml, then apply:
#   kubectl apply -f service.yaml
#
# --- service.yaml ---
apiVersion: v1
kind: Service
metadata:
    name: subscription-service
spec:
    type: LoadBalancer
    selector:
        app: subscription
    ports:
        - port: 80
         targetPort: 9696
# --- end service.yaml ---
#
# Notes:
#   - selector.app = subscription  (this is the answer for Q7)
#   - Service name can be anything consistent with your port-forward command:
#       kubectl port-forward service/subscription-service 9696:80
