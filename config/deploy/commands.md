# Endpoint Creation

az ml online-endpoint create -n dev-endpoint -f endpoint.yml --debug

# Deployment Creation
Make sure to move score.py

az ml online-deployment create -n green --endpoint dev-endpoint -f deployment.yml --debug