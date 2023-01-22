param (
    # endpoint
    [string]$endpoint = 'local-endpoint',
    
    # deployment
    [string]$deployment = 'blue'
)

az ml online-endpoint create --local -n $endpoint -f endpoint.yml --debug
az ml online-deployment create --local -n $deployment --endpoint $endpoint -f deployment.yml --debug
copy ../../score.py ./score.py
az ml online-endpoint show -n $endpoint --local