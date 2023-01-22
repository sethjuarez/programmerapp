param (
    # model name
    [string]$model = 'dev-model',
    
    # runid
    [string]$runid = 'runid',

    # model version
    [int]$version = 1
)

az ml model create --name $model --version $version --path runs:/$runid/outputs/model