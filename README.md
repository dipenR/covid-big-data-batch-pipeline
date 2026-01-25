# covid-big-data-batch-pipeline

## Prerequisites
### 1. Ensure Python and PIP are installed
To do this run `python --version` and `pip --version`.
Then run `pip install awscli`.
### 1.5 Troubleshooting
If python is install but pip isn't `python -m pip install ...` works too. The issue is most likely your path variables not being set up or being set up incorrectly.
### 2. Setup AWS
1. verify that aws was installed using aws --version, then run `aws configure`. 
```
   aws configure
   AWS Access Key ID: [paste key]
   AWS Secret Access Key: [paste secret]
   Default region: us-east-1
   Default output format: json
```
2. Verify access:
```
   aws s3 ls
```

3. Copy config template
```
   cp config.yaml.template config.yaml
```

4. Use values from config

```
import os

BUCKET_NAME = os.environ.get('RAW_DATA_BUCKET', 'default_bucket_name')
REGION = os.environ.get('AWS_REGION', 'us-east-1')
```
## Important
- Never commit credentials to GitHub
- Never share credentials in public channels